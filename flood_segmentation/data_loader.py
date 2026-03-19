import os
import json
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# --- SUPPRESS RASTERIO WARNINGS ---
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message=".*CPLE_AppDefined.*")
warnings.filterwarnings("ignore")
logging.getLogger("rasterio").setLevel(logging.ERROR)

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class FloodDataset(Dataset):
    def __init__(self, root, phase, split):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.split = split
        
        # Paths
        self.metadata_csv = self.root / "data_analysis" / "dataset_metadata.csv"
        self.split_txt = self.root / "data_analysis" / f"splits_{self.split}.txt"
        self.norm_json = self.root / "data_analysis" / "normalization_stats.json"
        
        # Configuration rules
        assert self.phase in ["pretrain", "finetune"], "phase must be 'pretrain' or 'finetune'"
        assert self.split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"
        
        # Load constraints
        self._load_metadata()
        self._compute_or_load_normalization()
        
        # Summary
        logger.info(f"Initialized Dataset - Phase: {self.phase}, Split: {self.split}")
        logger.info(f"Total tiles: {len(self.tiles)}, Hand: {self.hand_count}, Weak: {self.weak_count}")
        if hasattr(self, 'missing_modality_count') and self.missing_modality_count > 0:
            logger.warning(f"Tiles with >=1 missing/corrupt modality file: {self.missing_modality_count} (will be zero-filled)")

    def _load_metadata(self):
        try:
            df = pd.read_csv(self.metadata_csv)
            with open(self.split_txt, "r") as f:
                split_ids = set([line.strip() for line in f.readlines() if line.strip()])
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata or split file: {e}")

        # Filter by split
        df = df[df['tile_id'].isin(split_ids)].copy()
        
        # Validation/Test always uses ONLY hand-labeled tiles
        if self.split in ["val", "test"]:
            df = df[df['has_hand_label'] == True] if 'has_hand_label' in df.columns else df[df['has_label_hand']]
        
        # Finetune always uses ONLY hand-labeled tiles
        if self.phase == "finetune":
            df = df[df['has_hand_label'] == True] if 'has_hand_label' in df.columns else df[df['has_label_hand']]
            
        self.tiles = []
        self.hand_count = 0
        self.weak_count = 0
        
        # Column names fallback
        col_hand_lbl = 'has_hand_label' if 'has_hand_label' in df.columns else 'has_label_hand'
        
        for _, row in df.iterrows():
            tid = row['tile_id']
            is_hand = row[col_hand_lbl]
            
            # Extract paths handling NaNs
            s1_hand_path = row.get('s1_hand_path', "")
            s2_hand_path = row.get('s2_hand_path', "")
            s1_weak_path = row.get('s1_weak_path', "")
            s2_weak_path = row.get('s2_weak_path', "")
            lbl_hand_path = row.get('label_hand_path', "")
            s2_weak_label_path = row.get('s2_weak_label_path', "")
            pre_s1_path = row.get('pre_s1_path', "")
            pre_s2_path = row.get('pre_s2_path', "")
            aux_path = row.get('aux_path', "")
            
            # Path logic
            p_s1 = s1_hand_path if pd.notna(s1_hand_path) and s1_hand_path != "" else s1_weak_path
            p_s2 = s2_hand_path if pd.notna(s2_hand_path) and s2_hand_path != "" else s2_weak_path
            
            lbl_hand_val = lbl_hand_path if pd.notna(lbl_hand_path) else ""
            s2_weak_lbl_val = s2_weak_label_path if pd.notna(s2_weak_label_path) else ""
            
            if self.phase == "pretrain" and self.split == "train":
                p_lbl = lbl_hand_val if is_hand else s2_weak_lbl_val
            else:
                p_lbl = lbl_hand_val
            
            if is_hand:
                self.hand_count += 1
            else:
                self.weak_count += 1
                
            self.tiles.append({
                'tile_id': tid,
                'has_hand_label': is_hand,
                'p_s1': p_s1,
                'p_s2': p_s2,
                'pre_s1': pre_s1_path,
                'pre_s2': pre_s2_path,
                'aux': aux_path,
                'p_lbl': p_lbl
            })

    def _compute_or_load_normalization(self):
        if self.norm_json.exists():
            with open(self.norm_json, "r") as f:
                self.norm_stats = json.load(f)
            return

        assert self.split == "train", "Normalization stats missing! Run training set init to generate them before val/test."
        
        logger.info("Computing normalization statistics from training set...")
        stats = {
            's1': {'mean': np.zeros(2).tolist(), 'std': np.ones(2).tolist()},
            's2': {'mean': np.zeros(6).tolist(), 'std': np.ones(6).tolist()},
            'pre_s1': {'mean': np.zeros(2).tolist(), 'std': np.ones(2).tolist()},
            'pre_s2': {'mean': np.zeros(6).tolist(), 'std': np.ones(6).tolist()}
        }
        
        # Accumulators
        def get_sums(tiles, key, num_c):
            sums = np.zeros(num_c)
            sq_sums = np.zeros(num_c)
            counts = np.zeros(num_c)
            for t in tiles:
                p = t[key]
                if not pd.notna(p) or p == "": continue
                path = self.root / p
                if not path.exists(): continue
                try:
                    with rasterio.open(path) as src:
                        chans = min(src.count, num_c)
                        for c in range(chans):
                            d = src.read(c+1).flatten()
                            # Center crop equivalent logic for sums to match train time
                            if d.size > 512*512:
                                h, w = src.height, src.width
                                d = np.reshape(d, (h, w))[h//2-256:h//2+256, w//2-256:w//2+256].flatten()
                            elif d.size < 512*512:
                                continue # bad tile
                            
                            valid = np.isfinite(d)
                            d = d[valid].astype(np.float64)
                            sums[c] += np.sum(d)
                            sq_sums[c] += np.sum(d**2)
                            counts[c] += d.size
                except:
                    pass
            
            mean = sums / np.maximum(counts, 1)
            var = (sq_sums / np.maximum(counts, 1)) - (mean**2)
            std = np.sqrt(np.maximum(var, 1e-6))
            return mean.tolist(), std.tolist()

        stats['s1']['mean'], stats['s1']['std'] = get_sums(self.tiles, 'p_s1', 2)
        stats['s2']['mean'], stats['s2']['std'] = get_sums(self.tiles, 'p_s2', 6)
        stats['pre_s1']['mean'], stats['pre_s1']['std'] = get_sums(self.tiles, 'pre_s1', 2)
        stats['pre_s2']['mean'], stats['pre_s2']['std'] = get_sums(self.tiles, 'pre_s2', 6)

        with open(self.norm_json, "w") as f:
            json.dump(stats, f, indent=4)
        self.norm_stats = stats

    def _center_crop(self, array, target_h=512, target_w=512):
        # array can be 2D (H,W) or 3D (C,H,W)
        shape = array.shape
        if len(shape) == 3:
            c, h, w = shape
            if h == target_h and w == target_w: return array
            cy, cx = h // 2, w // 2
            return array[:, cy-256:cy+256, cx-256:cx+256]
        elif len(shape) == 2:
            h, w = shape
            if h == target_h and w == target_w: return array
            cy, cx = h // 2, w // 2
            return array[cy-256:cy+256, cx-256:cx+256]
        return array

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        tid = tile['tile_id']
        
        # Create output tensor (20, 512, 512) and label (512, 512)
        img = np.zeros((20, 512, 512), dtype=np.float32)
        lbl = np.full((512, 512), -1, dtype=np.int64)
        
        def read_raster(path_key, num_chans, crop=False, output_slice=None):
            p = tile[path_key]
            if pd.notna(p) and p != "":
                full_path = self.root / p
                if full_path.exists():
                    try:
                        with rasterio.open(full_path) as src:
                            read_idx = list(range(1, min(src.count + 1, num_chans + 1)))
                            if len(read_idx) == 1:
                                data = src.read(read_idx[0])[np.newaxis, ...]
                            else:
                                data = src.read(read_idx) # (C, H, W)
                                
                            if data.shape[0] < num_chans:
                                # zero-pad missing channels to required num
                                pad = np.zeros((num_chans - data.shape[0], data.shape[1], data.shape[2]))
                                data = np.concatenate([data, pad], axis=0)
                            elif data.shape[0] > num_chans:
                                data = data[:num_chans]
                            
                            if crop:
                                data = self._center_crop(data)
                            
                            if data.shape[1] != 512 or data.shape[2] != 512:
                                raise ValueError(f"Shape {data.shape} after crop!")
                                
                            if output_slice is not None:
                                img[output_slice] = data
                            return data
                    except Exception as e:
                        pass
            return None

        # 0-1: Event S1
        s1 = read_raster('p_s1', 2, crop=False, output_slice=slice(0, 2))
        if s1 is None:
            logger.warning(f"Tile {tid} lacks S1, zero-filling S1 channels")
            img[0:2] = np.nan
        
        # 2-7: Event S2 (bands 0-5)
        s2 = read_raster('p_s2', 6, crop=False, output_slice=slice(2, 8))
        if s2 is None:
            logger.warning(f"Tile {tid} lacks S2, zero-filling S2 channels")
            img[2:8] = np.nan

        # 8-9: Pre S1
        ps1 = read_raster('pre_s1', 2, crop=True, output_slice=slice(8, 10))
        if ps1 is None:
            logger.warning(f"Tile {tid} lacks pre_s1, zero-filling channels 8-9")
            img[8:10] = np.nan

        # 10-15: Pre S2
        ps2 = read_raster('pre_s2', 6, crop=True, output_slice=slice(10, 16))
        if ps2 is None:
            logger.warning(f"Tile {tid} lacks pre_s2, zero-filling channels 10-15")
            img[10:16] = np.nan

        # 16-19: Aux
        aux = read_raster('aux', 4, crop=True, output_slice=slice(16, 20))
        if aux is None:
            logger.warning(f"Tile {tid} lacks aux, zero-filling channels 16-19")
            img[16:20] = np.nan

        # (shape is guaranteed by construction - all channels explicitly filled above)

        # Label resolving
        p_lbl = tile['p_lbl']
        if pd.notna(p_lbl) and p_lbl != "":
            full_lbl = self.root / p_lbl
            if full_lbl.exists():
                try:
                    with rasterio.open(full_lbl) as src:
                        l = self._center_crop(src.read(1))
                        if l.shape == (512, 512):
                            lbl = l.astype(np.int64)
                            # Keep 0 as 0, 1 as 1, anything else to -1
                            invalid_mask = (lbl != 0) & (lbl != 1)
                            lbl[invalid_mask] = -1
                except Exception as e:
                    logger.warning(f"Tile {tid} label read error: {e}, leaving label as -1")

        # Normalization
        # S1 (0-1)
        for i in range(2):
            m, s = self.norm_stats['s1']['mean'][i], self.norm_stats['s1']['std'][i]
            img[i] = (img[i] - m) / s
        
        # S2 (2-7)
        for i in range(6):
            m, s = self.norm_stats['s2']['mean'][i], self.norm_stats['s2']['std'][i]
            img[i+2] = (img[i+2] - m) / s
            
        # Pre S1 (8-9)
        for i in range(2):
            m, s = self.norm_stats['pre_s1']['mean'][i], self.norm_stats['pre_s1']['std'][i]
            img[i+8] = (img[i+8] - m) / s
            
        # Pre S2 (10-15)
        for i in range(6):
            m, s = self.norm_stats['pre_s2']['mean'][i], self.norm_stats['pre_s2']['std'][i]
            img[i+10] = (img[i+10] - m) / s

        # Aux (16-19)
        # JRC is 0-100, normalize to 0-1
        img[16:20] = img[16:20] / 100.0

        # Replace remaining NaNs/Infs with 0 AFTER normalization
        valid_mask = np.isfinite(img)
        img[~valid_mask] = 0.0

        # Augmentations (Train Only)
        if self.split == "train":
            # H-flip
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            # V-flip
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[::-1, :].copy()
            # Rot90
            if random.random() < 0.5:
                k = random.choice([0, 1, 2, 3])
                img = np.rot90(img, k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k, axes=(0, 1)).copy()
            
            # Radiometric Jitter S1 Only
            if random.random() < 0.3:
                img[0] += np.random.normal(0, 0.1, (512, 512))
                img[1] += np.random.normal(0, 0.1, (512, 512))
                img[8] += np.random.normal(0, 0.1, (512, 512))
                img[9] += np.random.normal(0, 0.1, (512, 512))

        return {
            'image': torch.from_numpy(img).float(),
            'label': torch.from_numpy(lbl).long(),
            'tile_id': tid,
            'split': self.split,
            'phase': self.phase,
            'has_hand_label': tile['has_hand_label']
        }

def get_dataloaders(root, phase, batch_size=4):
    train_dataset = FloodDataset(root, phase, "train")
    val_dataset = FloodDataset(root, phase, "val")
    test_dataset = FloodDataset(root, phase, "test")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, test_loader

# --- VALIDATION BLOCK ---
if __name__ == '__main__':
    logging.info("Starting Dataloader Validation Script...")
    root_dir = "data"
    # Wait, testing dataloader root definition is the project root containing 'data/'. 
    # Provided root usually 'data' does not contain 'data/data_analysis', it should be `./` assuming cwd.
    # The requirement is "dataset root is at data/ relative to the script". 
    # Metadata CSV is "data_analysis/dataset_metadata.csv" relative to script.
    # So `root` is `.` (the project root).
    project_root = "."

    phases = ["pretrain", "finetune"]
    
    for p in phases:
        print(f"\n{'='*50}\nValidating Phase: {p.upper()}\n{'='*50}")
        tr_loader, vl_loader, ts_loader = get_dataloaders(project_root, p, batch_size=2)
        
        loaders_to_test = [("train", tr_loader), ("val", vl_loader)]
        for s_name, loader in loaders_to_test:
            print(f"--- Loading one batch from {s_name} split ---")
            batch = next(iter(loader))
            
            img = batch['image']
            lbl = batch['label']
            tids = batch['tile_id']
            phase = batch['phase']
            split = batch['split']
            
            print(f"Image Tensor: Shape={img.shape}, dtype={img.dtype}")
            print(f"Label Tensor: Shape={lbl.shape}, dtype={lbl.dtype}")
            print(f"Image Values: Min={img.min():.4f}, Max={img.max():.4f}")
            print(f"Label Unique Values: {torch.unique(lbl).tolist()}")
            print(f"Batch Tile IDs: {tids}")
            print(f"Phase: {phase}, Split: {split}")
            
            # Asserts
            if torch.isnan(img).any() or torch.isinf(img).any():
                raise AssertionError(f"NaN or Inf found in image tensor for tiles: {tids}")
            
            allowed_labels = set([-1, 0, 1])
            actual_labels = set(torch.unique(lbl).tolist())
            if not actual_labels.issubset(allowed_labels):
                raise AssertionError(f"Invalid label values found: {actual_labels} (Only -1, 0, 1 allowed)")
                
            assert img.shape == (2, 20, 512, 512), f"Image tensor shape invalid!"
            assert lbl.shape == (2, 512, 512), f"Label tensor shape invalid!"

    print("\n" + "="*50)
    print("DATALOADER VALIDATION PASSED")
    print("="*50)
