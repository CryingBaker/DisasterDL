import os
import json
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
        self.split_txt    = self.root / "data_analysis" / f"splits_{self.split}.txt"
        self.norm_json    = self.root / "data_analysis" / "normalization_stats.json"

        assert self.phase in ["pretrain", "finetune"], "phase must be 'pretrain' or 'finetune'"
        assert self.split in ["train", "val", "test"],  "split must be 'train', 'val', or 'test'"

        self._load_metadata()
        self._compute_or_load_normalization()

        logger.info(f"Initialized Dataset - Phase: {self.phase}, Split: {self.split}")
        logger.info(f"Total tiles: {len(self.tiles)}, Hand: {self.hand_count}, Weak: {self.weak_count}")

    # ------------------------------------------------------------------
    # METADATA LOADING
    # ------------------------------------------------------------------
    def _load_metadata(self):
        try:
            df = pd.read_csv(self.metadata_csv)
            with open(self.split_txt, "r") as f:
                split_ids = set(l.strip() for l in f if l.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata or split file: {e}")

        df = df[df['tile_id'].isin(split_ids)].copy()

        col_hand_lbl = 'has_hand_label' if 'has_hand_label' in df.columns else 'has_label_hand'

        # Val/Test + finetune → hand-labeled only
        if self.split in ["val", "test"] or self.phase == "finetune":
            df = df[df[col_hand_lbl] == True]

        self.tiles      = []
        self.hand_count = 0
        self.weak_count = 0

        for _, row in df.iterrows():
            tid      = row['tile_id']
            is_hand  = bool(row[col_hand_lbl])

            def _get(col):
                v = row.get(col, "")
                return v if pd.notna(v) else ""

            s1_hand_path       = _get('s1_hand_path')
            s2_hand_path       = _get('s2_hand_path')
            s1_weak_path       = _get('s1_weak_path')
            s2_weak_path       = _get('s2_weak_path')
            lbl_hand_path      = _get('label_hand_path')
            s2_weak_label_path = _get('s2_weak_label_path')
            pre_s1_path        = _get('pre_s1_path')
            pre_s2_path        = _get('pre_s2_path')
            aux_path           = _get('aux_path')

            p_s1 = s1_hand_path if s1_hand_path else s1_weak_path
            p_s2 = s2_hand_path if s2_hand_path else s2_weak_path

            if self.phase == "pretrain" and self.split == "train":
                p_lbl = lbl_hand_path if is_hand else s2_weak_label_path
            else:
                p_lbl = lbl_hand_path

            if is_hand:
                self.hand_count += 1
            else:
                self.weak_count += 1

            self.tiles.append({
                'tile_id':       tid,
                'has_hand_label': is_hand,
                'p_s1':          p_s1,
                'p_s2':          p_s2,
                'pre_s1':        pre_s1_path,
                'pre_s2':        pre_s2_path,
                'aux':           aux_path,
                'p_lbl':         p_lbl,
            })

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------
    def _compute_or_load_normalization(self):
        if self.norm_json.exists():
            with open(self.norm_json, "r") as f:
                self.norm_stats = json.load(f)
            return

        assert self.split == "train", \
            "Normalization stats missing! Init training set first."

        logger.info("Computing normalization statistics from training set...")

        def get_sums(tiles, key, num_c):
            sums    = np.zeros(num_c)
            sq_sums = np.zeros(num_c)
            counts  = np.zeros(num_c)
            for t in tiles:
                p = t[key]
                if not p:
                    continue
                path = self.root / p
                if not path.exists():
                    continue
                try:
                    with rasterio.open(path) as src:
                        chans = min(src.count, num_c)
                        for c in range(chans):
                            d = src.read(c + 1).flatten()
                            if d.size > 512 * 512:
                                h, w = src.height, src.width
                                d = np.reshape(d, (h, w))[
                                    h // 2 - 256:h // 2 + 256,
                                    w // 2 - 256:w // 2 + 256,
                                ].flatten()
                            elif d.size < 512 * 512:
                                continue
                            valid = np.isfinite(d)
                            d = d[valid].astype(np.float64)
                            sums[c]    += d.sum()
                            sq_sums[c] += (d ** 2).sum()
                            counts[c]  += d.size
                except Exception:
                    pass
            mean = sums / np.maximum(counts, 1)
            var  = (sq_sums / np.maximum(counts, 1)) - mean ** 2
            std  = np.sqrt(np.maximum(var, 1e-6))
            return mean.tolist(), std.tolist()

        stats = {
            's1':     {'mean': np.zeros(2).tolist(), 'std': np.ones(2).tolist()},
            's2':     {'mean': np.zeros(6).tolist(), 'std': np.ones(6).tolist()},
            'pre_s1': {'mean': np.zeros(2).tolist(), 'std': np.ones(2).tolist()},
            'pre_s2': {'mean': np.zeros(6).tolist(), 'std': np.ones(6).tolist()},
        }
        stats['s1']['mean'],     stats['s1']['std']     = get_sums(self.tiles, 'p_s1',   2)
        stats['s2']['mean'],     stats['s2']['std']     = get_sums(self.tiles, 'p_s2',   6)
        stats['pre_s1']['mean'], stats['pre_s1']['std'] = get_sums(self.tiles, 'pre_s1', 2)
        stats['pre_s2']['mean'], stats['pre_s2']['std'] = get_sums(self.tiles, 'pre_s2', 6)

        with open(self.norm_json, "w") as f:
            json.dump(stats, f, indent=4)
        self.norm_stats = stats

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _center_crop(self, array, target_h=512, target_w=512):
        if len(array.shape) == 3:
            c, h, w = array.shape
            if h == target_h and w == target_w:
                return array
            cy, cx = h // 2, w // 2
            return array[:, cy - 256:cy + 256, cx - 256:cx + 256]
        else:
            h, w = array.shape
            if h == target_h and w == target_w:
                return array
            cy, cx = h // 2, w // 2
            return array[cy - 256:cy + 256, cx - 256:cx + 256]

    def flood_pixel_fraction(self, idx):
        """Return estimated flood fraction for a tile (used for oversampling weights)."""
        tile = self.tiles[idx]
        p_lbl = tile['p_lbl']
        if not p_lbl:
            return 0.0
        full_lbl = self.root / p_lbl
        if not full_lbl.exists():
            return 0.0
        try:
            with rasterio.open(full_lbl) as src:
                lbl = self._center_crop(src.read(1))
                valid = (lbl == 0) | (lbl == 1)
                if valid.sum() == 0:
                    return 0.0
                return float((lbl[valid] == 1).mean())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # DATASET ITEM
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        tid  = tile['tile_id']

        img = np.zeros((20, 512, 512), dtype=np.float32)
        lbl = np.full((512, 512), -1, dtype=np.int64)

        def read_raster(path_key, num_chans, crop=False, output_slice=None):
            p = tile[path_key]
            if p:
                full_path = self.root / p
                if full_path.exists():
                    try:
                        with rasterio.open(full_path) as src:
                            read_idx = list(range(1, min(src.count + 1, num_chans + 1)))
                            if len(read_idx) == 1:
                                data = src.read(read_idx[0])[np.newaxis, ...]
                            else:
                                data = src.read(read_idx)

                            if data.shape[0] < num_chans:
                                pad  = np.zeros((num_chans - data.shape[0], data.shape[1], data.shape[2]))
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
                    except Exception:
                        pass
            return None

        # --- Load modalities ---
        if read_raster('p_s1',   2, crop=False, output_slice=slice(0, 2))   is None: img[0:2]   = np.nan
        if read_raster('p_s2',   6, crop=False, output_slice=slice(2, 8))   is None: img[2:8]   = np.nan
        if read_raster('pre_s1', 2, crop=True,  output_slice=slice(8, 10))  is None: img[8:10]  = np.nan
        if read_raster('pre_s2', 6, crop=True,  output_slice=slice(10, 16)) is None: img[10:16] = np.nan
        if read_raster('aux',    4, crop=True,  output_slice=slice(16, 20)) is None: img[16:20] = np.nan

        # --- Label ---
        p_lbl = tile['p_lbl']
        if p_lbl:
            full_lbl = self.root / p_lbl
            if full_lbl.exists():
                try:
                    with rasterio.open(full_lbl) as src:
                        l = self._center_crop(src.read(1))
                        if l.shape == (512, 512):
                            lbl = l.astype(np.int64)
                            invalid_mask = (lbl != 0) & (lbl != 1)
                            lbl[invalid_mask] = -1
                except Exception as e:
                    logger.warning(f"Tile {tid} label read error: {e}")

        # --- Normalization ---
        for i in range(2):
            m, s = self.norm_stats['s1']['mean'][i], self.norm_stats['s1']['std'][i]
            img[i] = (img[i] - m) / s
        for i in range(6):
            m, s = self.norm_stats['s2']['mean'][i], self.norm_stats['s2']['std'][i]
            img[i + 2] = (img[i + 2] - m) / s
        for i in range(2):
            m, s = self.norm_stats['pre_s1']['mean'][i], self.norm_stats['pre_s1']['std'][i]
            img[i + 8] = (img[i + 8] - m) / s
        for i in range(6):
            m, s = self.norm_stats['pre_s2']['mean'][i], self.norm_stats['pre_s2']['std'][i]
            img[i + 10] = (img[i + 10] - m) / s

        # Aux: JRC-style channels → 0-1
        img[16:20] = img[16:20] / 100.0

        # Replace NaN/Inf with 0
        img[~np.isfinite(img)] = 0.0

        # --- Augmentations (train only) ---
        if self.split == "train":
            # Geometric
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            if random.random() < 0.5:
                k = random.randint(0, 3)
                img = np.rot90(img, k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k, axes=(0, 1)).copy()

            # --- FLOOD-SPECIFIC AUGMENTATION ---
            # For flood tiles: extra copy-paste of flood region onto itself
            has_flood = (lbl == 1).any()
            if has_flood and random.random() < 0.5:
                flood_mask = (lbl == 1)
                # Random shift within ±64 px and paste flood region back
                dy = random.randint(-64, 64)
                dx = random.randint(-64, 64)
                shifted_mask = np.roll(np.roll(flood_mask, dy, axis=0), dx, axis=1)
                # Paste flood pixels from original into shifted location
                img[:, shifted_mask] = img[:, flood_mask]
                lbl[shifted_mask] = 1

            # Radiometric jitter (S1 channels only)
            if random.random() < 0.3:
                for ch in [0, 1, 8, 9]:
                    img[ch] += np.random.normal(0, 0.1, (512, 512))

            # Random channel dropout (zero out one full channel to force multi-modal robustness)
            if random.random() < 0.15:
                drop_ch = random.choice(list(range(20)))
                img[drop_ch] = 0.0

        return {
            'image':          torch.from_numpy(img).float(),
            'label':          torch.from_numpy(lbl).long(),
            'tile_id':        tid,
            'split':          self.split,
            'phase':          self.phase,
            'has_hand_label': tile['has_hand_label'],
        }


# ------------------------------------------------------------------
# WEIGHTED SAMPLER: oversample flood-heavy tiles
# ------------------------------------------------------------------
def build_flood_sampler(dataset, flood_multiplier=4.0):
    """
    Give higher sampling weight to tiles that contain flood pixels.
    flood_multiplier: how many times more likely a flood tile is sampled vs a no-flood tile.
    Reads labels once — slow on first call but cached implicitly via the dataset.
    """
    logger.info("Building WeightedRandomSampler (reading flood fractions)...")
    weights = []
    for idx in range(len(dataset)):
        frac = dataset.flood_pixel_fraction(idx)
        # Any tile with ≥1% flood pixels gets the boost
        w = flood_multiplier if frac >= 0.01 else 1.0
        weights.append(w)
    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    logger.info(
        f"Sampler built: {int((torch.tensor(weights) == flood_multiplier).sum())} flood tiles "
        f"vs {int((torch.tensor(weights) == 1.0).sum())} non-flood tiles"
    )
    return sampler


def get_dataloaders(root, phase, batch_size=4, use_flood_sampler=True):
    train_dataset = FloodDataset(root, phase, "train")
    val_dataset   = FloodDataset(root, phase, "val")
    test_dataset  = FloodDataset(root, phase, "test")

    # NOTE: num_workers=0 is safest on MPS/Mac due to rasterio + multiprocessing issues.
    # Set to 2 only if you're on Linux. Keep 0 on macOS.
    nw = 2

    if use_flood_sampler and phase in ["pretrain", "finetune"]:
        sampler = build_flood_sampler(train_dataset, flood_multiplier=4.0)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=sampler,        # sampler replaces shuffle=True
            drop_last=True, num_workers=nw, pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, drop_last=True, num_workers=nw, pin_memory=False,
            multiprocessing_context="spawn",
        )

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=nw, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=nw, pin_memory=False)

    return train_loader, val_loader, test_loader


# --- VALIDATION BLOCK ---
if __name__ == '__main__':
    project_root = "."
    phases = ["pretrain", "finetune"]

    for p in phases:
        print(f"\n{'='*50}\nValidating Phase: {p.upper()}\n{'='*50}")
        tr_loader, vl_loader, ts_loader = get_dataloaders(project_root, p, batch_size=2)

        for s_name, loader in [("train", tr_loader), ("val", vl_loader)]:
            print(f"--- Loading one batch from {s_name} split ---")
            batch = next(iter(loader))

            img = batch['image']
            lbl = batch['label']

            print(f"Image Tensor: Shape={img.shape}, dtype={img.dtype}")
            print(f"Label Tensor: Shape={lbl.shape}, dtype={lbl.dtype}")
            print(f"Image Values: Min={img.min():.4f}, Max={img.max():.4f}")
            print(f"Label Unique Values: {torch.unique(lbl).tolist()}")

            if torch.isnan(img).any() or torch.isinf(img).any():
                raise AssertionError(f"NaN or Inf in image tensor!")

            actual_labels = set(torch.unique(lbl).tolist())
            if not actual_labels.issubset({-1, 0, 1}):
                raise AssertionError(f"Invalid labels: {actual_labels}")

            assert img.shape == (2, 20, 512, 512)
            assert lbl.shape == (2, 512, 512)

    print("\n" + "="*50)
    print("DATALOADER VALIDATION PASSED")
    print("="*50)