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

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message=".*CPLE_AppDefined.*")
warnings.filterwarnings("ignore")
logging.getLogger("rasterio").setLevel(logging.ERROR)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class FloodDataset(Dataset):
    """
    Standard supervised split: uses ONLY hand-labeled tiles.
    """
    def __init__(self, root, split):
        super().__init__()
        self.root  = Path(root)
        self.split = split

        self.metadata_csv = self.root / "data_analysis" / "dataset_metadata.csv"
        self.split_txt    = self.root / "data_analysis" / f"splits_{self.split}.txt"
        self.norm_json    = self.root / "data_analysis" / "normalization_stats.json"

        assert self.split in ["train", "val", "test"]

        self._load_metadata()
        self._compute_or_load_normalization()

        logger.info(
            f"FloodDataset [{self.split}]: "
            f"{len(self.tiles)} tiles (100% hand-labeled)"
        )

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    def _load_metadata(self):
        try:
            df = pd.read_csv(self.metadata_csv)
            with open(self.split_txt, "r") as f:
                split_ids = set(l.strip() for l in f if l.strip())
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata or split file: {e}")

        df  = df[df['tile_id'].isin(split_ids)].copy()
        col = 'has_hand_label' if 'has_hand_label' in df.columns else 'has_label_hand'

        # STRICT ENFORCEMENT: Only keep hand-labeled data
        df = df[df[col] == True]

        self.tiles = []
        for _, row in df.iterrows():
            def _get(c):
                v = row.get(c, "")
                return v if pd.notna(v) else ""

            # S1/S2: prefer hand path, fall back to weak path if hand missing
            p_s1 = _get('s1_hand_path') or _get('s1_weak_path')
            p_s2 = _get('s2_hand_path') or _get('s2_weak_path')
            p_lbl = _get('label_hand_path')

            self.tiles.append({
                'tile_id':       row['tile_id'],
                'has_hand_label': True,
                'p_s1':          p_s1,
                'p_s2':          p_s2,
                'pre_s1':        _get('pre_s1_path'),
                'pre_s2':        _get('pre_s2_path'),
                'aux':           _get('aux_path'),
                'p_lbl':         p_lbl,
            })

    # ------------------------------------------------------------------
    # NORMALIZATION  (computed from hand-labeled train split)
    # ------------------------------------------------------------------
    def _compute_or_load_normalization(self):
        if self.norm_json.exists():
            with open(self.norm_json, "r") as f:
                self.norm_stats = json.load(f)
            return

        assert self.split == "train", (
            "normalization_stats.json missing — run train dataset first."
        )

        logger.info("Computing normalization stats from train split...")

        def _accumulate(tiles, path_key, num_chans):
            sums    = np.zeros(num_chans)
            sq_sums = np.zeros(num_chans)
            counts  = np.zeros(num_chans)
            for t in tiles:
                p = t[path_key]
                if not p:
                    continue
                path = self.root / p
                if not path.exists():
                    continue
                try:
                    with rasterio.open(path) as src:
                        for c in range(min(src.count, num_chans)):
                            d = src.read(c + 1).astype(np.float64).flatten()
                            if d.size > 512 * 512:
                                h, w = src.height, src.width
                                d = np.reshape(d, (h, w))[
                                    h // 2 - 256:h // 2 + 256,
                                    w // 2 - 256:w // 2 + 256,
                                ].flatten()
                            elif d.size < 512 * 512:
                                continue
                            d = d[np.isfinite(d)]
                            sums[c]    += d.sum()
                            sq_sums[c] += (d ** 2).sum()
                            counts[c]  += d.size
                except Exception:
                    pass
            mean = sums / np.maximum(counts, 1)
            std  = np.sqrt(np.maximum(
                sq_sums / np.maximum(counts, 1) - mean ** 2, 1e-6
            ))
            return mean.tolist(), std.tolist()

        stats = {}
        for key, nc in [('p_s1', 2), ('p_s2', 6), ('pre_s1', 2), ('pre_s2', 6), ('aux', 4)]:
            m, s = _accumulate(self.tiles, key, nc)
            stats[key] = {'mean': m, 'std': s}

        with open(self.norm_json, "w") as f:
            json.dump(stats, f, indent=4)
        self.norm_stats = stats
        logger.info("Normalization stats saved.")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _center_crop(self, arr, th=512, tw=512):
        if arr.ndim == 3:
            _, h, w = arr.shape
            if h == th and w == tw:
                return arr
            cy, cx = h // 2, w // 2
            return arr[:, cy - 256:cy + 256, cx - 256:cx + 256]
        else:
            h, w = arr.shape
            if h == th and w == tw:
                return arr
            cy, cx = h // 2, w // 2
            return arr[cy - 256:cy + 256, cx - 256:cx + 256]

    def flood_pixel_fraction(self, idx):
        tile = self.tiles[idx]
        p    = tile['p_lbl']
        if not p:
            return 0.0
        fp = self.root / p
        if not fp.exists():
            return 0.0
        try:
            with rasterio.open(fp) as src:
                lbl   = self._center_crop(src.read(1))
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

        def _read(path_key, num_chans, crop, out_slice):
            p = tile[path_key]
            if not p:
                return False
            fp = self.root / p
            if not fp.exists():
                return False
            try:
                with rasterio.open(fp) as src:
                    idx_list = list(range(1, min(src.count, num_chans) + 1))
                    data = src.read(idx_list[0])[np.newaxis] if len(idx_list) == 1 \
                        else src.read(idx_list)
                    if data.shape[0] < num_chans:
                        data = np.concatenate(
                            [data, np.zeros((num_chans - data.shape[0],
                                             *data.shape[1:]))], 0)
                    elif data.shape[0] > num_chans:
                        data = data[:num_chans]
                    if crop:
                        data = self._center_crop(data)
                    if data.shape[1] != 512 or data.shape[2] != 512:
                        return False
                    img[out_slice] = data
                    return True
            except Exception:
                return False

        _read('p_s1',   2, False, slice(0, 2))
        _read('p_s2',   6, False, slice(2, 8))
        _read('pre_s1', 2, True,  slice(8, 10))
        _read('pre_s2', 6, True,  slice(10, 16))
        _read('aux',    4, True,  slice(16, 20))

        # Label
        p_lbl = tile['p_lbl']
        if p_lbl:
            fp = self.root / p_lbl
            if fp.exists():
                try:
                    with rasterio.open(fp) as src:
                        l = self._center_crop(src.read(1))
                        if l.shape == (512, 512):
                            lbl = l.astype(np.int64)
                            lbl[(lbl != 0) & (lbl != 1)] = -1
                except Exception as e:
                    logger.warning(f"Tile {tid} label error: {e}")

        # ------------------------------------------------------------------
        # NORMALIZATION
        # ------------------------------------------------------------------
        for i in range(2):
            m = self.norm_stats['p_s1']['mean'][i]
            s = self.norm_stats['p_s1']['std'][i]
            img[i] = (img[i] - m) / max(s, 1e-6)

        for i in range(6):
            m = self.norm_stats['p_s2']['mean'][i]
            s = self.norm_stats['p_s2']['std'][i]
            img[i + 2] = (img[i + 2] - m) / max(s, 1e-6)

        for i in range(2):
            m = self.norm_stats['pre_s1']['mean'][i]
            s = self.norm_stats['pre_s1']['std'][i]
            img[i + 8] = (img[i + 8] - m) / max(s, 1e-6)

        for i in range(6):
            m = self.norm_stats['pre_s2']['mean'][i]
            s = self.norm_stats['pre_s2']['std'][i]
            img[i + 10] = (img[i + 10] - m) / max(s, 1e-6)

        if 'aux' in self.norm_stats and any(v > 0 for v in self.norm_stats['aux']['std']):
            for i in range(4):
                m = self.norm_stats['aux']['mean'][i]
                s = self.norm_stats['aux']['std'][i]
                img[i + 16] = (img[i + 16] - m) / max(s, 1e-6)
        else:
            img[16:20] = img[16:20] / 100.0

        img[~np.isfinite(img)] = 0.0

        # ------------------------------------------------------------------
        # AUGMENTATIONS (train only)
        # ------------------------------------------------------------------
        if self.split == "train":
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[::-1, :].copy()
            if random.random() < 0.5:
                k   = random.randint(1, 3)
                img = np.rot90(img, k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k, axes=(0, 1)).copy()
            if random.random() < 0.3:
                for ch in [0, 1, 8, 9]:
                    img[ch] += np.random.normal(0, 0.1, (512, 512)).astype(np.float32)
            if random.random() < 0.15:
                img[random.randint(0, 19)] = 0.0

        return {
            'image':          torch.from_numpy(img).float(),
            'label':          torch.from_numpy(lbl).long(),
            'tile_id':        tid,
            'split':          self.split,
            'has_hand_label': tile['has_hand_label'],
        }

# ------------------------------------------------------------------
# WEIGHTED SAMPLER
# ------------------------------------------------------------------
def build_flood_sampler(dataset, flood_multiplier=5.0):
    logger.info("Building WeightedRandomSampler...")
    weights = []
    for i in range(len(dataset)):
        frac = dataset.flood_pixel_fraction(i)
        weights.append(flood_multiplier if frac >= 0.01 else 1.0)
    w = torch.tensor(weights, dtype=torch.double)
    logger.info(
        f"Sampler: {int((w == flood_multiplier).sum())} flood tiles, "
        f"{int((w == 1.0).sum())} non-flood tiles"
    )
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)

# ------------------------------------------------------------------
# DATALOADERS
# ------------------------------------------------------------------
def get_dataloaders(root, batch_size=4):
    train_ds = FloodDataset(root, "train")
    val_ds   = FloodDataset(root, "val")
    test_ds  = FloodDataset(root, "test")

    sampler = build_flood_sampler(train_ds, flood_multiplier=5.0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, drop_last=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader, test_loader

# ------------------------------------------------------------------
# VALIDATION BLOCK
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("Validating dataloaders...")
    tr, vl, ts = get_dataloaders(".", batch_size=2)

    for name, loader in [("train", tr), ("val", vl)]:
        batch = next(iter(loader))
        img, lbl = batch['image'], batch['label']
        print(f"[{name}] img: {img.shape}  min={img.min():.3f} max={img.max():.3f}")
        print(f"[{name}] lbl: unique={torch.unique(lbl).tolist()}")
        assert not torch.isnan(img).any(), "NaN in images!"
        assert not torch.isinf(img).any(), "Inf in images!"
        assert set(torch.unique(lbl).tolist()).issubset({-1, 0, 1})
        assert img.shape == (2, 20, 512, 512)
        assert lbl.shape == (2, 512, 512)

    print("\nDATALOADER VALIDATION PASSED")