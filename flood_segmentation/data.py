from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from config import CFG, logger



def safe_mkdir(path: Path) -> None:
    """Create directory (and parents) if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)


def first_matching_file(folder: Path, tile_id: str) -> Optional[Path]:
    if not folder.exists():
        return None
    try:
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() == ".tif" and f.name.startswith(tile_id):
                return f
    except Exception:
        return None
    return None


def read_raster(path: Optional[Path], expected_bands: int, fill_value: float) -> np.ndarray:
    if path is None or not path.exists():
        return np.full((expected_bands, CFG.tile_size, CFG.tile_size), fill_value, dtype=np.float32)

    try:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            data = np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)

            c, h, w = data.shape
            out = np.full((max(c, expected_bands), CFG.tile_size, CFG.tile_size), fill_value, dtype=np.float32)

            h_min = min(h, CFG.tile_size)
            w_min = min(w, CFG.tile_size)
            out[:c, :h_min, :w_min] = data[:, :h_min, :w_min]

            out = out[:expected_bands]
            if out.shape[0] < expected_bands:
                pad = np.full((expected_bands - out.shape[0], CFG.tile_size, CFG.tile_size), fill_value, dtype=np.float32)
                out = np.concatenate([out, pad], axis=0)

            return out
    except Exception:
        return np.full((expected_bands, CFG.tile_size, CFG.tile_size), fill_value, dtype=np.float32)


def scale_sar(arr: np.ndarray, fill_value: float) -> np.ndarray:
    """Normalise SAR (dB) from roughly [-25, 0] to [0, 1]."""
    out = arr.copy()
    valid = out != fill_value
    out[valid] = (out[valid] + 25.0) / 25.0
    out[valid] = np.clip(out[valid], 0.0, 1.0)
    return out


def scale_aux(arr: np.ndarray, fill_value: float) -> np.ndarray:
    """Normalise auxiliary data (e.g. elevation in metres) to [0, 1]."""
    out = arr.copy()
    valid = out != fill_value
    out[valid] = (out[valid] - 30.0) / 100.0
    out[valid] = np.clip(out[valid], 0.0, 1.0)
    return out


def scale_optical(arr: np.ndarray, fill_value: float) -> np.ndarray:
    """Normalise Sentinel-2 reflectance values (0–10000) to [0, 1]."""
    out = arr.copy()
    valid = out != fill_value
    if valid.any() and np.nanmax(out[valid]) > 10:  # raw DN values, not already scaled
        out[valid] = out[valid] / 10000.0
    out[valid] = np.clip(out[valid], 0.0, 1.0)
    return out



def setup_scientific_data(
    data_root: Path = CFG.data_root,
    meta_dir: Path = CFG.meta_dir,
    test_regions: Sequence[str] = CFG.test_regions,
    val_regions: Sequence[str] = CFG.val_regions,
    min_weak_flood_pixels: int = CFG.min_weak_flood_pixels,
) -> pd.DataFrame:
    safe_mkdir(meta_dir)
    meta_path = meta_dir / "dataset_metadata.csv"

    if meta_path.exists():
        logger.info("⚡ Using cached dataset metadata (skipping partitioning).")
        df = pd.read_csv(meta_path)
        return df

    logger.info("📡 Starting Scientific Data Partitioning...")

    configs = [
        {"s1": "sen1floods11/S1Hand", "s2": "sen1floods11/S2Hand", "lbl": "sen1floods11/LabelHand", "is_hand": True},
        {"s1": "sen1floods11/S1Weak", "s2": "sen1floods11/S2Weak", "lbl": "sen1floods11/S2IndexLabelWeak", "is_hand": False},
    ]

    rows: List[Dict[str, object]] = []

    for cfg in configs:
        s1_dir = data_root / cfg["s1"]
        s2_dir = data_root / cfg["s2"]
        lbl_dir = data_root / cfg["lbl"]

        if not s1_dir.exists() or not s2_dir.exists() or not lbl_dir.exists():
            logger.warning(f"Missing folder(s) for config: {cfg}")
            continue

        for s1_file in sorted(s1_dir.iterdir()):
            if not (s1_file.is_file() and s1_file.suffix.lower() == ".tif"):
                continue

            tid = "_".join(s1_file.stem.split("_")[:2])
            region = s1_file.stem.split("_")[0]

            if region in test_regions and not cfg["is_hand"]:
                continue

            s2_file = first_matching_file(s2_dir, tid)
            lbl_file = first_matching_file(lbl_dir, tid)

            if s2_file is None or lbl_file is None:
                continue

            if not cfg["is_hand"]:
                # Skip weak tiles with almost no flood signal — too noisy to be useful
                try:
                    with rasterio.open(lbl_file) as src:
                        lbl = src.read(1)
                    if int((lbl == 1).sum()) < min_weak_flood_pixels:
                        continue
                except Exception:
                    continue

            pre_s1 = first_matching_file(data_root / "pre_s1", tid)
            pre_s2 = first_matching_file(data_root / "pre_s2", tid)
            aux = first_matching_file(data_root / "aux", tid)

            rows.append({
                "tile_id": tid,
                "region": region,
                "is_hand": bool(cfg["is_hand"]),
                "s1_path": str(s1_file),
                "s2_path": str(s2_file),
                "label_path": str(lbl_file),
                "pre_s1_path": str(pre_s1) if pre_s1 else "",
                "pre_s2_path": str(pre_s2) if pre_s2 else "",
                "aux_path": str(aux) if aux else "",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No tiles found. Check your folder structure and data_root.")

    # Assign splits — test/val only use hand-labelled tiles to ensure clean evaluation
    df["split"] = "train"
    df.loc[(df["region"].isin(test_regions)) & (df["is_hand"] == True), "split"] = "test"
    df.loc[(df["region"].isin(val_regions)) & (df["is_hand"] == True), "split"] = "val"

    df.to_csv(meta_path, index=False)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        split_df["tile_id"].to_csv(meta_dir / f"splits_{split}.txt", index=False, header=False)
        logger.info(f"📁 {split} split locked: {len(split_df)} tiles.")

    return df



class FloodDataset(Dataset):
    def __init__(self, root: Union[str, Path], split: str, use_weak: Optional[bool] = None):
        self.root = Path(root)
        self.split = split

        meta_path = CFG.meta_dir / "dataset_metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found. Run setup_scientific_data() first.")

        self.df = pd.read_csv(meta_path)

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.df = self.df[self.df["split"] == split].copy()

        if split == "train" and use_weak is not None:
            self.df = self.df[self.df["is_hand"] == (not use_weak)]
        elif split != "train":
            self.df = self.df[self.df["is_hand"] == True]

        self.df = self.df.reset_index(drop=True)
        self.tiles = self.df["tile_id"].tolist()

    def __len__(self) -> int:
        return len(self.tiles)

    @staticmethod
    def _ensure_2d_label(lbl: np.ndarray) -> np.ndarray:
        if lbl.ndim == 3:
            lbl = lbl.squeeze()
        return lbl

    def _load_group(self, path_str: str, bands: int, kind: str) -> np.ndarray:
        path = Path(path_str) if isinstance(path_str, str) and path_str else None
        fill = CFG.sentinel_value if kind != "label" else float(CFG.ignore_index)
        arr = read_raster(path, bands, fill)

        if kind == "sar":
            return scale_sar(arr, CFG.sentinel_value)
        if kind == "aux":
            return scale_aux(arr, CFG.sentinel_value)
        if kind == "optical":
            return scale_optical(arr, CFG.sentinel_value)
        if kind == "label":
            lbl = self._ensure_2d_label(arr).astype(np.int64)
            lbl[(lbl != 0) & (lbl != 1)] = CFG.ignore_index
            return lbl
        return arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        tid = row["tile_id"]

        img = np.full((27, CFG.tile_size, CFG.tile_size), CFG.sentinel_value, dtype=np.float32)

        img[0:2] = self._load_group(row["s1_path"], 2, kind="sar")
        img[2:15] = self._load_group(row["s2_path"], 13, kind="optical")
        if row["pre_s1_path"]:
            img[15:17] = self._load_group(row["pre_s1_path"], 2, kind="sar")
        if row["pre_s2_path"]:
            img[17:23] = self._load_group(row["pre_s2_path"], 6, kind="optical")
        if row["aux_path"]:
            img[23:27] = self._load_group(row["aux_path"], 4, kind="aux")

        lbl = self._load_group(row["label_path"], 1, kind="label")

        if self.split == "train":
            # Random crop: cut a 256×256 patch from the 512×512 tile
            th = tw = CFG.crop_size
            i = np.random.randint(0, CFG.tile_size - th + 1)
            j = np.random.randint(0, CFG.tile_size - tw + 1)

            img = img[:, i:i + th, j:j + tw]
            lbl = lbl[i:i + th, j:j + tw]

            # Random horizontal and vertical flips for data augmentation
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1].copy()
                lbl = lbl[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[::-1, :].copy()

        return {
            "image": torch.from_numpy(img.copy()).float(),
            "label": torch.from_numpy(lbl.copy()).long(),
            "id": tid,
        }
