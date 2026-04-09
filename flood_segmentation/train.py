import os
import sys
import time
import json
import warnings
import logging
import multiprocessing
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader


# =========================
# 0) MPS / PROCESS STARTUP
# =========================

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# =========================
# 1) CONFIG
# =========================

@dataclass(frozen=True)
class Config:
    data_root: Path = Path("data")
    meta_dir: Path = Path("data_analysis")
    plots_dir: Path = Path("plots")
    log_file: str = "disaster_dl_v15_mps.log"

    tile_size: int = 512
    crop_size: int = 256

    ignore_index: int = -1
    sentinel_value: float = -3.5
    min_weak_flood_pixels: int = 10

    weak_epochs: int = 5
    hand_epochs: int = 20
    weak_batch_size: int = 16
    hand_batch_size: int = 8
    weak_lr: float = 1e-4
    hand_lr: float = 1e-5
    grad_clip_norm: float = 1.0

    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"

    test_regions: Tuple[str, ...] = ("India", "Mekong")
    val_regions: Tuple[str, ...] = ("USA",)

    seed: int = 42
    log_every_n_batches: int = 10


CFG = Config()


# =========================
# 2) LOGGING / SEED
# =========================

os.environ["CPL_LOG"] = "/dev/null"
warnings.filterwarnings("ignore")
logging.getLogger("rasterio").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(CFG.log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("flood_ablation")


def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)


# =========================
# 3) FILE HELPERS
# =========================

def safe_mkdir(path: Path) -> None:
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
    out = arr.copy()
    valid = out != fill_value
    out[valid] = (out[valid] + 25.0) / 25.0
    out[valid] = np.clip(out[valid], 0.0, 1.0)
    return out


def scale_aux(arr: np.ndarray, fill_value: float) -> np.ndarray:
    out = arr.copy()
    valid = out != fill_value
    out[valid] = (out[valid] - 30.0) / 100.0
    out[valid] = np.clip(out[valid], 0.0, 1.0)
    return out


def scale_optical(arr: np.ndarray, fill_value: float) -> np.ndarray:
    out = arr.copy()
    valid = out != fill_value
    if valid.any() and np.nanmax(out[valid]) > 10:
        out[valid] = out[valid] / 10000.0
    out[valid] = np.clip(out[valid], 0.0, 1.0)
    return out


# =========================
# 4) METADATA / PARTITIONING
# =========================

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

    df["split"] = "train"
    df.loc[(df["region"].isin(test_regions)) & (df["is_hand"] == True), "split"] = "test"
    df.loc[(df["region"].isin(val_regions)) & (df["is_hand"] == True), "split"] = "val"

    df.to_csv(meta_path, index=False)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        split_df["tile_id"].to_csv(meta_dir / f"splits_{split}.txt", index=False, header=False)
        logger.info(f"📁 {split} split locked: {len(split_df)} tiles.")

    return df


# =========================
# 5) DATASET
# =========================

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
            th = tw = CFG.crop_size
            i = np.random.randint(0, CFG.tile_size - th + 1)
            j = np.random.randint(0, CFG.tile_size - tw + 1)

            img = img[:, i:i + th, j:j + tw]
            lbl = lbl[i:i + th, j:j + tw]

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


# =========================
# 6) MODEL
# =========================

class SentinelMaskingLayer(nn.Module):
    def __init__(self, sentinel_value: float = CFG.sentinel_value):
        super().__init__()
        self.sentinel_value = sentinel_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_sentinel = (x == self.sentinel_value).all(dim=(2, 3), keepdim=True)
        return torch.where(is_sentinel, torch.zeros_like(x), x)


class MaskedUNet(nn.Module):
    def __init__(self, in_channels: int, classes: int = 2):
        super().__init__()
        self.masking = SentinelMaskingLayer()
        self.unet = smp.Unet(
            encoder_name=CFG.encoder_name,
            encoder_weights=CFG.encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(self.masking(x))


# =========================
# 7) LOSS
# =========================

class HybridSegLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = CFG.ignore_index,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.tensor([0.35, 0.65], dtype=torch.float32),
        )
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index, from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if valid.any():
            ce_map = F.cross_entropy(
                logits,
                target,
                weight=self.class_weights.to(logits.device),
                ignore_index=self.ignore_index,
                reduction="none",
            )
            ce = ce_map[valid].mean()
            dice = self.dice(logits, target)
            return self.ce_weight * ce + self.dice_weight * dice
        return logits.sum() * 0.0


# =========================
# 8) DATALOADER HELPERS
# =========================

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
    )


# =========================
# 9) EVALUATION
# =========================

@torch.no_grad()
def evaluate_detailed(model: nn.Module, device: torch.device, chs: Sequence[int], split: str = "val") -> Tuple[float, float, float]:
    model.eval()
    loader = make_loader(FloodDataset(CFG.data_root, split), batch_size=8, shuffle=False)

    inter_f, union_f = 0, 0
    inter_d, union_d = 0, 0
    pred_flood_total = 0
    gt_flood_total = 0

    for b in loader:
        x = b["image"][:, chs].to(device)
        y = b["label"].to(device).long()

        pred = torch.argmax(model(x), dim=1)
        valid = y != CFG.ignore_index

        pred_flood_total += (pred == 1).sum().item()
        gt_flood_total += (y == 1).sum().item()

        inter_f += ((pred == 1) & (y == 1) & valid).sum().item()
        union_f += (((pred == 1) | (y == 1)) & valid).sum().item()

        inter_d += ((pred == 0) & (y == 0) & valid).sum().item()
        union_d += (((pred == 0) | (y == 0)) & valid).sum().item()

    iou_flood = inter_f / union_f if union_f > 0 else 0.0
    iou_dry = inter_d / union_d if union_d > 0 else 0.0
    mean_iou = (iou_flood + iou_dry) / 2.0
    pred_ratio = pred_flood_total / max(gt_flood_total, 1)

    return iou_flood, mean_iou, pred_ratio


# =========================
# 10) TRAINING
# =========================

def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


# def run_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     criterion: nn.Module,
#     device: torch.device,
#     chs: Sequence[int],
#     train: bool = True,
#     grad_clip_norm: float = 1.0,
#     epoch_name: str = "",
# ) -> float:
#     model.train(train)
#     total_loss = 0.0
#     num_batches = len(loader)

#     logger.info(f"{epoch_name} batches: {num_batches}")
#     print(f"{epoch_name} batches: {num_batches}", flush=True)

#     for batch_idx, b in enumerate(loader, start=1):
#         x = b["image"][:, chs].to(device)
#         y = b["label"].to(device).long()

#         if train:
#             optimizer.zero_grad(set_to_none=True)

#         logits = model(x)
#         loss = criterion(logits, y)

#         if train:
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
#             optimizer.step()

#         total_loss += float(loss.item())

#         if batch_idx == 1 or batch_idx % CFG.log_every_n_batches == 0 or batch_idx == num_batches:
#             logger.info(f"{epoch_name} batch {batch_idx}/{num_batches} | loss={loss.item():.4f}")
#             print(f"{epoch_name} batch {batch_idx}/{num_batches} | loss={loss.item():.4f}", flush=True)

#     return total_loss / max(num_batches, 1)

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from rich.console import Console

console = Console()

def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    chs,
    train=True,
    grad_clip_norm=1.0,
    epoch_name=""
):
    model.train(train)
    total_loss = 0.0
    num_batches = len(loader)

    logger.info(f"{epoch_name} batches: {num_batches}")
    console.print(f"[bold cyan]{epoch_name}[/bold cyan] batches: {num_batches}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(epoch_name, total=num_batches)

        for batch_idx, b in enumerate(loader, start=1):
            x = b["image"][:, chs].to(device)
            y = b["label"].to(device).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.item())

            progress.update(task, advance=1)

            # keep your debug logs
            if batch_idx == 1 or batch_idx % CFG.log_every_n_batches == 0 or batch_idx == num_batches:
                logger.info(f"{epoch_name} batch {batch_idx}/{num_batches} | loss={loss.item():.4f}")

    return total_loss / max(num_batches, 1)

def run_research() -> pd.DataFrame:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"🚀 Device: {device}")

    ABLATIONS = {
        "OPTICAL_ONLY":     list(range(2, 15)),
        "SAR_ONLY":         [0, 1],
        "OPTICAL_SAR":      list(range(0, 15)),
        "OPTICAL_AUX":      list(range(2, 15)) + list(range(23, 27)),
        "OPTICAL_TEMPORAL": list(range(2, 23)),
        "SAR_TEMPORAL":     [0, 1, 15, 16],
        "NO_AUX":           list(range(0, 23)),
        "FULL_TEAM":        list(range(0, 27)),
    }

    history: List[Dict[str, object]] = []

    for name, chs in ABLATIONS.items():
        logger.info(f"\n🔥 Starting Ablation: {name} ({len(chs)} channels)")
        print(f"🔥 Starting Ablation: {name} ({len(chs)} channels)", flush=True)

        model = MaskedUNet(in_channels=len(chs), classes=2).to(device)
        criterion = HybridSegLoss().to(device)

        best_iou = -1.0

        # Weak stage
        weak_ds = FloodDataset(CFG.data_root, "train", use_weak=True)
        logger.info(f"{name} weak tiles: {len(weak_ds)}")
        print(f"{name} weak tiles: {len(weak_ds)}", flush=True)

        if len(weak_ds) > 0:
            weak_loader = make_loader(weak_ds, batch_size=CFG.weak_batch_size, shuffle=True)
            optimizer = make_optimizer(model, CFG.weak_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(CFG.weak_epochs, 1))

            for ep in range(1, CFG.weak_epochs + 1):
                tag = f"[{name}] Weak {ep}"
                train_loss = run_epoch(
                    model=model,
                    loader=weak_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    chs=chs,
                    train=True,
                    grad_clip_norm=CFG.grad_clip_norm,
                    epoch_name=tag,
                )
                scheduler.step()
                val_f_iou, val_m_iou, pred_ratio = evaluate_detailed(model, device, chs, split="val")

                history.append({
                    "Model": name,
                    "Stage": "Weak",
                    "Epoch": ep,
                    "Loss": train_loss,
                    "Val_Flood_IoU": val_f_iou,
                    "Val_Mean_IoU": val_m_iou,
                    "PredFloodRatio": pred_ratio,
                })

                logger.info(
                    f"[{name}] W-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                    f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
                print(
                    f"[{name}] W-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                    f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}",
                    flush=True,
                )

        # Hand stage
        hand_ds = FloodDataset(CFG.data_root, "train", use_weak=False)
        logger.info(f"{name} hand tiles: {len(hand_ds)}")
        print(f"{name} hand tiles: {len(hand_ds)}", flush=True)

        if len(hand_ds) == 0:
            logger.warning(f"[{name}] No hand training tiles found. Skipping hand stage.")
            continue

        hand_loader = make_loader(hand_ds, batch_size=CFG.hand_batch_size, shuffle=True)
        optimizer = make_optimizer(model, CFG.hand_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(CFG.hand_epochs, 1))

        for ep in range(1, CFG.hand_epochs + 1):
            tag = f"[{name}] Hand {ep}"
            train_loss = run_epoch(
                model=model,
                loader=hand_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                chs=chs,
                train=True,
                grad_clip_norm=CFG.grad_clip_norm,
                epoch_name=tag,
            )
            scheduler.step()
            val_f_iou, val_m_iou, pred_ratio = evaluate_detailed(model, device, chs, split="val")

            history.append({
                "Model": name,
                "Stage": "Hand",
                "Epoch": ep + CFG.weak_epochs,
                "Loss": train_loss,
                "Val_Flood_IoU": val_f_iou,
                "Val_Mean_IoU": val_m_iou,
                "PredFloodRatio": pred_ratio,
            })

            logger.info(
                f"[{name}] H-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            print(
                f"[{name}] H-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}",
                flush=True,
            )

            if val_f_iou > best_iou:
                best_iou = val_f_iou
                torch.save(model.state_dict(), f"best_{name}_model.pth")

    result = pd.DataFrame(history)
    result.to_csv("ablation_results_full.csv", index=False)
    logger.info("✅ Training complete. Results saved to ablation_results_full.csv")
    return result


# =========================
# 11) PLOTS
# =========================

def generate_plots(results_csv: str = "ablation_results_full.csv") -> None:
    df = pd.read_csv(results_csv)
    safe_mkdir(CFG.plots_dir)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(13, 7))
    sns.lineplot(data=df, x="Epoch", y="Val_Flood_IoU", hue="Model", lw=2)
    plt.axvline(x=CFG.weak_epochs + 0.5, color="black", linestyle=":", linewidth=1.5, label="Hand Fine-tuning Start")
    plt.xlabel("Epoch")
    plt.ylabel("Flood IoU (Val)")
    plt.title("Ablation Study: Flood IoU across Channel Configurations")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_iou_main.png", dpi=300)
    plt.close()

    g = sns.FacetGrid(df, col="Model", col_wrap=4, height=3.2, sharey=False)
    g.map_dataframe(sns.lineplot, x="Epoch", y="Loss")
    for ax in g.axes.flatten():
        ax.axvline(x=CFG.weak_epochs + 0.5, color="red", linestyle="--", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_loss_facets.png", dpi=300)
    plt.close()

    final = df[df["Stage"] == "Hand"].groupby("Model", as_index=False)["Val_Flood_IoU"].max()
    final = final.sort_values("Val_Flood_IoU", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=final, x="Model", y="Val_Flood_IoU")
    for bar, val in zip(ax.patches, final["Val_Flood_IoU"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Best Val Flood IoU")
    plt.title("Peak Flood IoU by Ablation Configuration")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_final_bar.png", dpi=300)
    plt.close()

    hand_df = df[df["Stage"] == "Hand"].copy()
    pivot = hand_df.pivot_table(index="Model", columns="Epoch", values="Val_Mean_IoU")
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False, linewidths=0.3)
    plt.title("Mean IoU Heatmap (Hand Stage)")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_meaniou_heatmap.png", dpi=300)
    plt.close()

    logger.info(f"📊 All plots saved to {CFG.plots_dir.resolve()} at 300 DPI.")


# =========================
# 12) MAIN
# =========================

if __name__ == "__main__":
    setup_scientific_data()
    run_research()
    generate_plots()