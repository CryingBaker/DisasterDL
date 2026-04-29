import os
import sys
import warnings
import logging
import multiprocessing
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import torch



# Use spawn to avoid multiprocessing deadlocks on macOS MPS
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Speed up matmul on modern hardware with minor precision trade-off
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")



# frozen=True makes the config immutable — prevents accidental changes mid-run
@dataclass(frozen=True)
class Config:
    data_root: Path = Path("data")          # root folder for all satellite tiles
    meta_dir: Path = Path("data_analysis")  # where metadata CSVs are cached
    plots_dir: Path = Path("plots")         # output directory for generated plots
    log_file: str = "disaster_dl_v15_mps.log"

    tile_size: int = 512   # original satellite tile resolution (pixels)
    crop_size: int = 256   # random crop size applied during training

    ignore_index: int = -1       # label value excluded from loss computation
    sentinel_value: float = -3.5 # fill value used for missing satellite bands
    min_weak_flood_pixels: int = 10  # skip weak tiles with fewer flood pixels than this

    # Curriculum learning: short weak pre-training, longer hand fine-tuning
    weak_epochs: int = 5
    hand_epochs: int = 20
    weak_batch_size: int = 16
    hand_batch_size: int = 8
    weak_lr: float = 1e-4   # higher LR for weak stage
    hand_lr: float = 1e-5   # lower LR to fine-tune without forgetting
    grad_clip_norm: float = 1.0  # prevent exploding gradients

    encoder_name: str = "efficientnet-b4"  # pretrained ImageNet encoder for U-Net
    encoder_weights: str = "imagenet"

    # Held-out regions — never seen during training to test geographic generalisation
    test_regions: Tuple[str, ...] = ("India", "Mekong")
    val_regions: Tuple[str, ...] = ("USA",)

    seed: int = 42
    log_every_n_batches: int = 10


CFG = Config()



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
    """Fix all random seeds for full reproducibility."""
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # ensures identical conv results
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)
