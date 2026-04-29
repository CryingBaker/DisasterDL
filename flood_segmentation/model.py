from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

from config import CFG, logger
from data import FloodDataset



class SentinelMaskingLayer(nn.Module):
    """Zero out channels where ALL pixels are the fill value (missing band)."""
    def __init__(self, sentinel_value: float = CFG.sentinel_value):
        super().__init__()
        self.sentinel_value = sentinel_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # True for each channel where every spatial pixel is the fill value
        is_sentinel = (x == self.sentinel_value).all(dim=(2, 3), keepdim=True)
        return torch.where(is_sentinel, torch.zeros_like(x), x)


class MaskedUNet(nn.Module):
    """U-Net with EfficientNet-B4 encoder and a missing-band masking pre-layer."""
    def __init__(self, in_channels: int, classes: int = 2):
        super().__init__()
        self.masking = SentinelMaskingLayer()
        self.unet = smp.Unet(
            encoder_name=CFG.encoder_name,
            encoder_weights=CFG.encoder_weights,  # pretrained on ImageNet
            in_channels=in_channels,
            classes=classes,
            activation=None,  # raw logits — loss handles softmax internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(self.masking(x))



class HybridSegLoss(nn.Module):
    """CE + Dice loss. CE handles per-pixel accuracy; Dice optimises overlap.
    Weights [0.35, 0.65] upweight flood pixels to counter class imbalance.
    """
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
        # register_buffer moves weights with the model to the right device
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.tensor([0.35, 0.65], dtype=torch.float32),
        )
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index, from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index  # exclude ignored pixels
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
        return logits.sum() * 0.0  # all pixels ignored — return zero safely



def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
    )



@torch.no_grad()  # disable gradient tracking during evaluation
def evaluate_detailed(model: nn.Module, device: torch.device, chs: Sequence[int], split: str = "val") -> Tuple[float, float, float]:
    """Return (flood_iou, mean_iou, pred_flood_ratio) on the given split."""
    model.eval()
    loader = make_loader(FloodDataset(CFG.data_root, split), batch_size=8, shuffle=False)

    inter_f, union_f = 0, 0  # flood class intersection / union
    inter_d, union_d = 0, 0  # dry class intersection / union
    pred_flood_total = 0
    gt_flood_total = 0

    for b in loader:
        x = b["image"][:, chs].to(device)
        y = b["label"].to(device).long()

        pred = torch.argmax(model(x), dim=1)  # hard prediction
        valid = y != CFG.ignore_index          # mask out ignored pixels

        pred_flood_total += (pred == 1).sum().item()
        gt_flood_total += (y == 1).sum().item()

        inter_f += ((pred == 1) & (y == 1) & valid).sum().item()
        union_f += (((pred == 1) | (y == 1)) & valid).sum().item()

        inter_d += ((pred == 0) & (y == 0) & valid).sum().item()
        union_d += (((pred == 0) | (y == 0)) & valid).sum().item()

    iou_flood = inter_f / union_f if union_f > 0 else 0.0
    iou_dry   = inter_d / union_d if union_d > 0 else 0.0
    mean_iou  = (iou_flood + iou_dry) / 2.0
    pred_ratio = pred_flood_total / max(gt_flood_total, 1)  # >1 means over-predicting

    return iou_flood, mean_iou, pred_ratio
