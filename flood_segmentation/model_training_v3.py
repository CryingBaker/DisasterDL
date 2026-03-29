import os
import sys
import subprocess
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import logging
from pathlib import Path
import csv
from tqdm import tqdm

from data_loader import FloodDataset, get_dataloaders

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# MODEL
# =============================================================================

def build_model(in_channels=20, num_classes=2):
    """
    UNet with EfficientNet-B4 encoder, ImageNet pretrained.
    smp handles N-channel input by averaging pretrained 3-channel stem weights.
    decoder_attention_type='scse' adds squeeze-and-excitation in decoder blocks.
    """
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        decoder_attention_type="scse",
    )


# =============================================================================
# LOSS
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        mask = labels != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        probs = F.softmax(logits, dim=1)
        dice, valid = 0.0, 0
        for c in range(probs.shape[1]):
            target = (labels == c).float() * mask.float()
            pred   = probs[:, c] * mask.float()
            inter  = (pred * target).sum()
            union  = pred.sum() + target.sum()
            if union > 0:
                dice  += 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
                valid += 1
        return dice / valid if valid > 0 else \
            torch.tensor(0.0, device=logits.device, requires_grad=True)


class CombinedLoss(nn.Module):
    """
    Weighted CrossEntropy + Dice.
    Flood class gets 4x weight to handle ~90/10 class imbalance.
    CE is numerically stable on MPS unlike focal loss.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=-1,
                 class_weights=None):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.dice        = DiceLoss(smooth=1.0, ignore_index=ignore_index)
        if class_weights is None:
            class_weights = torch.tensor([1.0, 4.0])
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        ce_l  = F.cross_entropy(
            logits, labels,
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
        )
        d_l   = self.dice(logits, labels)
        total = self.ce_weight * ce_l + self.dice_weight * d_l
        return total, ce_l, d_l


# =============================================================================
# METRICS  —  GLOBAL PIXEL ACCUMULATOR (correct for segmentation papers)
# =============================================================================

class GlobalMetrics:
    """
    Accumulates TP/FP/FN/union counts across ALL batches in an epoch,
    then computes IoU globally. This is the correct way to report
    segmentation IoU — per-batch averaging gives different (usually lower)
    numbers and is NOT what papers report.
    """
    def __init__(self, ignore_index=-1):
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.inter_f  = 0
        self.union_f  = 0
        self.inter_nf = 0
        self.union_nf = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.total_loss = 0.0
        self.total_ce   = 0.0
        self.total_dice = 0.0
        self.n_batches  = 0

    def update(self, logits, labels, loss=None, ce=None, dice=None):
        pred = torch.argmax(logits, dim=1)
        mask = labels != self.ignore_index

        p_f  = (pred == 1) & mask
        g_f  = (labels == 1) & mask
        p_nf = (pred == 0) & mask
        g_nf = (labels == 0) & mask

        self.inter_f  += (p_f  & g_f ).sum().item()
        self.union_f  += (p_f  | g_f ).sum().item()
        self.inter_nf += (p_nf & g_nf).sum().item()
        self.union_nf += (p_nf | g_nf).sum().item()

        self.TP += ((pred == 1) & (labels == 1) & mask).sum().item()
        self.FP += ((pred == 1) & (labels == 0) & mask).sum().item()
        self.FN += ((pred == 0) & (labels == 1) & mask).sum().item()

        if loss is not None:
            self.total_loss += loss
            self.total_ce   += ce
            self.total_dice += dice
            self.n_batches  += 1

    def compute(self):
        iou_f  = self.inter_f  / self.union_f  if self.union_f  > 0 else 0.0
        iou_nf = self.inter_nf / self.union_nf if self.union_nf > 0 else 0.0
        prec   = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0
        rec    = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        n      = max(self.n_batches, 1)
        return {
            'loss':        self.total_loss / n,
            'ce':          self.total_ce   / n,
            'dice':        self.total_dice / n,
            'iou_flood':   iou_f,
            'iou_noflood': iou_nf,
            'mean_iou':    (iou_f + iou_nf) / 2,
            'precision':   prec,
            'recall':      rec,
            'f1':          f1,
        }


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    gm   = GlobalMetrics()
    pbar = tqdm(loader, desc="Train", smoothing=0)

    for batch in pbar:
        imgs   = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        total, ce, dice = criterion(logits, labels)

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Flush MPS command buffer every step — prevents 40min->2hr epoch drift
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        with torch.no_grad():
            gm.update(logits, labels,
                      loss=total.item(), ce=ce.item(), dice=dice.item())

        pbar.set_postfix({
            'Loss':     f"{total.item():.3f}",
            'FloodIoU': f"{gm.inter_f / max(gm.union_f, 1):.3f}",
        })

        del imgs, labels, logits, total, ce, dice

    return gm.compute()


def validate(model, loader, criterion, device):
    model.eval()
    gm = GlobalMetrics()

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val  ", smoothing=0)
        for batch in pbar:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(imgs)
            total, ce, dice = criterion(logits, labels)
            gm.update(logits, labels,
                      loss=total.item(), ce=ce.item(), dice=dice.item())

            pbar.set_postfix({
                'Loss':     f"{total.item():.3f}",
                'FloodIoU': f"{gm.inter_f / max(gm.union_f, 1):.3f}",
            })

            del imgs, labels, logits, total, ce, dice

    return gm.compute()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def run_training(model, train_loader, val_loader, criterion, epochs,
                 optimizer, scheduler, patience=15, debug=False,
                 start_epoch=1, best_iou=-1.0, run_name="main"):
    """
    run_name: used to name log/checkpoint files.
              'main' -> models/best_model.pth
              anything else -> models/{run_name}_best.pth
              Allows ablation runs without overwriting the main model.
    """
    os.makedirs("logs",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if run_name == "main":
        best_pt  = "models/best_model.pth"
        last_pt  = "models/last_model.pth"
        log_csv  = "logs/train_log.csv"
        txt_path = Path("logs/train_metrics.txt")
    else:
        best_pt  = f"models/{run_name}_best.pth"
        last_pt  = f"models/{run_name}_last.pth"
        log_csv  = f"logs/{run_name}_log.csv"
        txt_path = Path(f"logs/{run_name}_metrics.txt")

    if start_epoch == 1:
        txt_path.write_text("")

    csv_header = [
        'epoch', 'lr',
        'train_loss', 'train_ce', 'train_dice',
        'train_iou_flood', 'train_iou_noflood', 'train_mean_iou',
        'train_precision', 'train_recall', 'train_f1',
        'val_loss', 'val_ce', 'val_dice',
        'val_iou_flood', 'val_iou_noflood', 'val_mean_iou',
        'val_precision', 'val_recall', 'val_f1',
    ]
    with open(log_csv, 'w' if start_epoch == 1 else 'a', newline='') as f:
        if start_epoch == 1:
            csv.writer(f).writerow(csv_header)

    best_flood_iou     = best_iou
    early_stop_counter = 0

    for epoch in range(start_epoch, epochs + 1):
        lr = optimizer.param_groups[0]['lr']

        eff_train = itertools.islice(train_loader, 20) if debug else train_loader
        eff_val   = itertools.islice(val_loader,    5) if debug else val_loader

        t = train_one_epoch(model, eff_train, optimizer, criterion, device)
        v = validate(model, eff_val, criterion, device)

        scheduler.step()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # --- CSV ---
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, lr,
                t['loss'], t['ce'], t['dice'],
                t['iou_flood'], t['iou_noflood'], t['mean_iou'],
                t['precision'], t['recall'], t['f1'],
                v['loss'], v['ce'], v['dice'],
                v['iou_flood'], v['iou_noflood'], v['mean_iou'],
                v['precision'], v['recall'], v['f1'],
            ])

        # --- Checkpoints ---
        is_best = v['iou_flood'] > best_flood_iou
        if is_best:
            best_flood_iou     = v['iou_flood']
            early_stop_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss':             v['loss'],
                'val_iou_flood':        v['iou_flood'],
                'best_val_iou_flood':   best_flood_iou,
                'run_name':             run_name,
            }, best_pt)
            print(f"  ✓ New best -> Val Flood IoU: {best_flood_iou:.4f}")
        else:
            early_stop_counter += 1

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss':             v['loss'],
            'val_iou_flood':        v['iou_flood'],
            'best_val_iou_flood':   best_flood_iou,
            'run_name':             run_name,
        }, last_pt)

        # --- TXT log ---
        block = (
            f"========================================\n"
            f"Epoch {epoch}/{epochs}  LR: {lr:.6f}  [{run_name}]\n"
            f"----------------------------------------\n"
            f"TRAIN | Loss: {t['loss']:.4f} (CE: {t['ce']:.4f}  Dice: {t['dice']:.4f})\n"
            f"  FloodIoU: {t['iou_flood']:.4f}  MeanIoU: {t['mean_iou']:.4f}\n"
            f"  Prec: {t['precision']:.4f}  Rec: {t['recall']:.4f}  F1: {t['f1']:.4f}\n"
            f"VAL   | Loss: {v['loss']:.4f} (CE: {v['ce']:.4f}  Dice: {v['dice']:.4f})\n"
            f"  FloodIoU: {v['iou_flood']:.4f}  MeanIoU: {v['mean_iou']:.4f}  <- BEST: {best_flood_iou:.4f}\n"
            f"  Prec: {v['precision']:.4f}  Rec: {v['recall']:.4f}  F1: {v['f1']:.4f}\n"
            f"  EarlyStopping: {early_stop_counter}/{patience}\n"
            f"========================================\n"
        )
        with open(txt_path, 'a') as f:
            f.write(block)
            f.flush()

        print(
            f"[{run_name}][{epoch:03d}/{epochs}] LR:{lr:.5f} | "
            f"Tr Loss:{t['loss']:.4f} FloodIoU:{t['iou_flood']:.4f} | "
            f"Va Loss:{v['loss']:.4f} FloodIoU:{v['iou_flood']:.4f} | "
            f"Best:{best_flood_iou:.4f} ES:{early_stop_counter}/{patience}"
        )

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nDone [{run_name}]. Metrics -> {txt_path.resolve()}")
    return best_flood_iou


# =============================================================================
# MAIN
# =============================================================================

def main(debug=False, resume=False):
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        print("Caffeinate started")

    TRAIN = {"Bolivia", "Colombia", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "USA"}
    VAL   = {"Ghana", "Somalia", "Spain"}
    TEST  = {"Sri-Lanka"}
    assert not (TRAIN & VAL) and not (TRAIN & TEST) and not (VAL & TEST), \
        "Region overlap detected!"
    print(f"Regions — Train:{len(TRAIN)}  Val:{len(VAL)}  Test:{len(TEST)}  ✓")

    batch_size = 2 if debug else 4
    epochs     = 2 if debug else 80

    train_loader, val_loader, _ = get_dataloaders(".", batch_size=batch_size)

    model     = build_model(in_channels=20, num_classes=2).to(device)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, ignore_index=-1).to(device)

    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith('encoder')]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 5e-5},
        {'params': decoder_params, 'lr': 2e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 1
    best_iou    = -1.0

    if resume and os.path.exists("models/last_model.pth"):
        ckpt = torch.load("models/last_model.pth", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_iou    = ckpt.get('best_val_iou_flood', ckpt['val_iou_flood'])
        print(f"Resumed from epoch {ckpt['epoch']} (best IoU: {best_iou:.4f})")

    run_training(
        model, train_loader, val_loader, criterion,
        epochs=epochs, optimizer=optimizer, scheduler=scheduler,
        patience=15, debug=debug,
        start_epoch=start_epoch, best_iou=best_iou,
        run_name="main",
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("\n--- SANITY CHECK ---")
    m = build_model(20, 2).to(device)
    x = torch.randn(2, 20, 512, 512).to(device)
    o = m(x)
    assert o.shape == (2, 2, 512, 512), f"Bad output shape: {o.shape}"
    print(f"Forward pass: {o.shape} ✓")

    c  = CombinedLoss(ignore_index=-1).to(device)
    y  = torch.randint(-1, 2, (2, 512, 512)).to(device)
    tl, ce, dl = c(o, y)
    assert tl.item() > 0
    print(f"Loss -> Total:{tl.item():.4f}  CE:{ce.item():.4f}  Dice:{dl.item():.4f} ✓")

    gm = GlobalMetrics()
    gm.update(o.detach(), y, loss=tl.item(), ce=ce.item(), dice=dl.item())
    met = gm.compute()
    assert all(isinstance(v, float) for v in met.values())
    print("GlobalMetrics:", {k: f"{v:.4f}" for k, v in met.items()}, "✓")
    print("SANITY CHECK PASSED\n" + "=" * 50 + "\n")

    main(debug=args.debug, resume=args.resume)