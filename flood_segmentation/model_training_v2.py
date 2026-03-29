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

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# MODEL
# =============================================================================

def build_model(in_channels=20, num_classes=2):
    """
    UNet with EfficientNet-B4 encoder, ImageNet pretrained.
    smp handles 20-channel input by averaging the pretrained 3-channel
    stem weights across all 20 input channels — far better than random init.
    """
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
        activation=None,           # raw logits — we apply softmax in loss/metrics
        decoder_attention_type="scse",  # squeeze-and-excitation in decoder
    )
    return model


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
        dice  = 0.0
        valid = 0

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
    Flood class gets 4x weight to handle 90/10 class imbalance.
    CE is numerically stable on MPS; replaces the focal loss that was
    causing the val focal spikes (0.4 → 1.1 → 3.0) seen in earlier runs.
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
        ce_l = F.cross_entropy(
            logits, labels,
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
        )
        d_l   = self.dice(logits, labels)
        total = self.ce_weight * ce_l + self.dice_weight * d_l
        return total, ce_l, d_l


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(preds, labels, ignore_index=-1):
    pred_classes = torch.argmax(preds, dim=1)
    mask         = labels != ignore_index

    def _iou(c):
        p     = (pred_classes == c) & mask
        g     = (labels == c) & mask
        inter = (p & g).sum().float()
        union = (p | g).sum().float()
        return float(inter / union) if union > 0 else 0.0

    iou_nf   = _iou(0)
    iou_f    = _iou(1)
    valid    = sum(1 for v in [iou_nf, iou_f] if v > 0)
    mean_iou = (iou_nf + iou_f) / valid if valid > 0 else 0.0

    vp  = pred_classes[mask]
    vl  = labels[mask]
    acc = float((vp == vl).float().mean()) if mask.any() else 0.0

    TP   = ((pred_classes == 1) & (labels == 1) & mask).sum().float()
    FP   = ((pred_classes == 1) & (labels == 0) & mask).sum().float()
    FN   = ((pred_classes == 0) & (labels == 1) & mask).sum().float()
    prec = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    rec  = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    f1   = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        'mean_iou':        float(mean_iou),
        'iou_flood':       float(iou_f),
        'iou_noflood':     float(iou_nf),
        'accuracy':        float(acc),
        'precision_flood': float(prec),
        'recall_flood':    float(rec),
        'f1_flood':        float(f1),
    }


def _zero_metrics():
    return {k: 0.0 for k in [
        'mean_iou', 'iou_flood', 'iou_noflood',
        'accuracy', 'precision_flood', 'recall_flood', 'f1_flood'
    ]}


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_sum, ce_sum, dice_sum = 0.0, 0.0, 0.0
    metrics_sum = _zero_metrics()
    batches     = 0

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

        # KEY FIX: flush MPS command buffer every step to prevent
        # the queue buildup that causes 40min → 2hr epoch drift
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        m = compute_metrics(logits.detach(), labels)
        for k in metrics_sum:
            metrics_sum[k] += m[k]

        tv, cv, dv = total.item(), ce.item(), dice.item()
        total_sum += tv; ce_sum += cv; dice_sum += dv
        batches   += 1

        del imgs, labels, logits, total, ce, dice

        pbar.set_postfix({'Loss': f"{tv:.3f}", 'FloodIoU': f"{m['iou_flood']:.3f}"})

    if batches > 0:
        avg = {k: v / batches for k, v in metrics_sum.items()}
        return total_sum / batches, ce_sum / batches, dice_sum / batches, avg
    return 0.0, 0.0, 0.0, _zero_metrics()


def validate(model, loader, criterion, device):
    model.eval()
    total_sum, ce_sum, dice_sum = 0.0, 0.0, 0.0
    metrics_sum = _zero_metrics()
    batches     = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val  ", smoothing=0)
        for batch in pbar:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(imgs)
            total, ce, dice = criterion(logits, labels)
            m = compute_metrics(logits.detach(), labels)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

            tv, cv, dv = total.item(), ce.item(), dice.item()
            total_sum += tv; ce_sum += cv; dice_sum += dv
            batches   += 1

            del imgs, labels, logits, total, ce, dice
            pbar.set_postfix({'Loss': f"{tv:.3f}", 'FloodIoU': f"{m['iou_flood']:.3f}"})

    if batches > 0:
        avg = {k: v / batches for k, v in metrics_sum.items()}
        return total_sum / batches, ce_sum / batches, dice_sum / batches, avg
    return 0.0, 0.0, 0.0, _zero_metrics()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def run_training(model, train_loader, val_loader, criterion, epochs,
                 optimizer, scheduler, patience=12, debug=False,
                 start_epoch=1, best_iou=-1.0):

    os.makedirs("logs",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_csv  = "logs/train_log.csv"
    txt_path = Path("logs/train_metrics.txt")
    best_pt  = "models/best_model.pth"
    last_pt  = "models/last_model.pth"

    if start_epoch == 1:
        txt_path.write_text("")

    csv_header = [
        'epoch', 'lr',
        'train_loss', 'train_ce', 'train_dice',
        'train_mean_iou', 'train_iou_flood', 'train_iou_noflood',
        'train_accuracy', 'train_precision_flood', 'train_recall_flood', 'train_f1_flood',
        'val_loss', 'val_ce', 'val_dice',
        'val_mean_iou', 'val_iou_flood', 'val_iou_noflood',
        'val_accuracy', 'val_precision_flood', 'val_recall_flood', 'val_f1_flood',
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

        t_loss, t_ce, t_dice, t_m = train_one_epoch(
            model, eff_train, optimizer, criterion, device)
        v_loss, v_ce, v_dice, v_m = validate(
            model, eff_val, criterion, device)

        scheduler.step()

        # Flush MPS cache at epoch end
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # --- CSV ---
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, lr,
                t_loss, t_ce, t_dice,
                t_m['mean_iou'], t_m['iou_flood'], t_m['iou_noflood'],
                t_m['accuracy'], t_m['precision_flood'], t_m['recall_flood'], t_m['f1_flood'],
                v_loss, v_ce, v_dice,
                v_m['mean_iou'], v_m['iou_flood'], v_m['iou_noflood'],
                v_m['accuracy'], v_m['precision_flood'], v_m['recall_flood'], v_m['f1_flood'],
            ])

        # --- Checkpoints ---
        is_best = v_m['iou_flood'] > best_flood_iou
        if is_best:
            best_flood_iou     = v_m['iou_flood']
            early_stop_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss':             v_loss,
                'val_iou_flood':        v_m['iou_flood'],
                'best_val_iou_flood':   best_flood_iou,
            }, best_pt)
            print(f"  ✓ New best saved → Flood IoU: {best_flood_iou:.4f}")
        else:
            early_stop_counter += 1

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss':             v_loss,
            'val_iou_flood':        v_m['iou_flood'],
            'best_val_iou_flood':   best_flood_iou,
        }, last_pt)

        # --- TXT log ---
        block = (
            f"========================================\n"
            f"Epoch {epoch}/{epochs}  LR: {lr:.6f}\n"
            f"----------------------------------------\n"
            f"TRAIN | Loss: {t_loss:.4f} (CE: {t_ce:.4f}  Dice: {t_dice:.4f})\n"
            f"  FloodIoU: {t_m['iou_flood']:.4f}  MeanIoU: {t_m['mean_iou']:.4f}\n"
            f"  Prec: {t_m['precision_flood']:.4f}  Rec: {t_m['recall_flood']:.4f}  F1: {t_m['f1_flood']:.4f}\n"
            f"VAL   | Loss: {v_loss:.4f} (CE: {v_ce:.4f}  Dice: {v_dice:.4f})\n"
            f"  FloodIoU: {v_m['iou_flood']:.4f}  MeanIoU: {v_m['mean_iou']:.4f}  ← BEST: {best_flood_iou:.4f}\n"
            f"  Prec: {v_m['precision_flood']:.4f}  Rec: {v_m['recall_flood']:.4f}  F1: {v_m['f1_flood']:.4f}\n"
            f"  EarlyStopping: {early_stop_counter}/{patience}\n"
            f"========================================\n"
        )
        with open(txt_path, 'a') as f:
            f.write(block)
            f.flush()

        print(
            f"[{epoch:03d}/{epochs}] LR:{lr:.5f} | "
            f"Tr Loss:{t_loss:.4f} FloodIoU:{t_m['iou_flood']:.4f} | "
            f"Va Loss:{v_loss:.4f} FloodIoU:{v_m['iou_flood']:.4f} | "
            f"Best:{best_flood_iou:.4f} ES:{early_stop_counter}/{patience}"
        )

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nDone. Metrics → {txt_path.resolve()}")
    return best_flood_iou


# =============================================================================
# MAIN
# =============================================================================

def main(debug=False, resume=False):
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        print("Caffeinate started")

    # Region split sanity check
    TRAIN = {"Bolivia", "Colombia", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "USA"}
    VAL   = {"Ghana", "Somalia", "Spain"}
    TEST  = {"Sri-Lanka"}
    assert not (TRAIN & VAL) and not (TRAIN & TEST) and not (VAL & TEST), \
        "Region overlap detected!"
    print(f"Regions — Train:{len(TRAIN)}  Val:{len(VAL)}  Test:{len(TEST)}  ✓")

    batch_size = 2 if debug else 4
    epochs     = 2 if debug else 60

    # --- Data ---
    train_loader, val_loader, _ = get_dataloaders(".", batch_size=batch_size)

    # --- Model ---
    model     = build_model(in_channels=20, num_classes=2).to(device)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, ignore_index=-1).to(device)

    # --- Optimizer & Scheduler ---
    # Separate LRs: lower for pretrained encoder, higher for randomly-init decoder
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith('encoder')]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 5e-5},   # fine-tune pretrained encoder slowly
        {'params': decoder_params, 'lr': 2e-4},   # train decoder faster
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # --- Resume ---
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

    # --- Train ---
    run_training(
        model, train_loader, val_loader, criterion,
        epochs=epochs, optimizer=optimizer, scheduler=scheduler,
        patience=12, debug=debug,
        start_epoch=start_epoch, best_iou=best_iou,
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Sanity check
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
    print(f"Loss → Total:{tl.item():.4f}  CE:{ce.item():.4f}  Dice:{dl.item():.4f} ✓")

    met = compute_metrics(o, y)
    assert all(isinstance(v, float) for v in met.values())
    print("Metrics:", {k: f"{v:.4f}" for k, v in met.items()}, "✓")
    print("SANITY CHECK PASSED\n" + "=" * 50 + "\n")

    main(debug=args.debug, resume=args.resume)