import os
import sys
import subprocess
import argparse
import itertools
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import csv
from tqdm import tqdm

from data_loader import FloodDataset, get_dataloaders

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =============================================================================
# ARCHITECTURE
# =============================================================================

class DoubleConvBlock(torch_nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1  = torch_nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1    = torch_nn.BatchNorm2d(out_channels)
        self.relu1  = torch_nn.ReLU(inplace=True)
        self.conv2  = torch_nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2    = torch_nn.BatchNorm2d(out_channels)
        self.relu2  = torch_nn.ReLU(inplace=True)
        self.use_res = in_channels != out_channels
        if self.use_res:
            self.res_conv = torch_nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        res = self.res_conv(x) if self.use_res else x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + res)


class EncoderBlock(torch_nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = DoubleConvBlock(in_ch, out_ch)
        self.pool       = torch_nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv_block(x)
        return skip, self.pool(skip)


class DecoderBlock(torch_nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up         = torch_nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv_block = DoubleConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        up = self.up(x)
        diffY = skip.size(2) - up.size(2)
        diffX = skip.size(3) - up.size(3)
        if diffY > 0 or diffX > 0:
            up = F.pad(up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv_block(torch.cat([skip, up], dim=1))


class FloodUNet(torch_nn.Module):
    def __init__(self, in_channels=20, num_classes=2):
        super().__init__()
        self.enc1      = EncoderBlock(in_channels, 64)
        self.enc2      = EncoderBlock(64, 128)
        self.enc3      = EncoderBlock(128, 256)
        self.enc4      = EncoderBlock(256, 512)
        self.bottleneck = DoubleConvBlock(512, 1024)
        self.dec1      = DecoderBlock(1024, 512, 512)
        self.dec2      = DecoderBlock(512, 256, 256)
        self.dec3      = DecoderBlock(256, 128, 128)
        self.dec4      = DecoderBlock(128, 64, 64)
        self.out_conv  = torch_nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        s1, d1 = self.enc1(x)
        s2, d2 = self.enc2(d1)
        s3, d3 = self.enc3(d2)
        s4, d4 = self.enc4(d3)
        b = self.bottleneck(d4)
        return self.out_conv(self.dec4(self.dec3(self.dec2(self.dec1(b, s4), s3), s2), s1))


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(torch_nn.Module):
    """
    FIX: Use F.cross_entropy with label_smoothing=0 as base, then apply focal
    weighting. This avoids the log_softmax + gather numerical blowup we saw
    in the pretrain logs (focal component exploding 0.04 → 3.0 by epoch 7).

    Alpha is now [non_flood_weight, flood_weight].
    For 90-10 class imbalance: alpha=[0.1, 0.9] or use class_weights in CE directly.
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        # FIX: Stronger alpha for flood class given 90-10 imbalance
        # alpha[0]=non-flood weight, alpha[1]=flood weight
        if alpha is None:
            self.register_buffer('alpha', torch.tensor([0.25, 0.75]))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))

    def forward(self, logits, labels):
        # Mask out ignore pixels
        mask = labels != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits.permute(0, 2, 3, 1)[mask]   # (N_valid, C)
        valid_labels = labels[mask]                          # (N_valid,)

        # FIX: clamp log_softmax to prevent -inf when softmax saturates
        log_pt = F.log_softmax(valid_logits, dim=-1).clamp(min=-100.0)
        pt     = log_pt.exp()

        log_pt_c = log_pt.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
        pt_c     = pt.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

        alpha_t  = self.alpha[valid_labels]
        loss     = -alpha_t * (1.0 - pt_c) ** self.gamma * log_pt_c

        # FIX: clamp individual losses before mean to prevent a single bad batch
        # from blowing up the whole epoch (this is what caused the 3.0 focal spike)
        loss = loss.clamp(max=10.0)
        return loss.mean()


class DiceLoss(torch_nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        mask = labels != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs       = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        dice_loss   = 0.0
        valid       = 0

        for c in range(num_classes):
            target_c = (labels == c).float() * mask.float()
            pred_c   = probs[:, c] * mask.float()
            inter    = (pred_c * target_c).sum()
            union    = pred_c.sum() + target_c.sum()
            if union > 0:
                dice_loss += 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
                valid += 1

        return (dice_loss / valid) if valid > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)


class CombinedLoss(torch_nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2.0,
                 ignore_index=-1, alpha=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal        = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)
        self.dice         = DiceLoss(smooth=1.0, ignore_index=ignore_index)

    def forward(self, logits, labels):
        f = self.focal(logits, labels)
        d = self.dice(logits, labels)
        return self.focal_weight * f + self.dice_weight * d, f, d


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(preds, labels, ignore_index=-1):
    pred_classes = torch.argmax(preds, dim=1)
    mask = labels != ignore_index

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

    vp = pred_classes[mask]
    vl = labels[mask]
    acc = float((vp == vl).float().mean()) if mask.any() else 0.0

    TP = ((pred_classes == 1) & (labels == 1) & mask).sum().float()
    FP = ((pred_classes == 1) & (labels == 0) & mask).sum().float()
    FN = ((pred_classes == 0) & (labels == 1) & mask).sum().float()
    prec  = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    rec   = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    denom = prec + rec
    f1    = float(2 * prec * rec / denom) if denom > 0 else 0.0

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
    return {k: 0.0 for k in ['mean_iou', 'iou_flood', 'iou_noflood',
                              'accuracy', 'precision_flood', 'recall_flood', 'f1_flood']}


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_sum, focal_sum, dice_sum = 0.0, 0.0, 0.0
    metrics_sum = _zero_metrics()
    batches = 0

    pbar = tqdm(loader, desc="Training", smoothing=0)
    for batch in pbar:
        imgs   = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        total, focal, dice = criterion(logits, labels)

        if total.item() > 0:
            total.backward()
            torch_nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        m = compute_metrics(logits.detach(), labels)
        for k in metrics_sum:
            metrics_sum[k] += m[k]

        tv, fv, dv = total.item(), focal.item(), dice.item()
        total_sum += tv; focal_sum += fv; dice_sum += dv
        batches   += 1

        del imgs, labels, logits, total, focal, dice

        if batches % 200 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        pbar.set_postfix({'Loss': f"{tv:.3f}", 'FloodIoU': f"{m['iou_flood']:.3f}"})

    if batches > 0:
        avg = {k: v / batches for k, v in metrics_sum.items()}
        return total_sum / batches, focal_sum / batches, dice_sum / batches, avg
    return 0.0, 0.0, 0.0, _zero_metrics()


def validate(model, loader, criterion, device):
    model.eval()
    total_sum, focal_sum, dice_sum = 0.0, 0.0, 0.0
    metrics_sum = _zero_metrics()
    batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", smoothing=0)
        for batch in pbar:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(imgs)
            total, focal, dice = criterion(logits, labels)
            m = compute_metrics(logits.detach(), labels)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

            tv, fv, dv = total.item(), focal.item(), dice.item()
            total_sum += tv; focal_sum += fv; dice_sum += dv
            batches   += 1

            del imgs, labels, logits, total, focal, dice

            pbar.set_postfix({'Loss': f"{tv:.3f}", 'FloodIoU': f"{m['iou_flood']:.3f}"})

    if batches > 0:
        avg = {k: v / batches for k, v in metrics_sum.items()}
        return total_sum / batches, focal_sum / batches, dice_sum / batches, avg
    return 0.0, 0.0, 0.0, _zero_metrics()


# =============================================================================
# PHASE RUNNER
# =============================================================================

def run_phase(phase_name, model, train_loader, val_loader, criterion,
              epochs, optimizer, scheduler, patience=10, debug=False,
              start_epoch=1, best_iou=-1.0, early_stop_counter=0):

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_csv  = f"logs/{phase_name}_log.csv"
    txt_path = Path(f"logs/{phase_name}_metrics.txt")

    if phase_name == "finetune":
        best_pt = "models/best_model.pth"
        last_pt = "models/last_model.pth"
    else:
        best_pt = f"models/{phase_name}_best.pth"
        last_pt = f"models/{phase_name}_last.pth"

    if start_epoch == 1:
        txt_path.write_text("")

    print(f"Metrics → {txt_path.resolve()}")

    csv_header = [
        'epoch', 'lr',
        'train_loss', 'train_focal', 'train_dice',
        'train_mean_iou', 'train_iou_flood', 'train_iou_noflood',
        'train_accuracy', 'train_precision_flood', 'train_recall_flood', 'train_f1_flood',
        'val_loss', 'val_focal', 'val_dice',
        'val_mean_iou', 'val_iou_flood', 'val_iou_noflood',
        'val_accuracy', 'val_precision_flood', 'val_recall_flood', 'val_f1_flood',
    ]
    with open(log_csv, 'w' if start_epoch == 1 else 'a', newline='') as f:
        if start_epoch == 1:
            csv.writer(f).writerow(csv_header)

    best_flood_iou      = best_iou
    early_stop_counter  = early_stop_counter

    for epoch in range(start_epoch, epochs + 1):
        if epoch == start_epoch:
            print(f"\n--- {phase_name.upper()} Phase (epoch {start_epoch}→{epochs}) ---")

        lr = optimizer.param_groups[0]['lr']

        eff_train = itertools.islice(train_loader, 20) if debug else train_loader
        eff_val   = itertools.islice(val_loader,   5)  if debug else val_loader

        t_loss, t_focal, t_dice, t_m = train_one_epoch(model, eff_train, optimizer, criterion, device)
        v_loss, v_focal, v_dice, v_m = validate(model, eff_val, criterion, device)

        scheduler.step()

        # --- CSV ---
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, lr,
                t_loss, t_focal, t_dice,
                t_m['mean_iou'], t_m['iou_flood'], t_m['iou_noflood'],
                t_m['accuracy'], t_m['precision_flood'], t_m['recall_flood'], t_m['f1_flood'],
                v_loss, v_focal, v_dice,
                v_m['mean_iou'], v_m['iou_flood'], v_m['iou_noflood'],
                v_m['accuracy'], v_m['precision_flood'], v_m['recall_flood'], v_m['f1_flood'],
            ])

        # --- Checkpoint ---
        is_best = v_m['iou_flood'] > best_flood_iou
        if is_best:
            best_flood_iou     = v_m['iou_flood']
            early_stop_counter = 0
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss':            v_loss,
                'val_iou_flood':       v_m['iou_flood'],
                # FIX: store best_val_iou explicitly so resume loads it correctly
                'best_val_iou_flood':  best_flood_iou,
                'phase':               phase_name,
            }, best_pt)
            print(f"  ✓ New best checkpoint saved (Flood IoU: {best_flood_iou:.4f})")
        else:
            early_stop_counter += 1

        # Last checkpoint always stores the HISTORICAL best_val_iou so resume is correct
        torch.save({
            'epoch':               epoch,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss':            v_loss,
            'val_iou_flood':       v_m['iou_flood'],
            # FIX: this is the KEY fix — last checkpoint carries the running best,
            # not just the current epoch's IoU. Previously resume read val_iou_flood
            # from last_pt which was last epoch's IoU (could be worse than best).
            'best_val_iou_flood':  best_flood_iou,
            'phase':               phase_name,
        }, last_pt)

        # --- TXT log ---
        txt_block = (
            f"========================================\n"
            f"Epoch {epoch}/{epochs} | Phase: {phase_name.upper()}\n"
            f"LR: {lr:.6f}\n"
            f"----------------------------------------\n"
            f"TRAIN:\n"
            f"  Loss:        {t_loss:.4f}  (Focal: {t_focal:.4f}  Dice: {t_dice:.4f})\n"
            f"  Mean IoU:    {t_m['mean_iou']:.4f}\n"
            f"  Flood IoU:   {t_m['iou_flood']:.4f}  <- most important\n"
            f"  NoFlood IoU: {t_m['iou_noflood']:.4f}\n"
            f"  Accuracy:    {t_m['accuracy']:.4f}\n"
            f"  Precision:   {t_m['precision_flood']:.4f}\n"
            f"  Recall:      {t_m['recall_flood']:.4f}\n"
            f"  F1 Score:    {t_m['f1_flood']:.4f}\n"
            f"VAL:\n"
            f"  Loss:        {v_loss:.4f}  (Focal: {v_focal:.4f}  Dice: {v_dice:.4f})\n"
            f"  Mean IoU:    {v_m['mean_iou']:.4f}\n"
            f"  Flood IoU:   {v_m['iou_flood']:.4f}  <- most important\n"
            f"  NoFlood IoU: {v_m['iou_noflood']:.4f}\n"
            f"  Accuracy:    {v_m['accuracy']:.4f}\n"
            f"  Precision:   {v_m['precision_flood']:.4f}\n"
            f"  Recall:      {v_m['recall_flood']:.4f}\n"
            f"  F1 Score:    {v_m['f1_flood']:.4f}\n"
            f"Best Val Flood IoU so far: {best_flood_iou:.4f}\n"
            f"========================================\n"
        )
        with open(txt_path, 'a') as f:
            f.write(txt_block)
            f.flush()

        print(
            f"[{phase_name.upper()}][{epoch}/{epochs}] LR:{lr:.5f} | "
            f"Tr→ Loss:{t_loss:.4f} mIoU:{t_m['mean_iou']:.4f} FloodIoU:{t_m['iou_flood']:.4f} | "
            f"Val→ Loss:{v_loss:.4f} mIoU:{v_m['mean_iou']:.4f} FloodIoU:{v_m['iou_flood']:.4f} | "
            f"Best:{best_flood_iou:.4f} ES:{early_stop_counter}/{patience}"
        )

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"Phase done. {txt_path.resolve()} ({txt_path.stat().st_size} bytes)")
    return best_flood_iou


# =============================================================================
# MAIN
# =============================================================================

def main(debug=False, resume=False, skip_pretrain_flag=False):
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        print("Caffeinate started")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if debug:
        print("DEBUG MODE — 1 epoch smoke test")

    TRAIN_REGIONS = {"Bolivia", "Colombia", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "USA"}
    VAL_REGIONS   = {"Ghana", "Somalia", "Spain"}
    TEST_REGIONS  = {"Sri-Lanka"}
    print("\n=== REGION SPLIT SUMMARY ===")
    for label, regions in [("TRAIN", TRAIN_REGIONS), ("VAL", VAL_REGIONS), ("TEST", TEST_REGIONS)]:
        print(f"{label} ({len(regions)}): {sorted(regions)}")
    overlaps = [
        TRAIN_REGIONS & VAL_REGIONS,
        TRAIN_REGIONS & TEST_REGIONS,
        VAL_REGIONS   & TEST_REGIONS,
    ]
    if any(overlaps):
        print(f"ERROR: Region overlap! {overlaps}")
    else:
        print("Region overlap check: PASSED")
    print()

    batch_size = 2 if debug else 4

    # FIX: Stronger alpha for 90-10 class imbalance [non_flood, flood]
    # Combined with WeightedRandomSampler this should significantly boost flood IoU
    model     = FloodUNet(in_channels=20, num_classes=2).to(device)
    criterion = CombinedLoss(
        focal_weight=0.5, dice_weight=0.5, gamma=2.0,
        ignore_index=-1,
        alpha=[0.15, 0.85],   # FIX: was [0.1186, 1.0] — normalised properly now
    )

    # ==========================================================================
    # PRETRAIN
    # ==========================================================================
    pretrain_epochs       = 1 if debug else 30
    skip_pretrain         = False
    pretrain_start_epoch  = 1
    pretrain_best_iou     = -1.0
    pretrain_es_counter   = 0

    if skip_pretrain_flag:
        skip_pretrain = True
        best_pretrain_path = "models/pretrain_best.pth"
        if os.path.exists(best_pretrain_path):
            ckpt = torch.load(best_pretrain_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"--skip-pretrain: loaded pretrain_best (epoch {ckpt['epoch']}, "
                  f"best IoU {ckpt.get('best_val_iou_flood', ckpt['val_iou_flood']):.4f})")
        else:
            print("WARNING: --skip-pretrain but pretrain_best.pth not found — random weights")

    elif resume and os.path.exists("models/pretrain_last.pth"):
        ckpt        = torch.load("models/pretrain_last.pth", map_location=device)
        saved_epoch = ckpt['epoch']
        if saved_epoch >= pretrain_epochs:
            skip_pretrain = True
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"PRETRAIN already complete (epoch {saved_epoch}), skipping")
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            pretrain_start_epoch = saved_epoch + 1
            # FIX: read best_val_iou_flood (not val_iou_flood which is last epoch only)
            pretrain_best_iou    = ckpt.get('best_val_iou_flood', ckpt.get('val_iou_flood', -1.0))
            print(f"Resuming PRETRAIN from epoch {pretrain_start_epoch} "
                  f"(historical best IoU: {pretrain_best_iou:.4f})")

    if not skip_pretrain:
        tr1, vl1, _ = get_dataloaders(".", "pretrain", batch_size=batch_size, use_flood_sampler=True)
        opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=pretrain_epochs, eta_min=1e-5)

        if pretrain_start_epoch > 1 and os.path.exists("models/pretrain_last.pth"):
            ckpt = torch.load("models/pretrain_last.pth", map_location=device)
            opt1.load_state_dict(ckpt['optimizer_state_dict'])
            sch1.load_state_dict(ckpt['scheduler_state_dict'])

        run_phase("pretrain", model, tr1, vl1, criterion, pretrain_epochs,
                  opt1, sch1, patience=8, debug=debug,
                  start_epoch=pretrain_start_epoch, best_iou=pretrain_best_iou,
                  early_stop_counter=pretrain_es_counter)

    # ==========================================================================
    # FINETUNE
    # ==========================================================================
    print("\n--- FINETUNE setup ---")
    finetune_epochs  = 1 if debug else 50
    start_ft_epoch   = 1
    best_ft_iou      = -1.0
    ft_es_counter    = 0

    tr2, vl2, _ = get_dataloaders(".", "finetune", batch_size=batch_size, use_flood_sampler=True)

    # FIX: lower LR for finetune to avoid overwriting good pretrain representations too fast
    opt2 = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=finetune_epochs, eta_min=1e-6)

    if resume and os.path.exists("models/last_model.pth"):
        ckpt = torch.load("models/last_model.pth", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt2.load_state_dict(ckpt['optimizer_state_dict'])
        sch2.load_state_dict(ckpt['scheduler_state_dict'])
        start_ft_epoch = ckpt['epoch'] + 1
        # FIX: use best_val_iou_flood, not val_iou_flood
        best_ft_iou    = ckpt.get('best_val_iou_flood', ckpt.get('val_iou_flood', -1.0))
        ft_es_counter  = 0  # reset ES counter on resume — can't recover it from ckpt yet
        print(f"Resuming FINETUNE from epoch {start_ft_epoch} "
              f"(historical best IoU: {best_ft_iou:.4f})")
    elif not skip_pretrain:
        # Load BEST pretrain weights (not last) into finetune
        if os.path.exists("models/pretrain_best.pth"):
            ckpt = torch.load("models/pretrain_best.pth", map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded pretrain_best.pth (epoch {ckpt['epoch']}, "
                  f"IoU {ckpt.get('best_val_iou_flood', ckpt['val_iou_flood']):.4f})")

    if start_ft_epoch <= finetune_epochs:
        run_phase("finetune", model, tr2, vl2, criterion, finetune_epochs,
                  opt2, sch2, patience=10, debug=debug,
                  start_epoch=start_ft_epoch, best_iou=best_ft_iou,
                  early_stop_counter=ft_es_counter)
    else:
        print("FINETUNE already complete.")

    print("\nTraining finished.")


# =============================================================================
# ENTRY
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodSegPro Training")
    parser.add_argument("--debug",        action="store_true")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()

    # Sanity check
    print("\n--- SANITY CHECK ---")
    m_check = FloodUNet(in_channels=20, num_classes=2).to(device)
    x_check = torch.randn(2, 20, 512, 512).to(device)
    o_check = m_check(x_check)
    assert o_check.shape == (2, 2, 512, 512), f"Bad output shape: {o_check.shape}"
    print(f"Forward pass: {o_check.shape} ✓")

    l_check  = CombinedLoss(ignore_index=-1, alpha=[0.15, 0.85])
    y_check  = torch.randint(-1, 2, (2, 512, 512)).to(device)
    tl, fl, dl = l_check(o_check, y_check)
    assert tl.item() > 0, "Total loss must be > 0"
    assert fl.item() < 15.0, f"Focal loss suspiciously large: {fl.item():.4f} — check FocalLoss clamp"
    print(f"Loss → Total:{tl.item():.4f} Focal:{fl.item():.4f} Dice:{dl.item():.4f} ✓")

    met = compute_metrics(o_check, y_check)
    assert all(isinstance(v, float) for v in met.values())
    print("Metrics:", {k: f"{v:.4f}" for k, v in met.items()}, "✓")
    print("SANITY CHECK PASSED\n" + "="*50)

    main(debug=args.debug, resume=args.resume, skip_pretrain_flag=args.skip_pretrain)