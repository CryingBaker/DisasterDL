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

# --- ARCHITECTURE ---

class DoubleConvBlock(torch_nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch_nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch_nn.BatchNorm2d(out_channels)
        self.relu1 = torch_nn.ReLU(inplace=True)
        
        self.conv2 = torch_nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch_nn.BatchNorm2d(out_channels)
        self.relu2 = torch_nn.ReLU(inplace=True)
        
        self.use_res = in_channels != out_channels
        if self.use_res:
            self.res_conv = torch_nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            
    def forward(self, x):
        res = x
        if self.use_res:
            res = self.res_conv(res)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + res
        out = self.relu2(out)
        return out

class EncoderBlock(torch_nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = DoubleConvBlock(in_channels, out_channels)
        self.pool = torch_nn.MaxPool2d(2)
        
    def forward(self, x):
        skip = self.conv_block(x)
        downsampled = self.pool(skip)
        return skip, downsampled

class DecoderBlock(torch_nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = torch_nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConvBlock(out_channels + skip_channels, out_channels)
        
    def forward(self, x, skip):
        upsampled = self.up(x)
        # Pad if shapes don't perfectly match
        diffY = skip.size()[2] - upsampled.size()[2]
        diffX = skip.size()[3] - upsampled.size()[3]
        if diffY > 0 or diffX > 0:
            upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        concat = torch.cat([skip, upsampled], dim=1)
        return self.conv_block(concat)

class FloodUNet(torch_nn.Module):
    def __init__(self, in_channels=20, num_classes=2):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        self.bottleneck = DoubleConvBlock(512, 1024)
        
        self.dec1 = DecoderBlock(1024, 512, 512)
        self.dec2 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec4 = DecoderBlock(128, 64, 64)
        
        self.out_conv = torch_nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        s1, d1 = self.enc1(x)
        s2, d2 = self.enc2(d1)
        s3, d3 = self.enc3(d2)
        s4, d4 = self.enc4(d3)
        
        b = self.bottleneck(d4)
        
        dec1 = self.dec1(b, s4)
        dec2 = self.dec2(dec1, s3)
        dec3 = self.dec3(dec2, s2)
        dec4 = self.dec4(dec3, s1)
        
        return self.out_conv(dec4)

# --- LOSS FUNCTIONS ---

class FocalLoss(torch_nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.tensor([0.1186, 1.0])
        else:
            self.alpha = alpha
        self.ignore_index = ignore_index
        
    def forward(self, logits, labels):
        self.alpha = self.alpha.to(logits.device)
        
        mask = labels != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        valid_logits = logits.permute(0, 2, 3, 1)[mask] # (N, C)
        valid_labels = labels[mask] # (N,)
        
        log_pt = F.log_softmax(valid_logits, dim=-1)
        pt = torch.exp(log_pt)
        
        log_pt_c = log_pt.gather(1, valid_labels.unsqueeze(1)).squeeze(-1)
        pt_c = pt.gather(1, valid_labels.unsqueeze(1)).squeeze(-1)
        
        alpha_t = self.alpha[valid_labels]
        
        loss = -alpha_t * (1 - pt_c)**self.gamma * log_pt_c
        return loss.mean()

class DiceLoss(torch_nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, logits, labels):
        mask = labels != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        probs = F.softmax(logits, dim=1)
        
        num_classes = probs.shape[1]
        dice_loss = 0.0
        
        masked_probs = probs * mask.unsqueeze(1)
        
        valid_classes = 0
        for c in range(num_classes):
            target_c = (labels == c).float() * mask.float()
            pred_c = masked_probs[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union > 0:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_loss += (1.0 - dice)
                valid_classes += 1
                
        if valid_classes > 0:
            return dice_loss / valid_classes
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

class CombinedLoss(torch_nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=1.0, ignore_index=ignore_index)
        
    def forward(self, logits, labels):
        f_loss = self.focal(logits, labels)
        d_loss = self.dice(logits, labels)
        total = self.focal_weight * f_loss + self.dice_weight * d_loss
        return total, f_loss, d_loss

# --- METRIC ---

def compute_metrics(preds, labels, ignore_index=-1):
    """Returns a dict of per-class and aggregate metrics, excluding ignore_index pixels.
    All values are guaranteed to be plain Python floats (never torch tensors)."""
    pred_classes = torch.argmax(preds, dim=1)
    mask = labels != ignore_index

    # Per-class IoU
    def _iou(c):
        p = (pred_classes == c) & mask
        g = (labels == c) & mask
        inter = (p & g).sum().float()
        union = (p | g).sum().float()
        return float(inter.item() / union.item()) if union.item() > 0 else 0.0

    iou_noflood = _iou(0)
    iou_flood   = _iou(1)
    valid = sum(1 for v in [iou_noflood, iou_flood] if v > 0)
    mean_iou = float((iou_noflood + iou_flood) / valid) if valid > 0 else 0.0

    # Pixel accuracy
    valid_preds  = pred_classes[mask]
    valid_labels = labels[mask]
    accuracy = float((valid_preds == valid_labels).float().mean().item()) if mask.any() else 0.0

    # Flood (class 1) precision / recall / F1
    TP = ((pred_classes == 1) & (labels == 1) & mask).sum().float()
    FP = ((pred_classes == 1) & (labels == 0) & mask).sum().float()
    FN = ((pred_classes == 0) & (labels == 1) & mask).sum().float()

    precision_flood = float((TP / (TP + FP)).item()) if (TP + FP).item() > 0 else 0.0
    recall_flood    = float((TP / (TP + FN)).item()) if (TP + FN).item() > 0 else 0.0
    denom_f1 = precision_flood + recall_flood
    f1_flood = float(2 * precision_flood * recall_flood / denom_f1) if denom_f1 > 0 else 0.0

    return {
        'mean_iou':        float(mean_iou),
        'iou_flood':       float(iou_flood),
        'iou_noflood':     float(iou_noflood),
        'accuracy':        float(accuracy),
        'precision_flood': float(precision_flood),
        'recall_flood':    float(recall_flood),
        'f1_flood':        float(f1_flood),
    }


def _zero_metrics():
    return {k: 0.0 for k in ['mean_iou','iou_flood','iou_noflood',
                              'accuracy','precision_flood','recall_flood','f1_flood']}

# --- TRAINING FUNCTIONS ---

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss_sum = 0.0
    focal_loss_sum = 0.0
    dice_loss_sum  = 0.0
    metrics_sum    = _zero_metrics()
    batches = 0

    pbar = tqdm(loader, desc="Training", smoothing=0)  # FIX 3E: disable tqdm smoothing
    for batch in pbar:
        imgs   = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        total, focal, dice = criterion(logits, labels)
        if total.item() > 0:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        m = compute_metrics(logits.detach(), labels)
        for k in metrics_sum: metrics_sum[k] += m[k]

        total_loss_val = total.item()
        focal_loss_val = focal.item()
        dice_loss_val  = dice.item()
        total_loss_sum += total_loss_val
        focal_loss_sum += focal_loss_val
        dice_loss_sum  += dice_loss_val
        batches += 1

        # FIX 3A: delete intermediate tensors to free MPS memory
        del imgs, labels, logits, total, focal, dice

        # FIX 3D: periodic MPS cache clear every 200 batches
        if batches % 200 == 0 and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        pbar.set_postfix({'Loss': f"{total_loss_val:.3f}",
                          'FloodIoU': f"{m['iou_flood']:.3f}"})

    if batches > 0:
        avg = {k: v / batches for k, v in metrics_sum.items()}
        return total_loss_sum/batches, focal_loss_sum/batches, dice_loss_sum/batches, avg
    return 0.0, 0.0, 0.0, _zero_metrics()

def validate(model, loader, criterion, device):
    model.eval()

    total_loss_sum = 0.0
    focal_loss_sum = 0.0
    dice_loss_sum  = 0.0
    metrics_sum    = _zero_metrics()
    batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", smoothing=0)  # FIX 3E: disable tqdm smoothing
        for batch in pbar:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(imgs)
            total, focal, dice = criterion(logits, labels)
            # FIX 3B: detach logits before compute_metrics
            m = compute_metrics(logits.detach(), labels)
            for k in metrics_sum: metrics_sum[k] += m[k]

            total_loss_val = total.item()
            focal_loss_val = focal.item()
            dice_loss_val  = dice.item()
            total_loss_sum += total_loss_val
            focal_loss_sum += focal_loss_val
            dice_loss_sum  += dice_loss_val
            batches += 1

            # FIX 3B: delete intermediate tensors to free MPS memory
            del imgs, labels, logits, total, focal, dice

            pbar.set_postfix({'Loss': f"{total_loss_val:.3f}",
                              'FloodIoU': f"{m['iou_flood']:.3f}"})

    if batches > 0:
        avg = {k: v / batches for k, v in metrics_sum.items()}
        return total_loss_sum/batches, focal_loss_sum/batches, dice_loss_sum/batches, avg
    return 0.0, 0.0, 0.0, _zero_metrics()

# --- MAIN ---

def run_phase(phase_name, model, train_loader, val_loader, criterion, epochs, optimizer, scheduler, patience=10, debug=False, start_epoch=1, best_iou=-1.0, early_stop_counter=0):
    best_flood_iou   = best_iou
    best_val_flood_iou = best_iou  # alias used in txt block
    # early_stop_counter is passed as a parameter (default 0, or restored from checkpoint)

    # FIX 2: Reliable metrics txt setup at phase start
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log_csv  = f"logs/{phase_name}_log.csv"
    # Use Path object for txt — enables .resolve(), .stat(), .write_text(), .flush()
    txt_path = Path(f"logs/{phase_name}_metrics.txt")
    # Finetune phase produces the final deliverable; use canonical names
    if phase_name == "finetune":
        best_pt = "models/best_model.pth"
        last_pt = "models/last_model.pth"
    else:
        best_pt = f"models/{phase_name}_best.pth"
        last_pt = f"models/{phase_name}_last.pth"

    # Wipe the file clean at phase start ONLY IF not resuming
    if start_epoch == 1:
        txt_path.write_text("")
    print(f"Metrics will be saved to: {txt_path.resolve()}")

    csv_header = [
        'epoch', 'lr',
        'train_loss', 'train_focal', 'train_dice',
        'train_mean_iou', 'train_iou_flood', 'train_iou_noflood',
        'train_accuracy', 'train_precision_flood', 'train_recall_flood', 'train_f1_flood',
        'val_loss', 'val_focal', 'val_dice',
        'val_mean_iou', 'val_iou_flood', 'val_iou_noflood',
        'val_accuracy', 'val_precision_flood', 'val_recall_flood', 'val_f1_flood',
    ]
    csv_mode = 'w' if start_epoch == 1 else 'a'
    with open(log_csv, csv_mode, newline='') as f:
        if start_epoch == 1:
            csv.writer(f).writerow(csv_header)

    for epoch in range(start_epoch, epochs + 1):
        if epoch == start_epoch:
            print(f"\n--- Starting {phase_name.upper()} Phase (from epoch {start_epoch}) ---")

        lr = optimizer.param_groups[0]['lr']

        # Debug mode: limit to first N batches
        eff_train = itertools.islice(train_loader, 20) if debug else train_loader
        eff_val   = itertools.islice(val_loader, 5)   if debug else val_loader

        t_loss, t_focal, t_dice, t_m = train_one_epoch(model, eff_train, optimizer, criterion, device)
        v_loss, v_focal, v_dice, v_m = validate(model, eff_val, criterion, device)

        scheduler.step()

        # --- CSV log ---
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

        # --- Best checkpoint (monitor val_iou_flood) ---
        is_best = v_m['iou_flood'] > best_flood_iou
        if is_best:
            best_flood_iou = v_m['iou_flood']
            best_val_flood_iou = best_flood_iou  # keep alias in sync
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': v_loss,
                'val_iou_flood': v_m['iou_flood'],
                'phase': phase_name
            }, best_pt)
        else:
            early_stop_counter += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': v_loss,
            'val_iou_flood': v_m['iou_flood'],
            'phase': phase_name
        }, last_pt)

        # --- Human-readable TXT log (FIX 2) ---
        t_metrics = t_m  # alias for readability in format block
        v_metrics = v_m
        txt_block = (
            f"========================================\n"
            f"Epoch {epoch}/{epochs} | Phase: {phase_name.upper()}\n"
            f"LR: {lr:.6f}\n"
            f"----------------------------------------\n"
            f"TRAIN:\n"
            f"  Loss:        {t_loss:.4f}  (Focal: {t_focal:.4f}  Dice: {t_dice:.4f})\n"
            f"  Mean IoU:    {t_metrics['mean_iou']:.4f}\n"
            f"  Flood IoU:   {t_metrics['iou_flood']:.4f}  <- most important\n"
            f"  NoFlood IoU: {t_metrics['iou_noflood']:.4f}\n"
            f"  Accuracy:    {t_metrics['accuracy']:.4f}\n"
            f"  Precision:   {t_metrics['precision_flood']:.4f}\n"
            f"  Recall:      {t_metrics['recall_flood']:.4f}\n"
            f"  F1 Score:    {t_metrics['f1_flood']:.4f}\n"
            f"VAL:\n"
            f"  Loss:        {v_loss:.4f}  (Focal: {v_focal:.4f}  Dice: {v_dice:.4f})\n"
            f"  Mean IoU:    {v_metrics['mean_iou']:.4f}\n"
            f"  Flood IoU:   {v_metrics['iou_flood']:.4f}  <- most important\n"
            f"  NoFlood IoU: {v_metrics['iou_noflood']:.4f}\n"
            f"  Accuracy:    {v_metrics['accuracy']:.4f}\n"
            f"  Precision:   {v_metrics['precision_flood']:.4f}\n"
            f"  Recall:      {v_metrics['recall_flood']:.4f}\n"
            f"  F1 Score:    {v_metrics['f1_flood']:.4f}\n"
            f"Best Val Flood IoU so far: {best_val_flood_iou:.4f}\n"
            f"========================================\n"
        )
        with open(txt_path, 'a') as f:
            f.write(txt_block)
            f.flush()
        # Verify write succeeded
        if txt_path.stat().st_size == 0:
            print("WARNING: TXT WRITE FAILED")

        # --- Console line ---
        print(
            f"[{phase_name.upper()}][Epoch {epoch}/{epochs}] LR: {lr:.6f} | "
            f"Train → Loss: {t_loss:.4f} | mIoU: {t_m['mean_iou']:.4f} | "
            f"Flood IoU: {t_m['iou_flood']:.4f} | F1: {t_m['f1_flood']:.4f} | "
            f"Val → Loss: {v_loss:.4f} | mIoU: {v_m['mean_iou']:.4f} | "
            f"Flood IoU: {v_m['iou_flood']:.4f} | F1: {v_m['f1_flood']:.4f} | "
            f"Best: {best_flood_iou:.4f}"
        )

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no Flood IoU improvement for {patience} epochs)!")
            break

    # FIX 2: Print final path + file size at end of phase
    print(f"Training complete. Metrics saved to: {txt_path.resolve()}")
    print(f"File size: {txt_path.stat().st_size} bytes")

def main(debug=False, resume=False, skip_pretrain_flag=False):
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        print("Caffeinate started — Mac will not sleep during training")

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if debug:
        print("RUNNING IN DEBUG MODE - 1 EPOCH SMOKE TEST")
    
    # Print region-split summary to confirm zero overlap
    TRAIN_REGIONS = {"Bolivia", "Colombia", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "USA"}
    VAL_REGIONS   = {"Ghana", "Somalia", "Spain"}
    TEST_REGIONS  = {"Sri-Lanka"}
    print("\n=== REGION SPLIT SUMMARY ===")
    print(f"TRAIN regions ({len(TRAIN_REGIONS)}): {sorted(TRAIN_REGIONS)}")
    print(f"VAL   regions ({len(VAL_REGIONS)}):   {sorted(VAL_REGIONS)}")
    print(f"TEST  regions ({len(TEST_REGIONS)}):  {sorted(TEST_REGIONS)}")
    overlap_tv = TRAIN_REGIONS & VAL_REGIONS
    overlap_tt = TRAIN_REGIONS & TEST_REGIONS
    overlap_vt = VAL_REGIONS   & TEST_REGIONS
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"ERROR: Region overlaps detected! TV={overlap_tv} TT={overlap_tt} VT={overlap_vt}")
    else:
        print("Region overlap check: PASSED — zero overlap between train/val/test")
    print("="*28 + "\n")
    
    batch_size = 2 if debug else 4
    
    # Init Model & Loss
    model = FloodUNet(in_channels=20, num_classes=2).to(device)
    criterion = CombinedLoss(focal_weight=0.5, dice_weight=0.5, gamma=2.0, ignore_index=-1)
    
    # --- PRETRAIN PHASE ---
    pretrain_epochs = 1 if debug else 30
    skip_pretrain = False
    pretrain_start_epoch = 1
    pretrain_best_iou = -1.0
    pretrain_early_stop_counter = 0

    if skip_pretrain_flag:
        # Explicitly requested skip — load best pretrain weights and jump to finetune
        skip_pretrain = True
        best_pretrain_path = "models/pretrain_best.pth"
        if os.path.exists(best_pretrain_path):
            checkpoint = torch.load(best_pretrain_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Skipping PRETRAIN: loading pretrain_best.pth directly")
        else:
            print("WARNING: --skip-pretrain specified but models/pretrain_best.pth not found — starting finetune with random weights")

    elif resume and os.path.exists("models/pretrain_last.pth"):
        checkpoint = torch.load("models/pretrain_last.pth", map_location=device)
        saved_epoch = checkpoint['epoch']
        if saved_epoch >= pretrain_epochs:
            # Pretrain is fully complete — skip it
            skip_pretrain = True
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nPRETRAIN complete at epoch {saved_epoch}, skipping to finetune")
        else:
            # Pretrain is NOT complete — resume from where we left off
            skip_pretrain = False
            model.load_state_dict(checkpoint['model_state_dict'])
            pretrain_start_epoch = saved_epoch + 1
            pretrain_best_iou = checkpoint.get('val_iou_flood', -1.0)
            print(f"Resuming PRETRAIN from epoch {pretrain_start_epoch}/{pretrain_epochs}")

    if not skip_pretrain:
        tr_loader_1, vl_loader_1, _ = get_dataloaders(root=".", phase="pretrain", batch_size=batch_size)
        opt_1 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch_1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_1, T_max=pretrain_epochs, eta_min=1e-5)
        # If resuming mid-phase, restore optimizer and scheduler state
        if pretrain_start_epoch > 1 and os.path.exists("models/pretrain_last.pth"):
            resume_ckpt = torch.load("models/pretrain_last.pth", map_location=device)
            opt_1.load_state_dict(resume_ckpt['optimizer_state_dict'])
            sch_1.load_state_dict(resume_ckpt['scheduler_state_dict'])
        run_phase("pretrain", model, tr_loader_1, vl_loader_1, criterion, pretrain_epochs, opt_1, sch_1,
                  patience=8, debug=debug, start_epoch=pretrain_start_epoch, best_iou=pretrain_best_iou,
                  early_stop_counter=pretrain_early_stop_counter)
    
    # --- FINETUNE PHASE ---
    print("\n--- Starting FINETUNE Phase setup ---")
    finetune_epochs = 1 if debug else 50
    start_ft_epoch = 1
    best_ft_iou = -1.0
    
    tr_loader_2, vl_loader_2, _ = get_dataloaders(root=".", phase="finetune", batch_size=batch_size)
    opt_2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch_2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_2, T_max=finetune_epochs, eta_min=1e-6)

    # Resume finetune logic
    if resume and os.path.exists("models/last_model.pth"):
        checkpoint = torch.load("models/last_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_2.load_state_dict(checkpoint['optimizer_state_dict'])
        sch_2.load_state_dict(checkpoint['scheduler_state_dict'])
        start_ft_epoch = checkpoint['epoch'] + 1
        best_ft_iou = checkpoint['val_iou_flood']
        print(f"Resuming FINETUNE from epoch {start_ft_epoch} (checkpoint was at {checkpoint['epoch']} with Best IoU: {best_ft_iou:.4f})")
    elif not skip_pretrain:
        # If we didn't skip pretrain and didn't resume finetune, load best pretrain weights
        best_pretrain = "models/pretrain_best.pth"
        if os.path.exists(best_pretrain):
            checkpoint = torch.load(best_pretrain, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrain best checkpoint from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")

    if start_ft_epoch <= finetune_epochs:
        run_phase("finetune", model, tr_loader_2, vl_loader_2, criterion, finetune_epochs, opt_2, sch_2, patience=10, debug=debug, start_epoch=start_ft_epoch, best_iou=best_ft_iou)
    else:
        print("FINETUNE already completed all epochs.")
        
    print("\nTraining completed safely.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodSegPro Training")
    parser.add_argument("--debug", action="store_true", help="Run 1-epoch smoke test with limited batches")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints if available")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretrain phase and load pretrain_best.pth directly")
    args = parser.parse_args()

    print("\n--- SANITY CHECK BLOCK ---")
    sanity_model = FloodUNet(in_channels=20, num_classes=2).to(device)
    sanity_input = torch.randn(2, 20, 512, 512).to(device)
    sanity_out = sanity_model(sanity_input)
    print(f"Forward pass output shape: {sanity_out.shape}")
    assert sanity_out.shape == (2, 2, 512, 512), "Output shape is incorrect!"

    sanity_loss = CombinedLoss(ignore_index=-1)
    sanity_labels = torch.randint(-1, 2, (2, 512, 512)).to(device)
    t_loss, f_loss, d_loss = sanity_loss(sanity_out, sanity_labels)
    print(f"Losses -> Total: {t_loss.item():.4f}, Focal: {f_loss.item():.4f}, Dice: {d_loss.item():.4f}")
    assert t_loss.item() > 0, "Total loss must be positive!"

    sanity_metrics = compute_metrics(sanity_out, sanity_labels)
    print("Metrics sanity check:")
    for k, v in sanity_metrics.items():
        print(f"  {k}: {v:.4f}")
    assert all(isinstance(v, float) for v in sanity_metrics.values()), "Metrics must all be floats!"

    print("MODEL AND LOSS SANITY CHECK PASSED\n")
    print("="*50)
    main(debug=args.debug, resume=args.resume, skip_pretrain_flag=args.skip_pretrain)
