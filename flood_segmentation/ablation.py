import os
import sys
import argparse
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import csv
from tqdm import tqdm

from data_loader import FloodDataset, get_dataloaders, build_flood_sampler
from torch.utils.data import DataLoader
from model_training import CombinedLoss, GlobalMetrics, train_one_epoch, validate, run_training

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# ABLATION CHANNEL CONFIGURATIONS
# =============================================================================

ABLATION_CONFIGS = {
    "s1_only": {
        "channels":    list(range(0, 2)), 
        "in_channels": 2,
        "description": "Sentinel-1 SAR only (VV+VH)",
    },
    "s1_s2": {
        "channels":    list(range(0, 8)),
        "in_channels": 8,
        "description": "S1 + S2 optical (8ch)",
    },
    "s1_s2_pre": {
        "channels":    list(range(0, 16)), 
        "in_channels": 16,
        "description": "S1 + S2 + pre-event S1/S2 (16ch)",
    },
    "full_20ch": {
        "channels":    list(range(0, 20)),
        "in_channels": 20,
        "description": "Full 20-channel fusion (S1+S2+pre+aux)",
    },
}

# =============================================================================
# CHANNEL-MASKED DATASET WRAPPER
# =============================================================================

class ChannelMaskedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, keep_channels):
        self.base    = base_dataset
        self.keep    = keep_channels
        self.mask    = torch.zeros(20, dtype=torch.bool)
        for c in keep_channels:
            self.mask[c] = True

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img  = item['image'].clone()
        img  = img[self.mask]
        return {**item, 'image': img}

    def flood_pixel_fraction(self, idx):
        return self.base.flood_pixel_fraction(idx)


def get_ablation_dataloaders(root, variant_cfg, batch_size=4):
    train_ds = FloodDataset(root, "train")
    val_ds   = FloodDataset(root, "val")
    test_ds  = FloodDataset(root, "test")

    keep = variant_cfg["channels"]

    train_masked = ChannelMaskedDataset(train_ds, keep)
    val_masked   = ChannelMaskedDataset(val_ds,   keep)
    test_masked  = ChannelMaskedDataset(test_ds,  keep)

    sampler = build_flood_sampler(train_masked, flood_multiplier=5.0)

    train_loader = DataLoader(
        train_masked, batch_size=batch_size,
        sampler=sampler, drop_last=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_masked, batch_size=batch_size,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_masked, batch_size=batch_size,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def build_ablation_model(in_channels):
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=2,
        activation=None,
        decoder_attention_type="scse",
    )


def evaluate_on_test(model, root, variant_cfg, batch_size=4):
    test_ds     = FloodDataset(root, "test")
    test_masked = ChannelMaskedDataset(test_ds, variant_cfg["channels"])
    test_loader = DataLoader(
        test_masked, batch_size=batch_size,
        shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )

    model.eval()
    gm = GlobalMetrics()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test eval"):
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(imgs)
            gm.update(logits, labels)
            del imgs, labels, logits

    return gm.compute()


def run_ablation_variant(variant_name, variant_cfg, debug=False, epochs=60):
    print(f"\n{'='*60}")
    print(f"ABLATION: {variant_name} — {variant_cfg['description']}")
    print(f"{'='*60}")

    in_ch     = variant_cfg["in_channels"]
    batch_size = 2 if debug else 4
    criterion  = CombinedLoss(ce_weight=0.5, dice_weight=0.5, ignore_index=-1).to(device)

    tr, vl, _ = get_ablation_dataloaders(".", variant_cfg, batch_size)

    model  = build_ablation_model(in_ch).to(device)
    enc_p  = list(model.encoder.parameters())
    dec_p  = [p for n, p in model.named_parameters() if not n.startswith('encoder')]
    opt    = torch.optim.AdamW([
        {'params': enc_p, 'lr': 5e-5},
        {'params': dec_p, 'lr': 1e-4},
    ], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs if not debug else 1, eta_min=1e-6)

    # Monkey patch the save paths in run_training for ablations so they don't overwrite each other
    original_run_training = run_training
    
    def ablation_run_training(*args, **kwargs):
        # Using a slight hack to redirect save paths without rewriting run_training
        import train
        train.os.makedirs("models", exist_ok=True)
        train.os.makedirs("logs", exist_ok=True)
        
        # Manually alter the string paths in the train namespace
        old_best = "models/best_model.pth"
        old_last = "models/last_model.pth"
        old_csv  = "logs/training_log.csv"
        old_txt  = "logs/training_metrics.txt"
        
        try:
            # We rewrite the training loop variables inline
            pass
        except:
            pass
            
        return original_run_training(*args, **kwargs)

    # For pure safety without hacks, let's just use the train loop as-is but rename output immediately after
    # Pass the prefix here so it names everything perfectly from the start
    best_iou = run_training(
        model, tr, vl, criterion,
        epochs=1 if debug else epochs,
        optimizer=opt, scheduler=sch,
        patience=12, debug=debug,
        prefix=f"ablation_{variant_name}_" 
    )
    
    # --- TEST EVAL ---
    ft_path = f"models/ablation_{variant_name}_best_model.pth"
    
    # --- TEST EVAL ---
    if os.path.exists(ft_path):
        ckpt = torch.load(ft_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        test_m = evaluate_on_test(model, ".", variant_cfg)
        print(f"\n[{variant_name}] TEST RESULTS:")
        for k, v in test_m.items():
            if k not in ['loss', 'ce', 'dice']:
                print(f"  {k:<15} {v:.4f}")
        return test_m
    return None

def eval_all_ablations():
    print("\n" + "="*60)
    print("ABLATION EVALUATION SUMMARY")
    print("="*60)

    results = {}

    for variant_name, variant_cfg in ABLATION_CONFIGS.items():
        ft_path = f"models/ablation_{variant_name}_best.pth"

        if variant_name == "full_20ch" and not os.path.exists(ft_path):
            ft_path = "models/best_model.pth"

        if not os.path.exists(ft_path):
            print(f"  {variant_name:<15} — checkpoint not found, skipping")
            continue

        in_ch  = variant_cfg["in_channels"]
        model  = build_ablation_model(in_ch).to(device)
        ckpt   = torch.load(ft_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        m = evaluate_on_test(model, ".", variant_cfg)
        results[variant_name] = m
        print(f"\n  {variant_name} ({variant_cfg['description']})")
        print(f"    Test Flood IoU: {m['iou_flood']:.4f}")
        print(f"    Test Mean IoU:  {m['mean_iou']:.4f}")
        print(f"    Test F1:        {m['f1']:.4f}")

    if results:
        print("\n" + "="*60)
        print("SUMMARY TABLE (sorted by Test Flood IoU)")
        print("="*60)
        print(f"{'Variant':<15} {'Channels':>8} {'FloodIoU':>10} {'MeanIoU':>10} {'F1':>8}")
        print("-"*55)
        for name, m in sorted(results.items(), key=lambda x: x[1]['iou_flood'], reverse=True):
            cfg = ABLATION_CONFIGS[name]
            print(f"{name:<15} {cfg['in_channels']:>8} {m['iou_flood']:>10.4f} {m['mean_iou']:>10.4f} {m['f1']:>8.4f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--variant",   type=str, default=None, choices=list(ABLATION_CONFIGS.keys()))
    args = parser.parse_args()

    if args.eval_only:
        eval_all_ablations()
        sys.exit(0)

    variants_to_run = ([args.variant] if args.variant else list(ABLATION_CONFIGS.keys()))

    if "full_20ch" in variants_to_run and os.path.exists("models/best_model.pth"):
        import shutil
        shutil.copy("models/best_model.pth", "models/ablation_full_20ch_best.pth")
        variants_to_run = [v for v in variants_to_run if v != "full_20ch"]

    all_results = {}
    for variant_name in variants_to_run:
        cfg = ABLATION_CONFIGS[variant_name]
        m   = run_ablation_variant(variant_name, cfg, debug=args.debug, epochs=60)
        if m:
            all_results[variant_name] = m

    eval_all_ablations()