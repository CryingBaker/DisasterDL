"""
Training Script for Building Damage Classification.
Uses pre-extracted cached crops for fast training.
Run preprocess.py first to create the cache!

Saves checkpoints, metrics, graphs, and Excel reports.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import CachedBuildingDataset
from model import get_model
from utils import (
    compute_metrics,
    plot_training_curves,
    save_metrics_to_excel,
    CLASS_NAMES,
)


# ═══════════════════════ HYPERPARAMETERS ═══════════════════════
TRAIN_MANIFEST    = './data/cached_crops/train_manifest.json'
BATCH_SIZE        = 64
NUM_WORKERS       = 4
LEARNING_RATE     = 3e-4
WEIGHT_DECAY      = 1e-4
EPOCHS            = 25
OPTIMIZER         = 'AdamW'       # 'AdamW' | 'Adam' | 'SGD'
SCHEDULER         = 'CosineAnnealing'
DROPOUT           = 0.4
PRETRAINED        = True
MODELS_DIR        = './models/'
OUTPUTS_DIR       = './logs/'
# ═══════════════════════════════════════════════════════════════


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch. Returns avg loss, all labels, all preds."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for pre_crops, post_crops, labels in dataloader:
        pre_crops = pre_crops.to(device, non_blocking=True)
        post_crops = post_crops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(pre_crops, post_crops)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, all_labels, all_preds


def main():
    print("=" * 70)
    print("  BUILDING DAMAGE CLASSIFICATION — TRAINING")
    print("=" * 70)

    # ── Device ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # ── Directories ───────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Check cache exists ────────────────────────────────────
    if not os.path.exists(TRAIN_MANIFEST):
        print(f"\n  ✗ Cache not found: {TRAIN_MANIFEST}")
        print(f"  Run 'python preprocess.py' first to extract building crops.")
        return

    # ── Dataset (FAST — cached crops) ─────────────────────────
    print(f"\n  Loading cached training data...")
    train_dataset = CachedBuildingDataset(
        manifest_path=TRAIN_MANIFEST,
        augment=True
    )

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=True,
        drop_last=False
    )

    print(f"  Total buildings: {len(train_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────
    print("\n  Building Siamese ResNet-34 model...")
    model = get_model(num_classes=4, dropout=DROPOUT, pretrained=PRETRAINED)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Loss (class-weighted) ─────────────────────────────────
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"\n  Class weights: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer ─────────────────────────────────────────────
    if OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                       weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                     momentum=0.9, weight_decay=WEIGHT_DECAY)
    print(f"  Optimizer: {OPTIMIZER}")

    # ── Scheduler ─────────────────────────────────────────────
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    print(f"  Scheduler: {SCHEDULER}")

    # ── Training loop ─────────────────────────────────────────
    metrics_dict = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
    }

    print(f"\n  Starting training for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("-" * 70)

    best_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_labels, train_preds = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Metrics
        train_acc, train_f1, _ = compute_metrics(train_labels, train_preds)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Store metrics
        metrics_dict['train_loss'].append(round(train_loss, 6))
        metrics_dict['train_acc'].append(round(train_acc, 6))
        metrics_dict['train_f1'].append(round(train_f1, 6))

        epoch_time = time.time() - epoch_start

        # Print epoch results
        print(f"  Epoch [{epoch:2d}/{EPOCHS}]  "
              f"Loss: {train_loss:.4f}  "
              f"Acc: {train_acc:.4f}  "
              f"F1: {train_f1:.4f}  "
              f"LR: {current_lr:.6f}  "
              f"Time: {epoch_time:.1f}s")

        # Save checkpoint
        checkpoint_path = os.path.join(MODELS_DIR, f'model_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
        }, checkpoint_path)

        # Track best
        if train_f1 > best_f1:
            best_f1 = train_f1
            best_path = os.path.join(MODELS_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_f1': train_f1,
                'train_acc': train_acc,
            }, best_path)
            print(f"    ★ Best model saved (F1={best_f1:.4f})")

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"  Training completed in {total_time / 60:.1f} minutes")
    print(f"  Best F1: {best_f1:.4f}")

    # ── Save outputs ──────────────────────────────────────────
    print("\n  Saving outputs...")
    plot_training_curves(metrics_dict, save_dir=OUTPUTS_DIR)
    save_metrics_to_excel(metrics_dict, filepath=os.path.join(OUTPUTS_DIR, 'training_metrics.xlsx'))

    print("\n  All outputs saved!")
    print(f"  Models: {MODELS_DIR}/")
    print(f"  Graphs: {OUTPUTS_DIR}/")
    print(f"  Excel:  {OUTPUTS_DIR}/training_metrics.xlsx")
    print("=" * 70)


if __name__ == '__main__':
    main()
