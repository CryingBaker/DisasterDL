"""
Testing / Evaluation Script for Building Damage Classification.
Loads a trained model, evaluates on the test set, prints metrics,
and saves graphs + Excel report.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CachedBuildingDataset
from model import get_model
from utils import (
    compute_metrics,
    plot_training_curves,
    append_test_metrics_to_excel,
    save_metrics_to_excel,
    CLASS_NAMES,
)
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════ CONFIGURATION ═════════════════════════
TEST_MANIFEST     = './data/cached_crops/test_manifest.json'
MODEL_PATH        = './models/best_model.pth'
BATCH_SIZE        = 64
NUM_WORKERS       = 4
OUTPUTS_DIR       = './logs/'
EXCEL_PATH        = './logs/training_metrics.xlsx'
# ═══════════════════════════════════════════════════════════════


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a dataset. Returns avg loss, all labels, all preds."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc='  Evaluating', leave=False, ncols=100)
    with torch.no_grad():
        for pre_crops, post_crops, labels in pbar:
            pre_crops = pre_crops.to(device)
            post_crops = post_crops.to(device)
            labels = labels.to(device)

            outputs = model(pre_crops, post_crops)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, all_labels, all_preds


def plot_confusion_matrix(labels, preds, save_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')
    fig.colorbar(im, ax=ax)
    tick_marks = range(4)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label', fontsize=13)
    ax.set_xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("  BUILDING DAMAGE CLASSIFICATION — TESTING")
    print("=" * 70)

    # ── Device ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────
    if not os.path.exists(TEST_MANIFEST):
        print(f"\n  ✗ Cache not found: {TEST_MANIFEST}")
        print(f"  Run 'python preprocess.py' first.")
        return

    print(f"\n  Loading cached test data...")
    test_dataset = CachedBuildingDataset(
        manifest_path=TEST_MANIFEST,
        augment=False
    )

    use_cuda = device.type == 'cuda'
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=True
    )

    print(f"  Total test buildings: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────
    print(f"\n  Loading model from '{MODEL_PATH}'...")
    model = get_model(num_classes=4, dropout=0.0, pretrained=False)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"  Model loaded (trained epoch: {checkpoint.get('epoch', '?')})")

    # ── Loss ──────────────────────────────────────────────────
    class_weights = test_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Evaluate ──────────────────────────────────────────────
    print("\n  Running evaluation...")
    test_loss, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
    test_acc, test_f1, test_report = compute_metrics(test_labels, test_preds)

    # ── Print results ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST RESULTS")
    print("=" * 70)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test F1 Score: {test_f1:.4f}")
    print("\n  Per-class Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    # ── Save confusion matrix ────────────────────────────────
    cm_path = os.path.join(OUTPUTS_DIR, 'test_confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, cm_path)
    print(f"  [✓] Confusion matrix saved to {cm_path}")

    # ── Save test metrics to Excel ───────────────────────────
    test_metrics = {
        'Test Loss': round(test_loss, 6),
        'Test Accuracy': round(test_acc, 6),
        'Test F1 (weighted)': round(test_f1, 6),
    }

    # Add per-class metrics
    for cls_name in CLASS_NAMES:
        if cls_name in test_report:
            test_metrics[f"{cls_name} Precision"] = round(test_report[cls_name]['precision'], 4)
            test_metrics[f"{cls_name} Recall"] = round(test_report[cls_name]['recall'], 4)
            test_metrics[f"{cls_name} F1"] = round(test_report[cls_name]['f1-score'], 4)

    append_test_metrics_to_excel(test_metrics, filepath=EXCEL_PATH)

    # ── Also save a standalone test-only graph ────────────────
    # Bar chart of per-class accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    class_prec = [test_report.get(cn, {}).get('precision', 0) for cn in CLASS_NAMES]
    class_rec = [test_report.get(cn, {}).get('recall', 0) for cn in CLASS_NAMES]
    class_f1s = [test_report.get(cn, {}).get('f1-score', 0) for cn in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    ax.bar(x - width, class_prec, width, label='Precision', color='#2196F3')
    ax.bar(x, class_rec, width, label='Recall', color='#FF9800')
    ax.bar(x + width, class_f1s, width, label='F1-Score', color='#4CAF50')
    ax.set_xlabel('Damage Class', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Test Per-Class Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, 'test_per_class_metrics.png'), dpi=150)
    plt.close(fig)
    print(f"  [✓] Per-class metrics graph saved")

    print("\n  All test outputs saved!")
    print("=" * 70)


if __name__ == '__main__':
    main()
