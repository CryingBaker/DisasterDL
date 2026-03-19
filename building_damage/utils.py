"""
Utility functions for Building Damage Classification.
Includes: polygon parsing, metrics, plotting, Excel export, visualization.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import pandas as pd


# ───────────────────────── Damage mapping ─────────────────────────
DAMAGE_LABEL_MAP = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3,
}

DAMAGE_DISPLAY = {
    0: 'No Damage (1)',
    1: 'Minor Damage (2)',
    2: 'Major Damage (3)',
    3: 'Destroyed (4)',
}

DAMAGE_COLORS = {
    0: (0, 255, 0),       # Green  – No damage
    1: (255, 255, 0),     # Yellow – Minor damage
    2: (255, 165, 0),     # Orange – Major damage
    3: (255, 0, 0),       # Red    – Destroyed
}

DAMAGE_COLORS_NORM = {
    0: (0.0, 1.0, 0.0, 0.45),
    1: (1.0, 1.0, 0.0, 0.45),
    2: (1.0, 0.65, 0.0, 0.45),
    3: (1.0, 0.0, 0.0, 0.45),
}

CLASS_NAMES = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']


# ───────────────────────── Polygon parsing ────────────────────────
def parse_wkt_polygon(wkt_str):
    """Parse a WKT POLYGON string and return list of (x, y) tuples."""
    match = re.search(r'POLYGON\s*\(\((.*?)\)\)', wkt_str)
    if not match:
        return []
    coords_str = match.group(1)
    coords = []
    for pair in coords_str.split(','):
        parts = pair.strip().split()
        if len(parts) == 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords


def polygon_to_mask(coords, img_size=(1024, 1024)):
    """Create a binary mask from polygon coordinates."""
    mask = Image.new('L', img_size, 0)
    if len(coords) >= 3:
        int_coords = [(int(round(x)), int(round(y))) for x, y in coords]
        ImageDraw.Draw(mask).polygon(int_coords, fill=255)
    return np.array(mask)


def get_building_bbox(coords, img_size=(1024, 1024), padding=5):
    """Get bounding box of a polygon with padding, clipped to image size."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min = max(0, int(np.floor(min(xs))) - padding)
    y_min = max(0, int(np.floor(min(ys))) - padding)
    x_max = min(img_size[0], int(np.ceil(max(xs))) + padding)
    y_max = min(img_size[1], int(np.ceil(max(ys))) + padding)
    return x_min, y_min, x_max, y_max


def extract_building_crop(image, coords, crop_size=128, img_size=(1024, 1024)):
    """
    Extract a building region crop from an image using polygon coordinates.
    Applies polygon mask to isolate only the building pixels.
    Returns a PIL Image of size crop_size x crop_size.
    """
    if len(coords) < 3:
        return Image.new('RGB', (crop_size, crop_size), (0, 0, 0))

    x_min, y_min, x_max, y_max = get_building_bbox(coords, img_size)

    # Ensure valid bbox
    if x_max <= x_min or y_max <= y_min:
        return Image.new('RGB', (crop_size, crop_size), (0, 0, 0))

    # Create mask for the full image then crop
    mask = polygon_to_mask(coords, img_size)

    # Convert image to numpy
    img_arr = np.array(image)

    # Apply mask
    masked = img_arr.copy()
    mask_3d = np.stack([mask] * 3, axis=-1)
    masked = np.where(mask_3d > 0, masked, 0)

    # Crop to bounding box
    crop = masked[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return Image.new('RGB', (crop_size, crop_size), (0, 0, 0))

    # Resize to target size
    crop_img = Image.fromarray(crop.astype(np.uint8))
    crop_img = crop_img.resize((crop_size, crop_size), Image.BILINEAR)
    return crop_img


# ───────────────────────── Metrics ────────────────────────────────
def compute_metrics(all_labels, all_preds):
    """Compute accuracy, weighted F1, per-class report."""
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )
    return acc, f1, report


# ───────────────────────── Plotting ───────────────────────────────
def plot_training_curves(metrics_dict, save_dir='./logs/'):
    """Plot and save training accuracy, loss, and F1 curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(metrics_dict['train_loss']) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_dict['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    if 'val_loss' in metrics_dict and metrics_dict['val_loss']:
        ax.plot(epochs, metrics_dict['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Loss per Epoch', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_dict['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
    if 'val_acc' in metrics_dict and metrics_dict['val_acc']:
        ax.plot(epochs, metrics_dict['val_acc'], 'r-o', label='Validation Accuracy', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Accuracy per Epoch', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=150)
    plt.close(fig)

    # F1 curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_dict['train_f1'], 'b-o', label='Training F1 (weighted)', linewidth=2)
    if 'val_f1' in metrics_dict and metrics_dict['val_f1']:
        ax.plot(epochs, metrics_dict['val_f1'], 'r-o', label='Validation F1 (weighted)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('F1 Score per Epoch', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'f1_curve.png'), dpi=150)
    plt.close(fig)

    print(f"  [✓] Plots saved to {save_dir}/")


def save_metrics_to_excel(metrics_dict, filepath='./logs/training_metrics.xlsx'):
    """Save per-epoch metrics to an Excel file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    epochs = list(range(1, len(metrics_dict['train_loss']) + 1))
    data = {
        'Epoch': epochs,
        'Train Loss': metrics_dict['train_loss'],
        'Train Accuracy': metrics_dict['train_acc'],
        'Train F1': metrics_dict['train_f1'],
    }
    if 'val_loss' in metrics_dict and metrics_dict['val_loss']:
        data['Val Loss'] = metrics_dict['val_loss']
        data['Val Accuracy'] = metrics_dict['val_acc']
        data['Val F1'] = metrics_dict['val_f1']

    df = pd.DataFrame(data)
    df.to_excel(filepath, index=False, sheet_name='Metrics')
    print(f"  [✓] Metrics saved to {filepath}")


def append_test_metrics_to_excel(test_metrics, filepath='./logs/training_metrics.xlsx'):
    """Append test metrics as a new sheet in the existing Excel file."""
    test_df = pd.DataFrame([test_metrics])
    if os.path.exists(filepath):
        with pd.ExcelWriter(filepath, mode='a', engine='openpyxl',
                            if_sheet_exists='replace') as writer:
            test_df.to_excel(writer, index=False, sheet_name='Test Metrics')
    else:
        test_df.to_excel(filepath, index=False, sheet_name='Test Metrics')
    print(f"  [✓] Test metrics appended to {filepath}")


# ───────────────────────── Visualization ──────────────────────────
def overlay_damage_on_image(image, buildings_with_preds):
    """
    Overlay color-coded damage predictions on an image.
    buildings_with_preds: list of (polygon_coords, predicted_class)
    Returns a PIL Image with overlays.
    """
    img = image.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for coords, pred_class in buildings_with_preds:
        if len(coords) < 3:
            continue
        color = DAMAGE_COLORS.get(pred_class, (128, 128, 128))
        rgba_color = color + (100,)  # Semi-transparent
        int_coords = [(int(round(x)), int(round(y))) for x, y in coords]
        draw.polygon(int_coords, fill=rgba_color, outline=color + (255,))

    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')


def create_legend_image(save_path=None):
    """Create a damage color legend."""
    fig, ax = plt.subplots(figsize=(4, 2))
    for i, (cls, name) in enumerate(zip(range(4), CLASS_NAMES)):
        color = np.array(DAMAGE_COLORS[cls]) / 255.0
        ax.barh(i, 1, color=color, edgecolor='black', linewidth=0.5)
        ax.text(0.5, i, f"  {name}", va='center', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Damage Legend', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
