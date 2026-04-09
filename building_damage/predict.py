"""
Prediction & Visualization Script for Building Damage Classification.
Takes a pre-disaster image filename, finds the corresponding post-disaster
image and JSON files, runs model inference, and shows color-coded results.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import get_model
from utils import (
    parse_wkt_polygon,
    extract_building_crop,
    overlay_damage_on_image,
    DAMAGE_LABEL_MAP,
    DAMAGE_COLORS,
    CLASS_NAMES,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ═══════════════════════ CONFIGURATION ═════════════════════════

# ── SET THE PRE-DISASTER IMAGE FILENAME HERE ──
PRE_DISASTER_FILENAME = 'hurricane-florence_00000092_pre_disaster.png'

MODEL_PATH  = 'models/best_model.pth'
DATA_DIR    = 'data/test'   # or 'filtered_dataset/test'
CROP_SIZE   = 128
OUTPUTS_DIR = 'outputs'
# ═══════════════════════════════════════════════════════════════


def predict_single_image(pre_filename, model_path=MODEL_PATH,
                         data_dir=DATA_DIR, crop_size=CROP_SIZE):
    """
    Predict damage levels for all buildings in a given image pair.

    Args:
        pre_filename: filename of the pre-disaster image (e.g., 'hurricane_00001_pre_disaster.png')
        model_path: path to trained model checkpoint
        data_dir: path to dataset folder containing images/ and labels/
        crop_size: size of building crops

    Returns:
        results: list of dicts with polygon coords, predicted class, and class name
    """
    # ── Derive filenames ──────────────────────────────────────
    base_name = pre_filename.replace('_pre_disaster.png', '')
    post_filename = f"{base_name}_post_disaster.png"
    pre_json = f"{base_name}_pre_disaster.json"
    post_json = f"{base_name}_post_disaster.json"

    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    pre_img_path = os.path.join(images_dir, pre_filename)
    post_img_path = os.path.join(images_dir, post_filename)
    pre_json_path = os.path.join(labels_dir, pre_json)
    post_json_path = os.path.join(labels_dir, post_json)

    # ── Verify files exist ────────────────────────────────────
    for fpath, desc in [(pre_img_path, 'Pre-disaster image'),
                         (post_img_path, 'Post-disaster image'),
                         (pre_json_path, 'Pre-disaster JSON'),
                         (post_json_path, 'Post-disaster JSON')]:
        if not os.path.exists(fpath):
            print(f"  ✗ {desc} not found: {fpath}")
            return None
        print(f"  ✓ {desc}: {os.path.basename(fpath)}")

    # ── Load model ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    model = get_model(num_classes=4, dropout=0.0, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded from: {model_path}")

    # ── Load images ───────────────────────────────────────────
    pre_img = Image.open(pre_img_path).convert('RGB')
    post_img = Image.open(post_img_path).convert('RGB')
    img_size = (pre_img.width, pre_img.height)

    # ── Load JSON labels ──────────────────────────────────────
    with open(pre_json_path, 'r') as f:
        pre_data = json.load(f)
    with open(post_json_path, 'r') as f:
        post_data = json.load(f)

    # Build UID -> pre polygon mapping
    pre_polys = {}
    for feat in pre_data['features']['xy']:
        uid = feat['properties']['uid']
        pre_polys[uid] = parse_wkt_polygon(feat['wkt'])

    # ── Normalization (ImageNet) ──────────────────────────────
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # ── Predict each building ─────────────────────────────────
    results = []
    buildings = post_data['features']['xy']
    print(f"\n  Processing {len(buildings)} buildings...")

    with torch.no_grad():
        for feat in buildings:
            uid = feat['properties']['uid']
            post_coords = parse_wkt_polygon(feat['wkt'])
            pre_coords = pre_polys.get(uid, post_coords)

            if len(post_coords) < 3 or len(pre_coords) < 3:
                continue

            # Extract crops
            pre_crop = extract_building_crop(pre_img, pre_coords, crop_size, img_size)
            post_crop = extract_building_crop(post_img, post_coords, crop_size, img_size)

            # Convert to tensor
            pre_tensor = transforms.ToTensor()(pre_crop)
            post_tensor = transforms.ToTensor()(post_crop)
            pre_tensor = normalize(pre_tensor).unsqueeze(0).to(device)
            post_tensor = normalize(post_tensor).unsqueeze(0).to(device)

            # Forward pass
            output = model(pre_tensor, post_tensor)
            pred_class = output.argmax(dim=1).item()

            # Get ground truth if available
            gt_subtype = feat['properties'].get('subtype', 'unknown')
            gt_label = DAMAGE_LABEL_MAP.get(gt_subtype, -1)

            results.append({
                'uid': uid,
                'post_coords': post_coords,
                'predicted_class': pred_class,
                'predicted_name': CLASS_NAMES[pred_class],
                'ground_truth': gt_subtype,
                'gt_label': gt_label,
            })

    return results, pre_img, post_img


def main():
    print("=" * 70)
    print("  BUILDING DAMAGE CLASSIFICATION — PREDICTION")
    print("=" * 70)
    print(f"\n  Input: {PRE_DISASTER_FILENAME}")

    output = predict_single_image(PRE_DISASTER_FILENAME)
    if output is None:
        print("\n  Prediction failed. Check file paths.")
        return

    results, pre_img, post_img = output

    # ── Print results ─────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  PREDICTIONS ({len(results)} buildings)")
    print(f"  {'='*60}")

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    correct = 0
    total_with_gt = 0

    for i, r in enumerate(results):
        pred_cls = r['predicted_class']
        class_counts[pred_cls] = class_counts.get(pred_cls, 0) + 1

        marker = ''
        if r['gt_label'] >= 0:
            total_with_gt += 1
            if r['gt_label'] == pred_cls:
                correct += 1
                marker = ' ✓'
            else:
                gt_name = r['ground_truth']
                marker = f' ✗ (GT: {gt_name})'

        pred_name = r['predicted_name']
        print(f"  Building {i+1:3d}: {pred_name:16s}{marker}")

    print(f"\n  {'─'*60}")
    print(f"  Summary:")
    for cls_id in range(4):
        print(f"    {CLASS_NAMES[cls_id]:16s}: {class_counts[cls_id]} buildings")

    if total_with_gt > 0:
        print(f"\n  Accuracy (vs ground truth): {correct}/{total_with_gt} "
              f"({correct/total_with_gt*100:.1f}%)")

    # ── Visualization ─────────────────────────────────────────
    buildings_with_preds = [(r['post_coords'], r['predicted_class']) for r in results]
    result_img = overlay_damage_on_image(post_img, buildings_with_preds)

    # ── Display ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    axes[0].imshow(np.array(pre_img))
    axes[0].set_title('Pre-disaster', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(np.array(post_img))
    axes[1].set_title('Post-disaster', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(np.array(result_img))
    axes[2].set_title('Damage Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Legend
    legend_elements = [
        Patch(facecolor=np.array(DAMAGE_COLORS[0])/255.0, edgecolor='black',
              label='No Damage'),
        Patch(facecolor=np.array(DAMAGE_COLORS[1])/255.0, edgecolor='black',
              label='Minor Damage'),
        Patch(facecolor=np.array(DAMAGE_COLORS[2])/255.0, edgecolor='black',
              label='Major Damage'),
        Patch(facecolor=np.array(DAMAGE_COLORS[3])/255.0, edgecolor='black',
              label='Destroyed'),
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUTS_DIR, 'prediction_result.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  [✓] Visualization saved to {save_path}")

    plt.show()
    print("\n  Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()