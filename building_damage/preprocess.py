"""
Pre-extract building crops from the dataset.
Run this ONCE before training to cache all building crops as small images.
This eliminates the expensive polygon parsing + masking during training.

Usage:
    python preprocess.py
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import parse_wkt_polygon, extract_building_crop, DAMAGE_LABEL_MAP


# ═══════════════════════ CONFIGURATION ═════════════════════════
CROP_SIZE = 128
SPLITS = ['train', 'test']
DATASET_ROOT = './data/'
OUTPUT_ROOT = './data/cached_crops'
# ═══════════════════════════════════════════════════════════════


def process_image_pair(args):
    """Process a single pre/post image pair and extract all building crops."""
    post_json_path, pre_json_path, images_dir, output_dir, crop_size = args
    results = []

    try:
        with open(post_json_path, 'r') as f:
            post_data = json.load(f)
        with open(pre_json_path, 'r') as f:
            pre_data = json.load(f)

        # Build UID -> pre polygon mapping
        pre_polys = {}
        for feat in pre_data['features']['xy']:
            uid = feat['properties']['uid']
            pre_polys[uid] = parse_wkt_polygon(feat['wkt'])

        # Image filenames
        post_img_name = os.path.basename(post_json_path).replace('.json', '.png')
        pre_img_name = os.path.basename(pre_json_path).replace('.json', '.png')
        post_img_path = os.path.join(images_dir, post_img_name)
        pre_img_path = os.path.join(images_dir, pre_img_name)

        if not os.path.exists(post_img_path) or not os.path.exists(pre_img_path):
            return results

        # Load images ONCE per pair
        pre_img = Image.open(pre_img_path).convert('RGB')
        post_img = Image.open(post_img_path).convert('RGB')
        img_size = (pre_img.width, pre_img.height)

        for feat in post_data['features']['xy']:
            subtype = feat['properties'].get('subtype', '')
            if subtype not in DAMAGE_LABEL_MAP:
                continue

            uid = feat['properties']['uid']
            label = DAMAGE_LABEL_MAP[subtype]
            post_coords = parse_wkt_polygon(feat['wkt'])
            pre_coords = pre_polys.get(uid, post_coords)

            if len(post_coords) < 3 or len(pre_coords) < 3:
                continue

            # Extract crops
            pre_crop = extract_building_crop(pre_img, pre_coords, crop_size, img_size)
            post_crop = extract_building_crop(post_img, post_coords, crop_size, img_size)

            # Save crops
            safe_uid = uid.replace('-', '')[:16]
            base = os.path.basename(post_json_path).replace('_post_disaster.json', '')
            fname = f"{base}_{safe_uid}"

            pre_path = os.path.join(output_dir, f"{fname}_pre.png")
            post_path = os.path.join(output_dir, f"{fname}_post.png")

            pre_crop.save(pre_path)
            post_crop.save(post_path)

            results.append({
                'pre_crop': pre_path,
                'post_crop': post_path,
                'label': label,
                'uid': uid,
                'source': base,
            })

    except Exception as e:
        print(f"  Error processing {post_json_path}: {e}")

    return results


def preprocess_split(split):
    """Pre-extract all crops for a dataset split (train/test)."""
    labels_dir = os.path.join(DATASET_ROOT, split, 'labels')
    images_dir = os.path.join(DATASET_ROOT, split, 'images')
    output_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all post-disaster JSON files
    post_jsons = sorted([
        f for f in os.listdir(labels_dir)
        if f.endswith('.json') and 'post_disaster' in f
    ])

    print(f"\n  Processing {split}: {len(post_jsons)} image pairs...")

    # Build args list
    tasks = []
    for post_json_name in post_jsons:
        pre_json_name = post_json_name.replace('post_disaster', 'pre_disaster')
        post_json_path = os.path.join(labels_dir, post_json_name)
        pre_json_path = os.path.join(labels_dir, pre_json_name)
        if os.path.exists(pre_json_path):
            tasks.append((post_json_path, pre_json_path, images_dir, output_dir, CROP_SIZE))

    # Process sequentially with progress bar
    all_results = []
    for task_args in tqdm(tasks, desc=f'  {split}', ncols=80):
        results = process_image_pair(task_args)
        all_results.extend(results)

    # Save manifest (index file)
    manifest_path = os.path.join(OUTPUT_ROOT, f'{split}_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print stats
    labels = [r['label'] for r in all_results]
    class_names = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    print(f"\n  {split} — Total crops: {len(all_results)}")
    for i, name in enumerate(class_names):
        count = labels.count(i)
        print(f"    Class {i} ({name}): {count}")

    return all_results


def main():
    print("=" * 60)
    print("  BUILDING CROP PRE-EXTRACTION")
    print("=" * 60)
    print(f"  Crop size: {CROP_SIZE}x{CROP_SIZE}")
    print(f"  Output: {OUTPUT_ROOT}/")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for split in SPLITS:
        preprocess_split(split)

    print("\n" + "=" * 60)
    print("  DONE! Pre-extracted crops saved.")
    print(f"  Now run: python train.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
