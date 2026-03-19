"""
Building Damage Dataset for PyTorch.
Two modes:
  1. CachedBuildingDataset (FAST) — loads pre-extracted crop PNGs from cached_crops/
  2. BuildingDamageDataset (SLOW) — extracts crops on-the-fly from full images + JSON
Use preprocess.py to create the cache first, then train with CachedBuildingDataset.
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import parse_wkt_polygon, extract_building_crop, DAMAGE_LABEL_MAP


class CachedBuildingDataset(Dataset):
    """
    FAST dataset that loads pre-extracted building crops from disk.
    Run preprocess.py first to create the cache.

    Each sample = (pre_crop_tensor, post_crop_tensor, damage_label)
    """

    def __init__(self, manifest_path, augment=False):
        """
        Args:
            manifest_path: Path to the manifest JSON (e.g., 'cached_crops/train_manifest.json')
            augment: Whether to apply data augmentations.
        """
        self.augment = augment

        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)

        # Normalization (ImageNet stats for pretrained ResNet)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        self.to_tensor = transforms.ToTensor()

        # Print stats
        labels = [s['label'] for s in self.samples]
        class_names = ['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        print(f"  [CachedDataset] Loaded {len(self.samples)} samples from {manifest_path}")
        for i, name in enumerate(class_names):
            print(f"    Class {i} ({name}): {labels.count(i)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load small crop PNGs (128x128) — very fast!
        pre_crop = Image.open(sample['pre_crop']).convert('RGB')
        post_crop = Image.open(sample['post_crop']).convert('RGB')

        # Augmentations
        if self.augment:
            if random.random() > 0.5:
                pre_crop = pre_crop.transpose(Image.FLIP_LEFT_RIGHT)
                post_crop = post_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                pre_crop = pre_crop.transpose(Image.FLIP_TOP_BOTTOM)
                post_crop = post_crop.transpose(Image.FLIP_TOP_BOTTOM)
            pre_crop = self.color_jitter(pre_crop)
            post_crop = self.color_jitter(post_crop)

        # Convert to tensor and normalize
        pre_tensor = self.normalize(self.to_tensor(pre_crop))
        post_tensor = self.normalize(self.to_tensor(post_crop))
        label = torch.tensor(sample['label'], dtype=torch.long)

        return pre_tensor, post_tensor, label

    def get_class_weights(self):
        """Compute inverse frequency class weights for loss balancing."""
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels, minlength=4).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        total = len(labels)
        weights = total / (4.0 * class_counts)
        return torch.tensor(weights, dtype=torch.float32)


class BuildingDamageDataset(Dataset):
    """
    SLOW dataset — extracts crops on-the-fly from full images + JSON.
    Use CachedBuildingDataset instead for training.
    This class is kept for prediction/inference where you need polygon info.
    """

    def __init__(self, data_dir, crop_size=128, augment=False):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.crop_size = crop_size
        self.augment = augment

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

        self.samples = []
        self._build_sample_list()

    def _build_sample_list(self):
        label_files = [f for f in os.listdir(self.labels_dir)
                       if f.endswith('.json') and 'post_disaster' in f]

        for post_json_name in sorted(label_files):
            pre_json_name = post_json_name.replace('post_disaster', 'pre_disaster')
            post_json_path = os.path.join(self.labels_dir, post_json_name)
            pre_json_path = os.path.join(self.labels_dir, pre_json_name)

            if not os.path.exists(pre_json_path):
                continue

            with open(post_json_path, 'r') as f:
                post_data = json.load(f)
            with open(pre_json_path, 'r') as f:
                pre_data = json.load(f)

            pre_polys = {}
            for feat in pre_data['features']['xy']:
                uid = feat['properties']['uid']
                pre_polys[uid] = parse_wkt_polygon(feat['wkt'])

            post_img_name = post_json_name.replace('.json', '.png')
            pre_img_name = pre_json_name.replace('.json', '.png')
            post_img_path = os.path.join(self.images_dir, post_img_name)
            pre_img_path = os.path.join(self.images_dir, pre_img_name)

            if not os.path.exists(post_img_path) or not os.path.exists(pre_img_path):
                continue

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

                self.samples.append({
                    'pre_img_path': pre_img_path,
                    'post_img_path': post_img_path,
                    'pre_coords': pre_coords,
                    'post_coords': post_coords,
                    'label': label,
                    'uid': uid,
                })

        print(f"  [Dataset] Loaded {len(self.samples)} building samples from {self.data_dir}")
        labels = [s['label'] for s in self.samples]
        for cls_id, cls_name in enumerate(['No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']):
            print(f"    Class {cls_id} ({cls_name}): {labels.count(cls_id)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pre_img = Image.open(sample['pre_img_path']).convert('RGB')
        post_img = Image.open(sample['post_img_path']).convert('RGB')
        img_size = (pre_img.width, pre_img.height)

        pre_crop = extract_building_crop(pre_img, sample['pre_coords'], self.crop_size, img_size)
        post_crop = extract_building_crop(post_img, sample['post_coords'], self.crop_size, img_size)

        if self.augment:
            if random.random() > 0.5:
                pre_crop = pre_crop.transpose(Image.FLIP_LEFT_RIGHT)
                post_crop = post_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                pre_crop = pre_crop.transpose(Image.FLIP_TOP_BOTTOM)
                post_crop = post_crop.transpose(Image.FLIP_TOP_BOTTOM)
            pre_crop = self.color_jitter(pre_crop)
            post_crop = self.color_jitter(post_crop)

        pre_tensor = transforms.ToTensor()(pre_crop)
        post_tensor = transforms.ToTensor()(post_crop)
        pre_tensor = self.normalize(pre_tensor)
        post_tensor = self.normalize(post_tensor)

        label = torch.tensor(sample['label'], dtype=torch.long)
        return pre_tensor, post_tensor, label

    def get_class_weights(self):
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels, minlength=4).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        total = len(labels)
        weights = total / (4.0 * class_counts)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_info(self, idx):
        return self.samples[idx]
