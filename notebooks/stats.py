import rasterio
import numpy as np
from pathlib import Path
import csv
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "flood_sar" / "sen1floods11" / "v1.1"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
STATS_FILE = NOTEBOOKS_DIR / "stats.csv"

s1_folder = DATA_DIR / "data" / "flood_events" / "HandLabeled" / "S1Hand"
label_folder = DATA_DIR / "data" / "flood_events" / "HandLabeled" / "LabelHand"
split_files = {
    'train': DATA_DIR / "splits" / "flood_handlabeled" / "flood_train_data.csv",
    'valid': DATA_DIR / "splits" / "flood_handlabeled" / "flood_valid_data.csv"
}

stats = {'split': [], 'total_pixels': [], 'valid_pixels': [], 'flood_pixels': [], 'flood_ratio': []}

for split, split_file in split_files.items():
    image_pairs = []
    with open(split_file, 'r') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                s1_file = s1_folder / row[0].strip()
                label_file = label_folder / row[1].strip()
                if s1_file.exists() and label_file.exists():
                    image_pairs.append((s1_file, label_file))
    
    total_valid = 0
    total_flood = 0
    total_px = 0
    
    for s1_file, label_file in image_pairs:
        with rasterio.open(label_file) as src:
            label = src.read(1)
            h, w = label.shape
            total_px += h * w
            valid = (label != -1).sum()
            flood = (label == 1).sum()
            total_valid += valid
            total_flood += flood
    
    avg_flood_ratio = (total_flood / total_valid) if total_valid > 0 else 0
    stats['split'].append(split)
    stats['total_pixels'].append(total_px)
    stats['valid_pixels'].append(total_valid)
    stats['flood_pixels'].append(total_flood)
    stats['flood_ratio'].append(avg_flood_ratio * 100)  # %
    
    print(f"{split.capitalize()}: {len(image_pairs)} imgs | Valid px: {total_valid:,} | Flood px: {total_flood:,} | Ratio: {avg_flood_ratio:.2%}")

df = pd.DataFrame(stats)
df.to_csv(STATS_FILE, index=False)
print(f"\nFull stats saved to {STATS_FILE}")
