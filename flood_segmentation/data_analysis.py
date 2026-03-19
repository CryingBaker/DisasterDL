import os
import sys
import glob
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# --- SETUP & PATHS ---
DATA_ROOT = "data"
OUTPUT_DIR = "data_analysis"
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
LOG_FILE = os.path.join(OUTPUT_DIR, "analysis_log.txt")

os.makedirs(GRAPHS_DIR, exist_ok=True)

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")  # Append mode
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return self.terminal.isatty() if hasattr(self.terminal, 'isatty') else False

# --- UTILS ---
def get_tile_id(filename):
    fname = os.path.basename(filename)
    if fname.startswith("._"): fname = fname[2:]
    parts = fname.split("_")
    if len(parts) >= 2: return f"{parts[0]}_{parts[1]}"
    return fname.split(".")[0]

# --- CORE FUNCTIONS ---

def generate_metadata(data_root=DATA_ROOT, output_dir=OUTPUT_DIR):
    """Scans directories and generates dataset_metadata.csv and split files."""
    print("\n--- Generating Metadata ---")
    modalities = {
        "aux": os.path.join(data_root, "aux"),
        "pre_s1": os.path.join(data_root, "pre_s1"),
        "pre_s2": os.path.join(data_root, "pre_s2"),
        "label_hand": os.path.join(data_root, "sen1floods11", "LabelHand"),
        "s1_hand": os.path.join(data_root, "sen1floods11", "S1Hand"),
        "s2_hand": os.path.join(data_root, "sen1floods11", "S2Hand"),
        "s1_weak": os.path.join(data_root, "sen1floods11", "S1Weak"),
        "s2_weak": os.path.join(data_root, "sen1floods11", "S2Weak"),
        "s2_weak_label": os.path.join(data_root, "sen1floods11", "S2IndexLabelWeak"),
    }

    registry = {}
    for mod, path in modalities.items():
        if not os.path.exists(path): continue
        files = [f for f in os.listdir(path) if f.endswith(".tif") and not f.startswith("._")]
        for f in files:
            tid = get_tile_id(f)
            if tid not in registry: registry[tid] = {}
            registry[tid][mod] = os.path.join(path, f)

    tile_ids = sorted(list(registry.keys()))
    print(f"Total entries discovered: {len(tile_ids)}")

    # --- Split Assignment ---
    # IMPORTANT: Split files are the SOURCE OF TRUTH and must NOT be overwritten.
    # If splits_train.txt / splits_val.txt / splits_test.txt already exist, load them.
    # Only generate new split files if NONE of them exist yet.
    split_files = {s: os.path.join(output_dir, f'splits_{s}.txt') for s in ['train', 'val', 'test']}
    all_exist = all(os.path.exists(p) for p in split_files.values())

    if all_exist:
        print("Found existing split files — loading splits from disk (NOT regenerating).")
        split_map = {}
        for split_name, fpath in split_files.items():
            with open(fpath) as f:
                for line in f:
                    tid = line.strip()
                    if tid:
                        split_map[tid] = split_name
        # Assign 'train' for any tile not in the existing split files
        for tid in tile_ids:
            if tid not in split_map:
                split_map[tid] = 'train'
    else:
        print("No existing split files found — generating country-level split.")
        # Train = Bolivia, Colombia, Nigeria, Pakistan, Somalia, Sri-Lanka (and any unlisted)
        # Val   = Ghana, Somalia, Spain
        # Test  = Sri-Lanka
        # NOTE: Adjust these sets if your intended split differs!
        TEST_COUNTRIES = set()   # Fill in if regenerating from scratch
        VAL_COUNTRIES  = set()   # Fill in if regenerating from scratch

        def get_country(tid): return tid.split('_')[0]

        train_ids, val_ids, test_ids = [], [], []
        for tid in tile_ids:
            country = get_country(tid)
            if country in TEST_COUNTRIES:
                test_ids.append(tid)
            elif country in VAL_COUNTRIES:
                val_ids.append(tid)
            else:
                train_ids.append(tid)

        split_map = {tid: 'train' for tid in train_ids}
        split_map.update({tid: 'val' for tid in val_ids})
        split_map.update({tid: 'test' for tid in test_ids})

        for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            with open(split_files[split_name], 'w') as f:
                f.write('\n'.join(ids))

    # Print summary
    from collections import Counter
    cnt = Counter(split_map.values())
    print(f"Split summary: train={cnt['train']}, val={cnt['val']}, test={cnt['test']}")

    metadata_list = []
    for tid in tqdm(tile_ids, desc="Processing tiles"):
        paths = registry[tid]
        h, w, f_pct, iv_pct = np.nan, np.nan, np.nan, np.nan
        q, shape_c = "n/a", True
        shapes = []
        
        # Determine set type
        has_hand = "label_hand" in paths
        has_weak_lbl = "s2_weak_label" in paths
        st = "none"
        if has_hand: st = "both" if has_weak_lbl else "hand"
        elif has_weak_lbl: st = "weak"

        for mod, p in paths.items():
            try:
                with rasterio.open(p) as src:
                    if np.isnan(h): h, w = src.height, src.width
                    shapes.append((src.height, src.width))
                    if mod == "label_hand":
                        lbl = src.read(1)
                        total = lbl.size
                        inv = np.sum(lbl == -1)
                        flood = np.sum(lbl == 1)
                        nf = np.sum(lbl == 0)
                        valid = total - inv
                        if valid > 0:
                            f_pct = (flood / valid) * 100
                            iv_pct = (inv / total) * 100
                            if inv == total: q = "all_invalid"
                            elif flood == valid: q = "all_flood"
                            elif nf == valid: q = "all_noflood"
                            elif (flood/valid) < 0.01 or (flood/valid) > 0.99: q = "imbalanced"
                            else: q = "good"
                        else: iv_pct = 100; q = "all_invalid"
            except: pass
        
        if len(set(shapes)) > 1: shape_c = False
        
        complete = False
        if st in ["hand", "both"]: 
            complete = all(k in paths for k in ["label_hand", "s1_hand", "s2_hand", "pre_s1", "pre_s2", "aux"])
        elif st == "weak": 
            complete = all(k in paths for k in ["s1_weak", "s2_weak", "pre_s1", "pre_s2", "aux"])

        row = {
            "tile_id": tid, "split": split_map[tid], "set_type": st,
            "has_label_hand": has_hand, "has_s2_weak_label": has_weak_lbl,
            "all_modalities_present": complete, "label_flood_pct": f_pct, "label_novalid_pct": iv_pct,
            "label_quality": q, "image_height": h, "image_width": w, "shape_consistent": shape_c,
            "s1_hand_path": paths.get("s1_hand", ""),
            "s2_hand_path": paths.get("s2_hand", ""),
            "s1_weak_path": paths.get("s1_weak", ""),
            "s2_weak_path": paths.get("s2_weak", ""),
            "label_hand_path": paths.get("label_hand", ""),
            "s2_weak_label_path": paths.get("s2_weak_label", ""),
            "pre_s1_path": paths.get("pre_s1", ""),
            "pre_s2_path": paths.get("pre_s2", ""),
            "aux_path": paths.get("aux", "")
        }
        metadata_list.append(row)

    df = pd.DataFrame(metadata_list)
    df.to_csv(os.path.join(output_dir, "dataset_metadata.csv"), index=False)
    print(f"Metadata saved to {os.path.join(output_dir, 'dataset_metadata.csv')}")
    return df

def run_audits(df=None):
    """Performs deep sanity checks on labels and modalities (merges logic from step5, step12, check_weak_labels)."""
    if df is None:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_metadata.csv"))
    
    print("\n--- Running Dataset Audits ---")
    
    # 1. Weak Label Audit
    weak_df = df[df['set_type'].isin(['weak', 'both'])]
    print(f"Total tiles with weak labels: {len(weak_df)}")
    
    if len(weak_df) > 0:
        sample = weak_df.dropna(subset=['s2_weak_label_path']).sample(min(3, len(weak_df)), random_state=42)
        print("\nSampling 3 weak labels for value check:")
        for _, row in sample.iterrows():
            p = row['s2_weak_label_path']
            if os.path.exists(p):
                with rasterio.open(p) as src:
                    d = src.read(1)
                    unique = np.unique(d)
                    p1 = (d == 1).sum() / d.size * 100
                    p0 = (d == 0).sum() / d.size * 100
                    pm1 = (d == -1).sum() / d.size * 100
                    print(f"  {row['tile_id']}: Unique={unique}, Flood={p1:.1f}%, NoFlood={p0:.1f}%, Invalid={pm1:.1f}%")

    # 2. Hand Label Audit
    hand_df = df[df['has_label_hand']]
    print(f"\nTotal hand-labeled tiles: {len(hand_df)}")
    if len(hand_df) > 0:
        bad_q = hand_df[hand_df['label_quality'] == 'all_invalid']
        print(f"  Tiles with 100% invalid pixels: {len(bad_q)}")
        
    # 3. Shape Consistency
    inconsistent = df[~df['shape_consistent']]
    if len(inconsistent) > 0:
        print(f"\nWARNING: {len(inconsistent)} tiles have inconsistent modality shapes!")
        print(inconsistent[['tile_id', 'image_height', 'image_width']].head())
    else:
        print("\nShape consistency check: PASSED")

def generate_visuals(df=None):
    """Generates graphs and sample grids (from data_analysis_script)."""
    if df is None:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_metadata.csv"))

    print("\n--- Generating Visualizations ---")
    
    # Distribution Heatmap
    ct = pd.crosstab(df['set_type'], df['split'])
    plt.figure(figsize=(8,6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu').set_title("Set Type vs Split Distribution")
    plt.savefig(os.path.join(GRAPHS_DIR, "set_split_distribution.png"))
    plt.close()

    # Flood Ratio Hist
    df_hand = df[df['has_label_hand']].copy()
    if not df_hand.empty:
        plt.figure(figsize=(10,6))
        sns.histplot(df_hand['label_flood_pct'].dropna(), bins=50).set_title("Per-Tile Flood Ratio")
        plt.savefig(os.path.join(GRAPHS_DIR, "flood_ratio_hist.png"))
        plt.close()

    print(f"Graphs saved to {GRAPHS_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Consolidated Data Analysis Tool")
    parser.add_argument("mode", choices=["scan", "audit", "visualize", "full"], help="Analysis mode")
    parser.add_argument("--data", default=DATA_ROOT, help="Path to data directory")
    args = parser.parse_args()

    sys.stdout = TeeLogger(LOG_FILE)
    
    df = None
    if args.mode in ["scan", "full"]:
        df = generate_metadata(args.data)
    
    if args.mode in ["audit", "full"]:
        run_audits(df)
        
    if args.mode in ["visualize", "full"]:
        generate_visuals(df)

if __name__ == "__main__":
    main()
