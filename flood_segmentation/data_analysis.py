import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
# Assuming this script is run from the same root where OUTPUT_DIR = "data_analysis"
CSV_PATH = os.path.join("data_analysis", "dataset_metadata.csv")
OUTPUT_PATH = "Flood_dataset_analysis_paper.png"

def generate_exact_chart():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}. Make sure you ran your data script first.")
        return

    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Filter for hand-labelled tiles only (as per your paper description)
    df = df[df['has_label_hand'] == True].copy()
    
    # 2. Extract Region from tile_id (e.g., 'Bolivia_1234' -> 'Bolivia')
    df['Region'] = df['tile_id'].apply(lambda x: x.split('_')[0])
    
    # 3. Calculate exact pixel counts from percentages
    # total_pixels = height * width
    # valid_pixels = total_pixels * (1 - invalid_pct/100)
    # flood_pixels = valid_pixels * (flood_pct/100)
    # dry_pixels = valid_pixels - flood_pixels
    
    df['total_px'] = df['image_height'] * df['image_width']
    df['valid_px'] = df['total_px'] * (1 - df['label_novalid_pct'] / 100.0)
    df['flood_px'] = df['valid_px'] * (df['label_flood_pct'] / 100.0)
    df['dry_px'] = df['valid_px'] - df['flood_px']
    
    # 4. Aggregate by Region
    agg_df = df.groupby('Region')[['dry_px', 'flood_px']].sum().reset_index()
    
    # 5. Enforce specific ordering to match your paper's splits
    ordered_regions = [
        'Bolivia', 'Ghana', 'Nigeria', 'Pakistan', 'Paraguay', 
        'Somalia', 'Spain', 'Sri-Lanka', 'USA', 'India', 'Mekong'
    ]
    
    # Handle slight naming mismatches in raw data (e.g., "Sri Lanka" vs "Sri-Lanka")
    region_map = {r.replace('-', '').lower(): r for r in ordered_regions}
    agg_df['Matched_Region'] = agg_df['Region'].apply(lambda r: region_map.get(r.replace('-', '').lower(), r))
    
    # Filter to wanted regions and sort
    agg_df = agg_df[agg_df['Matched_Region'].isin(ordered_regions)]
    agg_df['Matched_Region'] = pd.Categorical(agg_df['Matched_Region'], categories=ordered_regions, ordered=True)
    agg_df = agg_df.sort_values('Matched_Region')
    
    regions = agg_df['Matched_Region'].tolist()
    dry_counts = agg_df['dry_px'].tolist()
    flood_counts = agg_df['flood_px'].tolist()

    print("Generating high-resolution plot...")

    # --- Plotting ---
    # Massive font sizes ensure readability in IEEE double-column format
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.title_fontsize': 16
    })

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(regions))
    width = 0.4

    # Bars
    rects1 = ax.bar(x - width/2, dry_counts, width, label='Dry', color='#8D9999', edgecolor='black', linewidth=1.2)
    rects2 = ax.bar(x + width/2, flood_counts, width, label='Flood', color='#265C83', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Total Pixel Count (Log Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_yscale('log')
    
    ax.legend(title='Pixel Class', loc='upper left', framealpha=1.0, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotations (Train/Val/Test lines at the bottom)
    y_line = -0.12
    y_text = -0.17

    def add_split_group(ax, start_idx, end_idx, text, color):
        x_start = (start_idx - 0.4) / (len(regions) - 1)
        x_end = (end_idx + 0.4) / (len(regions) - 1)
        ax.annotate('', xy=(x_start, y_line), xytext=(x_end, y_line),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', color=color, lw=3))
        center_x = (x_start + x_end) / 2
        ax.annotate(text, xy=(center_x, y_text), xycoords='axes fraction', 
                    ha='center', va='top', fontsize=14, fontweight='bold', color=color)

    # 0-7 = Train, 8 = Val (USA), 9-10 = Test (India/Mekong)
    add_split_group(ax, 0, 7, 'TRAINING SET', '#34495e')
    add_split_group(ax, 8, 8, 'VALIDATION', '#e67e22')
    add_split_group(ax, 9, 10, 'TESTING SET', '#27ae60')

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Success! Graph saved as '{OUTPUT_PATH}'")

if __name__ == "__main__":
    generate_exact_chart()