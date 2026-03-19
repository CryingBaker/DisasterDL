import os
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import FloodDataset
from model_training import FloodUNet, compute_metrics

# --- CONFIG ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PRIORITY = ["models/finetune_best.pth", "models/pretrain_best.pth"]
NORM_STATS_PATH = "data_analysis/normalization_stats.json"
NUM_TILES = 10
REGIONS = ["Sri-Lanka"]
SAVE_PATH = "logs/test_predictions.png"

def load_model(checkpoint_path):
    model = FloodUNet(in_channels=20, num_classes=2).to(DEVICE)
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded weights from {checkpoint_path} (Epoch {checkpoint['epoch']})")
    return model

def unnormalize_rgb(img_tensor, stats):
    # img_tensor: (20, 512, 512)
    # R: index 5 (B4), G: index 4 (B3), B: index 3 (B2)
    # stats: {'s2': {'mean': [...], 'std': [...]}}
    
    r_idx, g_idx, b_idx = 5, 4, 3
    
    def get_raw(channel_idx, stat_key, stat_idx):
        m = stats[stat_key]['mean'][stat_idx]
        s = stats[stat_key]['std'][stat_idx]
        return img_tensor[channel_idx].cpu().numpy() * s + m

    # S2 indices in stats match 0-5 in dataloader (bands 1-6)
    # img[2] is band 1 (stat index 0)
    # img[3] is band 2 (stat index 1) -> Blue
    # img[4] is band 3 (stat index 2) -> Green
    # img[5] is band 4 (stat index 3) -> Red
    
    r = get_raw(5, 's2', 3)
    g = get_raw(4, 's2', 2)
    b = get_raw(3, 's2', 1)
    
    rgb = np.stack([r, g, b], axis=-1)
    
    # Simple percentile clipping for visualization
    def clip_normalize(x):
        low, high = np.percentile(x, (2, 98))
        x = np.clip(x, low, high)
        return (x - low) / (high - low + 1e-6)
    
    return clip_normalize(rgb)

def main():
    # Load weights
    model = None
    for cp in CHECKPOINT_PRIORITY:
        model = load_model(cp)
        if model: break
    if not model:
        print("Error: No checkpoints found!")
        return

    # Load norm stats
    with open(NORM_STATS_PATH, 'r') as f:
        stats = json.load(f)

    # Initialize Dataset
    dataset = FloodDataset(root=".", phase="finetune", split="test")
    
    # Filter by region (Ghana, Somalia, Spain)
    # Tile IDs start with region name (e.g. Ghana_123)
    val_indices = [i for i, t in enumerate(dataset.tiles) if any(t['tile_id'].startswith(r) for r in REGIONS)]
    
    if not val_indices:
        print("Warning: No tiles found for requested regions, using all validation tiles.")
        val_indices = list(range(len(dataset)))
        
    random.seed(42)
    selected_indices = random.sample(val_indices, min(NUM_TILES, len(val_indices)))
    
    # Setup plot
    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(20, 5 * len(selected_indices)))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for i, idx in enumerate(selected_indices):
        batch = dataset[idx]
        img = batch['image'].unsqueeze(0).to(DEVICE) # (1, 20, 512, 512)
        target = batch['label'] # (512, 512)
        tid = batch['tile_id']
        
        with torch.no_grad():
            preds = model(img) # (1, 2, 512, 512)
            metrics = compute_metrics(preds, target.unsqueeze(0).to(DEVICE))
            pred_mask = torch.argmax(preds[0], dim=0).cpu().numpy()
        
        # 1. RGB
        rgb = unnormalize_rgb(batch['image'], stats)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Tile: {tid}\nPost-Event RGB (S2)")
        
        # 2. GT Mask
        # flood=red (1), no-flood=grey (0), ignore=-1=black
        gt = target.numpy()
        gt_viz = np.zeros((*gt.shape, 3))
        gt_viz[gt == 1] = [1, 0, 0]   # Red
        gt_viz[gt == 0] = [0.5, 0.5, 0.5] # Grey
        gt_viz[gt == -1] = [0, 0, 0]  # Black
        axes[i, 1].imshow(gt_viz)
        axes[i, 1].set_title(f"Ground Truth\nFlood IoU: {metrics['iou_flood']:.3f}")
        
        # 3. Prediction
        # flood=red (1), no-flood=grey (0)
        pred_viz = np.zeros((*pred_mask.shape, 3))
        pred_viz[pred_mask == 1] = [1, 0, 0]
        pred_viz[pred_mask == 0] = [0.5, 0.5, 0.5]
        axes[i, 2].imshow(pred_viz)
        axes[i, 2].set_title(f"Model Prediction\nMean IoU: {metrics['mean_iou']:.3f}")
        
        # 4. Overlay
        axes[i, 3].imshow(rgb)
        overlay = np.zeros((*pred_mask.shape, 4))
        overlay[pred_mask == 1] = [1, 0, 0, 0.4] # Transparent red
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f"Prediction Overlay\n(40% Alpha Red)")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
