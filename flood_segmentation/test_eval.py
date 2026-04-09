import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# Ensure this matches your training file name
from train import CFG, FloodDataset, MaskedUNet, make_loader, evaluate_detailed

def expand_weights_test(model, in_ch):
    if in_ch <= 3: return
    with torch.no_grad():
        old_conv = model.unet.encoder._conv_stem
        new_w = old_conv.weight.repeat(1, (in_ch // 3) + 1, 1, 1)[:, :in_ch, :, :]
        new_w *= (3.0 / in_ch)
        new_conv = nn.Conv2d(in_ch, old_conv.out_channels, 3, stride=2, padding=1, bias=False)
        new_conv.weight = nn.Parameter(new_w)
        model.unet.encoder._conv_stem = new_conv

def run_final_test_detailed():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🏁 Starting Detailed Final Test on {device}...")
    
    ABLATIONS = {
        "OPTICAL_ONLY":     list(range(2, 15)),
        "SAR_ONLY":         [0, 1],
        "OPTICAL_SAR":      list(range(0, 15)),
        "OPTICAL_AUX":      list(range(2, 15)) + list(range(23, 27)),
        "OPTICAL_TEMPORAL": list(range(2, 23)),
        "SAR_TEMPORAL":     [0, 1, 15, 16],
        "NO_AUX":           list(range(0, 23)),
        "FULL_TEAM":        list(range(27)),
    }

    test_results = []

    for name, chs in ABLATIONS.items():
        model_path = Path(f"best_{name}_model.pth")
        if not model_path.exists():
            print(f"⚠️  Skipping {name}: No checkpoint.")
            continue

        model = MaskedUNet(in_channels=len(chs), classes=2).to(device)
        expand_weights_test(model, len(chs))
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Metrics from evaluate_detailed
        f_iou, m_iou, pred_ratio = evaluate_detailed(model, device, chs, split="test")
        
        test_results.append({
            "Model": name,
            "Test_Flood_IoU": round(f_iou, 4),
            "Test_Mean_IoU": round(m_iou, 4),
            "Pred_GT_Ratio": round(pred_ratio, 3)
        })
        
        print(f"✅ {name:16} | IoU: {f_iou:.4f} | Ratio: {pred_ratio:.3f}")

    # --- 💾 1. SAVE COMPREHENSIVE LEADERBOARD ---
    df = pd.DataFrame(test_results).sort_values(by="Test_Flood_IoU", ascending=False)
    df.to_csv("final_detailed_test_results.csv", index=False)
    print("\n💾 Results saved to final_detailed_test_results.csv")

    # --- 📈 2. GENERATE DUAL-METRIC PLOT ---
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot A: Flood IoU (Efficiency)
    sns.barplot(data=df, x="Test_Flood_IoU", y="Model", ax=ax1, palette="viridis")
    ax1.set_title("Detection Accuracy (Flood IoU)", fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    for i, v in enumerate(df["Test_Flood_IoU"]):
        ax1.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

    # Plot B: Pred/GT Ratio (Calibration)
    # Add a vertical line at 1.0 for the 'Perfect Match' baseline
    sns.barplot(data=df, x="Pred_GT_Ratio", y="Model", ax=ax2, palette="magma")
    ax2.axvline(1.0, color='red', linestyle='--', alpha=0.6, label="Perfect GT Match")
    ax2.set_title("Prediction Bias (Pred/GT Ratio)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Ratio (Values > 1.0 = Over-prediction)")
    ax2.legend()
    for i, v in enumerate(df["Pred_GT_Ratio"]):
        ax2.text(v + 0.02, i, f'{v:.2f}', va='center')

    plt.tight_layout()
    plt.savefig("plots/final_detailed_metrics.png", dpi=300)
    print("🖼️  Detailed metrics chart saved to plots/final_detailed_metrics.png")

if __name__ == "__main__":
    run_final_test_detailed()