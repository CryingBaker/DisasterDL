import time
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from rich.console import Console
from typing import Dict, List, Sequence

from config import CFG, logger
from data import FloodDataset, setup_scientific_data
from model import MaskedUNet, HybridSegLoss, make_loader, evaluate_detailed


console = Console()



def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    chs: Sequence[int],
    train: bool = True,
    grad_clip_norm: float = 1.0,
    epoch_name: str = "",
) -> float:
    model.train(train)
    total_loss = 0.0
    num_batches = len(loader)

    logger.info(f"{epoch_name} batches: {num_batches}")
    console.print(f"[bold cyan]{epoch_name}[/bold cyan] batches: {num_batches}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(epoch_name, total=num_batches)

        for batch_idx, b in enumerate(loader, start=1):
            x = b["image"][:, chs].to(device)
            y = b["label"].to(device).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.item())

            progress.update(task, advance=1)

            if batch_idx == 1 or batch_idx % CFG.log_every_n_batches == 0 or batch_idx == num_batches:
                logger.info(f"{epoch_name} batch {batch_idx}/{num_batches} | loss={loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def run_research() -> pd.DataFrame:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"🚀 Device: {device}")

    ABLATIONS = {
        "OPTICAL_ONLY":     list(range(2, 15)),
        "SAR_ONLY":         [0, 1],
        "OPTICAL_SAR":      list(range(0, 15)),
        "OPTICAL_AUX":      list(range(2, 15)) + list(range(23, 27)),
        "OPTICAL_TEMPORAL": list(range(2, 23)),
        "SAR_TEMPORAL":     [0, 1, 15, 16],
        "NO_AUX":           list(range(0, 23)),
        "FULL_TEAM":        list(range(0, 27)),
    }

    history: List[Dict[str, object]] = []

    for name, chs in ABLATIONS.items():
        logger.info(f"\n🔥 Starting Ablation: {name} ({len(chs)} channels)")
        print(f"🔥 Starting Ablation: {name} ({len(chs)} channels)", flush=True)

        model = MaskedUNet(in_channels=len(chs), classes=2).to(device)
        criterion = HybridSegLoss().to(device)

        best_iou = -1.0

        # ── Weak stage ──────────────────────────────────────────────
        weak_ds = FloodDataset(CFG.data_root, "train", use_weak=True)
        logger.info(f"{name} weak tiles: {len(weak_ds)}")
        print(f"{name} weak tiles: {len(weak_ds)}", flush=True)

        if len(weak_ds) > 0:
            weak_loader = make_loader(weak_ds, batch_size=CFG.weak_batch_size, shuffle=True)
            optimizer = make_optimizer(model, CFG.weak_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(CFG.weak_epochs, 1))

            for ep in range(1, CFG.weak_epochs + 1):
                tag = f"[{name}] Weak {ep}"
                train_loss = run_epoch(
                    model=model,
                    loader=weak_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    chs=chs,
                    train=True,
                    grad_clip_norm=CFG.grad_clip_norm,
                    epoch_name=tag,
                )
                scheduler.step()
                val_f_iou, val_m_iou, pred_ratio = evaluate_detailed(model, device, chs, split="val")

                history.append({
                    "Model": name,
                    "Stage": "Weak",
                    "Epoch": ep,
                    "Loss": train_loss,
                    "Val_Flood_IoU": val_f_iou,
                    "Val_Mean_IoU": val_m_iou,
                    "PredFloodRatio": pred_ratio,
                })

                logger.info(
                    f"[{name}] W-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                    f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
                print(
                    f"[{name}] W-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                    f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}",
                    flush=True,
                )

        # ── Hand stage ──────────────────────────────────────────────
        hand_ds = FloodDataset(CFG.data_root, "train", use_weak=False)
        logger.info(f"{name} hand tiles: {len(hand_ds)}")
        print(f"{name} hand tiles: {len(hand_ds)}", flush=True)

        if len(hand_ds) == 0:
            logger.warning(f"[{name}] No hand training tiles found. Skipping hand stage.")
            continue

        hand_loader = make_loader(hand_ds, batch_size=CFG.hand_batch_size, shuffle=True)
        optimizer = make_optimizer(model, CFG.hand_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(CFG.hand_epochs, 1))

        for ep in range(1, CFG.hand_epochs + 1):
            tag = f"[{name}] Hand {ep}"
            train_loss = run_epoch(
                model=model,
                loader=hand_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                chs=chs,
                train=True,
                grad_clip_norm=CFG.grad_clip_norm,
                epoch_name=tag,
            )
            scheduler.step()
            val_f_iou, val_m_iou, pred_ratio = evaluate_detailed(model, device, chs, split="val")

            history.append({
                "Model": name,
                "Stage": "Hand",
                "Epoch": ep + CFG.weak_epochs,
                "Loss": train_loss,
                "Val_Flood_IoU": val_f_iou,
                "Val_Mean_IoU": val_m_iou,
                "PredFloodRatio": pred_ratio,
            })

            logger.info(
                f"[{name}] H-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            print(
                f"[{name}] H-Ep {ep} | Loss: {train_loss:.4f} | Flood IoU: {val_f_iou:.4f} | "
                f"Mean IoU: {val_m_iou:.4f} | Pred/GT Flood Ratio: {pred_ratio:.3f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}",
                flush=True,
            )

            if val_f_iou > best_iou:
                best_iou = val_f_iou
                torch.save(model.state_dict(), f"best_{name}_model.pth")

    result = pd.DataFrame(history)
    result.to_csv("ablation_results_full.csv", index=False)
    logger.info("✅ Training complete. Results saved to ablation_results_full.csv")
    return result



def generate_plots(results_csv: str = "ablation_results_full.csv") -> None:
    from data import safe_mkdir

    df = pd.read_csv(results_csv)
    safe_mkdir(CFG.plots_dir)
    sns.set_theme(style="whitegrid")

    # Plot 1: Flood IoU over epochs
    plt.figure(figsize=(13, 7))
    sns.lineplot(data=df, x="Epoch", y="Val_Flood_IoU", hue="Model", lw=2)
    plt.axvline(x=CFG.weak_epochs + 0.5, color="black", linestyle=":", linewidth=1.5, label="Hand Fine-tuning Start")
    plt.xlabel("Epoch")
    plt.ylabel("Flood IoU (Val)")
    plt.title("Ablation Study: Flood IoU across Channel Configurations")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_iou_main.png", dpi=300)
    plt.close()

    # Plot 2: Loss curves per ablation (facet grid)
    g = sns.FacetGrid(df, col="Model", col_wrap=4, height=3.2, sharey=False)
    g.map_dataframe(sns.lineplot, x="Epoch", y="Loss")
    for ax in g.axes.flatten():
        ax.axvline(x=CFG.weak_epochs + 0.5, color="red", linestyle="--", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_loss_facets.png", dpi=300)
    plt.close()

    # Plot 3: Best Flood IoU bar chart
    final = df[df["Stage"] == "Hand"].groupby("Model", as_index=False)["Val_Flood_IoU"].max()
    final = final.sort_values("Val_Flood_IoU", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=final, x="Model", y="Val_Flood_IoU")
    for bar, val in zip(ax.patches, final["Val_Flood_IoU"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Best Val Flood IoU")
    plt.title("Peak Flood IoU by Ablation Configuration")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_final_bar.png", dpi=300)
    plt.close()

    # Plot 4: Mean IoU heatmap (Hand stage only)
    hand_df = df[df["Stage"] == "Hand"].copy()
    pivot = hand_df.pivot_table(index="Model", columns="Epoch", values="Val_Mean_IoU")
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False, linewidths=0.3)
    plt.title("Mean IoU Heatmap (Hand Stage)")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(CFG.plots_dir / "plot_meaniou_heatmap.png", dpi=300)
    plt.close()

    logger.info(f"📊 All plots saved to {CFG.plots_dir.resolve()} at 300 DPI.")



if __name__ == "__main__":
    setup_scientific_data()
    run_research()
    generate_plots()