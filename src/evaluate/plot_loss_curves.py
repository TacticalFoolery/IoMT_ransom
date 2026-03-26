"""
Plots training loss curves for all models and datasets.
Saves to results/figures/loss_curves.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config

FIG_DIR  = "results/figures"
LOSS_DIR = "results/losses"


def load_losses(filename):
    path = os.path.join(LOSS_DIR, filename)
    if not os.path.exists(path):
        return None
    return np.load(path)


def main():
    cfg = Config()
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    panels = [
        {
            "ax_title": "Autoencoder (Reconstruction Loss)",
            "series": [
                ("TON-IoT",       "ae_ton_losses.npy",  "#059669", "-"),
                ("Simulated ICU", "ae_sim_losses.npy",  "#7C3AED", "--"),
            ],
        },
        {
            "ax_title": "Mamba Classifier (BCE Loss)",
            "series": [
                ("TON-IoT",       "mamba_ton_losses.npy", "#059669", "-"),
                ("Simulated ICU", "mamba_sim_losses.npy", "#7C3AED", "--"),
            ],
        },
        {
            "ax_title": "LSTM Classifier (BCE Loss)",
            "series": [
                ("TON-IoT",       "lstm_ton_losses.npy", "#059669", "-"),
                ("Simulated ICU", "lstm_sim_losses.npy", "#7C3AED", "--"),
            ],
        },
    ]

    for ax, panel in zip(axes, panels):
        plotted = False
        for label, filename, color, linestyle in panel["series"]:
            losses = load_losses(filename)
            if losses is None:
                print(f"  Warning: {filename} not found, skipping.")
                continue
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, color=color, linestyle=linestyle,
                    linewidth=2, marker="o", markersize=3, label=label)
            plotted = True

        ax.set_title(panel["ax_title"], fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "loss_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
