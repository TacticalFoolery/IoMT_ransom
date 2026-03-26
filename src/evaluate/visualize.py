"""
Generates all figures and saves them to results/figures/.

Figures produced

1. roc_curves.png         — ROC curves: Mamba vs LR on both datasets
2. confusion_matrices.png — Confusion matrix heatmaps (sim dataset)
3. early_detection.png    — Cumulative detection rate vs steps after attack onset
4. metrics_comparison.png — Bar chart: key metrics for Mamba vs LR on both datasets
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src.config import Config
from src.utils import set_seed, get_device
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier

ATTACK_START = 200
FIG_DIR = "results/figures"
STYLE = {
    "mamba":  {"color": "#2563EB", "label": "Mamba (sequence)"},
    "lr":     {"color": "#DC2626", "label": "LR (last row only)"},
    "ton":    {"color": "#059669", "label": "TON-IoT"},
    "sim":    {"color": "#7C3AED", "label": "Simulated ICU"},
}



# Helpers


def extract_latent_and_error(ae, X, device, batch_size=256):
    ae.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = ae(batch)
            err = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)
            out.append(torch.cat([z, err], dim=1).cpu().numpy())
    return np.vstack(out)


def run_mamba_inference(model, dataset, cfg, device):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=cfg.clf_batch_size, shuffle=False)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y_b.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def build_dataset(Z, y, group_ids, cfg, label_mode):
    return ArraySequenceDataset(
        features=Z, labels=y, group_ids=group_ids,
        seq_len=cfg.seq_len, label_mode=label_mode,
    )


def load_split(split_dir):
    return {k: np.load(os.path.join(split_dir, f"{k}.npy"), allow_pickle=True)
            for k in ("X_train", "X_test", "y_train", "y_test",
                      "group_ids_train", "group_ids_test")}


def lr_on_sequences(train_ds, test_ds, cfg):
    """Train LR on last-row features of train sequences, evaluate on test."""
    X_tr = train_ds.samples[:, -1, :].numpy()
    y_tr = train_ds.labels.numpy()
    X_te = test_ds.samples[:,  -1, :].numpy()
    lr = LogisticRegression(max_iter=1000, random_state=cfg.random_seed)
    lr.fit(X_tr, y_tr)
    prob = lr.predict_proba(X_te)[:, 1]
    return prob


def sliding_windows(Z, seq_len):
    N, F = Z.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=Z.dtype)
    idx = np.arange(seq_len)[None, :] + np.arange(N - seq_len + 1)[:, None]
    return Z[idx]



# Figure 1 — ROC curves


def plot_roc_curves(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ROC Curves: Mamba vs Logistic Regression Baseline", fontsize=14, fontweight="bold")

    for ax, (ds_name, ds_results) in zip(axes, results.items()):
        for model_name, (y_true, y_prob) in ds_results.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            s = STYLE[model_name]
            ax.plot(fpr, tpr, color=s["color"], lw=2,
                    label=f"{s['label']} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ds_label = "TON-IoT Dataset" if ds_name == "ton" else "Simulated ICU Dataset"
        ax.set_title(ds_label, fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")



# Figure 2 — Confusion matrices


def plot_confusion_matrices(cm_data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Confusion Matrices — Simulated ICU Dataset", fontsize=14, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list("blue_white", ["#ffffff", "#2563EB"])

    for ax, (title, (y_true, y_pred)) in zip(axes, cm_data.items()):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attack"], fontsize=10)
        ax.set_yticklabels(["Normal", "Attack"], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")



# Figure 3 — Early detection timeline


def plot_early_detection(mamba_lags, lr_lags, n_attacked, out_path):
    max_steps = 60
    steps = list(range(0, max_steps + 1))

    mamba_cum = [sum(1 for l in mamba_lags if l <= s) / n_attacked for s in steps]
    lr_cum    = [sum(1 for l in lr_lags    if l <= s) / n_attacked for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(steps, [v * 100 for v in mamba_cum],
            color=STYLE["mamba"]["color"], lw=2.5, label=STYLE["mamba"]["label"], marker="o", markersize=3)
    ax.plot(steps, [v * 100 for v in lr_cum],
            color=STYLE["lr"]["color"],    lw=2.5, label=STYLE["lr"]["label"],    marker="s", markersize=3,
            linestyle="--")

    ax.axvline(x=0, color="gray", linestyle=":", lw=1.5, label="Attack onset (t=200)")
    ax.fill_betweenx([0, 100], 0, 0, alpha=0.1, color="gray")

    ax.set_xlabel("Timesteps After Attack Onset", fontsize=12)
    ax.set_ylabel("Devices Detected (%)", fontsize=12)
    ax.set_title("Early Detection Rate: Mamba vs LR Baseline\n(Simulated ICU — Ransomware Onset)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim([0, max_steps])
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    for s, mv, lv in zip(steps, mamba_cum, lr_cum):
        if mv == 1.0 and mamba_cum[s - 1] < 1.0 if s > 0 else False:
            ax.annotate(f"Mamba 100%\n@ step {s}",
                        xy=(s, 100), xytext=(s + 3, 85),
                        arrowprops=dict(arrowstyle="->", color=STYLE["mamba"]["color"]),
                        color=STYLE["mamba"]["color"], fontsize=9)
        if lv == 1.0 and lr_cum[s - 1] < 1.0 if s > 0 else False:
            ax.annotate(f"LR 100%\n@ step {s}",
                        xy=(s, 100), xytext=(s + 3, 70),
                        arrowprops=dict(arrowstyle="->", color=STYLE["lr"]["color"]),
                        color=STYLE["lr"]["color"], fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")



# Figure 4 — Metrics comparison bar chart


def plot_metrics_comparison(metrics_data, out_path):
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    metric_keys  = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Performance Metrics: Mamba vs LR Baseline", fontsize=14, fontweight="bold")

    x = np.arange(len(metric_names))
    width = 0.35

    for ax, (ds_name, ds_metrics) in zip(axes, metrics_data.items()):
        mamba_vals = [ds_metrics["mamba"][k] for k in metric_keys]
        lr_vals    = [ds_metrics["lr"][k]    for k in metric_keys]

        bars1 = ax.bar(x - width / 2, mamba_vals, width,
                       label=STYLE["mamba"]["label"], color=STYLE["mamba"]["color"], alpha=0.85)
        bars2 = ax.bar(x + width / 2, lr_vals,    width,
                       label=STYLE["lr"]["label"],    color=STYLE["lr"]["color"],    alpha=0.85)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim([0.85, 1.02])
        ax.set_ylabel("Score", fontsize=11)
        ds_label = "TON-IoT Dataset" if ds_name == "ton" else "Simulated ICU Dataset"
        ax.set_title(ds_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")



# Main


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading models and data...")

    roc_results     = {}
    cm_data         = {}
    metrics_data    = {}

    for ds_name in ("ton", "sim"):
        label_mode  = "any" if ds_name == "sim" else "last"
        split_dir   = cfg.sim_splits_path if ds_name == "sim" else cfg.ton_splits_path
        ae_path     = cfg.sim_autoencoder_model_path if ds_name == "sim" else cfg.ton_autoencoder_model_path
        clf_path    = cfg.sim_classifier_model_path  if ds_name == "sim" else cfg.ton_classifier_model_path
        splits      = load_split(split_dir)

        # load dataset-specific AE
        ae = Autoencoder(
            input_dim=splits["X_train"].shape[1],
            hidden_dim1=cfg.ae_hidden_dim1,
            hidden_dim2=cfg.ae_hidden_dim2,
            latent_dim=cfg.latent_dim,
        ).to(device)
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        ae.eval()

        # load dataset-specific Mamba classifier
        mamba = MambaClassifier(
            input_dim=cfg.latent_dim + 1,
            d_model=cfg.d_model,
            n_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)
        mamba.load_state_dict(torch.load(clf_path, map_location=device))
        mamba.eval()

        print(f"\nEncoding {ds_name} features...")
        Z_train = extract_latent_and_error(ae, splits["X_train"], device)
        Z_test  = extract_latent_and_error(ae, splits["X_test"],  device)

        train_ds = build_dataset(Z_train, splits["y_train"], splits["group_ids_train"], cfg, label_mode)
        test_ds  = build_dataset(Z_test,  splits["y_test"],  splits["group_ids_test"],  cfg, label_mode)

        mamba_prob, y_true = run_mamba_inference(mamba, test_ds, cfg, device)
        lr_prob            = lr_on_sequences(train_ds, test_ds, cfg)

        mamba_pred = (mamba_prob >= cfg.threshold).astype(int)
        lr_pred    = (lr_prob    >= cfg.threshold).astype(int)

        roc_results[ds_name] = {
            "mamba": (y_true, mamba_prob),
            "lr":    (y_true, lr_prob),
        }

        if ds_name == "sim":
            cm_data["Mamba (sequence)"]      = (y_true, mamba_pred)
            cm_data["LR (last row only)"] = (y_true, lr_pred)

        def make_metrics(y_t, y_p, y_pb):
            return {
                "accuracy":  accuracy_score(y_t, y_p),
                "precision": precision_score(y_t, y_p, zero_division=0),
                "recall":    recall_score(y_t, y_p, zero_division=0),
                "f1":        f1_score(y_t, y_p, zero_division=0),
                "auc_roc":   roc_auc_score(y_t, y_pb),
            }

        metrics_data[ds_name] = {
            "mamba": make_metrics(y_true, mamba_pred, mamba_prob),
            "lr":    make_metrics(y_true, lr_pred,    lr_prob),
        }

    # early detection data (re-compute using sim models)
    print("\nComputing early detection lags...")
    splits   = load_split(cfg.sim_splits_path)
    ae = Autoencoder(
        input_dim=splits["X_train"].shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)
    ae.load_state_dict(torch.load(cfg.sim_autoencoder_model_path, map_location=device))
    ae.eval()

    mamba = MambaClassifier(
        input_dim=cfg.latent_dim + 1,
        d_model=cfg.d_model,
        n_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    mamba.load_state_dict(torch.load(cfg.sim_classifier_model_path, map_location=device))
    mamba.eval()

    Z_train = extract_latent_and_error(ae, splits["X_train"], device)
    Z_test  = extract_latent_and_error(ae, splits["X_test"],  device)

    # train LR for early detection
    unique_tr, inv_tr = np.unique(splits["group_ids_train"], return_inverse=True)
    last_rows_tr, labels_tr = [], []
    for gi in range(len(unique_tr)):
        rows = np.where(inv_tr == gi)[0]
        if len(rows) < cfg.seq_len:
            continue
        wins = sliding_windows(Z_train[rows], cfg.seq_len)
        y_r  = splits["y_train"][rows]
        labs = np.array([int(np.any(y_r[j:j + cfg.seq_len])) for j in range(len(rows) - cfg.seq_len + 1)])
        last_rows_tr.append(wins[:, -1, :])
        labels_tr.append(labs)
    lr_ed = LogisticRegression(max_iter=1000, random_state=cfg.random_seed)
    lr_ed.fit(np.vstack(last_rows_tr), np.concatenate(labels_tr))

    unique_te, inv_te = np.unique(splits["group_ids_test"], return_inverse=True)
    mamba_lags, lr_lags, n_attacked = [], [], 0

    for gi in range(len(unique_te)):
        rows   = np.where(inv_te == gi)[0]
        dev_id = unique_te[gi]
        if "attack" not in str(dev_id):
            continue
        n_attacked += 1
        Z_dev = Z_test[rows]
        if len(Z_dev) < cfg.seq_len:
            continue
        wins = sliding_windows(Z_dev, cfg.seq_len)
        # mamba
        with torch.no_grad():
            logits = mamba(torch.tensor(wins, dtype=torch.float32).to(device))
            m_preds = (torch.sigmoid(logits) >= cfg.threshold).cpu().numpy().astype(int)
        # lr
        l_preds = lr_ed.predict(wins[:, -1, :])

        def first_valid(preds):
            hits = np.where(preds == 1)[0]
            if not len(hits):
                return None
            t = hits[0] + cfg.seq_len - 1
            lag = t - ATTACK_START
            return lag if lag >= 0 else None

        ml = first_valid(m_preds)
        ll = first_valid(l_preds)
        if ml is not None:
            mamba_lags.append(ml)
        if ll is not None:
            lr_lags.append(ll)

    # generate figures
    print("\nGenerating figures...")
    plot_roc_curves(roc_results,                              os.path.join(FIG_DIR, "roc_curves.png"))
    plot_confusion_matrices(cm_data,                          os.path.join(FIG_DIR, "confusion_matrices.png"))
    plot_early_detection(mamba_lags, lr_lags, n_attacked,    os.path.join(FIG_DIR, "early_detection.png"))
    plot_metrics_comparison(metrics_data,                     os.path.join(FIG_DIR, "metrics_comparison.png"))

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
