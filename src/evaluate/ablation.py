
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

FIG_DIR = "results/figures"

from src.config import Config
from src.utils import set_seed, get_device
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier
from src.models.lstm_classifier import LSTMClassifier


# Helpers

def encode(ae, X, device, batch_size=256):
    ae.eval()
    latent, recon_err = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = ae(batch)
            err = torch.mean((batch - x_hat) ** 2, dim=1)
            latent.append(torch.cat([z, err.unsqueeze(1)], dim=1).cpu().numpy())
            recon_err.append(err.cpu().numpy())
    return np.vstack(latent), np.concatenate(recon_err)


def run_mamba(model, dataset, cfg, device):
    """Returns (attack_scores, labels) where scores are float in [0,1].
    Binary models: sigmoid output. Multi-class: 1 - P(normal class).
    Labels are always returned raw (collapsed to binary by caller if needed).
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=cfg.clf_batch_size, shuffle=False)
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            if logits.dim() == 1:  # binary (num_classes=1)
                scores.append(torch.sigmoid(logits).cpu().numpy())
            else:  # multi-class: P(attack) = 1 - P(normal)
                probs = torch.softmax(logits, dim=1)
                scores.append((1.0 - probs[:, 0]).cpu().numpy())
            labels.append(y_b.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, y_prob),
    }


def plot_ablation(all_results, out_path):
    """
    Grouped bar chart — one subplot per dataset, one group per metric,
    three bars per group (AE Only / AE+LR / AE+Mamba).
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    model_names  = ["AE Only", "AE + LR", "AE + LSTM", "AE + Mamba"]
    colors       = ["#059669", "#DC2626", "#F59E0B", "#2563EB"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("Ablation Study: Framework Component Contribution",
                 fontsize=14, fontweight="bold")

    x     = np.arange(len(metric_names))
    width = 0.20

    for ax, (ds_label, ds_results) in zip(axes, all_results.items()):
        offsets = [-1.5, -0.5, 0.5, 1.5]
        for i, (model_name, color, offset) in enumerate(zip(model_names, colors, offsets)):
            vals = [ds_results[model_name][m] for m in metric_names]
            bars = ax.bar(x + offset * width, vals, width,
                          label=model_name, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                        rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=10)
        ymin = 0.93 if ds_label == "TON-IoT" else 0.88
        ax.set_ylim([ymin, 1.02])
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(ds_label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def print_ablation_table(dataset_label, results):
    cols   = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    models = ["AE Only", "AE + LR", "AE + LSTM", "AE + Mamba"]
    width  = 11

    print(f"\n{'='*65}")
    print(f"  Ablation Results — {dataset_label}")
    print(f"{'='*65}")
    header = f"  {'Model':<16}" + "".join(f"{c:>{width}}" for c in cols)
    print(header)
    print(f"  {'-'*16}" + "-" * (width * len(cols)))

    for model_name in models:
        m = results[model_name]
        row = f"  {model_name:<16}" + "".join(f"{m[c]:>{width}.4f}" for c in cols)
        print(row)

    print(f"{'='*65}\n")


# Main

def main():
    cfg    = Config()
    set_seed(cfg.random_seed)
    device = get_device()

    datasets = {
        "TON-IoT": {
            "split_dir":   cfg.ton_splits_path,
            "ae_path":     cfg.ton_autoencoder_model_path,
            "clf_path":    cfg.ton_classifier_model_path,
            "lstm_path":   cfg.ton_lstm_model_path,
            "label_mode":  "last",
            "num_classes": 1,
        },
        "Simulated ICU": {
            "split_dir":   cfg.sim_splits_path,
            "ae_path":     cfg.sim_autoencoder_model_path,
            "clf_path":    cfg.sim_classifier_model_path,
            "lstm_path":   cfg.sim_lstm_model_path,
            "label_mode":  "max",
            "num_classes": cfg.sim_num_classes,
        },
    }

    all_results = {}

    for ds_label, ds_cfg in datasets.items():
        print(f"\nProcessing {ds_label}...")

        split_dir   = ds_cfg["split_dir"]
        label_mode  = ds_cfg["label_mode"]
        is_sim      = ds_label == "Simulated ICU"
        num_classes = ds_cfg["num_classes"]

        X_train         = np.load(os.path.join(split_dir, "X_train.npy"))
        X_test          = np.load(os.path.join(split_dir, "X_test.npy"))
        y_train         = np.load(os.path.join(split_dir, "y_train.npy"))
        y_test          = np.load(os.path.join(split_dir, "y_test.npy"))
        group_ids_train = np.load(os.path.join(split_dir, "group_ids_train.npy"), allow_pickle=True)
        group_ids_test  = np.load(os.path.join(split_dir, "group_ids_test.npy"),  allow_pickle=True)

        # load AE
        ae = Autoencoder(
            input_dim=X_train.shape[1],
            hidden_dim1=cfg.ae_hidden_dim1,
            hidden_dim2=cfg.ae_hidden_dim2,
            latent_dim=cfg.latent_dim,
        ).to(device)
        ae.load_state_dict(torch.load(ds_cfg["ae_path"], map_location=device))
        ae.eval()

        Z_train, err_train = encode(ae, X_train, device)
        Z_test,  err_test  = encode(ae, X_test,  device)

        train_ds = ArraySequenceDataset(Z_train, y_train, group_ids_train, cfg.seq_len, label_mode)
        test_ds  = ArraySequenceDataset(Z_test,  y_test,  group_ids_test,  cfg.seq_len, label_mode)

        # Ground-truth labels collapsed to binary (attack detected vs normal)
        ae_labels_raw = test_ds.labels.numpy()
        y_true = (ae_labels_raw > 0).astype(int) if is_sim else ae_labels_raw.astype(int)

        # 1. AE Only — threshold on reconstruction error of last row
        benign_err = err_train[y_train == 0]
        threshold  = benign_err.mean() + 2 * benign_err.std()
        ae_scores  = test_ds.samples[:, -1, -1].numpy()   # last row, last feature = recon error
        ae_pred    = (ae_scores >= threshold).astype(int)

        ae_metrics = metrics(y_true, ae_pred, ae_scores)

        # 2. AE + LR — train binary detection LR on last-row latent features
        X_tr_last   = train_ds.samples[:, -1, :].numpy()
        y_tr_binary = (train_ds.labels.numpy() > 0).astype(int) if is_sim else train_ds.labels.numpy().astype(int)
        X_te_last   = test_ds.samples[:,  -1, :].numpy()

        lr = LogisticRegression(max_iter=1000, random_state=cfg.random_seed)
        lr.fit(X_tr_last, y_tr_binary)
        lr_prob = lr.predict_proba(X_te_last)[:, 1]
        lr_pred = (lr_prob >= cfg.threshold).astype(int)

        lr_metrics = metrics(y_true, lr_pred, lr_prob)

        # 3. AE + Mamba
        mamba = MambaClassifier(
            input_dim=cfg.latent_dim + 1,
            d_model=cfg.d_model,
            n_layers=cfg.num_layers,
            dropout=cfg.dropout,
            num_classes=num_classes,
        ).to(device)
        mamba.load_state_dict(torch.load(ds_cfg["clf_path"], map_location=device))
        mamba.eval()

        mamba_score, _ = run_mamba(mamba, test_ds, cfg, device)
        mamba_pred     = (mamba_score >= cfg.threshold).astype(int)

        mamba_metrics = metrics(y_true, mamba_pred, mamba_score)

        # 4. AE + LSTM
        lstm = LSTMClassifier(
            input_dim=cfg.latent_dim + 1,
            hidden_dim=cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            num_classes=num_classes,
        ).to(device)
        lstm.load_state_dict(torch.load(ds_cfg["lstm_path"], map_location=device))
        lstm.eval()

        lstm_score, _ = run_mamba(lstm, test_ds, cfg, device)
        lstm_pred     = (lstm_score >= cfg.threshold).astype(int)

        lstm_metrics = metrics(y_true, lstm_pred, lstm_score)

        ds_results = {
            "AE Only":    ae_metrics,
            "AE + LR":    lr_metrics,
            "AE + LSTM":  lstm_metrics,
            "AE + Mamba": mamba_metrics,
        }
        print_ablation_table(ds_label, ds_results)
        all_results[ds_label] = ds_results

    # save figure
    print("Generating ablation figure...")
    plot_ablation(all_results, os.path.join(FIG_DIR, "ablation.png"))


if __name__ == "__main__":
    main()
