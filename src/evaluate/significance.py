"""
Statistical Significance Testing

Runs McNemar's test on all pairs of models to determine whether performance
differences are statistically significant.

McNemar's test operates on the 2x2 contingency table of discordant predictions:

           Model B correct   Model B wrong
Model A correct      n00              n01
Model A wrong        n10              n11

Statistic = (|n01 - n10| - 1)^2 / (n01 + n10)   [continuity-corrected]
Distributed as chi2(1) under the null hypothesis that both models have the
same error rate.

For the simulated ICU dataset (multi-class), all model predictions are
collapsed to binary (attack detected vs normal) so the test measures detection
ability consistently across all models.

Comparisons run (for each dataset):
  Mamba vs LR
  Mamba vs LSTM
  Mamba vs AE-only
  LSTM  vs LR       (context: does any sequence model beat LR?)
"""

import os
import numpy as np
import torch
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import set_seed, get_device
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier
from src.models.lstm_classifier import LSTMClassifier


# helpers

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


def run_seq_model(model, dataset, cfg, device):
    """Returns (predictions, labels).
    Binary models: sigmoid >= threshold. Multi-class: argmax.
    """
    loader = DataLoader(dataset, batch_size=cfg.clf_batch_size, shuffle=False)
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            if logits.dim() == 1:  # binary
                p = (torch.sigmoid(logits) >= cfg.threshold).cpu().numpy().astype(int)
            else:  # multi-class
                p = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            preds.append(p)
            labels.append(y_b.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def mcnemar_test(pred_a, pred_b, y_true):
    """
    Returns (statistic, p_value, n01, n10) for McNemar's test.

    pred_a, pred_b : binary prediction arrays (N,)
    y_true         : ground truth labels (N,)
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n01 = int(np.sum( correct_a & ~correct_b))  # A right, B wrong
    n10 = int(np.sum(~correct_a &  correct_b))  # A wrong, B right

    if n01 + n10 == 0:
        return 0.0, 1.0, n01, n10

    # McNemar's with continuity correction (Edwards, 1948)
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p    = 1.0 - chi2.cdf(stat, df=1)
    return stat, p, n01, n10


def print_mcnemar_table(dataset_label, results):
    """
    results : list of (comparison_label, stat, p, n01, n10, n_test)
    """
    col_w = 26
    print(f"\n{'='*75}")
    print(f"  McNemar's Test — {dataset_label}  (n = {results[0][5]:,} sequences)")
    print(f"{'='*75}")
    print(f"  {'Comparison':<{col_w}}  {'chi2':>7}  {'p-value':>10}  "
          f"{'A-only':>8}  {'B-only':>8}  {'sig':>5}")
    print(f"  {'-'*col_w}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5}")

    for label, stat, p, n01, n10, _ in results:
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {label:<{col_w}}  {stat:>7.3f}  {p:>10.4e}  "
              f"{n01:>8,}  {n10:>8,}  {sig:>5}")

    print(f"\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print(f"  A-only = sequences Model A got right but Model B missed")
    print(f"  B-only = sequences Model B got right but Model A missed")
    print(f"{'='*75}")


# main

def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()
    print("Using device:", device)

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

    for ds_label, ds_cfg in datasets.items():
        print(f"\nLoading {ds_label}...")

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

        # ── autoencoder ──────────────────────────────────────────────────────
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

        # ── build sequence datasets ───────────────────────────────────────────
        train_ds = ArraySequenceDataset(
            Z_train, y_train, group_ids_train, cfg.seq_len, label_mode
        )
        test_ds = ArraySequenceDataset(
            Z_test, y_test, group_ids_test, cfg.seq_len, label_mode
        )
        y_seq = test_ds.labels.numpy()

        # ── Mamba ─────────────────────────────────────────────────────────────
        mamba = MambaClassifier(
            input_dim=cfg.latent_dim + 1,
            d_model=cfg.d_model,
            n_layers=cfg.num_layers,
            dropout=cfg.dropout,
            num_classes=num_classes,
        ).to(device)
        mamba.load_state_dict(torch.load(ds_cfg["clf_path"], map_location=device))
        mamba.eval()
        mamba_pred, _ = run_seq_model(mamba, test_ds, cfg, device)

        # ── LSTM ──────────────────────────────────────────────────────────────
        lstm = LSTMClassifier(
            input_dim=cfg.latent_dim + 1,
            hidden_dim=cfg.d_model,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            num_classes=num_classes,
        ).to(device)
        lstm.load_state_dict(torch.load(ds_cfg["lstm_path"], map_location=device))
        lstm.eval()
        lstm_pred, _ = run_seq_model(lstm, test_ds, cfg, device)

        # ── LR (last row of each sequence) ────────────────────────────────────
        X_tr_last = train_ds.samples[:, -1, :].numpy()
        y_tr_seq  = train_ds.labels.numpy()
        X_te_last = test_ds.samples[:,  -1, :].numpy()

        lr = LogisticRegression(max_iter=1000, random_state=cfg.random_seed)
        lr.fit(X_tr_last, y_tr_seq)
        lr_pred = lr.predict(X_te_last)

        # ── AE-only (threshold on reconstruction error of last row) ───────────
        benign_err   = err_train[y_train == 0]
        ae_threshold = benign_err.mean() + 2 * benign_err.std()
        ae_scores    = test_ds.samples[:, -1, -1].numpy()   # last row, last feature = recon error
        ae_pred      = (ae_scores >= ae_threshold).astype(int)

        # ── Collapse to binary for McNemar (detection task: attack vs normal) ──
        # All models are compared on whether they correctly detect ANY attack,
        # regardless of variant. This is fair since AE-only cannot classify variants.
        y_true_binary = (y_seq       > 0).astype(int) if is_sim else y_seq.astype(int)
        mamba_binary  = (mamba_pred  > 0).astype(int) if is_sim else mamba_pred.astype(int)
        lstm_binary   = (lstm_pred   > 0).astype(int) if is_sim else lstm_pred.astype(int)
        lr_binary     = (lr_pred     > 0).astype(int) if is_sim else lr_pred.astype(int)

        # ── run McNemar's tests ───────────────────────────────────────────────
        n = len(y_true_binary)
        comparisons = [
            ("Mamba vs LR",       mamba_binary, lr_binary),
            ("Mamba vs LSTM",     mamba_binary, lstm_binary),
            ("Mamba vs AE-only",  mamba_binary, ae_pred),
            ("LSTM  vs LR",       lstm_binary,  lr_binary),
        ]

        rows = []
        for label, pred_a, pred_b in comparisons:
            stat, p, n01, n10 = mcnemar_test(pred_a, pred_b, y_true_binary)
            rows.append((label, stat, p, n01, n10, n))

        print_mcnemar_table(ds_label, rows)


if __name__ == "__main__":
    main()
