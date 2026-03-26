"""
Early Detection Analysis

Measures how quickly Mamba vs a per-row LR baseline detects a ransomware
attack after it begins on each ICU device.

Detection is defined as: the first sliding-window sequence that the model
classifies as an attack.  For Mamba this window spans seq_len=20 rows; for
LR the "window" is just the last row (same as per-row inference).

"""

import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.utils import set_seed, get_device
from src.models.autoencoder import Autoencoder
from src.models.mamba_classifier import MambaClassifier


ATTACK_START = 200   # must match simulate_icu.py


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


def sliding_windows(Z, seq_len):
    """Return (N - seq_len + 1, seq_len, F) array of sliding windows."""
    N, F = Z.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), dtype=Z.dtype)
    idx = np.arange(seq_len)[None, :] + np.arange(N - seq_len + 1)[:, None]
    return Z[idx]


def first_detection_timestep(preds, seq_len):
    """
    preds : binary array of length (N - seq_len + 1)
             preds[i] corresponds to the window ending at timestep i + seq_len - 1
    Returns the timestep of the first positive prediction, or None.
    """
    hits = np.where(preds == 1)[0]
    if len(hits) == 0:
        return None
    first_window = hits[0]
    return first_window + seq_len - 1   # last timestep of that window


def ascii_chart(title, x_vals, mamba_vals, lr_vals, width=50):
    print(f"\n  {title}")
    print(f"  {'Step':>5}  {'Mamba':>7}  {'LR':>7}  {'Mamba':^{width}}  {'LR':^{width}}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*width}  {'-'*width}")
    for x, mv, lv in zip(x_vals, mamba_vals, lr_vals):
        mb = int(mv * width)
        lb = int(lv * width)
        print(f"  {x:>5}  {mv:>6.1%}  {lv:>6.1%}  {'#'*mb:{width}}  {'#'*lb:{width}}")


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()
    seq_len = cfg.seq_len

    split_dir = cfg.sim_splits_path

    X_train         = np.load(os.path.join(split_dir, "X_train.npy"))
    X_test          = np.load(os.path.join(split_dir, "X_test.npy"))
    y_train         = np.load(os.path.join(split_dir, "y_train.npy"))
    y_test          = np.load(os.path.join(split_dir, "y_test.npy"))
    group_ids_train = np.load(os.path.join(split_dir, "group_ids_train.npy"), allow_pickle=True)
    group_ids_test  = np.load(os.path.join(split_dir, "group_ids_test.npy"),  allow_pickle=True)

    # load models
    ae = Autoencoder(
        input_dim=X_train.shape[1],
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

    print("Encoding features...")
    Z_train = extract_latent_and_error(ae, X_train, device)
    Z_test  = extract_latent_and_error(ae, X_test,  device)

    # train LR on last-row features of training sequences
    unique_train, inv_train = np.unique(group_ids_train, return_inverse=True)
    train_last, train_labels = [], []
    for gi in range(len(unique_train)):
        rows = np.where(inv_train == gi)[0]
        if len(rows) < seq_len:
            continue
        windows = sliding_windows(Z_train[rows], seq_len)   # (W, seq_len, F)
        last_rows = windows[:, -1, :]                        # (W, F)
        y_rows    = y_train[rows]
        labels    = np.array([int(np.any(y_rows[j:j + seq_len])) for j in range(len(rows) - seq_len + 1)])
        train_last.append(last_rows)
        train_labels.append(labels)

    lr = LogisticRegression(max_iter=1000, random_state=cfg.random_seed)
    lr.fit(np.vstack(train_last), np.concatenate(train_labels))

    # per-device early detection on test set
    unique_test, inv_test = np.unique(group_ids_test, return_inverse=True)

    mamba_lags, lr_lags = [], []
    mamba_missed, lr_missed = 0, 0
    n_attacked = 0

    for gi in range(len(unique_test)):
        rows = np.where(inv_test == gi)[0]
        dev_id = unique_test[gi]

        # only evaluate attacked devices
        if "attack" not in str(dev_id):
            continue

        n_attacked += 1
        Z_dev = Z_test[rows]
        y_dev = y_test[rows]

        if len(Z_dev) < seq_len:
            mamba_missed += 1
            lr_missed    += 1
            continue

        windows = sliding_windows(Z_dev, seq_len)   # (W, seq_len, F)

        # Mamba inference
        X_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = mamba(X_tensor)
            mamba_preds = (torch.sigmoid(logits) >= cfg.threshold).cpu().numpy().astype(int)

        # LR inference (last row of each window)
        last_rows = windows[:, -1, :]
        lr_preds  = lr.predict(last_rows)

        # detection timestep = first window predicted as attack, mapped back to
        # the device's local timestep axis (0 = first row in device's test slice)
        mt = first_detection_timestep(mamba_preds, seq_len)
        lt = first_detection_timestep(lr_preds,    seq_len)

        if mt is None:
            mamba_missed += 1
        else:
            lag = mt - ATTACK_START
            if lag < 0:
                # Model fired before attack onset — false positive, not early detection
                mamba_missed += 1
                print(f"  [FP] Mamba false alarm on {dev_id} at t={mt} ({lag} steps before attack)")
            else:
                mamba_lags.append(lag)

        if lt is None:
            lr_missed += 1
        else:
            lag = lt - ATTACK_START
            if lag < 0:
                lr_missed += 1
                print(f"  [FP] LR false alarm on {dev_id} at t={lt} ({lag} steps before attack)")
            else:
                lr_lags.append(lag)

    # summary statistics
    print(f"\nAttacked devices in test set : {n_attacked}")
    print(f"Mamba  — detected: {len(mamba_lags)}/{n_attacked}, missed: {mamba_missed}")
    print(f"LR     — detected: {len(lr_lags)}/{n_attacked},    missed: {lr_missed}")

    if mamba_lags:
        print(f"\nMamba  detection lag (steps after attack_start):")
        print(f"  Mean   : {np.mean(mamba_lags):.1f}")
        print(f"  Median : {np.median(mamba_lags):.1f}")
        print(f"  Min    : {np.min(mamba_lags)}")
        print(f"  Max    : {np.max(mamba_lags)}")

    if lr_lags:
        print(f"\nLR     detection lag (steps after attack_start):")
        print(f"  Mean   : {np.mean(lr_lags):.1f}")
        print(f"  Median : {np.median(lr_lags):.1f}")
        print(f"  Min    : {np.min(lr_lags)}")
        print(f"  Max    : {np.max(lr_lags)}")

    # cumulative detection rate over time
    max_steps = 100
    steps = list(range(0, max_steps + 1, 5))
    mamba_cum, lr_cum = [], []

    for s in steps:
        mamba_cum.append(sum(1 for l in mamba_lags if l <= s) / n_attacked)
        lr_cum.append(sum(1 for l in lr_lags    if l <= s) / n_attacked)

    print("\n\nCumulative Detection Rate vs Steps After Attack Onset")
    print(f"  {'Steps':>5}  {'Mamba':>7}  {'LR':>7}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}")
    for s, mv, lv in zip(steps, mamba_cum, lr_cum):
        print(f"  {s:>5}  {mv:>6.1%}  {lv:>6.1%}")

    ascii_chart(
        "Detection Rate (# = % detected)",
        steps, mamba_cum, lr_cum, width=30
    )


if __name__ == "__main__":
    main()
