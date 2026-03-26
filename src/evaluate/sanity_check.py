"""
Sanity check: compare Mamba (all seq_len rows) against a logistic regression
that sees only the LAST row of each sequence.

Both models are evaluated on the SAME sequences with the SAME labels,
giving a true apples-to-apples comparison of whether sequence context helps.

If LR (last row only) matches Mamba → sequence context adds nothing.
If Mamba beats LR               → temporal modelling is contributing.
"""

import os
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from src.config import Config
from src.utils import set_seed, get_device
from src.datasets.sequence_dataset import ArraySequenceDataset
from src.models.autoencoder import Autoencoder


def extract_latent_and_error(model, X, device, batch_size=256):
    model.eval()
    combined = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            z, x_hat = model(batch)
            recon_error = torch.mean((batch - x_hat) ** 2, dim=1, keepdim=True)
            combined.append(torch.cat([z, recon_error], dim=1).cpu().numpy())
    return np.vstack(combined)


def print_metrics(label, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'='*40}")
    print(f" {label}")
    print(f"{'='*40}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  :  {tn:>6}   {fp:>6}")
    print(f"  Actual 1  :  {fn:>6}   {tp:>6}")
    print(f"{'='*40}\n")


def main(dataset_name="ton"):
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device()

    if dataset_name.lower() == "ton":
        split_dir  = cfg.ton_splits_path
        label_mode = "last"
    elif dataset_name.lower() == "sim":
        split_dir  = cfg.sim_splits_path
        label_mode = "any"
    else:
        raise ValueError("dataset_name must be 'ton' or 'sim'")

    print(f"Running sanity check on: {dataset_name}  (label_mode={label_mode}, seq_len={cfg.seq_len})")

    X_train         = np.load(os.path.join(split_dir, "X_train.npy"))
    X_test          = np.load(os.path.join(split_dir, "X_test.npy"))
    y_train         = np.load(os.path.join(split_dir, "y_train.npy"))
    y_test          = np.load(os.path.join(split_dir, "y_test.npy"))
    group_ids_train = np.load(os.path.join(split_dir, "group_ids_train.npy"), allow_pickle=True)
    group_ids_test  = np.load(os.path.join(split_dir, "group_ids_test.npy"),  allow_pickle=True)

    ae = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim1=cfg.ae_hidden_dim1,
        hidden_dim2=cfg.ae_hidden_dim2,
        latent_dim=cfg.latent_dim,
    ).to(device)
    ae_path = cfg.ton_autoencoder_model_path if dataset_name.lower() == "ton" else cfg.sim_autoencoder_model_path
    ae.load_state_dict(torch.load(ae_path, map_location=device))

    print("Encoding features...")
    Z_train = extract_latent_and_error(ae, X_train, device)
    Z_test  = extract_latent_and_error(ae, X_test,  device)

    # Build the same sequences used by Mamba
    train_dataset = ArraySequenceDataset(
        features=Z_train, labels=y_train,
        group_ids=group_ids_train, seq_len=cfg.seq_len, label_mode=label_mode,
    )
    test_dataset = ArraySequenceDataset(
        features=Z_test, labels=y_test,
        group_ids=group_ids_test,  seq_len=cfg.seq_len, label_mode=label_mode,
    )

    # Extract last-row features and sequence labels from each sequence
    # Shape: (N, seq_len, features) ; take [:, -1, :] for last row
    X_train_last = train_dataset.samples[:, -1, :].numpy()   # (N_train, F)
    X_test_last  = test_dataset.samples[:,  -1, :].numpy()   # (N_test,  F)
    y_train_seq  = train_dataset.labels.numpy()
    y_test_seq   = test_dataset.labels.numpy()

    print(f"Sequences — train: {len(X_train_last)}, test: {len(X_test_last)}")
    print("Fitting logistic regression on last-row latent features only...")

    lr = LogisticRegression(max_iter=1000, random_state=cfg.random_seed)
    lr.fit(X_train_last, y_train_seq)

    y_pred = lr.predict(X_test_last)
    y_prob = lr.predict_proba(X_test_last)[:, 1]

    print_metrics(f"LR — last row only (no sequence context)", y_test_seq, y_pred, y_prob)

    auc = roc_auc_score(y_test_seq, y_prob)
    print("Interpretation:")
    if auc >= 0.999:
        print("  -> LR matches Mamba perfectly. Sequence context adds nothing.")
    elif auc >= 0.90:
        f1_lr = f1_score(y_test_seq, y_pred, zero_division=0)
        print(f"  -> Strong last-row baseline (AUC={auc:.4f}, F1={f1_lr:.4f}).")
        print("     Compare directly to Mamba's F1/AUC — any gap is the value of sequence context.")
    else:
        print("  -> Last-row baseline is weak. Sequence context is clearly contributing.")


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "ton"
    main(dataset)
