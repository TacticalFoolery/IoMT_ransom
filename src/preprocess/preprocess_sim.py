import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.config import Config


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    cfg = Config()

    ensure_dir(cfg.sim_splits_path)
    ensure_dir(cfg.processed_icu_path)

    data_path = os.path.join(cfg.raw_icu_path, "icu_simulation.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Simulation data not found at {data_path}. "
            "Run simulate_icu.py first."
        )

    df = pd.read_csv(data_path)
    print("Loaded simulation data:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())

    # Sort each device's rows by timestamp to preserve temporal order
    df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    group_ids = df["device_id"].values
    y = df["label"].values

    drop_cols = ["label", "timestamp", "device_id", "device_type", "attack_variant"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_df = X_df.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    # Group-aware split: entire device timelines go to train or test.
    # This prevents the model from seeing future attack patterns during training.
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_seed)
    train_idx, test_idx = next(gss.split(X_df, y, groups=group_ids))

    X_train_df      = X_df.iloc[train_idx]
    X_test_df       = X_df.iloc[test_idx]
    y_train         = y[train_idx]
    y_test          = y[test_idx]
    group_ids_train = group_ids[train_idx]
    group_ids_test  = group_ids[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)

    joblib.dump(scaler, os.path.join(cfg.processed_icu_path, "scaler.pkl"))

    np.save(os.path.join(cfg.sim_splits_path, "X_train.npy"),        X_train)
    np.save(os.path.join(cfg.sim_splits_path, "X_test.npy"),         X_test)
    np.save(os.path.join(cfg.sim_splits_path, "y_train.npy"),        y_train)
    np.save(os.path.join(cfg.sim_splits_path, "y_test.npy"),         y_test)
    np.save(os.path.join(cfg.sim_splits_path, "group_ids_train.npy"), np.array(group_ids_train, dtype=object))
    np.save(os.path.join(cfg.sim_splits_path, "group_ids_test.npy"),  np.array(group_ids_test,  dtype=object))

    print("Preprocessing complete.")
    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Train labels — 0:", int((y_train == 0).sum()), "| 1:", int((y_train == 1).sum()))
    print("Test  labels — 0:", int((y_test  == 0).sum()), "| 1:", int((y_test  == 1).sum()))


if __name__ == "__main__":
    main()
