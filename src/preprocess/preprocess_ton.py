import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import Config
from pathlib import Path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_conn_csvs(folder_path, label):
    all_dfs = []

    folder = Path(folder_path)
    conn_files = list(folder.rglob("conn.csv"))

    print(f"Searching in: {folder}")
    print(f"Found {len(conn_files)} conn.csv files")

    for file_path in conn_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df["label"] = label
            all_dfs.append(df)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Skipped {file_path}: {e}")

    return all_dfs

def main():
    cfg = Config()

    base_path = os.path.join(cfg.raw_ton_iot_path, "Network_dataset_Bro")

    normal_path = os.path.join(base_path, "normal_Bro")
    attack_path = os.path.join(base_path, "normal_attack_Bro")

    ensure_dir(cfg.ton_splits_path)
    ensure_dir(cfg.processed_ton_iot_path)

    print("Loading NORMAL data...")
    normal_dfs = load_conn_csvs(normal_path, label=0)

    print("Loading ATTACK data...")
    attack_dfs = load_conn_csvs(attack_path, label=1)

    print("Number of normal dataframes loaded:", len(normal_dfs))
    print("Number of attack dataframes loaded:", len(attack_dfs))

    if len(normal_dfs) == 0:
        raise ValueError("No normal conn.csv files were loaded.")
    if len(attack_dfs) == 0:
        raise ValueError("No attack conn.csv files were loaded.")

    normal_df = pd.concat(normal_dfs, ignore_index=True)
    attack_df = pd.concat(attack_dfs, ignore_index=True)

    print("Normal shape before sampling:", normal_df.shape)
    print("Attack shape before sampling:", attack_df.shape)

    max_rows = 100000
    half_rows = max_rows // 2

    if len(normal_df) > half_rows:
        normal_df = normal_df.sample(n=half_rows, random_state=cfg.random_seed)

    if len(attack_df) > half_rows:
        attack_df = attack_df.sample(n=half_rows, random_state=cfg.random_seed)

    df = pd.concat([normal_df, attack_df], ignore_index=True)
    df = df.reset_index(drop=True)

    print("Combined shape after sampling:", df.shape)
    # df = df.drop_duplicates().reset_index(drop=True)

    # Group IDs
    if "id.orig_h" in df.columns and "id.resp_h" in df.columns:
        group_ids = df["id.orig_h"].astype(str) + "_" + df["id.resp_h"].astype(str)
    elif "src_ip" in df.columns and "dst_ip" in df.columns:
        group_ids = df["src_ip"].astype(str) + "_" + df["dst_ip"].astype(str)
    else:
        group_ids = np.array([f"row_{i}" for i in range(len(df))], dtype=object)

    y = df["label"].values

    drop_cols = [
        "label", "ts", "timestamp",
        "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p",
        "src_ip", "dst_ip", "uid"
    ]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    categorical_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    for col in numeric_cols:
        X_df[col] = X_df[col].fillna(0)

    for col in categorical_cols:
        X_df[col] = X_df[col].fillna("unknown").astype(str)
        freq_map = X_df[col].value_counts(normalize=True)
        X_df[col] = X_df[col].map(freq_map).fillna(0)

    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train_df, X_test_df, y_train, y_test, group_ids_train, group_ids_test = train_test_split(
        X_df,
        y,
        group_ids,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    joblib.dump(scaler, os.path.join(cfg.processed_ton_iot_path, "scaler.pkl"))

    np.save(os.path.join(cfg.ton_splits_path, "X_train.npy"), X_train)
    np.save(os.path.join(cfg.ton_splits_path, "X_test.npy"), X_test)
    np.save(os.path.join(cfg.ton_splits_path, "y_train.npy"), y_train)
    np.save(os.path.join(cfg.ton_splits_path, "y_test.npy"), y_test)
    np.save(os.path.join(cfg.ton_splits_path, "group_ids_train.npy"), np.array(group_ids_train, dtype=object))
    np.save(os.path.join(cfg.ton_splits_path, "group_ids_test.npy"), np.array(group_ids_test, dtype=object))

    print("Preprocessing complete.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


if __name__ == "__main__":
    main()