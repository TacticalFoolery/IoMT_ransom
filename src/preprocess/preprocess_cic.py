import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from glob import glob

SEQ_LEN = 10
MAX_TRAIN_SAMPLES = 100000
MAX_TEST_SAMPLES = 20000


def get_label_from_filename(file):
    file = file.lower()

    if "benign" in file:
        return 0
    elif "mqtt" in file:
        return 1
    elif "tcp_ip" in file:
        return 2
    elif "recon" in file:
        return 3
    else:
        return 4


def load_files(paths):
    all_dfs = []

    for file in paths:
        print(f"Loading {file}")
        df = pd.read_csv(file)

        df["label"] = get_label_from_filename(file)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def create_sequences(X, y, seq_len):
    sequences = []
    labels = []

    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        labels.append(y[i+seq_len])

    return np.array(sequences, dtype=np.float32), np.array(labels)


def main():
    print("Loading data...")

    train_files = glob("data/raw/sim_raw/CICIoMT2024/**/train/*.csv", recursive=True)
    test_files  = glob("data/raw/sim_raw/CICIoMT2024/**/test/*.csv", recursive=True)

    train_df = load_files(train_files)
    test_df  = load_files(test_files)

    print("Sampling...")

    train_df = train_df.sample(n=min(len(train_df), MAX_TRAIN_SAMPLES), random_state=42)
    test_df  = test_df.sample(n=min(len(test_df), MAX_TEST_SAMPLES), random_state=42)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"].values

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].values

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
    X_test_seq, y_test_seq   = create_sequences(X_test, y_test, SEQ_LEN)

    os.makedirs("data/splits/sim_splits", exist_ok=True)

    np.save("data/splits/sim_splits/X_train.npy", X_train_seq)
    np.save("data/splits/sim_splits/y_train.npy", y_train_seq)
    np.save("data/splits/sim_splits/X_test.npy", X_test_seq)
    np.save("data/splits/sim_splits/y_test.npy", y_test_seq)

    print("Done.")


if __name__ == "__main__":
    main()