import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from glob import glob

SEQ_LEN = 10
MAX_TRAIN_SAMPLES = 100000
MAX_TEST_SAMPLES = 20000

# -------- CLASS MAPPING --------
def get_label_from_filename(file):
    if "Benign" in file:
        return 0
    elif "DDoS" in file:
        return 1
    elif "DoS" in file:
        return 2
    elif "Recon" in file:
        return 3
    elif "MQTT" in file:
        return 4
    else:
        return 4


def load_files(pattern):
    files = glob(pattern, recursive=True)

    if len(files) == 0:
        raise ValueError(f"No CSV files found for pattern: {pattern}")

    all_dfs = []

    for file in files:
        print(f"Loading {file}")
        df = pd.read_csv(file)

        label = get_label_from_filename(file)
        df["label"] = label

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

    train_path = "data/raw/sim_raw/CICIoMT2024/**/*.csv"
    test_path  = "data/raw/sim_raw/CICIoMT2024/**/*.csv"

    train_df = load_files(train_path)
    test_df  = train_df.copy() 

    print("Sampling...")

    train_df = train_df.sample(n=min(len(train_df), MAX_TRAIN_SAMPLES), random_state=42)

    print("Splitting...")

    X = train_df.drop(columns=["label"])
    y = train_df["label"].values

    print("Cleaning...")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("Scaling...")

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print("Creating sequences...")

    X_seq, y_seq = create_sequences(X, y, SEQ_LEN)

    split = int(0.8 * len(X_seq))

    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    print("Saving...")

    os.makedirs("data/splits/sim_splits", exist_ok=True)

    np.save("data/splits/sim_splits/X_train.npy", X_train)
    np.save("data/splits/sim_splits/y_train.npy", y_train)
    np.save("data/splits/sim_splits/X_test.npy", X_test)
    np.save("data/splits/sim_splits/y_test.npy", y_test)

    print("Done. Now your models actually have something to learn.")


if __name__ == "__main__":
    main()