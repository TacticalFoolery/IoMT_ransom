import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from glob import glob

SEQ_LEN = 10
MAX_TRAIN_SAMPLES = 20000



# CLASS MAPPING
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


# LOAD FILES
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


# CREATE SEQUENCES
def create_sequences(X, y, seq_len):
    sequences = []
    labels = []

    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        labels.append(y[i+seq_len])

    return np.array(sequences, dtype=np.float32), np.array(labels)


# MAIN
def main():
    print("Loading CIC data...")
    path = "data/raw/sim_raw/CICIoMT2024/**/*.csv"

    df = load_files(path)

    # SAMPLE
    print("Sampling...")
    df = df.sample(n=min(len(df), MAX_TRAIN_SAMPLES), random_state=42)

    # BALANCE (UNDERSAMPLING)
    print("Balancing dataset...")
    df_majority = df[df.label == 1]
    df_rest = df[df.label != 1]

    df_majority_downsampled = df_majority.sample(
        n=len(df_rest),
        random_state=42
    )

    df = pd.concat([df_majority_downsampled, df_rest])

    print("Class distribution AFTER balancing:")
    print(df["label"].value_counts())

    # CLEAN
    print("Cleaning...")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # SPLIT FEATURES
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # SCALE
    print("Scaling...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # SEQUENCES
    print("Creating sequences...")
    X_seq, y_seq = create_sequences(X, y, SEQ_LEN)

    # SHUFFLE (AFTER sequences!)
    indices = np.random.permutation(len(X_seq))
    X_seq = X_seq[indices]
    y_seq = y_seq[indices]

    # SPLIT TRAIN/TEST
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # SAVE
    os.makedirs("data/splits/cic_splits", exist_ok=True)

    np.save("data/splits/cic_splits/X_train.npy", X_train)
    np.save("data/splits/cic_splits/y_train.npy", y_train)
    np.save("data/splits/cic_splits/X_test.npy", X_test)
    np.save("data/splits/cic_splits/y_test.npy", y_test)

    print("Done. Dataset is balanced and usable.")


if __name__ == "__main__":
    main()