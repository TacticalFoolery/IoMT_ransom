import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import *

def load_data():
    path = "data/raw/cic_raw/cic_iomt2024.csv"
    df = pd.read_csv(path)
    return df

def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # Drop non-numeric columns if needed
    df = df.select_dtypes(include=[np.number])

    return df

def split_features_labels(df):
    # Adjust label column name depending on dataset
    label_col = "label"  # might be "Label" or "Attack"
    
    y = df[label_col].values
    X = df.drop(columns=[label_col]).values
    
    return X, y

def create_sequences(X, y, seq_len):
    sequences = []
    labels = []

    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        labels.append(y[i+seq_len])

    return np.array(sequences), np.array(labels)

def main():
    df = load_data()
    df = clean_data(df)

    X, y = split_features_labels(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    np.save("data/processed/cic_scaler.npy", scaler.mean_)  # optional, or joblib

    X_seq, y_seq = create_sequences(X, y, seq_len)

    split = int(0.8 * len(X_seq))

    np.save("data/splits/cic_X_train.npy", X_seq[:split])
    np.save("data/splits/cic_y_train.npy", y_seq[:split])
    np.save("data/splits/cic_X_test.npy", X_seq[split:])
    np.save("data/splits/cic_y_test.npy", y_seq[split:])

if __name__ == "__main__":
    main()