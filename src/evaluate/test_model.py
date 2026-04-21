import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.models.lstm_classifier import LSTMClassifier
from src.models.mamba_classifier import MambaClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
    print("Loading preprocessed data...")

    X_test = np.load("data/splits/sim_splits/X_test.npy")
    y_test = np.load("data/splits/sim_splits/y_test.npy")

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_test, y_test


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)

            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    acc = (preds == labels).mean()

    print("\nAccuracy:", acc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))

    print("\nClassification Report:")
    print(classification_report(labels, preds))

    return acc


def main():
    print("Loading dataset...")

    X_test, y_test = load_dataset()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=256,
        shuffle=False
    )

    input_dim = X_test.shape[-1]
    num_classes = 5   

    print("\nLoading LSTM...")
    lstm = LSTMClassifier(
        input_dim=input_dim,
        num_classes=num_classes
    ).to(DEVICE)

    lstm.load_state_dict(
        torch.load("models/lstm_classifier_sim.pt", map_location=DEVICE)
    )

    print("Evaluating LSTM...")
    evaluate(lstm, test_loader)

    print("\nLoading Mamba...")
    mamba = MambaClassifier(
        input_dim=input_dim,
        num_classes=num_classes
    ).to(DEVICE)

    mamba.load_state_dict(
        torch.load("models/mamba_classifier_sim.pt", map_location=DEVICE)
    )

    print("Evaluating Mamba...")
    evaluate(mamba, test_loader)


if __name__ == "__main__":
    main()