import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.models.lstm_classifier import LSTMClassifier
from src.models.mamba_classifier import MambaClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    X_test = np.load("data/splits/sim_splits/X_test.npy")
    y_test = np.load("data/splits/sim_splits/y_test.npy")
    return X_test, y_test


def evaluate(model, loader):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(predicted)
            labels.extend(y_batch.numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    print("\nAccuracy:", accuracy_score(labels, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    print("\nClassification Report:")
    print(classification_report(labels, preds))


def main():
    X_test, y_test = load_data()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    input_dim = X_test.shape[-1]

    lstm = LSTMClassifier(input_dim=input_dim, num_classes=5).to(DEVICE)
    lstm.load_state_dict(torch.load("models/lstm_classifier_sim.pt", map_location=DEVICE))

    print("LSTM:")
    evaluate(lstm, loader)

    mamba = MambaClassifier(input_dim=input_dim, num_classes=5).to(DEVICE)
    mamba.load_state_dict(torch.load("models/mamba_classifier_sim.pt", map_location=DEVICE))

    print("\nMamba:")
    evaluate(mamba, loader)


if __name__ == "__main__":
    main()