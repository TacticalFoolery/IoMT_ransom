import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize

from src.models.lstm_classifier import LSTMClassifier
from src.models.mamba_classifier import MambaClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 DEFINE ALL YOUR CLASSES HERE (EDIT THIS)
# Example: 0=Benign, 1=DDoS, 2=DoS, 3=Ransomware, 4=Other
ALL_CLASSES = np.array([0, 1, 2, 3, 4])


def load_data():
    print("Loading preprocessed data...")

    X_test = np.load("data/splits/sim_splits/X_test.npy")
    y_test = np.load("data/splits/sim_splits/y_test.npy")

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_test, y_test


def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # ===== BASIC METRICS =====
    acc = (preds == labels).mean()

    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    print("\n===== METRICS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall    (macro): {recall_macro:.4f}")
    print(f"F1-score  (macro): {f1_macro:.4f}")

    # ===== DEBUG CLASS COVERAGE =====
    print("\nClasses in test set:", np.unique(labels))
    print("All expected classes:", ALL_CLASSES)

    # ===== CONFUSION MATRIX =====
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds, labels=ALL_CLASSES))

    # ===== CLASSIFICATION REPORT =====
    print("\nClassification Report:")
    print(classification_report(
        labels,
        preds,
        labels=ALL_CLASSES,
        zero_division=0
    ))

    # ===== AUC-ROC =====
    try:
        labels_bin = label_binarize(labels, classes=ALL_CLASSES)
        auc = roc_auc_score(labels_bin, probs, multi_class="ovr")
        print("\nAUC-ROC:", auc)
    except Exception as e:
        print("\nAUC-ROC failed:", e)


def main():
    X_test, y_test = load_data()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=256,
        shuffle=False
    )

    input_dim = X_test.shape[-1]
    num_classes = len(ALL_CLASSES)

    # ===== LSTM =====
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

    # ===== MAMBA =====
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