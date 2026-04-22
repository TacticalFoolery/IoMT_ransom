import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mamba_classifier import MambaClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(dataset="sim"):
    print(f"Training Mamba ({dataset})")

    split_dir = f"data/splits/{dataset}_splits/"

    # =========================
    # LOAD DATA
    # =========================
    X_train = np.load(os.path.join(split_dir, "X_train.npy"))
    y_train = np.load(os.path.join(split_dir, "y_train.npy"))

    print("Train shape:", X_train.shape)
    print("Class distribution:", np.bincount(y_train))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=1024,   # 🔥 FAST
        shuffle=True
    )

    input_dim = X_train.shape[-1]
    num_classes = 5

    model = MambaClassifier(
        input_dim=input_dim,
        num_classes=num_classes
    ).to(DEVICE)

    # =========================
    # CLASS WEIGHTS
    # =========================
    class_counts = np.bincount(y_train.numpy())
    class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()

    print("Class weights:", class_weights)

    weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # =========================
    # FAST TRAINING
    # =========================
    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if X_batch.dim() == 2:
                X_batch = X_batch.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(X_batch)

            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/mamba_classifier_{dataset}.pt")

    print("Saved Mamba")


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "sim"
    main(dataset)