import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mamba_classifier import MambaClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(dataset="sim"):
    split_dir = "data/splits/sim_splits/"

    X_train = np.load(os.path.join(split_dir, "X_train.npy"))
    y_train = np.load(os.path.join(split_dir, "y_train.npy"))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    input_dim = X_train.shape[-1]
    num_classes = 5

    model = MambaClassifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        total_loss = 0
        model.train()

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/mamba_classifier_sim.pt")


if __name__ == "__main__":
    main()