import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MambaClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Stack multiple Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_dim)
        """

        x = self.input_projection(x)

        for layer in self.mamba_layers:
            x = layer(x)

        # Mean pooling over sequence
        x = x.mean(dim=1)

        x = self.dropout(x)

        logits = self.classifier(x).squeeze(1)

        return logits