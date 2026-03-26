import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    Pure-PyTorch Mamba block (no mamba_ssm / CUDA kernels required).
    Implements the selective SSM with a causal conv1d + linear recurrence.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner)"""
        B, L, D = x.shape
        d_state = self.d_state

        x_dbl = self.x_proj(x)                          # (B, L, d_state*2 + d_inner)
        delta = x_dbl[..., :D]                          # (B, L, d_inner)
        B_mat = x_dbl[..., D : D + d_state]             # (B, L, d_state)
        C_mat = x_dbl[..., D + d_state :]               # (B, L, d_state)

        delta = F.softplus(self.dt_proj(delta))         # (B, L, d_inner)
        A = -torch.exp(self.A_log)                      # (d_inner, d_state)

        dA = torch.exp(delta.unsqueeze(-1) * A)         # (B, L, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)   # (B, L, d_inner, d_state)

        h = torch.zeros(B, D, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)
            y = (h * C_mat[:, i].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)                      # (B, L, d_inner)
        return y + x * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)              # each (B, L, d_inner)

        x_branch = x_branch.transpose(1, 2)            # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[..., : x_branch.shape[-1]]
        x_branch = x_branch.transpose(1, 2)            # (B, L, d_inner)
        x_branch = F.silu(x_branch)

        y = self._ssm(x_branch)
        y = y * F.silu(z)

        return self.out_proj(y) + residual


class MambaClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, seq_len, input_dim)"""
        x = self.input_projection(x)

        for layer in self.mamba_layers:
            x = layer(x)

        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x).squeeze(1)
