"""Two-FC privacy projector + bias-free identity weight layer (paper Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivacyProjector(nn.Module):
    def __init__(self, in_dim: int = 512, private_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, private_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(private_dim, private_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class NormSoftmax(nn.Module):
    """Normalized softmax / ArcFace with margin m=0, scale s (paper Lr)."""

    def __init__(self, in_dim: int, n_classes: int, s: float = 64.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, p=2, dim=1)
        return self.s * (x @ w.t())
