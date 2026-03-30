from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinkPredictionMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = (in_channels * 2,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        x = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(x).squeeze(-1)
