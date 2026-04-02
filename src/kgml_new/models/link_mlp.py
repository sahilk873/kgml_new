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
        feature_mode: str = "concat_product",
    ):
        super().__init__()
        if feature_mode not in {"concat", "concat_product"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        self.feature_mode = feature_mode
        input_dim = in_channels * 2 if feature_mode == "concat" else in_channels * 3
        dims = (input_dim,) + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        if self.feature_mode == "concat_product":
            x = torch.cat([z_src, z_dst, z_src * z_dst], dim=-1)
        else:
            x = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(x).squeeze(-1)
