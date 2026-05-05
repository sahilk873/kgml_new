from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    requires_edge_type = True

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_relations: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * max(0, num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            RGCNConv(dims[i], dims[i + 1], num_relations=num_relations)
            for i in range(len(dims) - 1)
        )
        self.dropout = float(dropout)
        self.normalize_output = bool(normalize_output)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=-1)
        return x
