from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

NEIGHBOR_AGGREGATIONS = ("mean", "max")


class BaselineGraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
        normalize_output: bool = True,
        aggr: str = "mean",
        neighbor_aggr: str | None = None,
    ) -> None:
        super().__init__()
        if neighbor_aggr is not None:
            aggr = neighbor_aggr
        if aggr not in NEIGHBOR_AGGREGATIONS:
            raise ValueError(f"aggr must be one of {NEIGHBOR_AGGREGATIONS}, got {aggr!r}")
        dims = [in_dim] + [hidden_dim] * max(0, num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr=aggr) for i in range(len(dims) - 1)
        )
        self.dropout = float(dropout)
        self.normalize_output = bool(normalize_output)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = torch.relu(x)
                x = torch.dropout(x, p=self.dropout, train=self.training)
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=-1)
        return x
