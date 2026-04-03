from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.typing import Adj

# PyG `aggr`: "mean" is GraphSAGE mean; "max" is GraphSAGE pool (max over neighbors).
NEIGHBOR_AGGREGATIONS = ("mean", "max")


class BaselineGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        normalize_output: bool = True,
        neighbor_aggr: str = "mean",
    ):
        super().__init__()
        if neighbor_aggr not in NEIGHBOR_AGGREGATIONS:
            raise ValueError(
                f"neighbor_aggr must be one of {NEIGHBOR_AGGREGATIONS}, got {neighbor_aggr!r}"
            )
        self.convs = nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, aggr=neighbor_aggr)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=neighbor_aggr)
            )
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, aggr=neighbor_aggr)
        )
        self.dropout = dropout
        self.normalize_output = normalize_output

    def forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=-1)
        return x
