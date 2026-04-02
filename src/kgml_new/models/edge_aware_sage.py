from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, SAGEConv
from torch_geometric.typing import Adj, Tensor


class EdgeAwareSAGELayer(MessagePassing):
    def __init__(self, in_channels: int, edge_dim: int, out_channels: int, *, concat: bool = True) -> None:
        super().__init__(aggr="mean")
        self.concat = concat
        msg_in = in_channels + edge_dim if concat else in_channels
        self.lin = nn.Linear(msg_in, out_channels)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        m = torch.cat([x_j, edge_attr], dim=-1) if self.concat else x_j
        return self.lin(m)


class EdgeAwareGraphSAGE(nn.Module):
    """
    Edge-aware GraphSAGE encoder: per-edge message uses relation embedding.
    Produces L2-normalized node embeddings for dot-product link loss.
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        out_channels: int,
        relation_table: Tensor,
        num_layers: int = 2,
        dropout: float = 0.0,
        concat: bool = True,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.concat = concat
        self.normalize_output = normalize_output
        self.relation_table = nn.Parameter(relation_table, requires_grad=False)

        self.layers = nn.ModuleList()
        if num_layers <= 1:
            self.layers.append(EdgeAwareSAGELayer(in_channels, edge_dim, out_channels, concat=concat))
        else:
            self.layers.append(EdgeAwareSAGELayer(in_channels, edge_dim, hidden_channels, concat=concat))
            for _ in range(num_layers - 2):
                self.layers.append(EdgeAwareSAGELayer(hidden_channels, edge_dim, hidden_channels, concat=concat))
            self.layers.append(EdgeAwareSAGELayer(hidden_channels, edge_dim, out_channels, concat=concat))

        self.post = SAGEConv(out_channels, out_channels, aggr="mean", normalize=False)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        rel = self.relation_table[edge_attr]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, rel)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.post(h, edge_index)
        if self.normalize_output:
            h = F.normalize(h, p=2, dim=-1)
        return h
