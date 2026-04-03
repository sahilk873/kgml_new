from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, SAGEConv
from torch_geometric.typing import Adj, Tensor

EDGE_RELATION_MODES = ("concat", "gated", "basis_mixture")


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


class RelationGatedSAGELayer(MessagePassing):
    def __init__(self, in_channels: int, edge_dim: int, out_channels: int) -> None:
        super().__init__(aggr="mean")
        self.msg_linear = nn.Linear(in_channels, out_channels)
        self.gate_linear = nn.Linear(edge_dim, out_channels)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        base_msg = self.msg_linear(x_j)
        gate = torch.sigmoid(self.gate_linear(edge_attr))
        return base_msg * gate


class RelationBasisMixtureSAGELayer(MessagePassing):
    def __init__(self, in_channels: int, edge_dim: int, out_channels: int, *, num_bases: int) -> None:
        super().__init__(aggr="mean")
        if num_bases <= 0:
            raise ValueError("num_bases must be positive for basis mixture mode")
        self.num_bases = num_bases
        self.mix_linear = nn.Linear(edge_dim, num_bases)
        self.basis_linears = nn.ModuleList(
            nn.Linear(in_channels, out_channels) for _ in range(num_bases)
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        alpha = torch.softmax(self.mix_linear(edge_attr), dim=-1)
        basis_msgs = torch.stack([basis(x_j) for basis in self.basis_linears], dim=-1)
        return (basis_msgs * alpha.unsqueeze(1)).sum(dim=-1)


class EdgeAwareGraphSAGE(nn.Module):
    """
    Edge-aware GraphSAGE encoder: per-edge message uses relation embedding.
    Produces L2-normalized node embeddings for dot-product link loss.
    """

    edge_relation_mode = "concat"

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
        relation_input_dim = int(relation_table.size(-1))
        self.relation_input_dim = relation_input_dim
        self.edge_dim = edge_dim
        self.relation_projection = (
            nn.Linear(relation_input_dim, edge_dim, bias=False)
            if relation_input_dim != edge_dim
            else nn.Identity()
        )

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
        rel = self.relation_projection(self.relation_table[edge_attr])
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


class RelationGatedGraphSAGE(EdgeAwareGraphSAGE):
    edge_relation_mode = "gated"

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        out_channels: int,
        relation_table: Tensor,
        num_layers: int = 2,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.dropout = dropout
        self.concat = False
        self.normalize_output = normalize_output
        self.relation_table = nn.Parameter(relation_table, requires_grad=False)
        relation_input_dim = int(relation_table.size(-1))
        self.relation_input_dim = relation_input_dim
        self.edge_dim = edge_dim
        self.relation_projection = (
            nn.Linear(relation_input_dim, edge_dim, bias=False)
            if relation_input_dim != edge_dim
            else nn.Identity()
        )

        self.layers = nn.ModuleList()
        if num_layers <= 1:
            self.layers.append(RelationGatedSAGELayer(in_channels, edge_dim, out_channels))
        else:
            self.layers.append(RelationGatedSAGELayer(in_channels, edge_dim, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(RelationGatedSAGELayer(hidden_channels, edge_dim, hidden_channels))
            self.layers.append(RelationGatedSAGELayer(hidden_channels, edge_dim, out_channels))

        self.post = SAGEConv(out_channels, out_channels, aggr="mean", normalize=False)


class RelationBasisMixtureGraphSAGE(EdgeAwareGraphSAGE):
    edge_relation_mode = "basis_mixture"

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        out_channels: int,
        relation_table: Tensor,
        *,
        num_bases: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        normalize_output: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.dropout = dropout
        self.concat = False
        self.normalize_output = normalize_output
        self.relation_table = nn.Parameter(relation_table, requires_grad=False)
        relation_input_dim = int(relation_table.size(-1))
        self.relation_input_dim = relation_input_dim
        self.edge_dim = edge_dim
        self.num_bases = num_bases
        self.relation_projection = (
            nn.Linear(relation_input_dim, edge_dim, bias=False)
            if relation_input_dim != edge_dim
            else nn.Identity()
        )

        self.layers = nn.ModuleList()
        if num_layers <= 1:
            self.layers.append(
                RelationBasisMixtureSAGELayer(in_channels, edge_dim, out_channels, num_bases=num_bases)
            )
        else:
            self.layers.append(
                RelationBasisMixtureSAGELayer(in_channels, edge_dim, hidden_channels, num_bases=num_bases)
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    RelationBasisMixtureSAGELayer(
                        hidden_channels, edge_dim, hidden_channels, num_bases=num_bases
                    )
                )
            self.layers.append(
                RelationBasisMixtureSAGELayer(hidden_channels, edge_dim, out_channels, num_bases=num_bases)
            )

        self.post = SAGEConv(out_channels, out_channels, aggr="mean", normalize=False)


def build_edge_aware_model(
    *,
    edge_relation_mode: str,
    in_channels: int,
    edge_dim: int,
    hidden_channels: int,
    out_channels: int,
    relation_table: Tensor,
    num_layers: int = 2,
    dropout: float = 0.0,
    concat: bool = True,
    normalize_output: bool = True,
    num_relation_bases: int = 4,
) -> nn.Module:
    if edge_relation_mode == "concat":
        return EdgeAwareGraphSAGE(
            in_channels,
            edge_dim,
            hidden_channels,
            out_channels,
            relation_table=relation_table,
            num_layers=num_layers,
            dropout=dropout,
            concat=concat,
            normalize_output=normalize_output,
        )
    if edge_relation_mode == "gated":
        return RelationGatedGraphSAGE(
            in_channels,
            edge_dim,
            hidden_channels,
            out_channels,
            relation_table=relation_table,
            num_layers=num_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
    if edge_relation_mode == "basis_mixture":
        return RelationBasisMixtureGraphSAGE(
            in_channels,
            edge_dim,
            hidden_channels,
            out_channels,
            relation_table=relation_table,
            num_bases=num_relation_bases,
            num_layers=num_layers,
            dropout=dropout,
            normalize_output=normalize_output,
        )
    raise ValueError(
        f"Unsupported edge relation mode '{edge_relation_mode}'. Expected one of {EDGE_RELATION_MODES}."
    )
