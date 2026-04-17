from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Tensor

from kgml_new.models.baseline_sage import NEIGHBOR_AGGREGATIONS

EDGE_RELATION_MODES = ("concat", "gated", "basis_mixture", "film")


class EdgeAwareSAGELayer(MessagePassing):
    """Relation-conditioned neighbor aggregation + root transform (matches SAGEConv self term)."""

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        out_channels: int,
        *,
        concat: bool = True,
        root_weight: bool = True,
        aggr: str = "mean",
    ) -> None:
        if aggr not in NEIGHBOR_AGGREGATIONS:
            raise ValueError(f"aggr must be one of {NEIGHBOR_AGGREGATIONS}, got {aggr!r}")
        super().__init__(aggr=aggr)
        self.concat = concat
        msg_in = in_channels + edge_dim if concat else in_channels
        self.lin = nn.Linear(msg_in, out_channels)
        self.root_weight = root_weight
        self.lin_root = (
            nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.lin_root is not None:
            out = out + self.lin_root(x)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        m = torch.cat([x_j, edge_attr], dim=-1) if self.concat else x_j
        return self.lin(m)


class RelationGatedSAGELayer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        out_channels: int,
        *,
        root_weight: bool = True,
        aggr: str = "mean",
    ) -> None:
        if aggr not in NEIGHBOR_AGGREGATIONS:
            raise ValueError(f"aggr must be one of {NEIGHBOR_AGGREGATIONS}, got {aggr!r}")
        super().__init__(aggr=aggr)
        self.msg_linear = nn.Linear(in_channels, out_channels)
        self.gate_linear = nn.Linear(edge_dim, out_channels)
        self.root_weight = root_weight
        self.lin_root = (
            nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.lin_root is not None:
            out = out + self.lin_root(x)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        base_msg = self.msg_linear(x_j)
        gate = torch.sigmoid(self.gate_linear(edge_attr))
        return base_msg * gate


class RelationBasisMixtureSAGELayer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        out_channels: int,
        *,
        num_bases: int,
        root_weight: bool = True,
        aggr: str = "mean",
    ) -> None:
        if aggr not in NEIGHBOR_AGGREGATIONS:
            raise ValueError(f"aggr must be one of {NEIGHBOR_AGGREGATIONS}, got {aggr!r}")
        super().__init__(aggr=aggr)
        if num_bases <= 0:
            raise ValueError("num_bases must be positive for basis mixture mode")
        self.num_bases = num_bases
        self.mix_linear = nn.Linear(edge_dim, num_bases)
        self.basis_linears = nn.ModuleList(
            nn.Linear(in_channels, out_channels) for _ in range(num_bases)
        )
        self.root_weight = root_weight
        self.lin_root = (
            nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.lin_root is not None:
            out = out + self.lin_root(x)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        alpha = torch.softmax(self.mix_linear(edge_attr), dim=-1)
        weights = torch.stack([b.weight for b in self.basis_linears], dim=0)
        biases = torch.stack([b.bias for b in self.basis_linears], dim=0)
        stacked = torch.einsum("ei,koi->eko", x_j, weights) + biases.unsqueeze(0)
        return (stacked * alpha.unsqueeze(-1)).sum(dim=1)


class RelationFilmSAGELayer(MessagePassing):
    """FiLM: relation produces scale (gamma) and shift (beta) on a base message."""

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        out_channels: int,
        *,
        root_weight: bool = True,
        aggr: str = "mean",
    ) -> None:
        if aggr not in NEIGHBOR_AGGREGATIONS:
            raise ValueError(f"aggr must be one of {NEIGHBOR_AGGREGATIONS}, got {aggr!r}")
        super().__init__(aggr=aggr)
        self.msg_linear = nn.Linear(in_channels, out_channels)
        self.film_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, out_channels * 2),
        )
        self.root_weight = root_weight
        self.lin_root = (
            nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.lin_root is not None:
            out = out + self.lin_root(x)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        base_msg = self.msg_linear(x_j)
        gamma_beta = self.film_mlp(edge_attr)
        gamma_raw, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        return gamma * base_msg + beta


class EdgeAwareGraphSAGE(nn.Module):
    """
    Edge-aware encoder aligned with PyG GraphSAGE: each layer is
    aggregate(neighbor messages; mean or max pool) + W_root @ x (no neighbors -> root term only, like SAGEConv).
    Final hop is relation-aware (same family as the stack), not a relation-blind SAGEConv.
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
        neighbor_aggr: str = "mean",
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
            self.layers.append(
                EdgeAwareSAGELayer(
                    in_channels, edge_dim, out_channels, concat=concat, aggr=neighbor_aggr
                )
            )
        else:
            self.layers.append(
                EdgeAwareSAGELayer(
                    in_channels,
                    edge_dim,
                    hidden_channels,
                    concat=concat,
                    aggr=neighbor_aggr,
                )
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    EdgeAwareSAGELayer(
                        hidden_channels,
                        edge_dim,
                        hidden_channels,
                        concat=concat,
                        aggr=neighbor_aggr,
                    )
                )
            self.layers.append(
                EdgeAwareSAGELayer(
                    hidden_channels,
                    edge_dim,
                    out_channels,
                    concat=concat,
                    aggr=neighbor_aggr,
                )
            )

        self.post = EdgeAwareSAGELayer(
            out_channels, edge_dim, out_channels, concat=concat, aggr=neighbor_aggr
        )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # Project the relation-type table (R x D_in) before indexing edges (E rows).
        # Index-first materializes (E x D_in) and can OOM when E is millions and D_in is large.
        rel = self.relation_projection(self.relation_table)[edge_attr]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, rel)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.post(h, edge_index, rel)
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
        neighbor_aggr: str = "mean",
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
            self.layers.append(
                RelationGatedSAGELayer(
                    in_channels, edge_dim, out_channels, aggr=neighbor_aggr
                )
            )
        else:
            self.layers.append(
                RelationGatedSAGELayer(
                    in_channels, edge_dim, hidden_channels, aggr=neighbor_aggr
                )
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    RelationGatedSAGELayer(
                        hidden_channels, edge_dim, hidden_channels, aggr=neighbor_aggr
                    )
                )
            self.layers.append(
                RelationGatedSAGELayer(
                    hidden_channels, edge_dim, out_channels, aggr=neighbor_aggr
                )
            )

        self.post = RelationGatedSAGELayer(
            out_channels, edge_dim, out_channels, aggr=neighbor_aggr
        )


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
        neighbor_aggr: str = "mean",
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
                RelationBasisMixtureSAGELayer(
                    in_channels,
                    edge_dim,
                    out_channels,
                    num_bases=num_bases,
                    aggr=neighbor_aggr,
                )
            )
        else:
            self.layers.append(
                RelationBasisMixtureSAGELayer(
                    in_channels,
                    edge_dim,
                    hidden_channels,
                    num_bases=num_bases,
                    aggr=neighbor_aggr,
                )
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    RelationBasisMixtureSAGELayer(
                        hidden_channels,
                        edge_dim,
                        hidden_channels,
                        num_bases=num_bases,
                        aggr=neighbor_aggr,
                    )
                )
            self.layers.append(
                RelationBasisMixtureSAGELayer(
                    hidden_channels,
                    edge_dim,
                    out_channels,
                    num_bases=num_bases,
                    aggr=neighbor_aggr,
                )
            )

        self.post = RelationBasisMixtureSAGELayer(
            out_channels,
            edge_dim,
            out_channels,
            num_bases=num_bases,
            aggr=neighbor_aggr,
        )


class RelationFilmGraphSAGE(EdgeAwareGraphSAGE):
    edge_relation_mode = "film"

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
        neighbor_aggr: str = "mean",
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
            self.layers.append(
                RelationFilmSAGELayer(
                    in_channels, edge_dim, out_channels, aggr=neighbor_aggr
                )
            )
        else:
            self.layers.append(
                RelationFilmSAGELayer(
                    in_channels, edge_dim, hidden_channels, aggr=neighbor_aggr
                )
            )
            for _ in range(num_layers - 2):
                self.layers.append(
                    RelationFilmSAGELayer(
                        hidden_channels, edge_dim, hidden_channels, aggr=neighbor_aggr
                    )
                )
            self.layers.append(
                RelationFilmSAGELayer(
                    hidden_channels, edge_dim, out_channels, aggr=neighbor_aggr
                )
            )

        self.post = RelationFilmSAGELayer(
            out_channels, edge_dim, out_channels, aggr=neighbor_aggr
        )


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
    neighbor_aggr: str = "mean",
    film_semantic_inject: bool = False,
    rel_residual_scale: float = 0.25,
) -> nn.Module:
    if neighbor_aggr not in NEIGHBOR_AGGREGATIONS:
        raise ValueError(
            f"neighbor_aggr must be one of {NEIGHBOR_AGGREGATIONS}, got {neighbor_aggr!r}"
        )
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
            neighbor_aggr=neighbor_aggr,
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
            neighbor_aggr=neighbor_aggr,
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
            neighbor_aggr=neighbor_aggr,
        )
    if edge_relation_mode == "film":
        if film_semantic_inject:
            from kgml_new.models.semantic_relation_inject import (
                RelationFilmGraphSAGEAdapted,
            )

            return RelationFilmGraphSAGEAdapted(
                in_channels,
                edge_dim,
                hidden_channels,
                out_channels,
                relation_table=relation_table,
                num_layers=num_layers,
                dropout=dropout,
                normalize_output=normalize_output,
                neighbor_aggr=neighbor_aggr,
                rel_residual_scale=rel_residual_scale,
            )
        return RelationFilmGraphSAGE(
            in_channels,
            edge_dim,
            hidden_channels,
            out_channels,
            relation_table=relation_table,
            num_layers=num_layers,
            dropout=dropout,
            normalize_output=normalize_output,
            neighbor_aggr=neighbor_aggr,
        )
    raise ValueError(
        f"Unsupported edge relation mode '{edge_relation_mode}'. Expected one of {EDGE_RELATION_MODES}."
    )


def mixture_weights_from_relation_indices(
    model: nn.Module,
    relation_indices: Tensor,
) -> Tensor:
    """
    Softmax mixture weights for semantic alignment: uses the last
    RelationBasisMixtureSAGELayer in the model (typically the relation-aware post layer).
    """
    relation_indices = relation_indices.long()
    if relation_indices.numel() == 0:
        return relation_indices.new_empty(0).float().reshape(0, 1)

    rt = model.relation_table
    relation_indices = relation_indices.to(rt.device)
    rel_feat = model.relation_projection(rt)[relation_indices]

    basis_layers = [
        m
        for m in model.modules()
        if isinstance(m, RelationBasisMixtureSAGELayer)
    ]
    if not basis_layers:
        raise TypeError(
            "mixture_weights_from_relation_indices requires RelationBasisMixtureGraphSAGE "
            "(a model containing RelationBasisMixtureSAGELayer)."
        )
    basis_layer = basis_layers[-1]
    return F.softmax(basis_layer.mix_linear(rel_feat), dim=-1)
