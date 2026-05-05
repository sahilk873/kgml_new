from __future__ import annotations

import torch
from torch import Tensor, nn


class RelationFeatureMixin:
    relation_features: Tensor

    def _project_relations(self, relation_ids: Tensor) -> Tensor:
        return self.relation_projection(self.relation_features[relation_ids.long()])


class DistMultTripleDecoder(nn.Module, RelationFeatureMixin):
    def __init__(self, relation_features: Tensor, embedding_dim: int) -> None:
        super().__init__()
        self.register_buffer("relation_features", relation_features.detach().float().cpu())
        rel_dim = int(relation_features.size(-1))
        self.relation_projection = (
            nn.Linear(rel_dim, embedding_dim, bias=False)
            if rel_dim != int(embedding_dim)
            else nn.Identity()
        )

    def forward(self, z: Tensor, triples: Tensor) -> Tensor:
        h = z[triples[:, 0].long()]
        r = self._project_relations(triples[:, 1].long())
        t = z[triples[:, 2].long()]
        return (h * r * t).sum(dim=-1)


class MLPTripleDecoder(nn.Module, RelationFeatureMixin):
    def __init__(
        self,
        relation_features: Tensor,
        embedding_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("relation_features", relation_features.detach().float().cpu())
        rel_dim = int(relation_features.size(-1))
        self.relation_projection = (
            nn.Linear(rel_dim, embedding_dim, bias=False)
            if rel_dim != int(embedding_dim)
            else nn.Identity()
        )
        dims = [3 * int(embedding_dim), *list(hidden_dims), 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.ReLU(), nn.Dropout(float(dropout))])
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor, triples: Tensor) -> Tensor:
        h = z[triples[:, 0].long()]
        r = self._project_relations(triples[:, 1].long())
        t = z[triples[:, 2].long()]
        return self.net(torch.cat([h, r, t], dim=-1)).squeeze(-1)


class RelationAgnosticDotTripleDecoder(nn.Module):
    def forward(self, z: Tensor, triples: Tensor) -> Tensor:
        h = z[triples[:, 0].long()]
        t = z[triples[:, 2].long()]
        return (h * t).sum(dim=-1)


def build_triple_decoder(
    name: str,
    *,
    relation_features: Tensor,
    embedding_dim: int,
    hidden_dims: tuple[int, ...] = (256, 128),
    dropout: float = 0.1,
) -> nn.Module:
    if name == "distmult":
        return DistMultTripleDecoder(relation_features, embedding_dim)
    if name == "triple_mlp":
        return MLPTripleDecoder(
            relation_features,
            embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if name == "dot":
        return RelationAgnosticDotTripleDecoder()
    raise ValueError(f"Unsupported triple decoder: {name}")
