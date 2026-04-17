"""FiLM GraphSAGE with trainable relation residual + DistMult-style decode-time relation scoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.typing import Adj

from kgml_new.models.edge_aware_sage import RelationFilmGraphSAGE


class RelationFilmGraphSAGEAdapted(RelationFilmGraphSAGE):
    """
    Same as RelationFilmGraphSAGE, but relation rows are ``r + s * MLP(r)`` (MLP trainable,
    base table frozen) so task-specific LP gradients can reshape injected semantics.
    """

    def __init__(self, *args, rel_residual_scale: float = 0.25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d = int(self.relation_input_dim)
        self.rel_residual = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, d),
        )
        nn.init.zeros_(self.rel_residual[-1].weight)
        nn.init.zeros_(self.rel_residual[-1].bias)
        self._rel_scale = float(rel_residual_scale)

    def adapted_relation_table(self) -> Tensor:
        b = self.relation_table
        return b + self._rel_scale * self.rel_residual(b)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        adapted = self.adapted_relation_table()
        rel = self.relation_projection(adapted)[edge_attr]
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


class RelDistMultDecodeHead(nn.Module):
    """Decode-time DistMult-style score: sum(h_u ⊙ W(r) ⊙ h_v)."""

    def __init__(self, relation_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(relation_dim, emb_dim, bias=False)

    def forward(self, hu: Tensor, hv: Tensor, r: Tensor) -> Tensor:
        rp = self.proj(r)
        return (hu * rp * hv).sum(dim=-1)


class FilmSageSemanticInjectBundle(nn.Module):
    """Encoder (adapted FiLM SAGE) + relation-conditioned decode head (single optimizer)."""

    def __init__(
        self,
        encoder: RelationFilmGraphSAGEAdapted,
        decode_head: RelDistMultDecodeHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decode_head = decode_head

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.encoder(x, edge_index, edge_attr)

    def decode_logits(
        self, z: Tensor, edge_label_index: Tensor, relation_ids: Tensor
    ) -> Tensor:
        rt = self.encoder.adapted_relation_table()
        # Project full relation table (R x dim) before indexing batch edges — same as encoder forward.
        r_proj = self.decode_head.proj(rt)[relation_ids.clamp(min=0)]
        hu = z[edge_label_index[0]]
        hv = z[edge_label_index[1]]
        return (hu * r_proj * hv).sum(dim=-1)
