"""RotatE: relational rotation embeddings for knowledge graph link prediction."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RotatE(nn.Module):
    """
    RotatE scoring on homogeneous node ids with relation ids from ``edge_attr``.

    Entity embeddings live in ``C^K`` (stored as concatenated real/imag of shape ``2K``).
    Each relation is a rotation (phases ``theta_r in R^K``, applied per dimension).
    Score (higher is better) is ``-gamma * || h ∘ r - t ||_2^2`` (used as logits for BCE).
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        *,
        gamma: float = 12.0,
    ) -> None:
        super().__init__()
        if num_entities <= 0 or num_relations <= 0:
            raise ValueError("num_entities and num_relations must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gamma = float(gamma)

        self.entity_emb = nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        k = self.embedding_dim
        bound_ent = 6.0 / math.sqrt(k)
        nn.init.uniform_(self.entity_emb.weight.data, -bound_ent, bound_ent)
        nn.init.uniform_(self.relation_emb.weight.data, -math.pi, math.pi)

    def _split_complex(self, e: Tensor) -> tuple[Tensor, Tensor]:
        """Map ``[..., 2K]`` -> real ``[..., K]``, imag ``[..., K]``."""
        k = self.embedding_dim
        return e[..., :k], e[..., k:]

    def normalize_entities(self) -> None:
        """Project each entity onto the product of unit circles (per complex dimension)."""
        e = self.entity_emb.weight.data.view(-1, self.embedding_dim, 2)
        norms = (e * e).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
        e = e / norms
        self.entity_emb.weight.data = e.view(-1, self.embedding_dim * 2)

    def score_logits(self, src: Tensor, rel: Tensor, dst: Tensor) -> Tensor:
        """Pairwise RotatE logits ``[B]`` (same shape as inputs)."""
        re_h, im_h = self._split_complex(self.entity_emb(src))
        re_t, im_t = self._split_complex(self.entity_emb(dst))
        phase = self.relation_emb(rel)
        re_r = torch.cos(phase)
        im_r = torch.sin(phase)
        re_hr = re_h * re_r - im_h * im_r
        im_hr = re_h * im_r + im_h * re_r
        dist = ((re_hr - re_t) ** 2 + (im_hr - im_t) ** 2).sum(dim=-1)
        return -self.gamma * dist
