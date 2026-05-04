from __future__ import annotations

import torch
from torch import Tensor, nn


def _pair_features(h: Tensor, t: Tensor, feature_mode: str) -> Tensor:
    if feature_mode == "concat":
        return torch.cat([h, t], dim=-1)
    if feature_mode == "concat_product":
        return torch.cat([h, t, h * t], dim=-1)
    raise ValueError(f"feature_mode must be 'concat' or 'concat_product', got {feature_mode!r}")


class LinkPredictionMLP(nn.Module):
    """MLP scoring head for link prediction on pairs of node embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        feature_mode: str = "concat_product",
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.feature_mode = feature_mode
        in_dim = self._input_dim(self.embedding_dim)
        dims = [in_dim] + list(hidden_dims) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.ReLU(), nn.Dropout(float(dropout))])
        self.net = nn.Sequential(*layers)

    def _input_dim(self, embedding_dim: int) -> int:
        if self.feature_mode == "concat":
            return 2 * embedding_dim
        if self.feature_mode == "concat_product":
            return 3 * embedding_dim
        raise ValueError(
            f"feature_mode must be 'concat' or 'concat_product', got {self.feature_mode!r}"
        )

    def forward(self, z_src: Tensor, z_dst: Tensor) -> Tensor:
        x = _pair_features(z_src, z_dst, self.feature_mode)
        return self.net(x).squeeze(-1)
