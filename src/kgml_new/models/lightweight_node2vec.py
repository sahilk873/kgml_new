from __future__ import annotations

import torch
from torch import nn


class Node2VecEmbedding(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

    def forward(self) -> torch.Tensor:
        return self.embedding.weight


def train_lightweight_node2vec(*args, **kwargs):
    raise ImportError(
        "The lightweight node2vec fallback is unavailable in this checkout; "
        "install pyg-lib/torch-cluster so torch_geometric.nn.Node2Vec can run."
    )
