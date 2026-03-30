from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Defaults aligned with docs/semantic-edge-graphsage-pipeline.md §6."""

    in_dim: int = 64
    edge_dim: int = 32
    hidden_dim: int = 32
    out_dim: int = 64
    num_layers: int = 2
    epochs: int = 100
    neg_samples: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_neighbors: list[int] = field(default_factory=lambda: [10, 10])
    seed: int = 42
    dropout: float = 0.0
    concat: bool = True


@dataclass
class Node2VecConfig:
    """PyG `torch_geometric.nn.Node2Vec` training hyperparameters.

    Optimized for faster training:
    - Shorter walk_length (10 vs 20)
    - Fewer walks_per_node (1 vs 5)
    - Fewer epochs (20 vs 100)
    - Larger batch_size (512 vs 256)
    - More negative samples (5 vs 1)
    """

    embedding_dim: int = 64
    walk_length: int = 10
    context_size: int = 10
    walks_per_node: int = 1
    p: float = 1.0
    q: float = 1.0
    num_negative_samples: int = 5
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 0.01
    seed: int = 42


@dataclass
class LinkMLPConfig:
    """MLP scorer on `concat(z[src], z[dst])` for link prediction."""

    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    neg_ratio: int = 1
    seed: int = 42
