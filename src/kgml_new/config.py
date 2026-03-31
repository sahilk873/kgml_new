from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Defaults aligned with docs/semantic-edge-graphsage-pipeline.md §6.
    
    Optimized for GPU efficiency and publication-ready results:
    - Mixed precision (fp16) enabled by default on GPU
    - Gradient clipping for training stability
    - Early stopping to prevent overfitting
    - Learning rate scheduling with ReduceLROnPlateau
    - Validation-based model selection
    """

    in_dim: int = 64
    edge_dim: int = 32
    hidden_dim: int = 256
    out_dim: int = 256
    num_layers: int = 2
    epochs: int = 200
    neg_samples: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 512
    num_neighbors: list[int] = field(default_factory=lambda: [15, 10])
    seed: int = 42
    dropout: float = 0.2
    concat: bool = True
    early_stop_patience: int = 20
    grad_clip_norm: float = 1.0
    use_amp: bool = True


@dataclass
class Node2VecConfig:
    """PyG `torch_geometric.nn.Node2Vec` training hyperparameters.

    Optimized for GPU efficiency:
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Memory-efficient batch processing
    """

    embedding_dim: int = 256
    walk_length: int = 40
    context_size: int = 20
    walks_per_node: int = 10
    p: float = 1.0
    q: float = 1.0
    num_negative_samples: int = 5
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 0.01
    seed: int = 42
    early_stop_patience: int = 20
    grad_clip_norm: float = 1.0
    use_amp: bool = True


@dataclass
class LinkMLPConfig:
    """MLP scorer on `concat(z[src], z[dst])` for link prediction.
    
    Optimized for publication-ready results:
    - Multiple negative samples per positive edge
    - Full batch processing (not single batch)
    - Early stopping
    - Gradient clipping
    """

    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    neg_ratio: int = 3
    seed: int = 42
    early_stop_patience: int = 20
    grad_clip_norm: float = 1.0
    use_amp: bool = True
