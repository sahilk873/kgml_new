from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


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
    edge_relation_mode: str = "concat"
    num_relation_bases: int = 4
    neighbor_aggr: str = "mean"


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
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 20


@dataclass
class LinkMLPConfig:
    """MLP scorer on pair features for link prediction."""

    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    neg_ratio: int = 1
    seed: int = 42
    feature_mode: str = "concat_product"
    early_stop_patience: int = 20
    grad_clip_norm: float = 1.0
    use_amp: bool = True


@dataclass
class NodeClassificationConfig:
    """Generic node-classification training configuration."""

    in_dim: int = 64
    edge_dim: int = 32
    hidden_dim: int = 64
    embedding_dim: int = 64
    num_classes: int = 2
    num_layers: int = 2
    epochs: int = 50
    classifier_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    classifier_learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.1
    num_neighbors: list[int] = field(default_factory=lambda: [10, 10])
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    concat: bool = True
    edge_relation_mode: str = "concat"
    num_relation_bases: int = 4
    neighbor_aggr: str = "mean"


def _parse_num_neighbors_spec(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        return None
    return [int(x) for x in parts]


def train_config_from_run_gpu_method_args(ns: Any) -> TrainConfig:
    """Build ``TrainConfig`` from ``run_gpu_method`` argparse namespace (plus optional tuning fields)."""
    cfg = TrainConfig(
        in_dim=int(ns.in_dim),
        out_dim=int(ns.in_dim),
        epochs=int(ns.epochs),
        seed=int(ns.seed),
        neighbor_aggr=str(getattr(ns, "neighbor_aggr", "mean")),
        edge_relation_mode=str(getattr(ns, "edge_relation_mode", "concat")),
        num_relation_bases=int(getattr(ns, "num_relation_bases", 4)),
        concat=bool(getattr(ns, "concat", True)),
    )
    if getattr(ns, "learning_rate", None) is not None:
        cfg = replace(cfg, learning_rate=float(ns.learning_rate))
    if getattr(ns, "train_batch_size", None) is not None:
        cfg = replace(cfg, batch_size=int(ns.train_batch_size))
    if getattr(ns, "dropout", None) is not None:
        cfg = replace(cfg, dropout=float(ns.dropout))
    if getattr(ns, "num_layers", None) is not None:
        cfg = replace(cfg, num_layers=int(ns.num_layers))
    if getattr(ns, "hidden_dim", None) is not None:
        cfg = replace(cfg, hidden_dim=int(ns.hidden_dim))
    if getattr(ns, "edge_dim", None) is not None:
        cfg = replace(cfg, edge_dim=int(ns.edge_dim))
    nn_list = _parse_num_neighbors_spec(getattr(ns, "num_neighbors_spec", None))
    if nn_list is not None:
        cfg = replace(cfg, num_neighbors=nn_list)
    return cfg


def link_mlp_config_from_run_gpu_method_args(ns: Any, *, epochs: int | None = None) -> LinkMLPConfig:
    """Link MLP head config for ``run_gpu_method`` / link prediction."""
    ep = int(ns.epochs if epochs is None else epochs)
    cfg = LinkMLPConfig(epochs=ep, seed=int(ns.seed), feature_mode="concat_product")
    if getattr(ns, "link_mlp_learning_rate", None) is not None:
        cfg = replace(cfg, learning_rate=float(ns.link_mlp_learning_rate))
    if getattr(ns, "link_mlp_dropout", None) is not None:
        cfg = replace(cfg, dropout=float(ns.link_mlp_dropout))
    if getattr(ns, "link_mlp_batch_size", None) is not None:
        cfg = replace(cfg, batch_size=int(ns.link_mlp_batch_size))
    hd = getattr(ns, "link_mlp_hidden_dims", None)
    if hd is not None:
        dims = tuple(int(x) for x in str(hd).replace(" ", "").split(",") if x)
        if dims:
            cfg = replace(cfg, hidden_dims=dims)
    return cfg


def node2vec_config_from_run_gpu_method_args(ns: Any) -> Node2VecConfig:
    cfg = Node2VecConfig(epochs=int(ns.epochs), seed=int(ns.seed), embedding_dim=64)
    if getattr(ns, "n2v_embedding_dim", None) is not None:
        cfg = replace(cfg, embedding_dim=int(ns.n2v_embedding_dim))
    if getattr(ns, "n2v_walk_length", None) is not None:
        cfg = replace(cfg, walk_length=int(ns.n2v_walk_length))
    if getattr(ns, "n2v_context_size", None) is not None:
        cfg = replace(cfg, context_size=int(ns.n2v_context_size))
    if getattr(ns, "n2v_walks_per_node", None) is not None:
        cfg = replace(cfg, walks_per_node=int(ns.n2v_walks_per_node))
    if getattr(ns, "n2v_lr", None) is not None:
        cfg = replace(cfg, learning_rate=float(ns.n2v_lr))
    if getattr(ns, "n2v_batch_size", None) is not None:
        cfg = replace(cfg, batch_size=int(ns.n2v_batch_size))
    if getattr(ns, "n2v_num_negative_samples", None) is not None:
        cfg = replace(cfg, num_negative_samples=int(ns.n2v_num_negative_samples))
    return cfg


def train_config_from_link_prediction_args(ns: Any) -> TrainConfig:
    """``TrainConfig`` for ``run_link_prediction`` (homogeneous pickle graph)."""
    cfg = TrainConfig(
        epochs=int(ns.epochs),
        seed=int(ns.seed),
        neighbor_aggr=str(getattr(ns, "neighbor_aggr", "mean")),
        edge_relation_mode=str(getattr(ns, "edge_relation_mode", "concat")),
        num_relation_bases=int(getattr(ns, "num_relation_bases", 4)),
    )
    if getattr(ns, "learning_rate", None) is not None:
        cfg = replace(cfg, learning_rate=float(ns.learning_rate))
    if getattr(ns, "train_batch_size", None) is not None:
        cfg = replace(cfg, batch_size=int(ns.train_batch_size))
    if getattr(ns, "dropout", None) is not None:
        cfg = replace(cfg, dropout=float(ns.dropout))
    if getattr(ns, "num_layers", None) is not None:
        cfg = replace(cfg, num_layers=int(ns.num_layers))
    if getattr(ns, "hidden_dim", None) is not None:
        cfg = replace(cfg, hidden_dim=int(ns.hidden_dim))
    if getattr(ns, "edge_dim", None) is not None:
        cfg = replace(cfg, edge_dim=int(ns.edge_dim))
    nn_list = _parse_num_neighbors_spec(getattr(ns, "num_neighbors_spec", None))
    if nn_list is not None:
        cfg = replace(cfg, num_neighbors=nn_list)
    return cfg


def node_classification_config_from_args(ns: Any, *, num_classes: int) -> NodeClassificationConfig:
    """Build ``NodeClassificationConfig`` from ``run_node_classification`` argparse namespace."""
    cfg = NodeClassificationConfig(
        in_dim=int(ns.in_dim),
        edge_dim=int(ns.edge_dim),
        hidden_dim=int(ns.hidden_dim),
        embedding_dim=int(ns.embedding_dim),
        num_classes=int(num_classes),
        epochs=int(ns.epochs),
        classifier_epochs=int(ns.classifier_epochs),
        batch_size=int(ns.batch_size),
        seed=int(ns.seed),
        edge_relation_mode=str(getattr(ns, "edge_relation_mode", "concat")),
        num_relation_bases=int(getattr(ns, "num_relation_bases", 4)),
        neighbor_aggr=str(getattr(ns, "neighbor_aggr", "mean")),
    )
    if getattr(ns, "learning_rate", None) is not None:
        cfg = replace(cfg, learning_rate=float(ns.learning_rate))
    if getattr(ns, "classifier_learning_rate", None) is not None:
        cfg = replace(cfg, classifier_learning_rate=float(ns.classifier_learning_rate))
    if getattr(ns, "weight_decay", None) is not None:
        cfg = replace(cfg, weight_decay=float(ns.weight_decay))
    if getattr(ns, "dropout", None) is not None:
        cfg = replace(cfg, dropout=float(ns.dropout))
    if getattr(ns, "num_layers", None) is not None:
        cfg = replace(cfg, num_layers=int(ns.num_layers))
    nn_list = _parse_num_neighbors_spec(getattr(ns, "num_neighbors_spec", None))
    if nn_list is not None:
        cfg = replace(cfg, num_neighbors=nn_list)
    return cfg
