"""Homogeneous and hetero node classification trainers used by ``run_node_classification``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from kgml_new.config import Node2VecConfig, NodeClassificationConfig, TrainConfig
from kgml_new.embeddings.semantic import (
    DEFAULT_GLOSSARY_PATH,
    build_relation_tensor,
    relation_embeddings_from_graph,
)
from kgml_new.models.baseline_gcn import BaselineGCN
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import build_edge_aware_model
from kgml_new.training.eval import node_classification_scores
from kgml_new.training.link_unsupervised import compute_node_embeddings, train_unsupervised
from kgml_new.training.node2vec_train import train_node2vec_embeddings


@dataclass(frozen=True)
class NodeClassificationMethodSpec:
    """Routing metadata for ``NODE_CLASSIFICATION_METHODS``."""

    kind: Literal["native", "embedding", "hetero_native"]


NODE_CLASSIFICATION_METHODS: dict[str, NodeClassificationMethodSpec] = {
    "sage": NodeClassificationMethodSpec("native"),
    "gcn": NodeClassificationMethodSpec("native"),
    "edge_sage": NodeClassificationMethodSpec("native"),
    "node2vec": NodeClassificationMethodSpec("embedding"),
    "link_mlp": NodeClassificationMethodSpec("embedding"),
    "txgnn": NodeClassificationMethodSpec("hetero_native"),
}


@dataclass
class NodeClassificationHistory:
    epoch: list[int]
    train_loss: list[float]
    val_accuracy: list[float]
    val_macro_f1: list[float]


def _undirected_pos_edges(data: Data) -> torch.Tensor:
    """One undirected copy per edge (``src < dst``) for link-style pretraining."""
    ei = data.edge_index
    if ei.size(1) == 0:
        return ei
    src, dst = ei[0], ei[1]
    mask = src < dst
    if int(mask.sum()) > 0:
        return ei[:, mask]
    return ei


def _nc_to_train_config(cfg: NodeClassificationConfig) -> TrainConfig:
    return TrainConfig(
        in_dim=cfg.in_dim,
        out_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        num_layers=cfg.num_layers,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_neighbors=cfg.num_neighbors,
        seed=cfg.seed,
        dropout=cfg.dropout,
        neighbor_aggr=cfg.neighbor_aggr,
    )


def _build_relation_table(
    *,
    graph: nx.Graph,
    relation_lookup: dict[str, int],
    edge_dim: int,
    device: torch.device,
    use_semantic: bool,
    semantic_cache: Path | None,
    strict_semantic: bool,
) -> torch.Tensor:
    if use_semantic:
        rel_emb = relation_embeddings_from_graph(
            graph,
            edge_dim=edge_dim,
            cache_path=semantic_cache,
            embedding_model="openai",
            strict_embedding=strict_semantic,
            glossary_path=DEFAULT_GLOSSARY_PATH,
        )
    else:
        rel_emb = {k: torch.randn(edge_dim) * 0.02 for k in relation_lookup}
    return build_relation_tensor(rel_emb, relation_lookup, edge_dim, device)


def _build_native_model(
    method: str,
    cfg: NodeClassificationConfig,
    relation_table: torch.Tensor,
) -> nn.Module:
    """Encoder + linear head; output is class logits (``N x num_classes``)."""
    if method == "gcn":
        enc = BaselineGCN(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    elif method == "sage":
        enc = BaselineGraphSAGE(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            aggr=cfg.neighbor_aggr,
        )
    elif method == "edge_sage":
        enc = build_edge_aware_model(
            edge_relation_mode=cfg.edge_relation_mode,
            in_channels=cfg.in_dim,
            edge_dim=cfg.edge_dim,
            hidden_channels=cfg.hidden_dim,
            out_channels=cfg.hidden_dim,
            relation_table=relation_table,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            concat=cfg.concat,
            normalize_output=False,
            num_relation_bases=cfg.num_relation_bases,
            neighbor_aggr=cfg.neighbor_aggr,
        )
    else:
        raise ValueError(f"Unsupported native node classification method: {method!r}")

    return _EncoderClassifier(
        encoder=enc,
        dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        edge_aware=method == "edge_sage",
    )


class _EncoderClassifier(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        dim: int,
        num_classes: int,
        edge_aware: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, num_classes)
        self.edge_aware = edge_aware

    def forward(self, data: Data) -> torch.Tensor:
        if self.edge_aware:
            z = self.encoder(data.x, data.edge_index, data.edge_attr)
        else:
            z = self.encoder(data.x, data.edge_index)
        return self.head(z)


def train_native_node_classifier(
    method: str,
    data: Data,
    cfg: NodeClassificationConfig,
    device: torch.device,
    *,
    graph: nx.Graph,
    relation_lookup: dict[str, int],
    use_semantic: bool = False,
    semantic_cache: Path | None = None,
    strict_semantic: bool = False,
) -> tuple[nn.Module, NodeClassificationHistory, dict[str, float], dict[str, float]]:
    torch.manual_seed(cfg.seed)
    relation_table = _build_relation_table(
        graph=graph,
        relation_lookup=relation_lookup,
        edge_dim=cfg.edge_dim,
        device=device,
        use_semantic=use_semantic,
        semantic_cache=semantic_cache,
        strict_semantic=strict_semantic,
    )
    model = _build_native_model(method, cfg, relation_table).to(device)
    data_d = data.to(device)
    y = data_d.y
    train_m = data_d.train_mask
    val_m = data_d.val_mask
    test_m = data_d.test_mask

    labeled_train = train_m & (y >= 0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = NodeClassificationHistory([], [], [], [])
    val_metrics: dict[str, float] = {}
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data_d)
        loss = F.cross_entropy(logits[labeled_train], y[labeled_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data_d)
            val_metrics = node_classification_scores(
                logits[val_m & (y >= 0)],
                y[val_m & (y >= 0)],
            )
        history.epoch.append(epoch)
        history.train_loss.append(float(loss.detach()))
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_macro_f1.append(val_metrics["macro_f1"])

    model.eval()
    with torch.no_grad():
        logits = model(data_d)
        test_metrics = node_classification_scores(
            logits[test_m & (y >= 0)],
            y[test_m & (y >= 0)],
        )

    return model, history, val_metrics, test_metrics


def train_embedding_node_classifier(
    method: str,
    data: Data,
    cfg: NodeClassificationConfig,
    device: torch.device,
    *,
    relation_lookup: dict[str, int],
) -> tuple[None, None, NodeClassificationHistory, dict[str, float], dict[str, float]]:
    torch.manual_seed(cfg.seed)
    if method == "node2vec":
        n2v_cfg = Node2VecConfig(
            embedding_dim=cfg.embedding_dim,
            epochs=cfg.epochs,
            batch_size=min(cfg.batch_size, 512),
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
        )
        try:
            _, z, _ = train_node2vec_embeddings(data, n2v_cfg, device=device)
        except ImportError:
            gen = torch.Generator()
            gen.manual_seed(int(cfg.seed))
            z = torch.randn(
                int(data.num_nodes),
                int(cfg.embedding_dim),
                generator=gen,
                dtype=torch.float32,
            ).to(device)
    elif method == "link_mlp":
        tcfg = _nc_to_train_config(cfg)
        encoder = BaselineGraphSAGE(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.embedding_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            aggr=cfg.neighbor_aggr,
        )
        train_pos = _undirected_pos_edges(data)
        encoder, _ = train_unsupervised(
            encoder,
            data,
            train_pos,
            tcfg,
            device=device,
            edge_aware=False,
        )
        z = compute_node_embeddings(encoder, data, device, edge_aware=False)
    else:
        raise ValueError(f"Unsupported embedding node classification method: {method!r}")

    z = z.detach()
    return _fit_linear_classifier(z, data, cfg, device)


def _fit_linear_classifier(
    z: torch.Tensor,
    data: Data,
    cfg: NodeClassificationConfig,
    device: torch.device,
) -> tuple[None, None, NodeClassificationHistory, dict[str, float], dict[str, float]]:
    data_d = data.to(device)
    y = data_d.y.to(device)
    train_m = data_d.train_mask.to(device)
    val_m = data_d.val_mask.to(device)
    test_m = data_d.test_mask.to(device)
    z = z.to(device)

    head = nn.Linear(z.size(1), cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=cfg.classifier_learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = NodeClassificationHistory([], [], [], [])
    val_metrics: dict[str, float] = {}
    for epoch in range(cfg.classifier_epochs):
        head.train()
        optimizer.zero_grad()
        logits = head(z)
        loss = F.cross_entropy(
            logits[train_m & (y >= 0)],
            y[train_m & (y >= 0)],
        )
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            logits = head(z)
            val_metrics = node_classification_scores(
                logits[val_m & (y >= 0)],
                y[val_m & (y >= 0)],
            )
        history.epoch.append(epoch)
        history.train_loss.append(float(loss.detach()))
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_macro_f1.append(val_metrics["macro_f1"])

    with torch.no_grad():
        logits = head(z)
        test_metrics = node_classification_scores(
            logits[test_m & (y >= 0)],
            y[test_m & (y >= 0)],
        )

    return None, None, history, val_metrics, test_metrics


def train_txgnn_node_classifier(
    data: Any,
    target_node_type: str,
    cfg: NodeClassificationConfig,
    device: torch.device,
) -> tuple[None, None, NodeClassificationHistory, dict[str, float], dict[str, float]]:
    raise NotImplementedError(
        "TxGNN node classification is not available: TxGNN is stubbed in this checkout "
        "(see kgml_new.models.txgnn). Use sage, gcn, edge_sage, node2vec, or link_mlp."
    )
