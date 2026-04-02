from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.data import Data

from kgml_new.config import NodeClassificationConfig, TrainConfig
from kgml_new.embeddings.semantic import build_relation_tensor
from kgml_new.models.baseline_gcn import BaselineGCN
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.models.node_classifier import NodeClassificationMLP
from kgml_new.models.txgnn import TxGNN
from kgml_new.training.link_unsupervised import compute_node_embeddings, train_unsupervised
from kgml_new.training.node2vec_train import train_node2vec_embeddings


@dataclass
class NodeClassificationHistory:
    epoch: list[int]
    train_loss: list[float]
    val_accuracy: list[float]
    val_macro_f1: list[float]
    config: dict


@dataclass(frozen=True)
class NodeClassificationMethodSpec:
    name: str
    kind: str


NODE_CLASSIFICATION_METHODS: dict[str, NodeClassificationMethodSpec] = {
    "sage": NodeClassificationMethodSpec("sage", "native"),
    "gcn": NodeClassificationMethodSpec("gcn", "native"),
    "edge_sage": NodeClassificationMethodSpec("edge_sage", "native"),
    "node2vec": NodeClassificationMethodSpec("node2vec", "embedding"),
    "link_mlp": NodeClassificationMethodSpec("link_mlp", "embedding"),
    "txgnn": NodeClassificationMethodSpec("txgnn", "hetero_native"),
}


def node_classification_metrics(logits: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    accuracy = float((pred == y_true).float().mean().item())
    macro_f1 = float(
        f1_score(
            y_true.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _evaluate_masked_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if int(mask.sum()) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    return node_classification_metrics(logits[mask], y[mask])


def _build_native_model(
    method: str,
    cfg: NodeClassificationConfig,
    relation_lookup: dict[str, int] | None = None,
) -> nn.Module:
    if method == "sage":
        return BaselineGraphSAGE(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            normalize_output=False,
        )
    if method == "gcn":
        return BaselineGCN(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.num_classes,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            normalize_output=False,
        )
    if method == "edge_sage":
        if relation_lookup is None:
            raise ValueError("relation_lookup is required for edge_sage node classification")
        rel_emb = {rel: torch.randn(cfg.edge_dim) * 0.1 for rel in relation_lookup}
        relation_table = build_relation_tensor(
            rel_emb,
            relation_lookup,
            cfg.edge_dim,
            torch.device("cpu"),
        )
        return EdgeAwareGraphSAGE(
            cfg.in_dim,
            cfg.edge_dim,
            cfg.hidden_dim,
            cfg.num_classes,
            relation_table=relation_table,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            concat=cfg.concat,
            normalize_output=False,
        )
    raise ValueError(f"Unsupported native node classification method: {method}")


def train_native_node_classifier(
    method: str,
    data: Data,
    cfg: NodeClassificationConfig,
    *,
    device: torch.device | None = None,
    relation_lookup: dict[str, int] | None = None,
) -> tuple[nn.Module, NodeClassificationHistory, dict[str, float], dict[str, float]]:
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_native_model(method, cfg, relation_lookup=relation_lookup).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    history = NodeClassificationHistory(
        epoch=[],
        train_loss=[],
        val_accuracy=[],
        val_macro_f1=[],
        config=asdict(cfg),
    )

    use_batched = WITH_PYG_LIB or WITH_TORCH_SPARSE
    train_input_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
    cpu_data = data.cpu()
    loader = None
    if use_batched:
        loader = NeighborLoader(
            cpu_data,
            num_neighbors=cfg.num_neighbors,
            input_nodes=train_input_nodes,
            batch_size=cfg.batch_size,
            shuffle=True,
        )

    full_data = data.to(device)
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        if loader is None:
            optimizer.zero_grad()
            logits = (
                model(full_data.x, full_data.edge_index, full_data.edge_attr)
                if method == "edge_sage"
                else model(full_data.x, full_data.edge_index)
            )
            loss = F.cross_entropy(logits[full_data.train_mask], full_data.y[full_data.train_mask])
            loss.backward()
            optimizer.step()
            total_loss = float(loss.detach())
            num_batches = 1
        else:
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = (
                    model(batch.x, batch.edge_index, batch.edge_attr)
                    if method == "edge_sage"
                    else model(batch.x, batch.edge_index)
                )
                seed_logits = logits[: batch.batch_size]
                seed_labels = batch.y[: batch.batch_size]
                loss = F.cross_entropy(seed_logits, seed_labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                num_batches += 1

        model.eval()
        with torch.inference_mode():
            logits = (
                model(full_data.x, full_data.edge_index, full_data.edge_attr)
                if method == "edge_sage"
                else model(full_data.x, full_data.edge_index)
            )
        val_metrics = _evaluate_masked_logits(logits, full_data.y, full_data.val_mask)
        history.epoch.append(epoch)
        history.train_loss.append(total_loss / max(num_batches, 1))
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_macro_f1.append(val_metrics["macro_f1"])

    with torch.inference_mode():
        logits = (
            model(full_data.x, full_data.edge_index, full_data.edge_attr)
            if method == "edge_sage"
            else model(full_data.x, full_data.edge_index)
        )
    val_metrics = _evaluate_masked_logits(logits, full_data.y, full_data.val_mask)
    test_metrics = _evaluate_masked_logits(logits, full_data.y, full_data.test_mask)
    return model, history, val_metrics, test_metrics


def _positive_edges_from_data(data: Data) -> torch.Tensor:
    mask = data.edge_index[0] < data.edge_index[1]
    return data.edge_index[:, mask]


def _train_embeddings_for_method(
    method: str,
    data: Data,
    cfg: NodeClassificationConfig,
    *,
    device: torch.device,
    relation_lookup: dict[str, int] | None = None,
) -> torch.Tensor:
    if method == "node2vec":
        from kgml_new.config import Node2VecConfig

        node2vec_cfg = Node2VecConfig(
            embedding_dim=cfg.embedding_dim,
            epochs=max(1, cfg.epochs),
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
        )
        _, z, _ = train_node2vec_embeddings(data, node2vec_cfg, device=device)
        return z

    if method == "link_mlp":
        train_cfg = TrainConfig(
            in_dim=cfg.in_dim,
            edge_dim=cfg.edge_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.embedding_dim,
            num_layers=cfg.num_layers,
            epochs=max(1, cfg.epochs),
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            num_neighbors=cfg.num_neighbors,
            seed=cfg.seed,
            dropout=cfg.dropout,
            concat=cfg.concat,
        )
        encoder = BaselineGraphSAGE(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.embedding_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            normalize_output=True,
        )
        train_unsupervised(
            encoder,
            data,
            _positive_edges_from_data(data),
            train_cfg,
            device=device,
            edge_aware=False,
        )
        return compute_node_embeddings(encoder, data, device, edge_aware=False)

    raise ValueError(f"Unsupported embedding-based node classification method: {method}")


def train_embedding_node_classifier(
    method: str,
    data: Data,
    cfg: NodeClassificationConfig,
    *,
    device: torch.device | None = None,
    relation_lookup: dict[str, int] | None = None,
) -> tuple[torch.Tensor, nn.Module, NodeClassificationHistory, dict[str, float], dict[str, float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_data = data.to(device)
    z = _train_embeddings_for_method(
        method,
        data,
        cfg,
        device=device,
        relation_lookup=relation_lookup,
    ).to(device)

    classifier = NodeClassificationMLP(
        z.size(1),
        cfg.hidden_dim,
        cfg.num_classes,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=cfg.classifier_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    history = NodeClassificationHistory(
        epoch=[],
        train_loss=[],
        val_accuracy=[],
        val_macro_f1=[],
        config=asdict(cfg),
    )

    for epoch in range(cfg.classifier_epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(z)
        loss = F.cross_entropy(logits[full_data.train_mask], full_data.y[full_data.train_mask])
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.inference_mode():
            logits = classifier(z)
        val_metrics = _evaluate_masked_logits(logits, full_data.y, full_data.val_mask)
        history.epoch.append(epoch)
        history.train_loss.append(float(loss.detach()))
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_macro_f1.append(val_metrics["macro_f1"])

    with torch.inference_mode():
        logits = classifier(z)
    val_metrics = _evaluate_masked_logits(logits, full_data.y, full_data.val_mask)
    test_metrics = _evaluate_masked_logits(logits, full_data.y, full_data.test_mask)
    return z, classifier, history, val_metrics, test_metrics


def train_txgnn_node_classifier(
    data,
    target_node_type: str,
    cfg: NodeClassificationConfig,
    *,
    device: torch.device | None = None,
) -> tuple[TxGNN, nn.Module, NodeClassificationHistory, dict[str, float], dict[str, float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TxGNN(
        metadata=data.metadata(),
        in_channels=cfg.in_dim,
        hidden_channels=cfg.hidden_dim,
        out_channels=cfg.embedding_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_prototypes=True,
        prototype_k=5,
        prototype_alpha=0.5,
    ).to(device)
    classifier = NodeClassificationMLP(
        cfg.embedding_dim,
        cfg.hidden_dim,
        cfg.num_classes,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    history = NodeClassificationHistory(
        epoch=[],
        train_loss=[],
        val_accuracy=[],
        val_macro_f1=[],
        config=asdict(cfg),
    )

    data = data.to(device)
    target_store = data[target_node_type]
    for epoch in range(cfg.epochs):
        model.train()
        classifier.train()
        optimizer.zero_grad()
        z_dict = model.encode(data)
        logits = classifier(z_dict[target_node_type])
        loss = F.cross_entropy(logits[target_store.train_mask], target_store.y[target_store.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        classifier.eval()
        with torch.inference_mode():
            z_dict = model.encode(data)
            logits = classifier(z_dict[target_node_type])
        val_metrics = _evaluate_masked_logits(logits, target_store.y, target_store.val_mask)
        history.epoch.append(epoch)
        history.train_loss.append(float(loss.detach()))
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_macro_f1.append(val_metrics["macro_f1"])

    with torch.inference_mode():
        z_dict = model.encode(data)
        logits = classifier(z_dict[target_node_type])
    val_metrics = _evaluate_masked_logits(logits, target_store.y, target_store.val_mask)
    test_metrics = _evaluate_masked_logits(logits, target_store.y, target_store.test_mask)
    return model, classifier, history, val_metrics, test_metrics
