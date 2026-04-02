from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch

from kgml_new.config import NodeClassificationConfig
from kgml_new.data.datasets import (
    prepare_hetero_node_classification_dataset,
    prepare_node_classification_dataset,
)
from kgml_new.training.node_classification import (
    NODE_CLASSIFICATION_METHODS,
    train_embedding_node_classifier,
    train_native_node_classifier,
)


def _toy_labeled_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("a", label="drug")
    g.add_node("b", label="drug")
    g.add_node("c", label="disease")
    g.add_node("d", label="disease")
    g.add_edge("a", "b", relationship="similar")
    g.add_edge("a", "c", relationship="treats")
    g.add_edge("b", "d", relationship="treats")
    g.add_edge("c", "d", relationship="related")
    return g


def test_prepare_node_classification_dataset_smoke():
    dataset = prepare_node_classification_dataset(
        _toy_labeled_graph(),
        in_dim=16,
        label_attr="label",
        seed=0,
    )
    assert dataset.data.y.shape[0] == dataset.data.num_nodes
    assert int(dataset.train_mask.sum()) > 0
    assert len(dataset.label_lookup) == 2


def test_graphsage_node_classification_smoke():
    dataset = prepare_node_classification_dataset(
        _toy_labeled_graph(),
        in_dim=16,
        label_attr="label",
        seed=0,
    )
    cfg = NodeClassificationConfig(
        in_dim=16,
        hidden_dim=8,
        num_classes=len(dataset.label_lookup),
        epochs=2,
        batch_size=2,
        num_neighbors=[2, 2],
        seed=0,
    )
    _, history, val_metrics, test_metrics = train_native_node_classifier(
        "sage",
        dataset.data,
        cfg,
        device=torch.device("cpu"),
        relation_lookup=dataset.relation_lookup,
    )
    assert len(history.epoch) == 2
    assert "accuracy" in val_metrics
    assert "macro_f1" in test_metrics


def test_node2vec_node_classification_smoke():
    dataset = prepare_node_classification_dataset(
        _toy_labeled_graph(),
        in_dim=16,
        label_attr="label",
        seed=0,
    )
    cfg = NodeClassificationConfig(
        in_dim=16,
        hidden_dim=8,
        embedding_dim=16,
        num_classes=len(dataset.label_lookup),
        epochs=1,
        classifier_epochs=2,
        batch_size=8,
        seed=0,
    )
    _, _, history, val_metrics, test_metrics = train_embedding_node_classifier(
        "node2vec",
        dataset.data,
        cfg,
        device=torch.device("cpu"),
        relation_lookup=dataset.relation_lookup,
    )
    assert len(history.epoch) == 2
    assert "accuracy" in val_metrics
    assert "macro_f1" in test_metrics


def _primekg_node_classification_graph() -> nx.Graph:
    """
    Build a small PrimeKG subgraph with multiple node types.

    The first rows in kg.csv are mostly gene/protein edges, so we intentionally
    sample from the later mixed-type region to exercise node classification on
    a more representative PrimeKG slice.
    """
    kg_path = Path(__file__).parent.parent / "kg.csv"
    if not kg_path.exists():
        pytest.skip("PrimeKG CSV not found at kg.csv")

    selected = None
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            kg_path,
            chunksize=100000,
            usecols=["x_name", "y_name", "relation", "x_type", "y_type", "x_id", "y_id"],
        )
    ):
        if chunk_idx != 3:
            continue

        mask = (chunk["x_type"].astype(str) != "gene/protein") | (
            chunk["y_type"].astype(str) != "gene/protein"
        )
        if not mask.any():
            pytest.skip("PrimeKG chunk does not contain a mixed-type node classification slice")

        start_idx = int(mask.idxmax())
        selected = chunk.loc[start_idx : start_idx + 1999].copy()
        break

    if selected is None or selected.empty:
        pytest.skip("PrimeKG mixed-type slice could not be constructed")

    graph = nx.Graph()
    for row in selected.itertuples(index=False):
        graph.add_node(str(row.x_name), node_type=str(row.x_type), node_id=str(row.x_id))
        graph.add_node(str(row.y_name), node_type=str(row.y_type), node_id=str(row.y_id))
        graph.add_edge(str(row.x_name), str(row.y_name), relationship=str(row.relation))
    return graph


@pytest.mark.parametrize("method", ["sage", "gcn", "edge_sage", "node2vec", "link_mlp"])
def test_primekg_node_classification_smoke(method: str):
    graph = _primekg_node_classification_graph()
    dataset = prepare_node_classification_dataset(
        graph,
        in_dim=16,
        label_attr="node_type",
        seed=0,
    )
    assert len(dataset.label_lookup) >= 2

    cfg = NodeClassificationConfig(
        in_dim=16,
        hidden_dim=8,
        embedding_dim=16,
        num_classes=len(dataset.label_lookup),
        epochs=1,
        classifier_epochs=1,
        batch_size=64,
        seed=0,
    )
    if method in {"sage", "gcn", "edge_sage"}:
        _, history, val_metrics, test_metrics = train_native_node_classifier(
            method,
            dataset.data,
            cfg,
            device=torch.device("cpu"),
            relation_lookup=dataset.relation_lookup,
        )
    else:
        _, _, history, val_metrics, test_metrics = train_embedding_node_classifier(
            method,
            dataset.data,
            cfg,
            device=torch.device("cpu"),
            relation_lookup=dataset.relation_lookup,
        )
    assert len(history.epoch) == 1
    assert "accuracy" in val_metrics
    assert "macro_f1" in test_metrics


def test_node_classification_registry_covers_current_methods():
    assert {"sage", "gcn", "edge_sage", "node2vec", "link_mlp", "txgnn"} <= set(
        NODE_CLASSIFICATION_METHODS
    )


def test_node_classification_split_is_stratified_and_deterministic():
    graph = nx.Graph()
    labels = {
        "a0": "alpha",
        "a1": "alpha",
        "a2": "alpha",
        "b0": "beta",
        "b1": "beta",
        "b2": "beta",
        "c0": "gamma",
    }
    for node, label in labels.items():
        graph.add_node(node, label=label)
    for src, dst in zip(list(labels)[:-1], list(labels)[1:], strict=False):
        graph.add_edge(src, dst, relationship="related")

    first = prepare_node_classification_dataset(graph, in_dim=8, seed=7)
    second = prepare_node_classification_dataset(graph, in_dim=8, seed=7)

    assert torch.equal(first.train_mask, second.train_mask)
    assert torch.equal(first.val_mask, second.val_mask)
    assert torch.equal(first.test_mask, second.test_mask)

    gamma_idx = next(idx for label, idx in first.label_lookup.items() if label == "gamma")
    assert int(((first.data.y == gamma_idx) & first.train_mask).sum()) == 1

    for label_name, label_idx in first.label_lookup.items():
        label_mask = first.data.y == label_idx
        if int(label_mask.sum()) >= 3:
            assert int((label_mask & first.train_mask).sum()) > 0
            assert int((label_mask & first.test_mask).sum()) > 0


def test_hetero_node_classification_split_preserves_tiny_class_in_train():
    graph = nx.Graph()
    graph.add_node("drug_0", node_type="drug")
    graph.add_node("drug_1", node_type="drug")
    graph.add_node("drug_2", node_type="drug")
    graph.add_node("drug_3", node_type="drug")
    graph.add_node("drug_4", node_type="drug")
    graph.add_node("drug_5", node_type="drug")
    graph.add_node("disease_0", node_type="disease", label="rare")
    graph.add_node("disease_1", node_type="disease", label="common")
    graph.add_node("disease_2", node_type="disease", label="common")
    graph.add_node("disease_3", node_type="disease", label="common")
    for drug_idx in range(6):
        graph.add_edge(f"drug_{drug_idx}", f"disease_{drug_idx % 4}", relationship="treats")

    dataset = prepare_hetero_node_classification_dataset(
        graph,
        in_dim=8,
        target_node_type="disease",
        seed=3,
    )

    rare_idx = next(idx for label, idx in dataset.label_lookup.items() if label == "rare")
    rare_mask = dataset.data["disease"].y == rare_idx
    assert int((rare_mask & dataset.train_mask).sum()) == 1
