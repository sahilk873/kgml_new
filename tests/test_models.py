from __future__ import annotations

from pathlib import Path

import networkx as nx
import torch

from kgml_new.config import TrainConfig
from kgml_new.data.graph import networkx_to_data
from kgml_new.data.primekg import load_primekg_csv
from kgml_new.embeddings.semantic import build_relation_tensor
from kgml_new.models.baseline_gcn import BaselineGCN
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.eval import link_prediction_sklearn
from kgml_new.training.link_unsupervised import compute_node_embeddings, train_unsupervised


def _toy_data():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    data, relation_lookup = networkx_to_data(g, in_dim=16, seed=0)
    return data, relation_lookup


def test_baseline_forward():
    data, _ = _toy_data()
    m = BaselineGraphSAGE(16, 8, 32, num_layers=2, dropout=0.0)
    z = m(data.x, data.edge_index)
    assert z.shape == (data.num_nodes, 32)
    assert torch.allclose(z.norm(dim=-1), torch.ones(data.num_nodes), atol=1e-4)


def test_gcn_forward():
    data, _ = _toy_data()
    m = BaselineGCN(16, 8, 32, num_layers=2, dropout=0.0)
    z = m(data.x, data.edge_index)
    assert z.shape == (data.num_nodes, 32)
    assert torch.allclose(z.norm(dim=-1), torch.ones(data.num_nodes), atol=1e-4)


def test_edge_aware_differs_from_zero_relations():
    data, relation_lookup = _toy_data()
    device = torch.device("cpu")
    edge_dim = 8

    rel_emb = {k: torch.randn(edge_dim) * 0.1 for k in relation_lookup}
    rt = build_relation_tensor(rel_emb, relation_lookup, edge_dim, device)

    m = EdgeAwareGraphSAGE(
        in_channels=16,
        edge_dim=edge_dim,
        hidden_channels=8,
        out_channels=16,
        relation_table=rt,
        num_layers=2,
        dropout=0.0,
        concat=True,
    )
    z_sem = m(data.x, data.edge_index, data.edge_attr)

    rt_zero = torch.zeros_like(rt)
    m0 = EdgeAwareGraphSAGE(
        in_channels=16,
        edge_dim=edge_dim,
        hidden_channels=8,
        out_channels=16,
        relation_table=rt_zero,
        num_layers=2,
        dropout=0.0,
        concat=True,
    )
    z_zero = m0(data.x, data.edge_index, data.edge_attr)

    assert not torch.allclose(z_sem, z_zero, atol=1e-5)


def _get_primekg_path() -> Path:
    import pytest

    p = Path(__file__).parent.parent / "kg.csv"
    if not p.exists():
        pytest.skip("PrimeKG CSV not found at kg.csv")
    return p


def test_primekg_baseline_smoke():
    kg_path = _get_primekg_path()
    g = load_primekg_csv(kg_path, max_edges=1000)

    assert g.number_of_nodes() > 0
    assert g.number_of_edges() > 0

    data, _ = networkx_to_data(g, in_dim=64, seed=42)
    mask = data.edge_index[0] < data.edge_index[1]
    train_pos = data.edge_index[:, mask]

    cfg = TrainConfig(epochs=1, batch_size=64, neg_samples=2, learning_rate=0.01)
    model = BaselineGraphSAGE(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, num_layers=2)

    _, _ = train_unsupervised(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=False,
    )

    z = compute_node_embeddings(model, data, torch.device("cpu"), edge_aware=False)
    assert z.shape[0] == data.num_nodes
    assert z.shape[1] == cfg.out_dim


def test_primekg_edge_aware_smoke():
    kg_path = _get_primekg_path()
    g = load_primekg_csv(kg_path, max_edges=1000)

    data, relation_lookup = networkx_to_data(g, in_dim=64, seed=42)
    mask = data.edge_index[0] < data.edge_index[1]
    train_pos = data.edge_index[:, mask]

    edge_dim = 32
    rel_emb = {k: torch.randn(edge_dim) * 0.1 for k in relation_lookup}
    rt = build_relation_tensor(rel_emb, relation_lookup, edge_dim, torch.device("cpu"))

    cfg = TrainConfig(epochs=1, batch_size=64, neg_samples=2, learning_rate=0.01)
    model = EdgeAwareGraphSAGE(
        cfg.in_dim,
        edge_dim,
        cfg.hidden_dim,
        cfg.out_dim,
        relation_table=rt,
        num_layers=2,
        concat=cfg.concat,
    )

    _, _ = train_unsupervised(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=True,
    )

    z = compute_node_embeddings(model, data, torch.device("cpu"), edge_aware=True)
    assert z.shape[0] == data.num_nodes
    assert z.shape[1] == cfg.out_dim


def test_primekg_link_prediction_eval():
    kg_path = _get_primekg_path()
    g = load_primekg_csv(kg_path, max_edges=2000)

    data, _ = networkx_to_data(g, in_dim=64, seed=42)
    mask = data.edge_index[0] < data.edge_index[1]
    train_pos = data.edge_index[:, mask]

    cfg = TrainConfig(epochs=2, batch_size=128, neg_samples=3, learning_rate=0.01)
    model = BaselineGraphSAGE(cfg.in_dim, cfg.hidden_dim, cfg.out_dim, num_layers=2)

    _, _ = train_unsupervised(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=False,
    )

    z = compute_node_embeddings(model, data, torch.device("cpu"), edge_aware=False)

    n_pos = train_pos.shape[1]
    neg_edge_index = torch.randint(0, data.num_nodes, (2, n_pos))

    metrics = link_prediction_sklearn(z, train_pos, neg_edge_index)
    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1
