from __future__ import annotations

import networkx as nx
import torch

from kgml_new.config import LinkMLPConfig, Node2VecConfig, TrainConfig
from kgml_new.data.graph import networkx_to_data
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.link_mlp import LinkPredictionMLP
from kgml_new.training.eval import link_prediction_mlp_torch
from kgml_new.training.link_unsupervised import create_train_val_split, train_unsupervised
from kgml_new.training.node2vec_train import (
    train_node2vec_embeddings,
    train_node2vec_embeddings_with_validation,
)
from kgml_new.training.train_link_mlp import (
    train_link_mlp,
    train_link_mlp_with_validation,
)


def test_link_mlp_forward():
    m = LinkPredictionMLP(64, hidden_dims=(32,), dropout=0.0)
    z_src = torch.randn(10, 64)
    z_dst = torch.randn(10, 64)
    out = m(z_src, z_dst)
    assert out.shape == (10,)


def test_node2vec_train_smoke():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")

    data, _ = networkx_to_data(g, in_dim=16, seed=0)
    cfg = Node2VecConfig(epochs=1, batch_size=8, embedding_dim=32)

    _, z, _ = train_node2vec_embeddings(data, cfg, device=torch.device("cpu"))
    assert z.shape[0] == data.num_nodes
    assert z.shape[1] == cfg.embedding_dim


def test_node2vec_train_with_validation_smoke():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")

    data, _ = networkx_to_data(g, in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]
    train_pos, val_pos, val_neg = create_train_val_split(pos, val_ratio=0.34, seed=0)

    cfg = Node2VecConfig(epochs=1, batch_size=8, embedding_dim=32)
    _, z, last_epoch, history = train_node2vec_embeddings_with_validation(
        data,
        cfg,
        device=torch.device("cpu"),
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
    )

    assert last_epoch == 0
    assert z.shape == (data.num_nodes, cfg.embedding_dim)
    assert history.epoch == [0]
    assert len(history.val_auc) == 1
    assert len(history.val_ap) == 1
    assert len(history.val_loss) == 1


def test_train_link_mlp_smoke():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")

    data, _ = networkx_to_data(g, in_dim=16, seed=0)
    cfg = TrainConfig(epochs=1, batch_size=8, neg_samples=2, learning_rate=0.05)
    model = BaselineGraphSAGE(16, 8, 16, num_layers=2)

    mask = data.edge_index[0] < data.edge_index[1]
    train_pos = data.edge_index[:, mask]

    _, _ = train_unsupervised(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=False,
    )

    z = model(data.x, data.edge_index)

    pos = train_pos[:, :1]
    mlp_cfg = LinkMLPConfig(epochs=1, batch_size=8)
    mlp, _ = train_link_mlp(z, pos, mlp_cfg, device=torch.device("cpu"))

    neg = torch.randint(0, data.num_nodes, (2, 1))
    metrics = link_prediction_mlp_torch(mlp, pos, neg, z)
    assert "roc_auc" in metrics
    assert "average_precision" in metrics


def test_train_link_mlp_with_validation_smoke():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")

    data, _ = networkx_to_data(g, in_dim=16, seed=0)
    cfg = TrainConfig(epochs=1, batch_size=8, neg_samples=2, learning_rate=0.05)
    model = BaselineGraphSAGE(16, 8, 16, num_layers=2)

    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]
    train_pos, val_pos, val_neg = create_train_val_split(pos, val_ratio=0.34, seed=0)

    _, _ = train_unsupervised(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=False,
    )

    z = model(data.x, data.edge_index).detach()
    mlp_cfg = LinkMLPConfig(epochs=1, batch_size=8)
    mlp, last_epoch, history = train_link_mlp_with_validation(
        z,
        train_pos,
        mlp_cfg,
        device=torch.device("cpu"),
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
    )

    assert last_epoch == 0
    assert history.epoch == [0]
    assert len(history.train_loss) == 1
    assert len(history.val_auc) == 1
    assert len(history.val_loss) == 1
    metrics = link_prediction_mlp_torch(mlp, val_pos, val_neg, z)
    assert "roc_auc" in metrics
