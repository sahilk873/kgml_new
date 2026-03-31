from __future__ import annotations

import networkx as nx
import torch

from kgml_new.config import TrainConfig
from kgml_new.data.graph import networkx_to_data
from kgml_new.embeddings.semantic import build_relation_tensor
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.link_unsupervised import (
    create_train_val_split,
    train_unsupervised,
    train_unsupervised_batched,
    train_unsupervised_fullgraph,
)
from kgml_new.training.splits import build_train_graph_data, create_edge_split


def test_train_unsupervised_runs():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")

    data, _ = networkx_to_data(g, in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    train_pos = data.edge_index[:, mask]

    cfg = TrainConfig(epochs=2, batch_size=8, neg_samples=2, learning_rate=0.05)
    model = BaselineGraphSAGE(16, 8, 16, num_layers=2)
    _, _ = train_unsupervised(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=False,
    )


def _toy_graph():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")
    g.add_edge("c", "d", relationship="X")
    return g


def test_create_train_val_split_shapes():
    data, _ = networkx_to_data(_toy_graph(), in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]

    train_pos, val_pos, val_neg = create_train_val_split(pos, val_ratio=0.5, seed=0)

    assert train_pos.shape[0] == 2
    assert val_pos.shape[0] == 2
    assert val_neg.shape[0] == 2
    assert train_pos.shape[1] + val_pos.shape[1] == pos.shape[1]
    assert val_neg.shape[1] == val_pos.shape[1]


def test_create_edge_split_negatives_are_true_non_edges():
    data, _ = networkx_to_data(_toy_graph(), in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]

    split = create_edge_split(
        pos,
        num_src_nodes=int(data.num_nodes),
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        undirected=True,
    )

    positive_pairs = {
        tuple(sorted((int(pos[0, i]), int(pos[1, i]))))
        for i in range(pos.size(1))
    }
    for neg_edges in (split.val_neg_edge_index, split.test_neg_edge_index):
        for i in range(neg_edges.size(1)):
            pair = tuple(sorted((int(neg_edges[0, i]), int(neg_edges[1, i]))))
            assert pair not in positive_pairs
            assert pair[0] != pair[1]


def test_build_train_graph_data_excludes_held_out_edges():
    data, _ = networkx_to_data(_toy_graph(), in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]

    split = create_edge_split(
        pos,
        num_src_nodes=int(data.num_nodes),
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
        undirected=True,
    )
    train_data = build_train_graph_data(data, split.train_pos_edge_index)

    directed_pairs = {
        (int(train_data.edge_index[0, i]), int(train_data.edge_index[1, i]))
        for i in range(train_data.edge_index.size(1))
    }
    for held_out in (split.val_pos_edge_index, split.test_pos_edge_index):
        for i in range(held_out.size(1)):
            src = int(held_out[0, i])
            dst = int(held_out[1, i])
            assert (src, dst) not in directed_pairs
            assert (dst, src) not in directed_pairs


def test_train_unsupervised_fullgraph_with_validation_smoke(tmp_path):
    data, _ = networkx_to_data(_toy_graph(), in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]
    train_pos, val_pos, val_neg = create_train_val_split(pos, val_ratio=0.25, seed=0)

    cfg = TrainConfig(epochs=2, batch_size=8, neg_samples=2, learning_rate=0.05)
    model = BaselineGraphSAGE(16, 8, 16, num_layers=2)
    history_path = tmp_path / "fullgraph_history.pkl"

    _, last_epoch, history = train_unsupervised_fullgraph(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=False,
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
        history_path=history_path,
    )

    assert last_epoch == 1
    assert history.epoch == [0, 1]
    assert len(history.train_loss) == 2
    assert len(history.val_auc) == 2
    assert len(history.val_ap) == 2
    assert history_path.exists()


def test_train_unsupervised_edge_aware_with_validation_smoke():
    data, relation_lookup = networkx_to_data(_toy_graph(), in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]
    train_pos, val_pos, val_neg = create_train_val_split(pos, val_ratio=0.25, seed=0)

    edge_dim = 8
    torch.manual_seed(0)
    rel_emb = {k: torch.randn(edge_dim) * 0.1 for k in relation_lookup}
    relation_table = build_relation_tensor(
        rel_emb, relation_lookup, edge_dim, torch.device("cpu")
    )

    cfg = TrainConfig(
        in_dim=16,
        edge_dim=edge_dim,
        hidden_dim=8,
        out_dim=16,
        epochs=2,
        batch_size=8,
        neg_samples=2,
        learning_rate=0.05,
    )
    model = EdgeAwareGraphSAGE(
        16, edge_dim, 8, 16, relation_table=relation_table, num_layers=2, concat=True
    )

    _, last_epoch, history = train_unsupervised_fullgraph(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=True,
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
    )

    assert last_epoch == 1
    assert len(history.val_auc) == 2


def test_train_unsupervised_batched_with_validation_smoke():
    data, _ = networkx_to_data(_toy_graph(), in_dim=16, seed=0)
    mask = data.edge_index[0] < data.edge_index[1]
    pos = data.edge_index[:, mask]
    train_pos, val_pos, val_neg = create_train_val_split(pos, val_ratio=0.25, seed=0)

    cfg = TrainConfig(
        epochs=1,
        batch_size=2,
        neg_samples=1,
        learning_rate=0.05,
        num_neighbors=[2, 2],
    )
    model = BaselineGraphSAGE(16, 8, 16, num_layers=2)

    _, last_epoch, history = train_unsupervised_batched(
        model,
        data,
        train_pos,
        cfg,
        device=torch.device("cpu"),
        edge_aware=False,
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
    )

    assert last_epoch == 0
    assert history.epoch == [0]
    assert len(history.train_loss) == 1
