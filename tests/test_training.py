from __future__ import annotations

import networkx as nx
import torch

from kgml_new.config import TrainConfig
from kgml_new.data.graph import networkx_to_data
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.training.link_unsupervised import train_unsupervised


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
