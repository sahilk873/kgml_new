from __future__ import annotations

import networkx as nx
import torch

from kgml_new.data.hetero import networkx_to_heterodata
from kgml_new.models.hgt import HGTLinkPredictor
from kgml_new.scripts.run_hgt import _resolve_target_edge_type
from kgml_new.training.hgt_train import train_hgt


def _toy_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("d1", node_type="drug")
    g.add_node("d2", node_type="drug")
    g.add_node("dz1", node_type="disease")
    g.add_node("dz2", node_type="disease")
    g.add_edge("d1", "dz1", relationship="indication")
    g.add_edge("d2", "dz1", relationship="contraindication")
    g.add_edge("d2", "dz2", relationship="indication")
    return g


def test_hgt_forward_and_training_smoke():
    data = networkx_to_heterodata(_toy_graph(), in_dim=16, seed=0)
    model = HGTLinkPredictor(
        metadata=data.metadata(),
        in_channels=16,
        hidden_channels=16,
        out_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    z_dict = model.encode(data)
    assert z_dict["drug"].shape[1] == 16
    assert z_dict["disease"].shape[1] == 16

    edge_type = ("drug", "indication", "disease")
    history = train_hgt(
        model,
        data,
        edge_type,
        epochs=1,
        batch_size=2,
        neg_samples=1,
        learning_rate=1e-2,
        device=torch.device("cpu"),
    )
    assert len(history.train_loss) == 1


def test_resolve_target_edge_type_hgt():
    class Args:
        relation = "indication"
        source_node_type = None
        target_node_type = None

    edge_type = _resolve_target_edge_type(
        Args(),
        [
            ("compound", "indication", "condition"),
            ("condition", "rev_indication", "compound"),
        ],
    )

    assert edge_type == ("compound", "indication", "condition")
