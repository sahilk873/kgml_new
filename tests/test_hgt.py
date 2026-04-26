from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import torch

from kgml_new.data.hetero import networkx_to_heterodata
from kgml_new.models.hgt import HGTLinkPredictor
from kgml_new.scripts.run_hgt import _link_split_for_hgt, _resolve_target_edge_type
from kgml_new.training.hgt_train import evaluate_hgt_relation, neighbor_sampler_backend_available, train_hgt
from kgml_new.training.splits import create_bipartite_node_split


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


def test_hgt_freebase_relation_with_dots_parameterdict():
    """nn.ParameterDict rejects '.'; some FB15k predicates include '.' (e.g. ``...nominations./award...``)."""
    metadata = (
        ["entity"],
        [
            ("entity", "/people/person/profession", "entity"),
            ("entity", "/award/award_nominee/award_nominations./award/award_nomination/award", "entity"),
        ],
    )
    model = HGTLinkPredictor(
        metadata=metadata,
        in_channels=16,
        hidden_channels=16,
        out_channels=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    keys = list(model.rel_emb.keys())
    assert all("." not in k for k in keys)
    assert any("_DOT_" in k for k in keys)


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
        neighbor_sampling=neighbor_sampler_backend_available(),
    )
    assert len(history.train_loss) == 1


def test_hgt_neighbor_sampling_eval_smoke():
    """Chunked LinkNeighborLoader evaluation when sampler backend exists; falls back otherwise."""
    data = networkx_to_heterodata(_toy_graph(), in_dim=16, seed=0)
    edge_type = ("drug", "indication", "disease")
    ei = data[edge_type].edge_index
    n_pos = ei.size(1)
    assert n_pos > 0
    num_src = data["drug"].num_nodes
    num_dst = data["disease"].num_nodes
    torch.manual_seed(0)
    neg = torch.stack(
        [
            torch.randint(0, num_src, (n_pos * 2,)),
            torch.randint(0, num_dst, (n_pos * 2,)),
        ],
        dim=0,
    )
    device = torch.device("cpu")

    model = HGTLinkPredictor(
        metadata=data.metadata(),
        in_channels=16,
        hidden_channels=16,
        out_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    roc, ap = evaluate_hgt_relation(
        model,
        data,
        edge_type,
        ei,
        neg,
        device,
        neighbor_sampling=True,
        eval_batch_size_pos=2,
        num_layers=2,
    )
    assert isinstance(roc, float)
    assert isinstance(ap, float)


def test_hgt_train_explicit_no_neighbor_sampling():
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
        neighbor_sampling=False,
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


def _toy_homogeneous_graph() -> nx.Graph:
    g = nx.Graph()
    for i in range(8):
        g.add_node(f"e{i}", node_type="entity")
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]:
        g.add_edge(f"e{a}", f"e{b}", relationship="R")
    return g


def _toy_bipartite_dense() -> nx.Graph:
    g = nx.Graph()
    for i in range(6):
        g.add_node(f"d{i}", node_type="drug")
        g.add_node(f"z{i}", node_type="disease")
    for i in range(6):
        for j in range(6):
            g.add_edge(f"d{i}", f"z{j}", relationship="indication")
    return g


def test_link_split_bipartite_node():
    g = _toy_bipartite_dense()
    data = networkx_to_heterodata(g, in_dim=8, seed=0)
    args = SimpleNamespace(
        split_protocol="node",
        val_ratio=0.1,
        test_ratio=0.1,
        seed=123,
        negative_sampling_mode="global",
    )
    edge_type = ("drug", "indication", "disease")
    tp, vp, vn, tep, tn, bipartite_ns = _link_split_for_hgt(
        args=args, data=data, edge_type=edge_type, graph=g
    )
    assert bipartite_ns is True
    assert tp.size(1) > 0
    assert vn.size(1) > 0


def test_link_split_homogeneous_node():
    g = _toy_homogeneous_graph()
    data = networkx_to_heterodata(g, in_dim=8, seed=0)
    args = SimpleNamespace(
        split_protocol="node",
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        negative_sampling_mode="global",
    )
    edge_type = ("entity", "R", "entity")
    tp, vp, vn, tep, tn, bipartite_ns = _link_split_for_hgt(
        args=args, data=data, edge_type=edge_type, graph=g
    )
    assert bipartite_ns is False
    assert tp.size(1) > 0


def test_create_bipartite_node_split_returns_edgesplit():
    ei = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 0]], dtype=torch.long)
    split = create_bipartite_node_split(
        ei,
        num_src_nodes=6,
        num_dst_nodes=6,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=0,
        negatives_per_pos=2,
        negative_sampling_mode="global",
    )
    assert split.train_pos_edge_index.shape[0] == 2
    assert split.val_neg_edge_index.shape[0] == 2
