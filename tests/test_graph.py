from __future__ import annotations

import networkx as nx

from kgml_new.data.graph import networkx_to_data


def test_networkx_to_data():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")
    data, relation_lookup = networkx_to_data(g, in_dim=16, seed=0)
    assert data.num_nodes == 3
    assert data.edge_index.shape[1] == 6
    assert data.x.shape == (3, 16)
    assert "T" in relation_lookup
    assert "N" in relation_lookup
