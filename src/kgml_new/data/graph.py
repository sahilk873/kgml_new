from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from kgml_new.data.relations import build_relation_lookup, get_edge_relation


def _relation_index(relation_lookup: dict[str, int], rel: str) -> int:
    if rel in relation_lookup:
        return relation_lookup[rel]
    u = rel.upper()
    if u in relation_lookup:
        return relation_lookup[u]
    l = rel.lower()
    if l in relation_lookup:
        return relation_lookup[l]
    return relation_lookup["UNK"]


def _node_features(
    graph: nx.Graph,
    node_list: list,
    feat_dim: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for node in node_list:
        if "feat" in graph.nodes[node]:
            feat = np.asarray(graph.nodes[node]["feat"], dtype=np.float32).ravel()
        else:
            feat = rng.standard_normal(feat_dim).astype(np.float32)
        if feat.size < feat_dim:
            feat = np.pad(feat, (0, feat_dim - feat.size))
        elif feat.size > feat_dim:
            feat = feat[:feat_dim]
        rows.append(feat)
    return torch.from_numpy(np.stack(rows, axis=0))


def networkx_to_data(
    graph: nx.Graph,
    *,
    in_dim: int,
    relation_lookup: dict[str, int] | None = None,
    seed: int = 42,
    add_self_loops: bool = False,
) -> tuple[Data, dict[str, int]]:
    """
    Convert an undirected NetworkX graph to PyG Data.

    - x: node features (random Gaussian if no node['feat']).
    - edge_index: bidirectional edges.
    - edge_attr: long tensor of relation indices per directed edge (UNK=0).

    If relation_lookup is None, builds it from edge types present in the graph.
    """
    node_list = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    if relation_lookup is None:
        keys: set[str] = set()
        for _, _, data in graph.edges(data=True):
            keys.add(get_edge_relation(data))
        rel_keys = sorted(keys)
        relation_lookup, _ = build_relation_lookup(rel_keys)

    sources: list[int] = []
    targets: list[int] = []
    edge_attrs: list[int] = []

    for u, v, data in graph.edges(data=True):
        rel = get_edge_relation(data)
        idx = _relation_index(relation_lookup, rel)
        iu, iv = node_to_idx[u], node_to_idx[v]
        sources.extend([iu, iv])
        targets.extend([iv, iu])
        edge_attrs.extend([idx, idx])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    x = _node_features(graph, node_list, in_dim, seed)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    if add_self_loops:
        from torch_geometric.utils import add_self_loops as pyg_add_self_loops

        edge_index, edge_attr = pyg_add_self_loops(
            edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            fill_value=relation_lookup["UNK"],
        )
        data.edge_index = edge_index
        data.edge_attr = edge_attr

    return data, relation_lookup
