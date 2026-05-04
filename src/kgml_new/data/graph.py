from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

_LOG = logging.getLogger(__name__)


def _extract_node_features(
    graph: nx.Graph, feature_attr: str | None = None
) -> dict[str, Tensor]:
    if feature_attr is None:
        return {}
    feature_dict = {}
    for node, attrs in graph.nodes(data=True):
        if feature_attr in attrs:
            feature_dict[node] = torch.tensor(attrs[feature_attr])
    return feature_dict


def convert_nx_node_attrs_to_tensor(
    graph: nx.Graph,
    attr: str,
    dtype: type = torch.float32,
) -> dict[int, Tensor]:
    result = {}
    for node, attrs in graph.nodes(data=True):
        if attr in attrs:
            val = attrs[attr]
            if isinstance(val, Sequence):
                result[node] = torch.tensor(val, dtype=dtype)
            else:
                result[node] = torch.tensor([val], dtype=dtype)
    return result


def networkx_to_data(
    graph: nx.Graph,
    *,
    in_dim: int,
    relation_lookup: dict[str, int] | None = None,
    seed: int = 0,
    add_self_loops: bool = False,
) -> tuple[Data, dict[str, int]]:
    """Homogeneous ``nx.Graph`` → PyG ``Data`` with bidirected edges and relation ids."""
    from kgml_new.data.relations import get_edge_relation

    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        raise ValueError("networkx_to_data requires a non-empty graph")

    rel_on_edge: list[str] = []
    for _, _, attrs in graph.edges(data=True):
        rel_on_edge.append(get_edge_relation(attrs))

    lookup = dict(relation_lookup) if relation_lookup is not None else {}
    next_id = max(lookup.values(), default=-1) + 1
    for rel in sorted(set(rel_on_edge)):
        if rel not in lookup:
            lookup[rel] = next_id
            next_id += 1
    if "UNK" not in lookup:
        lookup["UNK"] = next_id
        next_id += 1

    torch.manual_seed(seed)
    x = torch.randn((n, in_dim), dtype=torch.float32)

    src_list: list[int] = []
    dst_list: list[int] = []
    attr_list: list[int] = []
    for u, v, attrs in graph.edges(data=True):
        rel = get_edge_relation(attrs)
        rid = int(lookup[rel])
        ui, vi = node_to_idx[u], node_to_idx[v]
        src_list.extend((ui, vi))
        dst_list.extend((vi, ui))
        attr_list.extend((rid, rid))

    if add_self_loops:
        fallback = int(next(iter(lookup.values())))
        for i in range(n):
            src_list.append(i)
            dst_list.append(i)
            attr_list.append(fallback)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, lookup


def networkx_to_heterodata(
    graph: nx.Graph,
    *,
    in_dim: int,
    seed: int = 0,
    add_reverse_edges: bool = True,
) -> HeteroData:
    """``HeteroData`` keyed by ``node_type``; edge stores use ``(src_type, rel, dst_type)``."""
    from kgml_new.data.relations import get_edge_relation

    type_order: list[str] = []
    type_to_nodes: dict[str, list[Any]] = {}
    for node, attrs in graph.nodes(data=True):
        nt = str(attrs.get("node_type", "entity"))
        if nt not in type_to_nodes:
            type_order.append(nt)
            type_to_nodes[nt] = []
        type_to_nodes[nt].append(node)

    data = HeteroData()
    torch.manual_seed(seed)
    node_key: dict[Any, tuple[str, int]] = {}
    for nt in type_order:
        nodes = type_to_nodes[nt]
        num = len(nodes)
        data[nt].x = torch.randn(num, in_dim, dtype=torch.float32)
        for local_i, node in enumerate(nodes):
            node_key[node] = (nt, local_i)

    triple_to_pairs: dict[tuple[str, str, str], list[list[int]]] = {}
    for u, v, attrs in graph.edges(data=True):
        if u not in node_key or v not in node_key:
            continue
        tu, iu = node_key[u]
        tv, iv = node_key[v]
        rel = get_edge_relation(attrs)
        triple_to_pairs.setdefault((tu, rel, tv), []).append([iu, iv])
        if add_reverse_edges:
            triple_to_pairs.setdefault((tv, rel, tu), []).append([iv, iu])

    for key, pairs in triple_to_pairs.items():
        if not pairs:
            continue
        ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[key].edge_index = ei

    return data