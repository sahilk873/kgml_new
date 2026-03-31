from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from kgml_new.data.relations import get_edge_relation


def networkx_to_heterodata(
    graph: nx.Graph,
    *,
    in_dim: int,
    seed: int = 42,
    add_reverse_edges: bool = True,
) -> HeteroData:
    """
    Convert a typed NetworkX graph into PyG HeteroData.

    Expected node attribute:
      - node_type (defaults to "entity")
      - feat (optional feature vector)

    Expected edge attribute:
      - relationship / predicate / relationship_type / label / edge_type
    """
    rng = np.random.default_rng(seed)
    data = HeteroData()

    nodes_by_type: dict[str, list[object]] = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("node_type", "entity"))
        nodes_by_type[node_type].append(node)

    local_index: dict[tuple[str, object], int] = {}
    for node_type, nodes in nodes_by_type.items():
        rows: list[np.ndarray] = []
        for idx, node in enumerate(nodes):
            local_index[(node_type, node)] = idx
            attrs = graph.nodes[node]
            feat = attrs.get("feat")
            if feat is None:
                vec = rng.standard_normal(in_dim).astype(np.float32)
            else:
                vec = np.asarray(feat, dtype=np.float32).ravel()
                if vec.size < in_dim:
                    vec = np.pad(vec, (0, in_dim - vec.size))
                elif vec.size > in_dim:
                    vec = vec[:in_dim]
            rows.append(vec)
        data[node_type].x = torch.from_numpy(np.stack(rows, axis=0))

    edge_buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for src, dst, attrs in graph.edges(data=True):
        src_type = str(graph.nodes[src].get("node_type", "entity"))
        dst_type = str(graph.nodes[dst].get("node_type", "entity"))
        relation = get_edge_relation(attrs)
        src_idx = local_index[(src_type, src)]
        dst_idx = local_index[(dst_type, dst)]
        edge_buckets[(src_type, relation, dst_type)].append((src_idx, dst_idx))
        if add_reverse_edges:
            edge_buckets[(dst_type, f"rev_{relation}", src_type)].append((dst_idx, src_idx))

    for edge_type, pairs in edge_buckets.items():
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[edge_type].edge_index = edge_index

    return data
