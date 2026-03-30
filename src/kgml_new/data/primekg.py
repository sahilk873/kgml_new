from __future__ import annotations

import networkx as nx
import pandas as pd
from pathlib import Path


def load_primekg_csv(
    csv_path: Path,
    *,
    max_edges: int | None = None,
    relation_col: str = "relation",
    source_col: str = "x_name",
    target_col: str = "y_name",
) -> nx.Graph:
    """
    Load a subset of PrimeKG from CSV into a NetworkX graph.

    Args:
        csv_path: Path to PrimeKG CSV file
        max_edges: Maximum number of edges to load (None = all)
        relation_col: Column name for relation type
        source_col: Column name for source node
        target_col: Column name for target node
    """
    df = pd.read_csv(csv_path)

    if max_edges is not None:
        df = df.head(max_edges)

    g = nx.Graph()

    for _, row in df.iterrows():
        src = str(row[source_col])
        tgt = str(row[target_col])
        rel = str(row[relation_col])

        g.add_node(src, node_type=row.get("x_type", "unknown"))
        g.add_node(tgt, node_type=row.get("y_type", "unknown"))
        g.add_edge(src, tgt, relationship=rel)

    return g
