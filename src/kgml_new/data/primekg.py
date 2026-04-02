from __future__ import annotations

from pathlib import Path

import networkx as nx

from kgml_new.data.loaders import PRIMEKG_CSV_SPEC, load_graph_csv


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
    spec = PRIMEKG_CSV_SPEC
    if (
        relation_col != PRIMEKG_CSV_SPEC.relation_col
        or source_col != PRIMEKG_CSV_SPEC.source_col
        or target_col != PRIMEKG_CSV_SPEC.target_col
    ):
        spec = type(PRIMEKG_CSV_SPEC)(
            source_col=source_col,
            target_col=target_col,
            relation_col=relation_col,
            source_type_col=PRIMEKG_CSV_SPEC.source_type_col,
            target_type_col=PRIMEKG_CSV_SPEC.target_type_col,
            source_id_col=PRIMEKG_CSV_SPEC.source_id_col,
            target_id_col=PRIMEKG_CSV_SPEC.target_id_col,
            node_type_attr=PRIMEKG_CSV_SPEC.node_type_attr,
            relation_attr=PRIMEKG_CSV_SPEC.relation_attr,
            default_node_type=PRIMEKG_CSV_SPEC.default_node_type,
            edge_attr_cols=PRIMEKG_CSV_SPEC.edge_attr_cols,
            directed=PRIMEKG_CSV_SPEC.directed,
        )
    return load_graph_csv(csv_path, spec=spec, max_edges=max_edges)
