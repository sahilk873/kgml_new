from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class GraphCSVSpec:
    source_col: str = "x_name"
    target_col: str = "y_name"
    relation_col: str = "display_relation"
    source_type_col: str = "x_type"
    target_type_col: str = "y_type"
    source_id_col: str = "x_id"
    target_id_col: str = "y_id"
    node_type_attr: str = "node_type"
    relation_attr: str = "relationship"
    default_node_type: str = "entity"
    edge_attr_cols: tuple[str, ...] = field(default_factory=tuple)
    directed: bool = False
    delimiter: str | None = None


PRIMEKG_CSV_SPEC = GraphCSVSpec()


def _node_type_from_name_or_default(name: str, *, default: str) -> str:
    """DRKG-style ``Gene::2157`` → ``gene``; otherwise ``default``."""
    s = str(name)
    if "::" in s:
        return s.split("::", 1)[0].strip().lower()
    return default


def load_graph_csv(
    path: str | Path,
    *,
    spec: GraphCSVSpec = PRIMEKG_CSV_SPEC,
    max_edges: int | None = None,
) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if spec.directed else nx.Graph()
    usecols = {
        spec.source_col,
        spec.target_col,
        spec.relation_col,
        spec.source_type_col,
        spec.target_type_col,
    }
    usecols.update(spec.edge_attr_cols)
    path = Path(path)
    probe = pd.read_csv(path, nrows=1)
    if spec.source_col in probe.columns and spec.target_col in probe.columns:
        df_iter = pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=500_000)
    else:
        # Headerless TSV: source TAB relation TAB target (column names from ``spec``).
        names = [spec.source_col, spec.relation_col, spec.target_col]
        df_iter = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=names,
            usecols=list(range(len(names))),
            chunksize=500_000,
        )
    seen = 0
    for df in df_iter:
        if max_edges is not None:
            remaining = max_edges - seen
            if remaining <= 0:
                break
            df = df.head(remaining)
        for row in df.itertuples(index=False):
            rec = row._asdict()
            src = str(rec[spec.source_col])
            dst = str(rec[spec.target_col])
            if spec.source_type_col in rec and rec.get(spec.source_type_col) is not None:
                src_type = str(rec.get(spec.source_type_col, spec.default_node_type))
            else:
                src_type = _node_type_from_name_or_default(src, default=spec.default_node_type)
            if spec.target_type_col in rec and rec.get(spec.target_type_col) is not None:
                dst_type = str(rec.get(spec.target_type_col, spec.default_node_type))
            else:
                dst_type = _node_type_from_name_or_default(dst, default=spec.default_node_type)
            rel = str(rec.get(spec.relation_col, "UNK"))
            graph.add_node(src, **{spec.node_type_attr: src_type})
            graph.add_node(dst, **{spec.node_type_attr: dst_type})
            attrs: dict[str, Any] = {spec.relation_attr: rel}
            for col in spec.edge_attr_cols:
                if col in rec:
                    attrs[col] = rec[col]
            graph.add_edge(src, dst, **attrs)
        seen += len(df)
    return graph


def load_primekg_csv(path: str | Path, *, max_edges: int | None = None) -> nx.Graph:
    """Load PrimeKG ``kg.csv`` (columns match ``PRIMEKG_CSV_SPEC``)."""
    return load_graph_csv(path, spec=PRIMEKG_CSV_SPEC, max_edges=max_edges)


def load_pickled_graph(path: str | Path) -> nx.Graph:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, nx.Graph):
        raise TypeError(f"Expected a networkx graph in {path}, got {type(obj)}")
    return obj
