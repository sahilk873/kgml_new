from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class GraphCSVSpec:
    source_col: str
    target_col: str
    relation_col: str = "relation"
    source_type_col: str | None = None
    target_type_col: str | None = None
    source_id_col: str | None = None
    target_id_col: str | None = None
    node_type_attr: str = "node_type"
    relation_attr: str = "relationship"
    default_node_type: str = "entity"
    edge_attr_cols: tuple[str, ...] = field(default_factory=tuple)
    directed: bool = False
    delimiter: str | None = None


PRIMEKG_CSV_SPEC = GraphCSVSpec(
    source_col="x_name",
    target_col="y_name",
    relation_col="relation",
    source_type_col="x_type",
    target_type_col="y_type",
    source_id_col="x_id",
    target_id_col="y_id",
    edge_attr_cols=("display_relation", "x_source", "y_source"),
)


def load_pickled_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, nx.Graph):
        raise TypeError(f"Expected networkx.Graph, got {type(obj)}")
    return obj


def load_graph_csv(
    csv_path: Path,
    *,
    spec: GraphCSVSpec,
    max_edges: int | None = None,
    low_memory: bool = False,
) -> nx.Graph:
    graph_cls = nx.DiGraph if spec.directed else nx.Graph
    graph = graph_cls()
    df = _read_graph_table(csv_path, spec=spec, low_memory=low_memory)

    if max_edges is not None:
        df = df.head(max_edges)

    for _, row in df.iterrows():
        src = str(row[spec.source_col])
        dst = str(row[spec.target_col])
        relation = str(row[spec.relation_col]).strip()

        src_attrs = {spec.node_type_attr: spec.default_node_type}
        dst_attrs = {spec.node_type_attr: spec.default_node_type}
        if spec.source_type_col:
            src_attrs[spec.node_type_attr] = str(row.get(spec.source_type_col, spec.default_node_type))
        else:
            inferred_src_type = _infer_node_type(src)
            if inferred_src_type is not None:
                src_attrs[spec.node_type_attr] = inferred_src_type
        if spec.target_type_col:
            dst_attrs[spec.node_type_attr] = str(row.get(spec.target_type_col, spec.default_node_type))
        else:
            inferred_dst_type = _infer_node_type(dst)
            if inferred_dst_type is not None:
                dst_attrs[spec.node_type_attr] = inferred_dst_type
        if spec.source_id_col:
            src_attrs["node_id"] = str(row.get(spec.source_id_col, src))
        if spec.target_id_col:
            dst_attrs["node_id"] = str(row.get(spec.target_id_col, dst))

        graph.add_node(src, **src_attrs)
        graph.add_node(dst, **dst_attrs)

        edge_attrs = {spec.relation_attr: relation}
        for col in spec.edge_attr_cols:
            value = row.get(col)
            if value is not None and not pd.isna(value):
                edge_attrs[col] = value
        graph.add_edge(src, dst, **edge_attrs)

    return graph


def _infer_delimiter(path: Path, explicit_delimiter: str | None) -> str:
    if explicit_delimiter:
        return explicit_delimiter
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    if "\t" in first_line and "," not in first_line:
        return "\t"
    return ","


def _has_header(path: Path, *, delimiter: str, spec: GraphCSVSpec) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return False
    tokens = {token.strip() for token in first_line.split(delimiter)}
    expected = {
        spec.source_col,
        spec.target_col,
        spec.relation_col,
    }
    if spec.source_type_col:
        expected.add(spec.source_type_col)
    if spec.target_type_col:
        expected.add(spec.target_type_col)
    return len(tokens & expected) >= 2


def _read_graph_table(
    csv_path: Path,
    *,
    spec: GraphCSVSpec,
    low_memory: bool,
) -> pd.DataFrame:
    delimiter = _infer_delimiter(csv_path, spec.delimiter)
    if _has_header(csv_path, delimiter=delimiter, spec=spec):
        return pd.read_csv(csv_path, sep=delimiter, low_memory=low_memory)

    preview = pd.read_csv(csv_path, sep=delimiter, header=None, nrows=5, low_memory=low_memory)
    if preview.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns in headerless edge list, found {preview.shape[1]} in {csv_path}"
        )

    base_names = [spec.source_col, spec.relation_col, spec.target_col]
    extra_names = [f"extra_{idx}" for idx in range(preview.shape[1] - len(base_names))]
    column_names = base_names + extra_names
    return pd.read_csv(
        csv_path,
        sep=delimiter,
        header=None,
        names=column_names,
        low_memory=low_memory,
    )


def _infer_node_type(node_name: str) -> str | None:
    if "::" not in node_name:
        return None
    prefix, _ = node_name.split("::", 1)
    prefix = prefix.strip()
    if not prefix:
        return None
    return prefix.lower()
