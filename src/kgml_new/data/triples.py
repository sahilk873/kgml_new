from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data

from kgml_new.data.loaders import GraphCSVSpec, _node_type_from_name_or_default


@dataclass
class TripleKGDataset:
    node_names: list[str]
    node_types: list[str]
    relation_lookup: dict[str, int]
    triples: torch.Tensor
    train_triples: torch.Tensor
    val_triples: torch.Tensor
    test_triples: torch.Tensor
    full_data: Data
    train_data: Data
    split_protocol: str
    val_ratio: float
    test_ratio: float


def _read_triple_rows(
    path: str | Path,
    *,
    spec: GraphCSVSpec,
    max_edges: int | None,
) -> pd.DataFrame:
    path = Path(path)
    probe = pd.read_csv(path, nrows=1)
    usecols = {
        spec.source_col,
        spec.target_col,
        spec.relation_col,
        spec.source_type_col,
        spec.target_type_col,
    }
    if spec.source_col in probe.columns and spec.target_col in probe.columns:
        frames = []
        seen = 0
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=500_000):
            if max_edges is not None:
                remaining = max_edges - seen
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)
            frames.append(chunk)
            seen += len(chunk)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    names = [spec.source_col, spec.relation_col, spec.target_col]
    frames = []
    seen = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=names,
        usecols=list(range(len(names))),
        chunksize=500_000,
    ):
        if max_edges is not None:
            remaining = max_edges - seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        frames.append(chunk)
        seen += len(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _rows_from_graph(graph: nx.Graph, *, relation_attr: str = "relationship") -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for u, v, attrs in graph.edges(data=True):
        src = str(u)
        dst = str(v)
        rel = str(attrs.get(relation_attr, attrs.get("predicate", attrs.get("edge_type", "UNK"))))
        src_type = str(graph.nodes[u].get("node_type", "entity"))
        dst_type = str(graph.nodes[v].get("node_type", "entity"))
        rows.append((src, rel, dst, src_type, dst_type))
    return rows


def load_triple_kg_dataset(
    source: str | Path | nx.Graph,
    *,
    spec: GraphCSVSpec,
    in_dim: int,
    seed: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_protocol: str = "edge",
    max_edges: int | None = None,
) -> TripleKGDataset:
    if split_protocol not in {"edge", "node", "node_category"}:
        raise ValueError(f"Unsupported KG completion split_protocol: {split_protocol}")
    if split_protocol == "node_category":
        raise ValueError("KG completion does not yet support split_protocol=node_category")

    if isinstance(source, nx.Graph):
        raw_rows = _rows_from_graph(source, relation_attr=spec.relation_attr)
        if max_edges is not None:
            raw_rows = raw_rows[:max_edges]
    else:
        df = _read_triple_rows(source, spec=spec, max_edges=max_edges)
        raw_rows = []
        for row in df.itertuples(index=False):
            rec: dict[str, Any] = row._asdict()
            src = str(rec[spec.source_col])
            dst = str(rec[spec.target_col])
            rel = str(rec.get(spec.relation_col, "UNK"))
            src_type = (
                str(rec.get(spec.source_type_col, spec.default_node_type))
                if spec.source_type_col in rec and pd.notna(rec.get(spec.source_type_col))
                else _node_type_from_name_or_default(src, default=spec.default_node_type)
            )
            dst_type = (
                str(rec.get(spec.target_type_col, spec.default_node_type))
                if spec.target_type_col in rec and pd.notna(rec.get(spec.target_type_col))
                else _node_type_from_name_or_default(dst, default=spec.default_node_type)
            )
            raw_rows.append((src, rel, dst, src_type, dst_type))

    if not raw_rows:
        raise ValueError("KG completion requires at least one triple")

    node_names: list[str] = []
    node_to_idx: dict[str, int] = {}
    node_types_map: dict[str, str] = {}
    for src, _, dst, src_type, dst_type in raw_rows:
        for name, ntype in ((src, src_type), (dst, dst_type)):
            if name not in node_to_idx:
                node_to_idx[name] = len(node_names)
                node_names.append(name)
                node_types_map[name] = ntype

    relations = sorted({rel for _, rel, _, _, _ in raw_rows})
    relation_lookup = {rel: i for i, rel in enumerate(relations)}
    triples = torch.tensor(
        [[node_to_idx[src], relation_lookup[rel], node_to_idx[dst]] for src, rel, dst, _, _ in raw_rows],
        dtype=torch.long,
    )

    train_idx, val_idx, test_idx = _split_triple_indices(
        triples,
        num_nodes=len(node_names),
        split_protocol=split_protocol,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    node_types = [node_types_map[name] for name in node_names]
    x = torch.randn((len(node_names), in_dim), generator=torch.Generator().manual_seed(seed))
    full_data = _triples_to_data(x, triples)
    train_data = _triples_to_data(x, triples[train_idx])
    return TripleKGDataset(
        node_names=node_names,
        node_types=node_types,
        relation_lookup=relation_lookup,
        triples=triples,
        train_triples=triples[train_idx],
        val_triples=triples[val_idx],
        test_triples=triples[test_idx],
        full_data=full_data,
        train_data=train_data,
        split_protocol=split_protocol,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def _split_triple_indices(
    triples: torch.Tensor,
    *,
    num_nodes: int,
    split_protocol: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    if split_protocol == "edge":
        perm = torch.randperm(triples.size(0), generator=gen)
        n_val = max(1, int(triples.size(0) * val_ratio)) if val_ratio > 0 else 0
        n_test = max(1, int(triples.size(0) * test_ratio)) if test_ratio > 0 and triples.size(0) - n_val > 1 else 0
        if n_val + n_test >= triples.size(0) and triples.size(0) > 1:
            n_test = max(0, triples.size(0) - n_val - 1)
        return perm[n_val + n_test :], perm[:n_val], perm[n_val : n_val + n_test]

    node_perm = torch.randperm(num_nodes, generator=gen)
    n_val = max(1, int(num_nodes * val_ratio)) if val_ratio > 0 else 0
    n_test = max(1, int(num_nodes * test_ratio)) if test_ratio > 0 and num_nodes - n_val > 1 else 0
    if n_val + n_test >= num_nodes and num_nodes > 1:
        n_test = max(0, num_nodes - n_val - 1)
    val_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    test_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    val_nodes[node_perm[:n_val]] = True
    test_nodes[node_perm[n_val : n_val + n_test]] = True
    h, t = triples[:, 0], triples[:, 2]
    test_mask = test_nodes[h] | test_nodes[t]
    val_mask = ~test_mask & (val_nodes[h] | val_nodes[t])
    train_mask = ~(test_mask | val_mask)
    return torch.where(train_mask)[0], torch.where(val_mask)[0], torch.where(test_mask)[0]


def _triples_to_data(x: torch.Tensor, triples: torch.Tensor) -> Data:
    data = Data(x=x.clone(), num_nodes=x.size(0))
    if triples.numel() == 0:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_attr = torch.empty((0,), dtype=torch.long)
        return data
    edge_index = torch.stack([triples[:, 0], triples[:, 2]], dim=0)
    rev_edge_index = edge_index.flip(0)
    data.edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    data.edge_attr = torch.cat([triples[:, 1], triples[:, 1]], dim=0)
    return data
