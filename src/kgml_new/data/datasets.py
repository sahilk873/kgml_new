from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

from kgml_new.data.graph import networkx_to_data, networkx_to_heterodata
from kgml_new.training.splits import (
    EdgeSplit,
    NodeSplit,
    build_train_graph_data,
    create_edge_split,
    create_edge_split_relation_holdout,
    create_node_category_split,
    create_node_split,
    create_node_split_relation_holdout,
)

_LOG = logging.getLogger(__name__)


@dataclass
class LinkPredictionDataset:
    graph: nx.Graph
    data: Data
    full_data: Data
    relation_lookup: dict[str, int]
    split: EdgeSplit | NodeSplit
    split_protocol: str
    negative_sampling_mode: str
    negatives_per_pos: int
    decoder: str
    shuffle_relations: bool
    node_list: list
    node_types: list[str]
    train_data: Data
    positive_edge_index: torch.Tensor
    positive_edge_attr: torch.Tensor | None
    train_pos_edge_attr: torch.Tensor | None
    held_out_relations: frozenset[str] | None = None
    held_out_relation_ids: frozenset[int] | None = None
    held_out_node_categories: frozenset[str] | None = None


def _shuffle_graph_relations(
    graph: nx.Graph,
    *,
    relation_attr: str = "relationship",
    seed: int,
) -> nx.Graph:
    shuffled = graph.copy()
    edge_list = list(shuffled.edges())
    relations = [str(shuffled[u][v].get(relation_attr, "UNK")) for u, v in edge_list]
    if len(relations) <= 1:
        return shuffled

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(len(relations), generator=generator).tolist()
    shuffled_relations = [relations[idx] for idx in perm]
    for (u, v), relation in zip(edge_list, shuffled_relations, strict=False):
        shuffled[u][v][relation_attr] = relation
    return shuffled


def compute_relation_diversity_buckets(
    graph: nx.Graph,
    node_list: list,
    *,
    relation_attr: str = "relationship",
) -> tuple[dict[int, int], dict[int, str]]:
    diversity_counts: dict[int, int] = {}
    diversity_buckets: dict[int, str] = {}
    for node_idx, node in enumerate(node_list):
        relation_types = {
            str(graph[node][neighbor].get(relation_attr, "UNK"))
            for neighbor in graph.neighbors(node)
        }
        diversity = len(relation_types)
        diversity_counts[node_idx] = diversity
        if diversity <= 2:
            bucket = "low"
        elif diversity <= 5:
            bucket = "medium"
        else:
            bucket = "high"
        diversity_buckets[node_idx] = bucket
    return diversity_counts, diversity_buckets


@dataclass
class NodeClassificationDataset:
    graph: nx.Graph
    data: Data
    relation_lookup: dict[str, int]
    label_lookup: dict[str, int]
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


@dataclass
class HeteroNodeClassificationDataset:
    graph: nx.Graph
    data: HeteroData
    target_node_type: str
    label_lookup: dict[str, int]
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


def _allocate_stratified_counts(
    class_size: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if class_size <= 0:
        return 0, 0, 0
    if class_size == 1:
        return 1, 0, 0
    if class_size == 2:
        if test_ratio > 0:
            return 1, 0, 1
        return 1, 1, 0

    ratios = [train_ratio, val_ratio, test_ratio]
    counts = [int(class_size * ratio) for ratio in ratios]
    remainder = class_size - sum(counts)
    fractional_order = sorted(
        range(3),
        key=lambda idx: (class_size * ratios[idx]) - counts[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        counts[idx] += 1

    if counts[0] == 0:
        donor = next((idx for idx in (1, 2) if counts[idx] > 0), None)
        if donor is not None:
            counts[donor] -= 1
        counts[0] += 1

    for idx, ratio in ((1, val_ratio), (2, test_ratio)):
        if ratio <= 0 or counts[idx] > 0:
            continue
        donor = next((candidate for candidate in (0, 3 - idx) if counts[candidate] > 1), None)
        if donor is not None:
            counts[donor] -= 1
            counts[idx] += 1

    return counts[0], counts[1], counts[2]


def _stratified_split_labeled_indices(
    *,
    y: torch.Tensor,
    labeled_idx: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not labeled_idx:
        raise ValueError("No labeled nodes available for splitting")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    by_label: dict[int, list[int]] = {}
    for idx in labeled_idx:
        label = int(y[idx])
        if label < 0:
            continue
        by_label.setdefault(label, []).append(idx)

    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for label in sorted(by_label):
        label_indices = torch.tensor(by_label[label], dtype=torch.long)
        perm = torch.randperm(label_indices.numel(), generator=generator)
        label_indices = label_indices[perm]
        num_train, num_val, num_test = _allocate_stratified_counts(
            label_indices.numel(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_parts.append(label_indices[:num_train])
        val_parts.append(label_indices[num_train : num_train + num_val])
        test_parts.append(label_indices[num_train + num_val : num_train + num_val + num_test])

    def _finalize(parts: list[torch.Tensor]) -> torch.Tensor:
        if not parts:
            return torch.empty(0, dtype=torch.long)
        non_empty = [part for part in parts if part.numel() > 0]
        if not non_empty:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(non_empty).sort().values

    return _finalize(train_parts), _finalize(val_parts), _finalize(test_parts)


def _undirected_positive_edges(data: Data) -> tuple[torch.Tensor, torch.Tensor | None]:
    mask = data.edge_index[0] < data.edge_index[1]
    pos_edge_index = data.edge_index[:, mask].clone()
    pos_edge_attr = data.edge_attr[mask].clone() if hasattr(data, "edge_attr") else None
    return pos_edge_index, pos_edge_attr


def resolve_held_out_relation_ids(
    held_out_relations: list[str],
    relation_lookup: dict[str, int],
) -> tuple[frozenset[int], list[str], list[str]]:
    """
    Map user-provided relation names to ids in ``relation_lookup``.

    Returns ``(ids, resolved_names, unknown_names)``.
    """
    ids: set[int] = set()
    resolved_names: list[str] = []
    unknown: list[str] = []
    for raw in held_out_relations:
        name = str(raw).strip()
        if not name:
            continue
        rid: int | None = None
        resolved_key: str | None = None
        for candidate in (name, name.lower(), name.upper()):
            if candidate in relation_lookup:
                rid = int(relation_lookup[candidate])
                resolved_key = candidate
                break
        if rid is None:
            unknown.append(name)
            continue
        ids.add(rid)
        resolved_names.append(resolved_key or name)
    return frozenset(ids), resolved_names, unknown


def _select_train_edge_attr(
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor | None,
    train_pos_edge_index: torch.Tensor,
) -> torch.Tensor | None:
    if positive_edge_attr is None:
        return None

    key_to_idx = {
        (int(positive_edge_index[0, i]), int(positive_edge_index[1, i])): i
        for i in range(positive_edge_index.size(1))
    }
    train_indices = [
        key_to_idx[(int(train_pos_edge_index[0, i]), int(train_pos_edge_index[1, i]))]
        for i in range(train_pos_edge_index.size(1))
    ]
    return positive_edge_attr[torch.tensor(train_indices, dtype=torch.long)]


def prepare_link_prediction_dataset(
    graph: nx.Graph,
    *,
    in_dim: int,
    seed: int = 42,
    relation_lookup: dict[str, int] | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_protocol: str = "node",
    negative_sampling_mode: str | None = None,
    negatives_per_pos: int = 20,
    decoder: str = "dot",
    shuffle_relations: bool = False,
    add_self_loops: bool = False,
    held_out_relations: list[str] | None = None,
    held_out_node_categories: list[str] | None = None,
) -> LinkPredictionDataset:
    working_graph = (
        _shuffle_graph_relations(graph, seed=seed)
        if shuffle_relations
        else graph
    )
    data, relation_lookup = networkx_to_data(
        working_graph,
        in_dim=in_dim,
        relation_lookup=relation_lookup,
        seed=seed,
        add_self_loops=add_self_loops,
    )
    full_data = data.clone()
    positive_edge_index, positive_edge_attr = _undirected_positive_edges(data)
    node_list = list(working_graph.nodes())
    node_types = [
        str(working_graph.nodes[node].get("node_type", "entity"))
        for node in node_list
    ]
    if negative_sampling_mode is None:
        negative_sampling_mode = (
            "type_matched"
            if split_protocol in ("node", "node_category")
            else "global"
        )

    ho_cat = (
        frozenset(str(x).strip() for x in held_out_node_categories if str(x).strip())
        if held_out_node_categories
        else frozenset()
    )
    if ho_cat and split_protocol != "node_category":
        raise ValueError(
            "held_out_node_categories is only supported with split_protocol=node_category"
        )
    if split_protocol == "node_category" and not ho_cat:
        raise ValueError(
            "split_protocol=node_category requires a non-empty held_out_node_categories list"
        )

    held_ids: frozenset[int] | None = None
    held_names: frozenset[str] | None = None
    if held_out_relations:
        if positive_edge_attr is None:
            raise ValueError(
                "held_out_relations requires relation labels on edges (edge_attr); "
                "check that the graph has a relation attribute."
            )
        hid, resolved, unknown = resolve_held_out_relation_ids(
            held_out_relations, relation_lookup
        )
        for u in unknown:
            _LOG.warning("Unknown held-out relation name (not in vocabulary): %s", u)
        if not hid:
            raise ValueError(
                "held_out_relations did not resolve to any relation id in the graph vocabulary."
            )
        held_ids = hid
        held_names = frozenset(resolved)

    if held_ids is not None and ho_cat:
        raise ValueError(
            "Combine only one of held_out_relations and held_out_node_categories, not both."
        )

    if held_ids is not None:
        if split_protocol == "edge":
            split = create_edge_split_relation_holdout(
                positive_edge_index,
                positive_edge_attr,
                held_out_relation_ids=held_ids,
                num_src_nodes=int(data.num_nodes),
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                undirected=True,
                node_types=node_types,
                negative_sampling_mode=negative_sampling_mode,
                negatives_per_pos=negatives_per_pos,
            )
        elif split_protocol == "node":
            split = create_node_split_relation_holdout(
                positive_edge_index,
                positive_edge_attr,
                held_out_relation_ids=held_ids,
                num_src_nodes=int(data.num_nodes),
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                undirected=True,
                node_types=node_types,
                negative_sampling_mode=negative_sampling_mode,
                negatives_per_pos=negatives_per_pos,
            )
        else:
            raise ValueError(f"Unsupported split_protocol: {split_protocol}")
    elif split_protocol == "edge":
        split = create_edge_split(
            positive_edge_index,
            num_src_nodes=int(data.num_nodes),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            undirected=True,
            node_types=node_types,
            negative_sampling_mode=negative_sampling_mode,
            negatives_per_pos=negatives_per_pos,
        )
    elif split_protocol == "node":
        split = create_node_split(
            positive_edge_index,
            num_src_nodes=int(data.num_nodes),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            undirected=True,
            node_types=node_types,
            negative_sampling_mode=negative_sampling_mode,
            negatives_per_pos=negatives_per_pos,
        )
    elif split_protocol == "node_category":
        split = create_node_category_split(
            positive_edge_index,
            node_types=node_types,
            held_out_node_categories=ho_cat,
            num_src_nodes=int(data.num_nodes),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            undirected=True,
            negative_sampling_mode=negative_sampling_mode,
            negatives_per_pos=negatives_per_pos,
        )
    else:
        raise ValueError(f"Unsupported split_protocol: {split_protocol}")
    train_pos_edge_attr = _select_train_edge_attr(
        positive_edge_index,
        positive_edge_attr,
        split.train_pos_edge_index,
    )
    train_data = build_train_graph_data(
        data,
        split.train_pos_edge_index,
        train_pos_edge_attr=train_pos_edge_attr,
    )
    return LinkPredictionDataset(
        graph=working_graph,
        data=data,
        full_data=full_data,
        relation_lookup=relation_lookup,
        split=split,
        split_protocol=split_protocol,
        negative_sampling_mode=negative_sampling_mode,
        negatives_per_pos=negatives_per_pos,
        decoder=decoder,
        shuffle_relations=shuffle_relations,
        node_list=node_list,
        node_types=node_types,
        train_data=train_data,
        positive_edge_index=positive_edge_index,
        positive_edge_attr=positive_edge_attr,
        train_pos_edge_attr=train_pos_edge_attr,
        held_out_relations=held_names,
        held_out_relation_ids=held_ids,
        held_out_node_categories=ho_cat if ho_cat else None,
    )


def prepare_node_classification_dataset(
    graph: nx.Graph,
    *,
    in_dim: int,
    label_attr: str = "label",
    relation_lookup: dict[str, int] | None = None,
    label_lookup: dict[str, int] | None = None,
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    add_self_loops: bool = False,
) -> NodeClassificationDataset:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")

    node_list = list(graph.nodes())
    raw_labels = [graph.nodes[node].get(label_attr) for node in node_list]
    present_labels = sorted({str(label) for label in raw_labels if label is not None})
    if not present_labels:
        raise ValueError(f"No node labels found under attr '{label_attr}'")

    if label_lookup is None:
        label_lookup = {label: idx for idx, label in enumerate(present_labels)}

    data, relation_lookup = networkx_to_data(
        graph,
        in_dim=in_dim,
        relation_lookup=relation_lookup,
        seed=seed,
        add_self_loops=add_self_loops,
    )

    y = torch.full((len(node_list),), -1, dtype=torch.long)
    labeled_idx: list[int] = []
    for idx, label in enumerate(raw_labels):
        if label is None:
            continue
        y[idx] = label_lookup[str(label)]
        labeled_idx.append(idx)

    train_idx, val_idx, test_idx = _stratified_split_labeled_indices(
        y=y,
        labeled_idx=labeled_idx,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_mask = torch.zeros(len(node_list), dtype=torch.bool)
    val_mask = torch.zeros(len(node_list), dtype=torch.bool)
    test_mask = torch.zeros(len(node_list), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.y = y
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return NodeClassificationDataset(
        graph=graph,
        data=data,
        relation_lookup=relation_lookup,
        label_lookup=label_lookup,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def prepare_hetero_node_classification_dataset(
    graph: nx.Graph,
    *,
    in_dim: int,
    target_node_type: str,
    label_attr: str = "label",
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    add_reverse_edges: bool = True,
) -> HeteroNodeClassificationDataset:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")

    data = networkx_to_heterodata(
        graph,
        in_dim=in_dim,
        seed=seed,
        add_reverse_edges=add_reverse_edges,
    )

    target_nodes = [
        node for node, attrs in graph.nodes(data=True)
        if str(attrs.get("node_type", "entity")) == target_node_type
    ]
    if not target_nodes:
        raise ValueError(f"No nodes found for target node type '{target_node_type}'")

    raw_labels = [graph.nodes[node].get(label_attr) for node in target_nodes]
    present_labels = sorted({str(label) for label in raw_labels if label is not None})
    if not present_labels:
        raise ValueError(
            f"No node labels found under attr '{label_attr}' for node type '{target_node_type}'"
        )
    label_lookup = {label: idx for idx, label in enumerate(present_labels)}

    y = torch.full((len(target_nodes),), -1, dtype=torch.long)
    labeled_idx: list[int] = []
    for idx, label in enumerate(raw_labels):
        if label is None:
            continue
        y[idx] = label_lookup[str(label)]
        labeled_idx.append(idx)

    train_idx, val_idx, test_idx = _stratified_split_labeled_indices(
        y=y,
        labeled_idx=labeled_idx,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
    val_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
    test_mask = torch.zeros(len(target_nodes), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data[target_node_type].y = y
    data[target_node_type].train_mask = train_mask
    data[target_node_type].val_mask = val_mask
    data[target_node_type].test_mask = test_mask

    return HeteroNodeClassificationDataset(
        graph=graph,
        data=data,
        target_node_type=target_node_type,
        label_lookup=label_lookup,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
