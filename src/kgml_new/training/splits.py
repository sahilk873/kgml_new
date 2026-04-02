from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


@dataclass
class EdgeSplit:
    train_pos_edge_index: torch.Tensor
    val_pos_edge_index: torch.Tensor
    test_pos_edge_index: torch.Tensor
    val_neg_edge_index: torch.Tensor
    test_neg_edge_index: torch.Tensor
    negatives_per_pos: int


@dataclass
class NodeSplit:
    train_node_mask: torch.Tensor
    val_node_mask: torch.Tensor
    test_node_mask: torch.Tensor
    train_pos_edge_index: torch.Tensor
    val_pos_edge_index: torch.Tensor
    test_pos_edge_index: torch.Tensor
    val_neg_edge_index: torch.Tensor
    test_neg_edge_index: torch.Tensor
    negatives_per_pos: int


def _sorted_membership(sorted_values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if sorted_values.numel() == 0 or query.numel() == 0:
        return torch.zeros_like(query, dtype=torch.bool)
    idx = torch.searchsorted(sorted_values, query)
    in_bounds = idx < sorted_values.numel()
    clamped = idx.clamp(max=sorted_values.numel() - 1)
    return in_bounds & (sorted_values[clamped] == query)


def _canonicalize_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    pos = edge_index.cpu().long()
    if pos.numel() == 0:
        return pos.reshape(2, 0)
    src = torch.minimum(pos[0], pos[1])
    dst = torch.maximum(pos[0], pos[1])
    return torch.unique(torch.stack([src, dst], dim=1), dim=0, sorted=True).t()


def _edge_keys(edge_index: torch.Tensor, *, num_dst_nodes: int, undirected: bool) -> torch.Tensor:
    src = edge_index[0].cpu().long()
    dst = edge_index[1].cpu().long()
    if undirected:
        canon_src = torch.minimum(src, dst)
        canon_dst = torch.maximum(src, dst)
        src, dst = canon_src, canon_dst
    return src * num_dst_nodes + dst


def _orient_query_edges(
    edge_index: torch.Tensor,
    *,
    primary_node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    edge_index = edge_index.cpu().long()
    if edge_index.numel() == 0 or primary_node_mask is None:
        return edge_index
    primary_node_mask = primary_node_mask.cpu().bool()
    src = edge_index[0]
    dst = edge_index[1]
    swap = (~primary_node_mask[src]) & primary_node_mask[dst]
    oriented_src = torch.where(swap, dst, src)
    oriented_dst = torch.where(swap, src, dst)
    return torch.stack([oriented_src, oriented_dst], dim=0)


def _build_type_to_nodes(node_types: list[str]) -> dict[str, torch.Tensor]:
    buckets: dict[str, list[int]] = {}
    for idx, node_type in enumerate(node_types):
        buckets.setdefault(str(node_type), []).append(idx)
    return {
        node_type: torch.tensor(indices, dtype=torch.long)
        for node_type, indices in buckets.items()
    }


def split_edge_indices(
    num_edges: int,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator)

    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    if num_edges > 0 and num_val == 0 and val_ratio > 0:
        num_val = 1
    if num_edges - num_val > 1 and num_test == 0 and test_ratio > 0:
        num_test = 1
    if num_val + num_test >= num_edges and num_edges > 1:
        num_test = max(0, num_edges - num_val - 1)

    val_idx = perm[:num_val]
    test_idx = perm[num_val : num_val + num_test]
    train_idx = perm[num_val + num_test :]
    return train_idx, val_idx, test_idx


def split_node_indices(
    num_nodes: int,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1, got {val_ratio + test_ratio}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    num_val = int(num_nodes * val_ratio)
    num_test = int(num_nodes * test_ratio)
    if num_nodes > 0 and num_val == 0 and val_ratio > 0:
        num_val = 1
    if num_nodes - num_val > 1 and num_test == 0 and test_ratio > 0:
        num_test = 1
    if num_val + num_test >= num_nodes and num_nodes > 1:
        num_test = max(0, num_nodes - num_val - 1)

    val_idx = perm[:num_val]
    test_idx = perm[num_val : num_val + num_test]
    train_idx = perm[num_val + num_test :]
    return train_idx, val_idx, test_idx


def sample_negative_edges(
    *,
    num_src_nodes: int,
    num_dst_nodes: int,
    positive_edge_index: torch.Tensor,
    num_samples: int,
    seed: int = 42,
    undirected: bool,
    anchor_node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if num_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    pos = positive_edge_index.cpu().long()
    src = pos[0]
    dst = pos[1]
    if undirected:
        src, dst = torch.minimum(src, dst), torch.maximum(src, dst)
    existing_keys = torch.unique(src * num_dst_nodes + dst, sorted=True)

    if anchor_node_mask is not None:
        anchor_node_mask = anchor_node_mask.cpu().bool().view(-1)
        if anchor_node_mask.numel() < max(num_src_nodes, num_dst_nodes):
            raise ValueError("anchor_node_mask is smaller than the node space")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    chosen = torch.empty(0, dtype=torch.long)

    while chosen.numel() < num_samples:
        remaining = num_samples - chosen.numel()
        candidate_count = max(remaining * 4, 1024)
        cand_src = torch.randint(0, num_src_nodes, (candidate_count,), generator=generator)
        cand_dst = torch.randint(0, num_dst_nodes, (candidate_count,), generator=generator)

        if undirected:
            keep = cand_src != cand_dst
            cand_src = cand_src[keep]
            cand_dst = cand_dst[keep]
            cand_src, cand_dst = torch.minimum(cand_src, cand_dst), torch.maximum(
                cand_src, cand_dst
            )

        if anchor_node_mask is not None:
            keep = anchor_node_mask[cand_src] | anchor_node_mask[cand_dst]
            cand_src = cand_src[keep]
            cand_dst = cand_dst[keep]

        cand_keys = torch.unique(cand_src * num_dst_nodes + cand_dst, sorted=True)
        if cand_keys.numel() == 0:
            continue

        is_existing = _sorted_membership(existing_keys, cand_keys)
        if chosen.numel() > 0:
            is_existing |= _sorted_membership(chosen, cand_keys)
        fresh = cand_keys[~is_existing]
        if fresh.numel() == 0:
            continue

        chosen = torch.cat([chosen, fresh]).unique(sorted=True)

    keys = chosen[:num_samples]
    neg_src = keys // num_dst_nodes
    neg_dst = keys % num_dst_nodes
    return torch.stack([neg_src, neg_dst], dim=0)


def sample_query_negative_edges(
    query_edge_index: torch.Tensor,
    *,
    positive_edge_index: torch.Tensor,
    num_src_nodes: int,
    num_dst_nodes: int | None = None,
    node_types: list[str] | None,
    negatives_per_pos: int,
    mode: str,
    seed: int = 42,
    undirected: bool = True,
    bipartite: bool | None = None,
) -> torch.Tensor:
    if negatives_per_pos <= 0 or query_edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if mode not in {"global", "type_matched"}:
        raise ValueError(f"Unsupported negative sampling mode: {mode}")
    if mode == "type_matched" and node_types is None:
        raise ValueError("node_types are required for type_matched negative sampling")
    if num_dst_nodes is None:
        num_dst_nodes = num_src_nodes
    if bipartite is None:
        bipartite = num_src_nodes != num_dst_nodes
    if node_types is not None and len(node_types) != num_dst_nodes:
        raise ValueError("node_types must align with the destination node space")

    positive_edge_index = positive_edge_index.cpu().long()
    query_edge_index = query_edge_index.cpu().long()
    # Keep the existence check directed so homogeneous undirected graphs can
    # still use the reverse orientation as a negative example.
    existing_keys = set(
        _edge_keys(
            positive_edge_index,
            num_dst_nodes=num_dst_nodes,
            undirected=False,
        ).tolist()
    )
    all_nodes = torch.arange(num_dst_nodes, dtype=torch.long)
    type_to_nodes = _build_type_to_nodes(node_types or ["entity"] * num_dst_nodes)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    neg_src_parts: list[torch.Tensor] = []
    neg_dst_parts: list[torch.Tensor] = []
    for edge_idx in range(query_edge_index.size(1)):
        src = int(query_edge_index[0, edge_idx])
        dst = int(query_edge_index[1, edge_idx])
        if mode == "type_matched":
            candidate_pool = type_to_nodes[str(node_types[dst])]
        else:
            candidate_pool = all_nodes

        valid_candidates: list[int] = []
        for candidate in candidate_pool.tolist():
            if candidate == dst or candidate == src:
                continue
            key = src * num_dst_nodes + candidate
            if key in existing_keys:
                continue
            valid_candidates.append(candidate)

        if not valid_candidates:
            raise ValueError(
                f"No valid negatives available for query edge ({src}, {dst}) under mode={mode}"
            )

        valid_tensor = torch.tensor(valid_candidates, dtype=torch.long)
        if valid_tensor.numel() >= negatives_per_pos:
            perm = torch.randperm(valid_tensor.numel(), generator=generator)
            chosen = valid_tensor[perm[:negatives_per_pos]]
        else:
            indices = torch.randint(
                0,
                valid_tensor.numel(),
                (negatives_per_pos,),
                generator=generator,
            )
            chosen = valid_tensor[indices]

        neg_src_parts.append(torch.full((negatives_per_pos,), src, dtype=torch.long))
        neg_dst_parts.append(chosen)

    return torch.stack(
        [torch.cat(neg_src_parts, dim=0), torch.cat(neg_dst_parts, dim=0)],
        dim=0,
    )


def create_edge_split(
    edge_index: torch.Tensor,
    *,
    num_src_nodes: int,
    num_dst_nodes: int | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    undirected: bool = True,
    node_types: list[str] | None = None,
    negative_sampling_mode: str = "global",
    negatives_per_pos: int = 20,
    bipartite: bool | None = None,
) -> EdgeSplit:
    if num_dst_nodes is None:
        num_dst_nodes = num_src_nodes
    if bipartite is None:
        bipartite = num_src_nodes != num_dst_nodes

    train_idx, val_idx, test_idx = split_edge_indices(
        edge_index.size(1),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    canonical_edge_index = (
        _canonicalize_undirected_edges(edge_index) if undirected else edge_index.cpu().long()
    )
    train_pos = canonical_edge_index[:, train_idx]
    val_pos = canonical_edge_index[:, val_idx]
    test_pos = canonical_edge_index[:, test_idx]
    val_pos = _orient_query_edges(val_pos)
    test_pos = _orient_query_edges(test_pos)

    val_neg = sample_query_negative_edges(
        val_pos,
        positive_edge_index=canonical_edge_index,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        node_types=node_types,
        negatives_per_pos=negatives_per_pos,
        mode=negative_sampling_mode,
        seed=seed + 1,
        undirected=undirected,
        bipartite=bipartite,
    )
    test_neg = sample_query_negative_edges(
        test_pos,
        positive_edge_index=canonical_edge_index,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        node_types=node_types,
        negatives_per_pos=negatives_per_pos,
        mode=negative_sampling_mode,
        seed=seed + 2,
        undirected=undirected,
        bipartite=bipartite,
    )

    return EdgeSplit(
        train_pos_edge_index=train_pos,
        val_pos_edge_index=val_pos,
        test_pos_edge_index=test_pos,
        val_neg_edge_index=val_neg,
        test_neg_edge_index=test_neg,
        negatives_per_pos=negatives_per_pos,
    )


def create_node_split(
    edge_index: torch.Tensor,
    *,
    num_src_nodes: int,
    num_dst_nodes: int | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    undirected: bool = True,
    node_types: list[str] | None = None,
    negative_sampling_mode: str = "type_matched",
    negatives_per_pos: int = 20,
) -> NodeSplit:
    if num_dst_nodes is None:
        num_dst_nodes = num_src_nodes
    if num_src_nodes != num_dst_nodes:
        raise ValueError("node split currently expects a homogeneous graph")

    train_idx, val_idx, test_idx = split_node_indices(
        num_src_nodes,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_node_mask = torch.zeros(num_src_nodes, dtype=torch.bool)
    val_node_mask = torch.zeros(num_src_nodes, dtype=torch.bool)
    test_node_mask = torch.zeros(num_src_nodes, dtype=torch.bool)
    train_node_mask[train_idx] = True
    val_node_mask[val_idx] = True
    test_node_mask[test_idx] = True

    pos = _canonicalize_undirected_edges(edge_index) if undirected else edge_index.cpu().long()

    src, dst = pos[0], pos[1]
    test_edge_mask = test_node_mask[src] | test_node_mask[dst]
    val_edge_mask = ~test_edge_mask & (val_node_mask[src] | val_node_mask[dst])
    train_edge_mask = ~(test_edge_mask | val_edge_mask)

    train_pos = pos[:, train_edge_mask]
    val_pos = _orient_query_edges(pos[:, val_edge_mask], primary_node_mask=val_node_mask)
    test_pos = _orient_query_edges(pos[:, test_edge_mask], primary_node_mask=test_node_mask)

    val_neg = sample_query_negative_edges(
        val_pos,
        positive_edge_index=pos,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        node_types=node_types,
        negatives_per_pos=negatives_per_pos,
        mode=negative_sampling_mode,
        seed=seed + 1,
        undirected=undirected,
        bipartite=False,
    )
    test_neg = sample_query_negative_edges(
        test_pos,
        positive_edge_index=pos,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        node_types=node_types,
        negatives_per_pos=negatives_per_pos,
        mode=negative_sampling_mode,
        seed=seed + 2,
        undirected=undirected,
        bipartite=False,
    )

    return NodeSplit(
        train_node_mask=train_node_mask,
        val_node_mask=val_node_mask,
        test_node_mask=test_node_mask,
        train_pos_edge_index=train_pos,
        val_pos_edge_index=val_pos,
        test_pos_edge_index=test_pos,
        val_neg_edge_index=val_neg,
        test_neg_edge_index=test_neg,
        negatives_per_pos=negatives_per_pos,
    )


def build_train_graph_data(
    data: Data,
    train_pos_edge_index: torch.Tensor,
    *,
    train_pos_edge_attr: torch.Tensor | None = None,
) -> Data:
    train_data = Data(x=data.x, num_nodes=data.num_nodes)
    rev_edge_index = train_pos_edge_index.flip(0)
    train_data.edge_index = torch.cat([train_pos_edge_index, rev_edge_index], dim=1)
    if train_pos_edge_attr is not None:
        train_data.edge_attr = torch.cat([train_pos_edge_attr, train_pos_edge_attr], dim=0)
    return train_data
