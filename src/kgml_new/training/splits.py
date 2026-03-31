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


def _sorted_membership(sorted_values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if sorted_values.numel() == 0 or query.numel() == 0:
        return torch.zeros_like(query, dtype=torch.bool)
    idx = torch.searchsorted(sorted_values, query)
    in_bounds = idx < sorted_values.numel()
    clamped = idx.clamp(max=sorted_values.numel() - 1)
    return in_bounds & (sorted_values[clamped] == query)


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


def sample_negative_edges(
    *,
    num_src_nodes: int,
    num_dst_nodes: int,
    positive_edge_index: torch.Tensor,
    num_samples: int,
    seed: int = 42,
    undirected: bool,
) -> torch.Tensor:
    if num_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    pos = positive_edge_index.cpu().long()
    src = pos[0]
    dst = pos[1]
    if undirected:
        src, dst = torch.minimum(src, dst), torch.maximum(src, dst)
    existing_keys = torch.unique(src * num_dst_nodes + dst, sorted=True)

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


def create_edge_split(
    edge_index: torch.Tensor,
    *,
    num_src_nodes: int,
    num_dst_nodes: int | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    undirected: bool = True,
) -> EdgeSplit:
    if num_dst_nodes is None:
        num_dst_nodes = num_src_nodes

    train_idx, val_idx, test_idx = split_edge_indices(
        edge_index.size(1),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_pos = edge_index[:, train_idx]
    val_pos = edge_index[:, val_idx]
    test_pos = edge_index[:, test_idx]

    val_neg = sample_negative_edges(
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        positive_edge_index=edge_index,
        num_samples=val_pos.size(1),
        seed=seed + 1,
        undirected=undirected,
    )
    test_neg = sample_negative_edges(
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_nodes,
        positive_edge_index=edge_index,
        num_samples=test_pos.size(1),
        seed=seed + 2,
        undirected=undirected,
    )

    return EdgeSplit(
        train_pos_edge_index=train_pos,
        val_pos_edge_index=val_pos,
        test_pos_edge_index=test_pos,
        val_neg_edge_index=val_neg,
        test_neg_edge_index=test_neg,
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
