from __future__ import annotations

import torch

import numpy as np

from kgml_new.training.eval import compute_relation_holdout_metrics
from kgml_new.training.splits import (
    create_edge_split_relation_holdout,
    create_node_split_relation_holdout,
    split_edge_indices_val_test_only,
)


def test_compute_relation_holdout_metrics_groups_by_held_ids() -> None:
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    ea = torch.tensor([1, 2], dtype=torch.long)
    q = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    pos = np.array([0.9, 0.1], dtype=np.float32)
    neg = np.array([[0.1, 0.2], [0.8, 0.7]], dtype=np.float32)
    out = compute_relation_holdout_metrics(
        query_pos_edge_index=q,
        positive_edge_index=ei,
        positive_edge_attr=ea,
        held_out_relation_ids=frozenset({2}),
        pos_scores=pos,
        neg_scores=neg,
    )
    assert out["n_seen_queries"] == 1
    assert out["n_unseen_queries"] == 1
    assert out["seen_relations"]["roc_auc"] == 1.0
    assert out["unseen_relations"]["roc_auc"] == 0.0


def test_split_edge_indices_val_test_only_partitions_all() -> None:
    v, t = split_edge_indices_val_test_only(10, val_ratio=0.1, test_ratio=0.1, seed=0)
    assert v.numel() + t.numel() == 10
    assert torch.unique(torch.cat([v, t])).numel() == 10


def _edge_to_rid(ei: torch.Tensor, ea: torch.Tensor, u: int, v: int) -> int:
    a, b = (u, v) if u <= v else (v, u)
    for j in range(ei.size(1)):
        if int(ei[0, j]) == a and int(ei[1, j]) == b:
            return int(ea[j].item())
    raise AssertionError("edge not found")


def test_edge_split_relation_holdout_excludes_held_from_train() -> None:
    # Canonical (min,max): path 0—1—2—3—4—5 plus chord (0,5); rel 1 on first 3, rel 2 on last 3
    ei = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], dtype=torch.long)
    ea = torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.long)
    sp = create_edge_split_relation_holdout(
        ei,
        ea,
        held_out_relation_ids={2},
        num_src_nodes=6,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        undirected=False,
        negative_sampling_mode="global",
        negatives_per_pos=2,
    )
    for i in range(sp.train_pos_edge_index.size(1)):
        u, v = int(sp.train_pos_edge_index[0, i]), int(sp.train_pos_edge_index[1, i])
        assert _edge_to_rid(ei, ea, u, v) != 2


def test_node_split_relation_holdout_excludes_held_from_train() -> None:
    ei = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 5]], dtype=torch.long)
    ea = torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.long)
    sp = create_node_split_relation_holdout(
        ei,
        ea,
        held_out_relation_ids={2},
        num_src_nodes=6,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        undirected=False,
        negative_sampling_mode="global",
        negatives_per_pos=2,
    )
    for i in range(sp.train_pos_edge_index.size(1)):
        u, v = int(sp.train_pos_edge_index[0, i]), int(sp.train_pos_edge_index[1, i])
        assert _edge_to_rid(ei, ea, u, v) != 2
