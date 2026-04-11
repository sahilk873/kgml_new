from __future__ import annotations

import numpy as np
import torch

from kgml_new.eval.ood_difficulty import (
    OODDifficultyConfig,
    RELATION_ID_NOT_IN_CATALOG,
    compute_ood_difficulty_for_split,
    compute_ood_features_for_positives,
)
from kgml_new.training.eval import grouped_link_prediction_metrics


def test_grouped_metrics_subset_matches_full():
    """Bucket that includes all positives should match full metrics."""
    rng = np.random.default_rng(0)
    pos_scores = rng.random(12).astype(np.float32)
    neg_scores = rng.random((12, 5)).astype(np.float32)
    full = grouped_link_prediction_metrics(pos_scores, neg_scores)
    idx = np.arange(12)
    ps = pos_scores[idx]
    ns = neg_scores[idx]
    part = grouped_link_prediction_metrics(ps, ns)
    assert np.isclose(full["roc_auc"], part["roc_auc"], rtol=1e-5)


def test_ood_features_toy_graph():
    # 4 nodes, train edges (0-1), (1-2), (2-3)
    num_nodes = 4
    train_pos = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    train_attr = torch.tensor([1, 1, 2], dtype=torch.long)
    full_edge = train_pos.clone()
    positive_edge_index = train_pos.clone()
    positive_edge_attr = train_attr.clone()
    query = torch.tensor([[0], [3]], dtype=torch.long)
    relation_lookup = {"UNK": 0, "R1": 1, "R2": 2}
    feats = compute_ood_features_for_positives(
        query_pos_edge_index=query,
        train_pos_edge_index=train_pos,
        train_pos_edge_attr=train_attr,
        full_edge_index=full_edge,
        positive_edge_index=positive_edge_index,
        positive_edge_attr=positive_edge_attr,
        num_nodes=num_nodes,
        relation_lookup=relation_lookup,
    )
    assert int(feats["src_train_degree"][0]) >= 1
    assert feats["novelty_bucket"][0] in ("both_seen", "one_unseen", "both_unseen")


def test_relation_unique_endpoint_count_matches_train_stats():
    """Per-relation unique endpoint count in train must match hand count (regression for shadowing bug)."""
    num_nodes = 4
    # Train: (0,1) and (1,2) both relation 1 -> unique endpoints {0,1,2} -> count 3
    train_pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    train_attr = torch.tensor([1, 1], dtype=torch.long)
    # Catalog includes train edges plus query edge (0,2) with same relation id 1
    positive_edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    positive_edge_attr = torch.tensor([1, 1, 1], dtype=torch.long)
    full_edge = torch.cat([train_pos, torch.tensor([[0], [2]], dtype=torch.long)], dim=1)
    query = torch.tensor([[0], [2]], dtype=torch.long)
    relation_lookup = {"UNK": 0, "R1": 1}
    feats = compute_ood_features_for_positives(
        query_pos_edge_index=query,
        train_pos_edge_index=train_pos,
        train_pos_edge_attr=train_attr,
        full_edge_index=full_edge,
        positive_edge_index=positive_edge_index,
        positive_edge_attr=positive_edge_attr,
        num_nodes=num_nodes,
        relation_lookup=relation_lookup,
    )
    assert int(feats["relation_train_count"][0]) == 2
    assert int(feats["relation_unique_endpoint_count_train"][0]) == 3
    assert bool(feats["relation_in_catalog"][0]) is True
    assert int(feats["relation_id"][0]) == 1


def test_edge_not_in_catalog_gets_minus_one_and_flag():
    """Missing (u,v) in positive catalog -> relation_id -1, not confused with UNK id 0."""
    num_nodes = 4
    train_pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    train_attr = torch.tensor([1, 1], dtype=torch.long)
    positive_edge_index = train_pos.clone()
    positive_edge_attr = train_attr.clone()
    full_edge = train_pos.clone()
    # (0,3) not in catalog
    query = torch.tensor([[0], [3]], dtype=torch.long)
    relation_lookup = {"UNK": 0, "R1": 1}
    feats = compute_ood_features_for_positives(
        query_pos_edge_index=query,
        train_pos_edge_index=train_pos,
        train_pos_edge_attr=train_attr,
        full_edge_index=full_edge,
        positive_edge_index=positive_edge_index,
        positive_edge_attr=positive_edge_attr,
        num_nodes=num_nodes,
        relation_lookup=relation_lookup,
    )
    assert int(feats["relation_id"][0]) == RELATION_ID_NOT_IN_CATALOG
    assert bool(feats["relation_in_catalog"][0]) is False
    assert int(feats["relation_train_count"][0]) == 0
    assert bool(feats["relation_seen_in_train"][0]) is False


def test_compute_ood_split_smoke():
    train_pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    train_attr = torch.tensor([1, 1], dtype=torch.long)
    full_edge = train_pos.clone()
    pos_cat = train_pos.clone()
    attr_cat = train_attr.clone()
    query = torch.tensor([[0], [2]], dtype=torch.long)
    pos_scores = np.array([0.8], dtype=np.float32)
    neg_scores = np.array([[0.1, 0.2]], dtype=np.float32)
    relation_lookup = {"UNK": 0, "R1": 1}
    out = compute_ood_difficulty_for_split(
        split_name="test",
        train_pos_edge_index=train_pos,
        train_pos_edge_attr=train_attr,
        full_edge_index=full_edge,
        positive_edge_index=pos_cat,
        positive_edge_attr=attr_cat,
        query_pos_edge_index=query,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        negatives_per_pos=2,
        num_nodes=3,
        relation_lookup=relation_lookup,
        split_protocol="edge",
        config=OODDifficultyConfig(),
    )
    assert "metrics" in out
    assert "novelty" in out["metrics"]
