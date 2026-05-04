"""Lightweight RotatE tests (no PrimeKG / semantic embedding imports)."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from kgml_new.config import RotatEConfig
from kgml_new.models.rotate import RotatE
from kgml_new.training.relation_decode_batch import build_canonical_uv_relation_lookup
from kgml_new.training.rotate_train import (
    evaluate_rotate_link_prediction,
    train_rotate_with_validation,
)


def test_train_rotate_with_validation_smoke():
    # 4 nodes, training edges (0,1) and (1,2); val query (2,3) with negatives
    num_nodes = 4
    train_data = Data(
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        num_nodes=num_nodes,
    )
    train_pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    train_attr = torch.tensor([0, 1], dtype=torch.long)
    n_neg = 2
    val_pos = torch.tensor([[2], [3]], dtype=torch.long)
    val_neg = torch.tensor([[2, 2], [0, 1]], dtype=torch.long)
    test_pos = torch.tensor([[0], [3]], dtype=torch.long)
    test_neg = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    positive_edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 3, 3]], dtype=torch.long
    )
    positive_edge_attr = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.long)
    query_uv_relation = build_canonical_uv_relation_lookup(
        positive_edge_index, positive_edge_attr
    )

    model = RotatE(
        num_entities=num_nodes,
        num_relations=2,
        embedding_dim=4,
        gamma=6.0,
    )
    cfg = RotatEConfig(
        embedding_dim=4,
        epochs=2,
        batch_size=8,
        learning_rate=0.1,
        neg_samples=2,
        early_stop_patience=100,
        eval_batch_size=128,
        gamma=6.0,
        seed=0,
        weight_decay=0.0,
    )

    _, last_epoch, history, val_m, test_m = train_rotate_with_validation(
        model,
        train_data,
        train_pos,
        train_attr,
        cfg,
        torch.device("cpu"),
        val_pos_edge_index=val_pos,
        val_neg_edge_index=val_neg,
        negatives_per_pos=n_neg,
        query_uv_relation=query_uv_relation,
        test_pos_edge_index=test_pos,
        test_neg_edge_index=test_neg,
        return_scores=False,
    )

    assert last_epoch >= 0
    assert len(history.train_loss) >= 1
    assert "roc_auc" in val_m
    assert test_m is not None
    assert "roc_auc" in test_m


def test_evaluate_rotate_link_prediction_matches_layout():
    model = RotatE(6, 2, 3, gamma=2.0)
    model.eval()
    pos = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # Flat negatives: n_pos * negatives_per_pos columns (matches evaluate_inductive layout).
    neg = torch.tensor([[0, 0, 0, 0], [3, 4, 3, 4]], dtype=torch.long)
    uv = {(0, 1): 0, (1, 2): 1}
    out = evaluate_rotate_link_prediction(
        model,
        pos_edge_index=pos,
        neg_edge_index=neg,
        negatives_per_pos=2,
        query_uv_relation=uv,
        device=torch.device("cpu"),
        batch_size=64,
        return_scores=True,
    )
    assert "roc_auc" in out
    assert out["pos_scores"].shape == (2,)
    assert out["neg_scores"].shape == (2, 2)
