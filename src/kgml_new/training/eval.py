from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


def _binary_metrics_from_scores(
    pos_scores: torch.Tensor | np.ndarray,
    neg_scores: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    if isinstance(pos_scores, torch.Tensor):
        pos_scores = pos_scores.detach().cpu().numpy()
    if isinstance(neg_scores, torch.Tensor):
        neg_scores = neg_scores.detach().cpu().numpy()

    y = np.concatenate(
        [
            np.ones(len(pos_scores), dtype=np.float32),
            np.zeros(len(neg_scores), dtype=np.float32),
        ],
        axis=0,
    )
    scores = np.concatenate([pos_scores, neg_scores], axis=0)

    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y, scores))
    except ValueError:
        ap = float("nan")

    return {"roc_auc": auc, "average_precision": ap}


def link_prediction_dot_product(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
) -> dict[str, float]:
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    return _binary_metrics_from_scores(pos_scores, neg_scores)


def link_prediction_sklearn(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    *,
    max_iter: int = 200,
) -> dict[str, float]:
    del max_iter
    return link_prediction_dot_product(z, pos_edge_index, neg_edge_index)


def link_prediction_mlp_torch(
    model: nn.Module,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    z: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    pos_src, pos_dst = pos_edge_index[0], pos_edge_index[1]
    neg_src, neg_dst = neg_edge_index[0], neg_edge_index[1]

    pos_logit = model(z[pos_src], z[pos_dst])
    neg_logit = model(z[neg_src], z[neg_dst])

    pos_prob = torch.sigmoid(pos_logit)
    neg_prob = torch.sigmoid(neg_logit)

    return _binary_metrics_from_scores(pos_prob, neg_prob)
