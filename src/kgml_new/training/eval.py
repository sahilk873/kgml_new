from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


def link_prediction_sklearn(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    *,
    max_iter: int = 200,
) -> dict[str, float]:
    """
    Concatenate src/dst embeddings and fit a logistic regression (lightweight analogue
    to kgml MLP link classifier).
    """
    z_np = z.detach().cpu().numpy()
    pos_src = z_np[pos_edge_index[0].cpu().numpy()]
    pos_dst = z_np[pos_edge_index[1].cpu().numpy()]
    neg_src = z_np[neg_edge_index[0].cpu().numpy()]
    neg_dst = z_np[neg_edge_index[1].cpu().numpy()]

    x_pos = np.concatenate([pos_src, pos_dst], axis=1)
    x_neg = np.concatenate([neg_src, neg_dst], axis=1)
    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([np.ones(len(x_pos), dtype=np.float32), np.zeros(len(x_neg), dtype=np.float32)], axis=0)

    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    clf.fit(x, y)
    prob = clf.predict_proba(x)[:, 1]

    try:
        auc = float(roc_auc_score(y, prob))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y, prob))
    except ValueError:
        ap = float("nan")

    return {"roc_auc": auc, "average_precision": ap}


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

    y = torch.cat([torch.ones_like(pos_prob), torch.zeros_like(neg_prob)])
    prob = torch.cat([pos_prob, neg_prob])

    try:
        auc = roc_auc_score(y.cpu().numpy(), prob.detach().cpu().numpy())
    except ValueError:
        auc = float("nan")

    try:
        ap = average_precision_score(y.cpu().numpy(), prob.detach().cpu().numpy())
    except ValueError:
        ap = float("nan")

    return {"roc_auc": auc, "average_precision": ap}
