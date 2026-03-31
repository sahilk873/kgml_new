from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn
from typing import Optional


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


def _compute_bootstrap_ci(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Args:
        pos_scores: Positive edge scores
        neg_scores: Negative edge scores
        metric_fn: Function to compute the metric
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        (metric_value, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    y = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores))
    ])
    scores = np.concatenate([pos_scores, neg_scores])
    
    metric_values = []
    n_total = len(y)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n_total, size=n_total, replace=True)
        y_boot = y[indices]
        scores_boot = scores[indices]
        try:
            metric_values.append(metric_fn(y_boot, scores_boot))
        except ValueError:
            continue
    
    if not metric_values:
        return float("nan"), float("nan"), float("nan")
    
    metric_values = np.array(metric_values)
    metric_mean = float(np.mean(metric_values))
    ci_lower = float(np.percentile(metric_values, (1 - ci_level) / 2 * 100))
    ci_upper = float(np.percentile(metric_values, (1 + ci_level) / 2 * 100))
    
    return metric_mean, ci_lower, ci_upper


def compute_metrics_with_ci(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute metrics with bootstrap confidence intervals for publication.
    
    Returns a dictionary containing:
    - roc_auc, roc_auc_ci_lower, roc_auc_ci_upper
    - average_precision, ap_ci_lower, ap_ci_upper
    """
    metrics = _binary_metrics_from_scores(pos_scores, neg_scores)
    
    auc_mean, auc_ci_lo, auc_ci_hi = _compute_bootstrap_ci(
        pos_scores, neg_scores, roc_auc_score, n_bootstrap, ci_level, seed
    )
    ap_mean, ap_ci_lo, ap_ci_hi = _compute_bootstrap_ci(
        pos_scores, neg_scores, average_precision_score, n_bootstrap, ci_level, seed
    )
    
    return {
        "roc_auc": metrics["roc_auc"],
        "roc_auc_ci_lower": auc_ci_lo,
        "roc_auc_ci_upper": auc_ci_hi,
        "average_precision": metrics["average_precision"],
        "ap_ci_lower": ap_ci_lo,
        "ap_ci_upper": ap_ci_hi,
        "n_positive": len(pos_scores),
        "n_negative": len(neg_scores),
    }


def link_prediction_dot_product(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
) -> dict[str, float]:
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    return _binary_metrics_from_scores(pos_scores, neg_scores)


def link_prediction_dot_product_with_ci(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Link prediction metrics with bootstrap confidence intervals."""
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1).detach().cpu().numpy()
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1).detach().cpu().numpy()
    return compute_metrics_with_ci(pos_scores, neg_scores, n_bootstrap, ci_level, seed)


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

    with torch.inference_mode():
        pos_logit = model(z[pos_src], z[pos_dst])
        neg_logit = model(z[neg_src], z[neg_dst])

    pos_prob = torch.sigmoid(pos_logit)
    neg_prob = torch.sigmoid(neg_logit)

    return _binary_metrics_from_scores(pos_prob, neg_prob)


def link_prediction_mlp_with_ci(
    model: nn.Module,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    z: torch.Tensor,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Link prediction MLP metrics with bootstrap confidence intervals."""
    model.eval()
    pos_src, pos_dst = pos_edge_index[0], pos_edge_index[1]
    neg_src, neg_dst = neg_edge_index[0], neg_edge_index[1]

    with torch.inference_mode():
        pos_logit = model(z[pos_src], z[pos_dst])
        neg_logit = model(z[neg_src], z[neg_dst])

    pos_prob = torch.sigmoid(pos_logit).detach().cpu().numpy()
    neg_prob = torch.sigmoid(neg_logit).detach().cpu().numpy()

    return compute_metrics_with_ci(pos_prob, neg_prob, n_bootstrap, ci_level, seed)
