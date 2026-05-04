"""OOD-style difficulty stratification by training relation frequency (tail / rare relations)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from kgml_new.training.eval import grouped_link_prediction_metrics


@dataclass(frozen=True)
class OODDifficultyConfig:
    num_quantile_buckets: int
    tail_quantile: float
    bucket_mode: str


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_uv_to_relation_id(
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor,
) -> dict[tuple[int, int], int]:
    ei = positive_edge_index.cpu().long()
    ea = positive_edge_attr.cpu().long().view(-1)
    out: dict[tuple[int, int], int] = {}
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        a, b = (u, v) if u <= v else (v, u)
        rid = int(ea[i].item())
        out[(a, b)] = rid
    return out


def strip_scores_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Drop score tensors/arrays so metrics dict is JSON-safe."""
    return {k: v for k, v in metrics.items() if k not in ("pos_scores", "neg_scores")}


def compute_ood_difficulty_for_split(
    *,
    split_name: str,
    train_pos_edge_index: torch.Tensor,
    train_pos_edge_attr: torch.Tensor | None,
    full_edge_index: torch.Tensor,
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor | None,
    query_pos_edge_index: torch.Tensor,
    pos_scores: np.ndarray | torch.Tensor,
    neg_scores: np.ndarray | torch.Tensor,
    negatives_per_pos: int,
    num_nodes: int,
    relation_lookup: dict[str, int],
    split_protocol: str,
    config: OODDifficultyConfig,
) -> dict[str, Any] | None:
    """
    Bucket query edges by training frequency of their relation type, then report per-bucket LP metrics.
    """
    del full_edge_index, num_nodes, relation_lookup, split_protocol  # reserved / API parity

    if train_pos_edge_attr is None or positive_edge_attr is None:
        return {
            "split_name": split_name,
            "error": "missing_edge_attr",
            "buckets": [],
            "features": None,
        }

    pos_scores_np = _to_numpy(pos_scores).reshape(-1).astype(np.float64)
    neg_scores_np = _to_numpy(neg_scores)
    if neg_scores_np.ndim == 1:
        neg_scores_np = neg_scores_np.reshape(-1, negatives_per_pos)
    n_q = query_pos_edge_index.size(1)
    if n_q == 0:
        return {"split_name": split_name, "buckets": [], "features": None}

    if pos_scores_np.shape[0] != n_q or neg_scores_np.shape[0] != n_q:
        return {
            "split_name": split_name,
            "error": "score_shape_mismatch",
            "buckets": [],
            "features": None,
        }

    ta = train_pos_edge_attr.cpu().long().view(-1).numpy()
    max_rid = int(ta.max()) if ta.size else 0
    freq = np.bincount(ta, minlength=max_rid + 1)

    uv_rel = _build_uv_to_relation_id(positive_edge_index, positive_edge_attr)
    qp = query_pos_edge_index.cpu().long().numpy()
    rid_q = np.zeros(n_q, dtype=np.int64)
    for i in range(n_q):
        u, v = int(qp[0, i]), int(qp[1, i])
        a, b = (u, v) if u <= v else (v, u)
        rid_q[i] = uv_rel.get((a, b), int(ta[0]) if ta.size else 0)

    train_count = freq[rid_q].astype(np.float64)
    try:
        nuniq = len(np.unique(train_count))
        q = min(config.num_quantile_buckets, max(1, nuniq))
        bucket_s = pd.qcut(
            train_count,
            q=q,
            labels=False,
            duplicates="drop",
        )
        bucket_id = np.asarray(bucket_s, dtype=np.float64)
        bucket_id = np.nan_to_num(bucket_id, nan=0.0).astype(np.int64).reshape(-1)
    except (ValueError, TypeError):
        bucket_id = np.zeros(n_q, dtype=np.int64)

    summaries: list[dict[str, Any]] = []
    for b in sorted(int(x) for x in np.unique(bucket_id)):
        mask = bucket_id == b
        if not np.any(mask):
            continue
        m = grouped_link_prediction_metrics(
            torch.from_numpy(pos_scores_np[mask]),
            torch.from_numpy(neg_scores_np[mask]),
        )
        summaries.append(
            {
                "bucket_id": int(b),
                "num_edges": int(mask.sum()),
                "mean_train_relation_freq": float(train_count[mask].mean()),
                "metrics": {k: float(v) for k, v in m.items()},
            }
        )

    features = {
        "u": qp[0],
        "v": qp[1],
        "relation_id": rid_q,
        "bucket_id": bucket_id,
        "train_relation_freq": train_count,
    }

    return {
        "split_name": split_name,
        "buckets": summaries,
        "features": features,
    }


def build_ood_json_payload(
    *,
    val_block: dict[str, Any] | None = None,
    test_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable OOD payload (strips large ``features`` arrays)."""

    def _trim(block: dict[str, Any] | None) -> dict[str, Any] | None:
        if block is None:
            return None
        out = {k: v for k, v in block.items() if k != "features"}
        return out

    payload: dict[str, Any] = {}
    tv = _trim(val_block)
    tt = _trim(test_block)
    if tv is not None:
        payload["val"] = tv
    if tt is not None:
        payload["test"] = tt
    return payload


def write_edge_predictions_csv(
    csv_path: Path,
    *,
    split_name: str,
    features: dict[str, Any],
    pos_scores: np.ndarray | torch.Tensor,
    neg_scores: np.ndarray | torch.Tensor,
    query_neg_edge_index: torch.Tensor,
    negatives_per_pos: int,
    relation_lookup: dict[str, int],
    seed: int,
    model_name: str,
    edge_relation_mode: str | None,
    split_protocol: str,
    append: bool,
) -> None:
    """Append per-edge prediction rows for downstream analysis."""
    del query_neg_edge_index, relation_lookup, seed, edge_relation_mode
    pos_scores_np = _to_numpy(pos_scores).reshape(-1)
    neg_scores_np = _to_numpy(neg_scores)
    if neg_scores_np.ndim == 1:
        neg_scores_np = neg_scores_np.reshape(-1, negatives_per_pos)
    neg_mean = neg_scores_np.mean(axis=1)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if not append:
            w.writerow(
                [
                    "split",
                    "model",
                    "split_protocol",
                    "u",
                    "v",
                    "relation_id",
                    "bucket_id",
                    "train_relation_freq",
                    "pos_score",
                    "neg_mean_score",
                ]
            )
        u = features["u"]
        v = features["v"]
        rid = features["relation_id"]
        bid = features["bucket_id"]
        tf = features["train_relation_freq"]
        for i in range(len(pos_scores_np)):
            w.writerow(
                [
                    split_name,
                    model_name,
                    split_protocol,
                    int(u[i]),
                    int(v[i]),
                    int(rid[i]),
                    int(bid[i]),
                    float(tf[i]),
                    float(pos_scores_np[i]),
                    float(neg_mean[i]),
                ]
            )
