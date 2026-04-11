"""
Post-hoc OOD / difficulty analysis for link prediction evaluation.

Graph policy (see ood_difficulty.meta in JSON output):
- Degrees, novelty, relation stats, local support: train positive edges only (undirected).
- Distance to train support: multi-source BFS on undirected graph from full_data.edge_index;
  seeds = nodes incident to at least one train positive edge.

Relation ids: ``RELATION_ID_NOT_IN_CATALOG`` (-1) means the query edge (canonical u,v) was
not found in ``positive_edge_index``; use ``relation_in_catalog`` to distinguish from
relation id 0 (e.g. UNK) that appears in the catalog.
"""

from __future__ import annotations

import csv
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kgml_new.training.eval import grouped_link_prediction_metrics

_LOG = logging.getLogger(__name__)

# Query edge (u,v) not present in positive_edge_index / edge catalog (not UNK id 0).
RELATION_ID_NOT_IN_CATALOG = -1


def strip_scores_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of metrics safe for JSON (no numpy score arrays)."""
    out = {k: v for k, v in metrics.items() if k not in ("pos_scores", "neg_scores")}
    return out


def _canonical_edges(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ei = edge_index.cpu().long()
    a = torch.minimum(ei[0], ei[1])
    b = torch.maximum(ei[0], ei[1])
    return a, b


def _build_uv_to_relation(
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor | None,
) -> dict[tuple[int, int], int]:
    """Map canonical (u,v) -> relation id for positive catalog edges."""
    a, b = _canonical_edges(positive_edge_index)
    out: dict[tuple[int, int], int] = {}
    for i in range(a.numel()):
        key = (int(a[i]), int(b[i]))
        rid = 0
        if positive_edge_attr is not None:
            rid = int(positive_edge_attr[i].item())
        out[key] = rid
    return out


def _train_degrees(
    train_pos: torch.Tensor,
    num_nodes: int,
) -> np.ndarray:
    """Undirected degree from train positive edges."""
    deg = np.zeros(num_nodes, dtype=np.int64)
    ei = train_pos.cpu().long()
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        deg[u] += 1
        deg[v] += 1
    return deg


def _train_seen_nodes(train_pos: torch.Tensor, num_nodes: int) -> np.ndarray:
    seen = np.zeros(num_nodes, dtype=np.bool_)
    ei = train_pos.cpu().long()
    for i in range(ei.size(1)):
        seen[int(ei[0, i])] = True
        seen[int(ei[1, i])] = True
    return seen


def _relation_counts_train(
    train_pos: torch.Tensor,
    train_edge_attr: torch.Tensor | None,
    num_relations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-relation positive count and unique endpoint count in train."""
    counts = np.zeros(max(1, num_relations), dtype=np.int64)
    endpoints: list[set[int]] = [set() for _ in range(max(1, num_relations))]
    ei = train_pos.cpu().long()
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        rid = 0
        if train_edge_attr is not None:
            rid = int(train_edge_attr[i].item())
        if rid >= len(counts):
            # extend if needed
            new_counts = np.zeros(rid + 1, dtype=np.int64)
            new_counts[: len(counts)] = counts
            counts = new_counts
            while len(endpoints) <= rid:
                endpoints.append(set())
        counts[rid] += 1
        endpoints[rid].add(u)
        endpoints[rid].add(v)
    unique_endpoints = np.array([len(s) for s in endpoints], dtype=np.int64)
    return counts, unique_endpoints


def _build_train_adjacency(train_pos: torch.Tensor, num_nodes: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    ei = train_pos.cpu().long()
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _common_neighbors_jaccard(
    adj: list[list[int]],
    u: int,
    v: int,
) -> tuple[int, float]:
    if u == v:
        return 0, 0.0
    nu = adj[u] if u < len(adj) else []
    nv = adj[v] if v < len(adj) else []
    if not nu or not nv:
        su, sv = set(nu), set(nv)
    else:
        su, sv = set(nu), set(nv)
    inter = len(su & sv)
    union = len(su | sv)
    jacc = float(inter / union) if union > 0 else 0.0
    return inter, jacc


def _multi_source_bfs_distances(
    full_edge_index: torch.Tensor,
    num_nodes: int,
    seeds: np.ndarray,
) -> np.ndarray:
    """Hop distance from nearest seed; unreachable -> large sentinel."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    ei = full_edge_index.cpu().long()
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u < num_nodes and v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)
    dist = np.full(num_nodes, -1, dtype=np.int32)
    q: deque[int] = deque()
    for s in seeds:
        if 0 <= s < num_nodes and dist[s] < 0:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    unreachable = np.iinfo(np.int32).max // 4
    out = np.where(dist >= 0, dist, unreachable)
    return out


def _quantile_thresholds(x: np.ndarray, n_buckets: int) -> np.ndarray:
    """Strictly increasing thresholds between buckets (n_buckets-1 boundaries)."""
    if x.size == 0:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0, 100, n_buckets + 1)[1:-1]
    return np.percentile(x, qs)


def _assign_quantile_bucket(
    values: np.ndarray,
    thresholds: np.ndarray,
    names: list[str],
) -> np.ndarray:
    """Assign each value to names[0]..names[k-1] by tertile-like thresholds."""
    out = np.empty(values.shape[0], dtype=object)
    if thresholds.size == 0:
        mid = len(names) // 2
        out[:] = names[mid]
        return out
    for i, v in enumerate(values):
        b = 0
        for t in thresholds:
            if v > t:
                b += 1
            else:
                break
        b = min(b, len(names) - 1)
        out[i] = names[b]
    return out


def _metrics_subset(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int]:
    if indices.size == 0:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "hits@1": float("nan"),
            "hits@3": float("nan"),
            "hits@10": float("nan"),
            "count": 0,
            "positive_count": 0,
            "negative_count": 0,
        }
    ps = pos_scores[indices]
    ns = neg_scores[indices]
    m = grouped_link_prediction_metrics(ps, ns)
    n_pos = int(ps.shape[0])
    n_neg = int(ns.size)
    m["count"] = n_pos + n_neg
    m["positive_count"] = n_pos
    m["negative_count"] = n_neg
    return m


@dataclass
class OODDifficultyConfig:
    num_quantile_buckets: int = 3
    tail_quantile: float = 0.1
    bucket_mode: str = "quantile"


def _inverse_relation_lookup(relation_lookup: dict[str, int]) -> dict[int, str]:
    inv: dict[int, str] = {}
    for name, idx in relation_lookup.items():
        inv.setdefault(int(idx), str(name))
    return inv


def compute_ood_features_for_positives(
    *,
    query_pos_edge_index: torch.Tensor,
    train_pos_edge_index: torch.Tensor,
    train_pos_edge_attr: torch.Tensor | None,
    full_edge_index: torch.Tensor,
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor | None,
    num_nodes: int,
    relation_lookup: dict[str, int],
) -> dict[str, Any]:
    """Per-positive-edge difficulty features (numpy arrays)."""
    num_relations = max(relation_lookup.values(), default=0) + 1
    deg = _train_degrees(train_pos_edge_index, num_nodes)
    train_seen = _train_seen_nodes(train_pos_edge_index, num_nodes)
    rel_counts, rel_unique_ep_per_relation = _relation_counts_train(
        train_pos_edge_index, train_pos_edge_attr, num_relations
    )
    uv_rel = _build_uv_to_relation(positive_edge_index, positive_edge_attr)

    seeds = np.where(train_seen)[0]
    dist_all = _multi_source_bfs_distances(full_edge_index, num_nodes, seeds)

    train_adj = _build_train_adjacency(train_pos_edge_index, num_nodes)

    qp = query_pos_edge_index.cpu().long()
    nq = qp.size(1)
    src = np.zeros(nq, dtype=np.int64)
    dst = np.zeros(nq, dtype=np.int64)
    src_train_degree = np.zeros(nq, dtype=np.int64)
    dst_train_degree = np.zeros(nq, dtype=np.int64)
    min_ep_deg = np.zeros(nq, dtype=np.int64)
    max_ep_deg = np.zeros(nq, dtype=np.float64)
    mean_ep_deg = np.zeros(nq, dtype=np.float64)
    src_seen = np.zeros(nq, dtype=np.bool_)
    dst_seen = np.zeros(nq, dtype=np.bool_)
    src_dist = np.zeros(nq, dtype=np.int64)
    dst_dist = np.zeros(nq, dtype=np.int64)
    edge_dist = np.zeros(nq, dtype=np.int64)
    max_dist = np.zeros(nq, dtype=np.int64)
    rel_id = np.full(nq, RELATION_ID_NOT_IN_CATALOG, dtype=np.int64)
    rel_train_count = np.zeros(nq, dtype=np.int64)
    rel_seen = np.zeros(nq, dtype=np.bool_)
    rel_unique_ep_query = np.zeros(nq, dtype=np.int64)
    rel_in_catalog = np.zeros(nq, dtype=np.bool_)
    cn = np.zeros(nq, dtype=np.int64)
    jac = np.zeros(nq, dtype=np.float64)

    inv_rel = _inverse_relation_lookup(relation_lookup)

    for i in range(nq):
        u, v = int(qp[0, i]), int(qp[1, i])
        cu, cv = (u, v) if u <= v else (v, u)
        src[i], dst[i] = u, v
        sdeg, ddeg = int(deg[u]), int(deg[v])
        src_train_degree[i] = sdeg
        dst_train_degree[i] = ddeg
        min_ep_deg[i] = min(sdeg, ddeg)
        max_ep_deg[i] = max(sdeg, ddeg)
        mean_ep_deg[i] = (sdeg + ddeg) / 2.0
        src_seen[i] = bool(train_seen[u])
        dst_seen[i] = bool(train_seen[v])
        src_dist[i] = int(dist_all[u])
        dst_dist[i] = int(dist_all[v])
        edge_dist[i] = min(src_dist[i], dst_dist[i])
        max_dist[i] = max(src_dist[i], dst_dist[i])

        key = (cu, cv)
        if key in uv_rel:
            rid = int(uv_rel[key])
            rel_in_catalog[i] = True
            rel_id[i] = rid
            if 0 <= rid < rel_counts.shape[0]:
                rel_train_count[i] = int(rel_counts[rid])
                if rid < rel_unique_ep_per_relation.shape[0]:
                    rel_unique_ep_query[i] = int(rel_unique_ep_per_relation[rid])
            rel_seen[i] = rel_train_count[i] > 0
        else:
            rel_id[i] = RELATION_ID_NOT_IN_CATALOG
            rel_seen[i] = False

        cni, ji = _common_neighbors_jaccard(train_adj, u, v)
        cn[i] = cni
        jac[i] = ji

    novelty = np.empty(nq, dtype=object)
    for i in range(nq):
        s, d = src_seen[i], dst_seen[i]
        if s and d:
            novelty[i] = "both_seen"
        elif s ^ d:
            novelty[i] = "one_unseen"
        else:
            novelty[i] = "both_unseen"

    dist_bucket = np.empty(nq, dtype=object)
    for i in range(nq):
        ed = int(edge_dist[i])
        if ed >= 3:
            dist_bucket[i] = "dist_3plus"
        elif ed == 2:
            dist_bucket[i] = "dist_2"
        elif ed == 1:
            dist_bucket[i] = "dist_1"
        else:
            dist_bucket[i] = "dist_0"

    return {
        "src": src,
        "dst": dst,
        "src_train_degree": src_train_degree,
        "dst_train_degree": dst_train_degree,
        "min_endpoint_degree": min_ep_deg,
        "max_endpoint_degree": max_ep_deg,
        "mean_endpoint_degree": mean_ep_deg,
        "src_seen_in_train": src_seen,
        "dst_seen_in_train": dst_seen,
        "novelty_bucket": novelty,
        "src_dist_to_train_support": src_dist,
        "dst_dist_to_train_support": dst_dist,
        "edge_dist_to_train_support": edge_dist,
        "max_dist_to_train_support": max_dist,
        "distance_bucket": dist_bucket,
        "relation_id": rel_id,
        "relation_train_count": rel_train_count,
        "relation_seen_in_train": rel_seen,
        "relation_unique_endpoint_count_train": rel_unique_ep_query,
        "relation_in_catalog": rel_in_catalog,
        "common_neighbors_train": cn,
        "jaccard_train_neighbors": jac,
        "inv_relation_lookup": inv_rel,
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def _assign_buckets_and_metrics(
    features: dict[str, Any],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    config: OODDifficultyConfig,
) -> dict[str, Any]:
    nq = pos_scores.shape[0]
    min_deg = features["min_endpoint_degree"].astype(np.float64)
    edge_dist = features["edge_dist_to_train_support"].astype(np.float64)
    rel_tc = features["relation_train_count"].astype(np.float64)
    rel_uep = features["relation_unique_endpoint_count_train"].astype(np.float64)
    cn = features["common_neighbors_train"].astype(np.float64)
    rel_seen = features["relation_seen_in_train"]

    n_b = config.num_quantile_buckets
    if n_b == 3:
        names_nf = ["nodefreq_low", "nodefreq_mid", "nodefreq_high"]
    else:
        names_nf = [f"nodefreq_q{i}" for i in range(n_b)]

    thr_nf = _quantile_thresholds(min_deg, n_b)
    nodefreq_bucket = _assign_quantile_bucket(min_deg, thr_nf, names_nf)

    tail_q = config.tail_quantile
    low_thr = np.percentile(min_deg, tail_q * 100)
    high_thr = np.percentile(min_deg, (1 - tail_q) * 100)
    nodefreq_tail = min_deg <= low_thr
    nodefreq_head = min_deg >= high_thr

    if n_b == 3:
        names_rf = ["relfreq_low", "relfreq_mid", "relfreq_high"]
    else:
        names_rf = [f"relfreq_q{i}" for i in range(n_b)]
    thr_rf = _quantile_thresholds(rel_tc[rel_seen], n_b) if np.any(rel_seen) else np.array([])
    relfreq_bucket = np.empty(nq, dtype=object)
    for i in range(nq):
        if not rel_seen[i]:
            relfreq_bucket[i] = "relation_unseen"
        else:
            rv = rel_tc[i]
            if thr_rf.size == 0:
                relfreq_bucket[i] = names_rf[0]
                continue
            b = 0
            for t in thr_rf:
                if rv > t:
                    b += 1
                else:
                    break
            b = min(b, len(names_rf) - 1)
            relfreq_bucket[i] = names_rf[b]

    thr_sp = _quantile_thresholds(rel_uep, n_b)
    if n_b == 3:
        names_sp = ["relspan_low", "relspan_mid", "relspan_high"]
    else:
        names_sp = [f"relspan_q{i}" for i in range(n_b)]
    relspan_bucket = _assign_quantile_bucket(rel_uep, thr_sp, names_sp)

    thr_sup = _quantile_thresholds(cn.astype(np.float64), n_b)
    if n_b == 3:
        names_sup = ["support_low", "support_mid", "support_high"]
    else:
        names_sup = [f"support_q{i}" for i in range(n_b)]
    support_bucket = _assign_quantile_bucket(cn.astype(np.float64), thr_sup, names_sup)

    hard_deg = -_zscore(min_deg)
    hard_dist = _zscore(edge_dist)
    hard_rel = -_zscore(np.log1p(rel_tc))
    hard_sup = -_zscore(cn)
    combined = (hard_deg + hard_dist + hard_rel + hard_sup) / 4.0
    thr_c = _quantile_thresholds(combined, n_b)
    if n_b == 3:
        names_c = ["combined_easy", "combined_mid", "combined_hard"]
    else:
        names_c = [f"combined_q{i}" for i in range(n_b)]
    combined_bucket = _assign_quantile_bucket(combined, thr_c, names_c)

    features = {
        **features,
        "nodefreq_bucket": nodefreq_bucket,
        "nodefreq_tail": nodefreq_tail,
        "nodefreq_head": nodefreq_head,
        "relfreq_bucket": relfreq_bucket,
        "relspan_bucket": relspan_bucket,
        "support_bucket": support_bucket,
        "combined_difficulty_score": combined,
        "combined_bucket": combined_bucket,
    }

    out: dict[str, Any] = {
        "node_frequency": {
            "thresholds": {
                "quantile_boundaries": thr_nf.tolist(),
                "tail_low_at_most": float(low_thr),
                "head_high_at_least": float(high_thr),
            },
            "buckets": {},
        },
        "novelty": {"buckets": {}},
        "distance_to_train_support": {"buckets": {}},
        "relation_frequency": {
            "thresholds": {"quantile_boundaries": thr_rf.tolist()},
            "buckets": {},
        },
        "relation_span": {
            "thresholds": {"quantile_boundaries": thr_sp.tolist()},
            "buckets": {},
        },
        "structural_support": {
            "thresholds": {"quantile_boundaries": thr_sup.tolist()},
            "buckets": {},
        },
        "combined": {
            "thresholds": {"quantile_boundaries": thr_c.tolist()},
            "buckets": {},
        },
    }

    def fill_bucket(group_key: str, labels: np.ndarray, bucket_names: list[str]):
        for bn in bucket_names:
            idx = np.where(labels == bn)[0]
            out[group_key]["buckets"][bn] = _metrics_subset(pos_scores, neg_scores, idx)

    fill_bucket(
        "node_frequency",
        nodefreq_bucket,
        sorted(set(nodefreq_bucket.tolist())),
    )
    for section, name_list in (
        ("novelty", ["both_seen", "one_unseen", "both_unseen"]),
        ("distance_to_train_support", ["dist_0", "dist_1", "dist_2", "dist_3plus"]),
    ):
        for bn in name_list:
            lab = features["novelty_bucket"] if section == "novelty" else features["distance_bucket"]
            idx = np.where(lab == bn)[0]
            out[section]["buckets"][bn] = _metrics_subset(pos_scores, neg_scores, idx)

    rf_names = sorted(set(relfreq_bucket.tolist()))
    for bn in rf_names:
        idx = np.where(relfreq_bucket == bn)[0]
        out["relation_frequency"]["buckets"][bn] = _metrics_subset(pos_scores, neg_scores, idx)

    rs_names = sorted(set(relspan_bucket.tolist()))
    for bn in rs_names:
        idx = np.where(relspan_bucket == bn)[0]
        out["relation_span"]["buckets"][bn] = _metrics_subset(pos_scores, neg_scores, idx)

    sup_names = sorted(set(support_bucket.tolist()))
    for bn in sup_names:
        idx = np.where(support_bucket == bn)[0]
        out["structural_support"]["buckets"][bn] = _metrics_subset(pos_scores, neg_scores, idx)

    cb_names = sorted(set(combined_bucket.tolist()))
    for bn in cb_names:
        idx = np.where(combined_bucket == bn)[0]
        out["combined"]["buckets"][bn] = _metrics_subset(pos_scores, neg_scores, idx)

    ext_tail = _metrics_subset(pos_scores, neg_scores, np.where(nodefreq_tail)[0])
    ext_head = _metrics_subset(pos_scores, neg_scores, np.where(nodefreq_head)[0])
    out["node_frequency"]["buckets"]["nodefreq_tail"] = ext_tail
    out["node_frequency"]["buckets"]["nodefreq_head"] = ext_head

    return out, features


def compute_ood_difficulty_for_split(
    *,
    split_name: str,
    train_pos_edge_index: torch.Tensor,
    train_pos_edge_attr: torch.Tensor | None,
    full_edge_index: torch.Tensor,
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor | None,
    query_pos_edge_index: torch.Tensor,
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    negatives_per_pos: int,
    num_nodes: int,
    relation_lookup: dict[str, int],
    split_protocol: str,
    config: OODDifficultyConfig | None = None,
) -> dict[str, Any]:
    """
    Compute OOD difficulty features, bucketed metrics, and metadata for one split (val or test).
    """
    cfg = config or OODDifficultyConfig()
    pos_scores = np.asarray(pos_scores, dtype=np.float32).reshape(-1)
    neg_scores = np.asarray(neg_scores, dtype=np.float32)
    if pos_scores.shape[0] == 0:
        return {
            "split": split_name,
            "metrics": {},
            "meta": {
                "split_protocol": split_protocol,
                "distance_graph": "full_data.edge_index_undirected_bfs",
                "support_graph": "train_pos_edges_only",
                "note": "empty query split",
            },
        }

    feats = compute_ood_features_for_positives(
        query_pos_edge_index=query_pos_edge_index,
        train_pos_edge_index=train_pos_edge_index,
        train_pos_edge_attr=train_pos_edge_attr,
        full_edge_index=full_edge_index,
        positive_edge_index=positive_edge_index,
        positive_edge_attr=positive_edge_attr,
        num_nodes=num_nodes,
        relation_lookup=relation_lookup,
    )

    metrics_block, feats = _assign_buckets_and_metrics(feats, pos_scores, neg_scores, cfg)

    return {
        "split": split_name,
        "metrics": metrics_block,
        "features": feats,
        "meta": {
            "split_protocol": split_protocol,
            "distance_graph": "full_data.edge_index (undirected), multi_source BFS from train-seen nodes",
            "support_graph": "train positive edges only",
            "degree_source": "train positive edges (undirected)",
            "bucket_mode": cfg.bucket_mode,
            "num_quantile_buckets": cfg.num_quantile_buckets,
            "tail_quantile": cfg.tail_quantile,
        },
    }


def write_edge_predictions_csv(
    path: Path,
    *,
    split_name: str,
    features: dict[str, Any],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    query_neg_edge_index: torch.Tensor,
    negatives_per_pos: int,
    relation_lookup: dict[str, int],
    seed: int,
    model_name: str,
    edge_relation_mode: str | None,
    split_protocol: str,
    append: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inv = features.get("inv_relation_lookup", _inverse_relation_lookup(relation_lookup))
    neg_ei = query_neg_edge_index.cpu().long()

    rows: list[dict[str, Any]] = []
    n = pos_scores.shape[0]
    for i in range(n):
        rid = int(features["relation_id"][i])
        in_cat = bool(features["relation_in_catalog"][i])
        if rid == RELATION_ID_NOT_IN_CATALOG or not in_cat:
            rname = "NOT_IN_CATALOG"
        else:
            rname = inv.get(rid, str(rid))
        base = {
            "split": split_name,
            "label": 1,
            "score": float(pos_scores[i]),
            "src": int(features["src"][i]),
            "dst": int(features["dst"][i]),
            "relation_id": rid,
            "relation_name": rname,
            "relation_in_catalog": in_cat,
            "seed": seed,
            "model_name": model_name,
            "edge_relation_mode": edge_relation_mode or "",
            "split_protocol": split_protocol,
            "src_train_degree": int(features["src_train_degree"][i]),
            "dst_train_degree": int(features["dst_train_degree"][i]),
            "min_endpoint_degree": int(features["min_endpoint_degree"][i]),
            "src_seen_in_train": bool(features["src_seen_in_train"][i]),
            "dst_seen_in_train": bool(features["dst_seen_in_train"][i]),
            "novelty_bucket": str(features["novelty_bucket"][i]),
            "src_dist_to_train_support": int(features["src_dist_to_train_support"][i]),
            "dst_dist_to_train_support": int(features["dst_dist_to_train_support"][i]),
            "edge_dist_to_train_support": int(features["edge_dist_to_train_support"][i]),
            "distance_bucket": str(features["distance_bucket"][i]),
            "relation_train_count": int(features["relation_train_count"][i]),
            "relation_seen_in_train": bool(features["relation_seen_in_train"][i]),
            "relation_freq_bucket": str(features["relfreq_bucket"][i]),
            "relation_unique_endpoint_count_train": int(
                features["relation_unique_endpoint_count_train"][i]
            ),
            "relation_span_bucket": str(features["relspan_bucket"][i]),
            "common_neighbors_train": int(features["common_neighbors_train"][i]),
            "jaccard_train_neighbors": float(features["jaccard_train_neighbors"][i]),
            "support_bucket": str(features["support_bucket"][i]),
            "combined_difficulty_score": float(features["combined_difficulty_score"][i]),
            "combined_bucket": str(features["combined_bucket"][i]),
        }
        rows.append(base)
        row_off = i * negatives_per_pos
        for j in range(negatives_per_pos):
            ndst = int(neg_ei[1, row_off + j])
            nsrc = int(neg_ei[0, row_off + j])
            rows.append(
                {
                    **base,
                    "label": 0,
                    "score": float(neg_scores[i, j]),
                    "src": nsrc,
                    "dst": ndst,
                }
            )

    fieldnames = list(rows[0].keys()) if rows else []
    mode = "a" if append else "w"
    with path.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not append and fieldnames:
            w.writeheader()
        w.writerows(rows)


def build_ood_json_payload(
    *,
    val_block: dict[str, Any] | None,
    test_block: dict[str, Any] | None,
) -> dict[str, Any]:
    """Strip heavy `features` from JSON (keep only metrics + meta per split)."""

    def slim(b: dict[str, Any] | None) -> dict[str, Any] | None:
        if b is None:
            return None
        return {"metrics": b.get("metrics", {}), "meta": b.get("meta", {})}

    out: dict[str, Any] = {}
    if val_block:
        out["val"] = slim(val_block)
    if test_block:
        out["test"] = slim(test_block)
    return out
