from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn

_LOG = logging.getLogger(__name__)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _binary_metrics_from_scores(
    pos_scores: torch.Tensor | np.ndarray,
    neg_scores: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    pos_scores_np = _to_numpy(pos_scores)
    neg_scores_np = _to_numpy(neg_scores)

    y = np.concatenate(
        [
            np.ones(len(pos_scores_np), dtype=np.float32),
            np.zeros(len(neg_scores_np), dtype=np.float32),
        ],
        axis=0,
    )
    scores = np.concatenate([pos_scores_np, neg_scores_np], axis=0)

    try:
        auc = float(roc_auc_score(y, scores))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(y, scores))
    except ValueError:
        ap = float("nan")

    return {"roc_auc": auc, "average_precision": ap}


def _hits_at_k(pos_scores: np.ndarray, neg_scores: np.ndarray, k: int) -> float:
    if pos_scores.size == 0:
        return float("nan")
    hits = 0.0
    for pos_score, neg_group in zip(pos_scores, neg_scores, strict=False):
        rank = 1 + int(np.sum(neg_group > pos_score))
        hits += float(rank <= k)
    return hits / float(pos_scores.shape[0])


def grouped_link_prediction_metrics(
    pos_scores: torch.Tensor | np.ndarray,
    neg_scores: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    pos_scores_np = _to_numpy(pos_scores).reshape(-1)
    neg_scores_np = _to_numpy(neg_scores)
    flat_metrics = _binary_metrics_from_scores(pos_scores_np, neg_scores_np.reshape(-1))
    flat_metrics["hits@1"] = _hits_at_k(pos_scores_np, neg_scores_np, 1)
    flat_metrics["hits@3"] = _hits_at_k(pos_scores_np, neg_scores_np, 3)
    flat_metrics["hits@10"] = _hits_at_k(pos_scores_np, neg_scores_np, 10)
    return flat_metrics


def _empty_subgroup_metrics() -> dict[str, float]:
    return {
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "hits@1": float("nan"),
        "hits@3": float("nan"),
        "hits@10": float("nan"),
    }


def _build_uv_to_relation_id(
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor,
) -> dict[tuple[int, int], int]:
    """Map canonical undirected (u,v) -> relation id."""
    ei = positive_edge_index.cpu().long()
    ea = positive_edge_attr.cpu().long().view(-1)
    out: dict[tuple[int, int], int] = {}
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        a, b = (u, v) if u <= v else (v, u)
        rid = int(ea[i].item())
        out[(a, b)] = rid
    return out


def compute_relation_holdout_metrics(
    *,
    query_pos_edge_index: torch.Tensor,
    positive_edge_index: torch.Tensor,
    positive_edge_attr: torch.Tensor,
    held_out_relation_ids: frozenset[int],
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> dict[str, Any]:
    """
    Split link-prediction scores by whether the query positive's relation type was held out
    from training (zero-shot relations).
    """
    if not held_out_relation_ids:
        return {}
    pos_scores = np.asarray(pos_scores, dtype=np.float32).reshape(-1)
    neg_scores = np.asarray(neg_scores, dtype=np.float32)
    uv_rel = _build_uv_to_relation_id(positive_edge_index, positive_edge_attr)
    qp = query_pos_edge_index.cpu().long()
    nq = qp.size(1)
    is_unseen = np.zeros(nq, dtype=bool)
    for i in range(nq):
        u, v = int(qp[0, i]), int(qp[1, i])
        a, b = (u, v) if u <= v else (v, u)
        rid = int(uv_rel.get((a, b), 0))
        is_unseen[i] = rid in held_out_relation_ids

    seen_idx = np.where(~is_unseen)[0]
    unseen_idx = np.where(is_unseen)[0]

    def _sub(idx: np.ndarray) -> dict[str, float]:
        if idx.size == 0:
            return _empty_subgroup_metrics()
        return grouped_link_prediction_metrics(pos_scores[idx], neg_scores[idx])

    return {
        "seen_relations": _sub(seen_idx),
        "unseen_relations": _sub(unseen_idx),
        "n_seen_queries": int(seen_idx.size),
        "n_unseen_queries": int(unseen_idx.size),
    }


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


def _undirected_edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    a = torch.minimum(src, dst)
    b = torch.maximum(src, dst)
    return a * num_nodes + b


def _prune_supervision_edges(batch, *, device: torch.device):
    if batch.edge_index.numel() == 0:
        return batch.edge_index, getattr(batch, "edge_attr", None)

    global_nodes = batch.n_id.cpu()
    edge_index_cpu = batch.edge_index.cpu()
    edge_keys = _undirected_edge_keys(global_nodes[edge_index_cpu], int(global_nodes.numel() + global_nodes.max().item()))

    supervised_keys = _undirected_edge_keys(
        global_nodes[batch.edge_label_index.cpu()],
        int(global_nodes.numel() + global_nodes.max().item()),
    ).unique()
    keep_mask = ~torch.isin(edge_keys, supervised_keys)
    pruned_edge_index = batch.edge_index[:, keep_mask.to(batch.edge_index.device)].to(device)

    pruned_edge_attr = None
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        pruned_edge_attr = batch.edge_attr[keep_mask.to(batch.edge_attr.device)].to(device)
    return pruned_edge_index, pruned_edge_attr


def _score_query_edges(
    *,
    z: torch.Tensor,
    edge_label_index: torch.Tensor,
    decoder: str,
    decoder_model: nn.Module | None,
    rel_decode_bundle: nn.Module | None = None,
    relation_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    src = edge_label_index[0]
    dst = edge_label_index[1]
    if rel_decode_bundle is not None and relation_ids is not None:
        logits = rel_decode_bundle.decode_logits(z, edge_label_index, relation_ids)
        return torch.sigmoid(logits)
    if decoder == "dot":
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))
    if decoder == "mlp":
        if decoder_model is None:
            raise ValueError("decoder_model is required for decoder='mlp'")
        return torch.sigmoid(decoder_model(z[src], z[dst]))
    raise ValueError(f"Unsupported decoder: {decoder}")


def _bucket_metrics(
    *,
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    query_buckets: list[str] | None,
) -> list[dict[str, float | str | int]]:
    if query_buckets is None:
        return []

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, bucket in enumerate(query_buckets):
        grouped[bucket].append(idx)

    results: list[dict[str, float | str | int]] = []
    for bucket_name in ("low", "medium", "high"):
        indices = grouped.get(bucket_name, [])
        if indices:
            metrics = grouped_link_prediction_metrics(
                pos_scores[np.array(indices)],
                neg_scores[np.array(indices)],
            )
        else:
            metrics = {
                "roc_auc": float("nan"),
                "average_precision": float("nan"),
                "hits@1": float("nan"),
                "hits@3": float("nan"),
                "hits@10": float("nan"),
            }
        metrics["bucket"] = bucket_name
        metrics["count"] = len(indices)
        results.append(metrics)
    return results


def _evaluate_inductive_link_prediction_fullgraph(
    model: nn.Module,
    data,
    *,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    negatives_per_pos: int,
    device: torch.device,
    edge_aware: bool,
    decoder: str = "dot",
    decoder_model: nn.Module | None = None,
    batch_size: int = 64,
    query_buckets: list[str] | None = None,
    return_scores: bool = False,
    rel_decode_bundle: nn.Module | None = None,
    query_uv_relation: dict[tuple[int, int], int] | None = None,
) -> dict[str, float] | dict[str, object]:
    """
    Same metrics as sampled eval, but one full-graph forward per batch with
    query edges pruned from the adjacency (no pyg-lib / torch-sparse).
    """
    from kgml_new.training.relation_decode_batch import (
        build_canonical_uv_relation_lookup,
        relation_ids_for_query_batch,
    )

    model = model.to(device)
    model.eval()
    if rel_decode_bundle is not None:
        rel_decode_bundle = rel_decode_bundle.to(device)
        rel_decode_bundle.eval()
    if decoder_model is not None:
        decoder_model = decoder_model.to(device)
        decoder_model.eval()

    x = data.x.to(device)
    base_edge_index = data.edge_index
    base_edge_attr = getattr(data, "edge_attr", None)
    num_nodes = int(data.num_nodes)

    pos_edge_index = pos_edge_index.cpu().long()
    neg_edge_index = neg_edge_index.cpu().long()

    if rel_decode_bundle is not None and query_uv_relation is None:
        query_uv_relation = build_canonical_uv_relation_lookup(
            data.edge_index.cpu(), data.edge_attr.cpu()
        )

    if pos_edge_index.numel() == 0:
        empty: dict[str, float | list | np.ndarray | object] = {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "hits@1": float("nan"),
            "hits@3": float("nan"),
            "hits@10": float("nan"),
            "bucket_metrics": _bucket_metrics(
                pos_scores=np.empty((0,), dtype=np.float32),
                neg_scores=np.empty((0, negatives_per_pos), dtype=np.float32),
                query_buckets=query_buckets,
            ),
        }
        if return_scores:
            empty["pos_scores"] = np.empty((0,), dtype=np.float32)
            empty["neg_scores"] = np.empty((0, negatives_per_pos), dtype=np.float32)
        return empty

    pos_scores_parts: list[torch.Tensor] = []
    neg_scores_parts: list[torch.Tensor] = []

    for start in range(0, pos_edge_index.size(1), batch_size):
        end = min(start + batch_size, pos_edge_index.size(1))
        batch_pos = pos_edge_index[:, start:end]
        batch_neg = neg_edge_index[:, start * negatives_per_pos : end * negatives_per_pos]
        edge_label_index = torch.cat([batch_pos, batch_neg], dim=1)
        supervised_keys = _undirected_edge_keys(edge_label_index, num_nodes).unique()

        if base_edge_index.numel() == 0:
            pruned_edge_index = base_edge_index.to(device)
            pruned_edge_attr = None
        else:
            edge_keys = _undirected_edge_keys(base_edge_index.cpu(), num_nodes)
            keep_mask = ~torch.isin(edge_keys, supervised_keys)
            pruned_edge_index = base_edge_index[:, keep_mask.to(base_edge_index.device)].to(
                device
            )
            pruned_edge_attr = None
            if base_edge_attr is not None:
                pruned_edge_attr = base_edge_attr[
                    keep_mask.to(base_edge_attr.device)
                ].to(device)

        with torch.no_grad():
            if edge_aware:
                z = model(x, pruned_edge_index, pruned_edge_attr)
            else:
                z = model(x, pruned_edge_index)
            eli = edge_label_index.to(device)
            rel_kw: dict = {}
            if rel_decode_bundle is not None and query_uv_relation is not None:
                rel_kw = {
                    "rel_decode_bundle": rel_decode_bundle,
                    "relation_ids": relation_ids_for_query_batch(
                        batch_pos,
                        batch_neg,
                        query_uv_relation,
                        negatives_per_pos,
                        device=device,
                    ),
                }
            scores = _score_query_edges(
                z=z,
                edge_label_index=eli,
                decoder=decoder,
                decoder_model=decoder_model,
                **rel_kw,
            )

        pos_count = batch_pos.size(1)
        pos_scores_parts.append(scores[:pos_count].cpu())
        neg_scores_parts.append(scores[pos_count:].view(pos_count, negatives_per_pos).cpu())

    pos_scores = torch.cat(pos_scores_parts, dim=0).numpy()
    neg_scores = torch.cat(neg_scores_parts, dim=0).numpy()
    metrics: dict[str, float | list | np.ndarray | object] = grouped_link_prediction_metrics(
        pos_scores, neg_scores
    )
    metrics["bucket_metrics"] = _bucket_metrics(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        query_buckets=query_buckets,
    )
    if return_scores:
        metrics["pos_scores"] = pos_scores
        metrics["neg_scores"] = neg_scores
    return metrics


def evaluate_inductive_link_prediction(
    model: nn.Module,
    data,
    *,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    negatives_per_pos: int,
    device: torch.device,
    edge_aware: bool,
    num_neighbors: list[int],
    decoder: str = "dot",
    decoder_model: nn.Module | None = None,
    batch_size: int = 64,
    query_buckets: list[str] | None = None,
    return_scores: bool = False,
    rel_decode_bundle: nn.Module | None = None,
    query_uv_relation: dict[tuple[int, int], int] | None = None,
) -> dict[str, float] | dict[str, object]:
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

    from kgml_new.training.relation_decode_batch import (
        build_canonical_uv_relation_lookup,
        relation_ids_for_query_batch,
    )

    if not WITH_PYG_LIB and not WITH_TORCH_SPARSE:
        _LOG.warning(
            "Neighbor sampling backend unavailable; using full-graph inductive eval "
            "(query edges pruned from adjacency per batch)."
        )
        return _evaluate_inductive_link_prediction_fullgraph(
            model=model,
            data=data,
            pos_edge_index=pos_edge_index,
            neg_edge_index=neg_edge_index,
            negatives_per_pos=negatives_per_pos,
            device=device,
            edge_aware=edge_aware,
            decoder=decoder,
            decoder_model=decoder_model,
            batch_size=batch_size,
            query_buckets=query_buckets,
            return_scores=return_scores,
            rel_decode_bundle=rel_decode_bundle,
            query_uv_relation=query_uv_relation,
        )

    model = model.to(device)
    model.eval()
    if rel_decode_bundle is not None:
        rel_decode_bundle = rel_decode_bundle.to(device)
        rel_decode_bundle.eval()
    if decoder_model is not None:
        decoder_model = decoder_model.to(device)
        decoder_model.eval()

    pos_edge_index = pos_edge_index.cpu().long()
    neg_edge_index = neg_edge_index.cpu().long()

    if rel_decode_bundle is not None and query_uv_relation is None:
        query_uv_relation = build_canonical_uv_relation_lookup(
            data.edge_index.cpu(), data.edge_attr.cpu()
        )

    if pos_edge_index.numel() == 0:
        empty: dict[str, float | list | np.ndarray | object] = {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "hits@1": float("nan"),
            "hits@3": float("nan"),
            "hits@10": float("nan"),
            "bucket_metrics": _bucket_metrics(
                pos_scores=np.empty((0,), dtype=np.float32),
                neg_scores=np.empty((0, negatives_per_pos), dtype=np.float32),
                query_buckets=query_buckets,
            ),
        }
        if return_scores:
            empty["pos_scores"] = np.empty((0,), dtype=np.float32)
            empty["neg_scores"] = np.empty((0, negatives_per_pos), dtype=np.float32)
        return empty

    pos_scores_parts: list[torch.Tensor] = []
    neg_scores_parts: list[torch.Tensor] = []

    for start in range(0, pos_edge_index.size(1), batch_size):
        end = min(start + batch_size, pos_edge_index.size(1))
        batch_pos = pos_edge_index[:, start:end]
        batch_neg = neg_edge_index[:, start * negatives_per_pos : end * negatives_per_pos]
        edge_label_index = torch.cat([batch_pos, batch_neg], dim=1)
        edge_label = torch.cat(
            [
                torch.ones(batch_pos.size(1), dtype=torch.float32),
                torch.zeros(batch_neg.size(1), dtype=torch.float32),
            ],
            dim=0,
        )

        loader = LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            batch_size=edge_label_index.size(1),
            shuffle=False,
            neg_sampling_ratio=0.0,
        )
        batch = next(iter(loader)).to(device)
        pruned_edge_index, pruned_edge_attr = _prune_supervision_edges(batch, device=device)

        with torch.no_grad():
            if edge_aware:
                z = model(batch.x, pruned_edge_index, pruned_edge_attr)
            else:
                z = model(batch.x, pruned_edge_index)
            rel_kw: dict = {}
            if rel_decode_bundle is not None and query_uv_relation is not None:
                rel_ids = relation_ids_for_query_batch(
                    batch_pos.to(device),
                    batch_neg.to(device),
                    query_uv_relation,
                    negatives_per_pos,
                    device=device,
                )
                rel_kw = {
                    "rel_decode_bundle": rel_decode_bundle,
                    "relation_ids": rel_ids,
                }
            scores = _score_query_edges(
                z=z,
                edge_label_index=batch.edge_label_index,
                decoder=decoder,
                decoder_model=decoder_model,
                **rel_kw,
            )

        pos_count = batch_pos.size(1)
        pos_scores_parts.append(scores[:pos_count].cpu())
        neg_scores_parts.append(scores[pos_count:].view(pos_count, negatives_per_pos).cpu())

    pos_scores = torch.cat(pos_scores_parts, dim=0).numpy()
    neg_scores = torch.cat(neg_scores_parts, dim=0).numpy()
    metrics: dict[str, float | list | np.ndarray | object] = grouped_link_prediction_metrics(
        pos_scores, neg_scores
    )
    metrics["bucket_metrics"] = _bucket_metrics(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        query_buckets=query_buckets,
    )
    if return_scores:
        metrics["pos_scores"] = pos_scores
        metrics["neg_scores"] = neg_scores
    return metrics


def node_classification_scores(
    logits: torch.Tensor,
    y_true: torch.Tensor,
) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    accuracy = float((pred == y_true).float().mean().item())
    macro_f1 = float(
        f1_score(
            y_true.detach().cpu().numpy(),
            pred.detach().cpu().numpy(),
            average="macro",
            zero_division=0,
        )
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1}
