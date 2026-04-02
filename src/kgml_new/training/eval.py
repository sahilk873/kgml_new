from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn


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
) -> torch.Tensor:
    src = edge_label_index[0]
    dst = edge_label_index[1]
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
) -> dict[str, float] | dict[str, object]:
    from torch_geometric.loader import LinkNeighborLoader

    model = model.to(device)
    model.eval()
    if decoder_model is not None:
        decoder_model = decoder_model.to(device)
        decoder_model.eval()

    pos_edge_index = pos_edge_index.cpu().long()
    neg_edge_index = neg_edge_index.cpu().long()

    if pos_edge_index.numel() == 0:
        empty = {
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
            scores = _score_query_edges(
                z=z,
                edge_label_index=batch.edge_label_index,
                decoder=decoder,
                decoder_model=decoder_model,
            )

        pos_count = batch_pos.size(1)
        pos_scores_parts.append(scores[:pos_count].cpu())
        neg_scores_parts.append(scores[pos_count:].view(pos_count, negatives_per_pos).cpu())

    pos_scores = torch.cat(pos_scores_parts, dim=0).numpy()
    neg_scores = torch.cat(neg_scores_parts, dim=0).numpy()
    metrics = grouped_link_prediction_metrics(pos_scores, neg_scores)
    metrics["bucket_metrics"] = _bucket_metrics(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        query_buckets=query_buckets,
    )
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
