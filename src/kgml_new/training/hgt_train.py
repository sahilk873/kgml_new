from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

from kgml_new.models.hgt import HGTLinkPredictor
from kgml_new.training.eval import (
    grouped_link_prediction_metrics,
    mean_bce_logits_link_prediction,
)

_LOG = logging.getLogger(__name__)


def _roc_ap_safe(y: np.ndarray | torch.Tensor, s: np.ndarray | torch.Tensor) -> tuple[float, float]:
    """ROC-AUC / AP with sklearn; returns (nan, nan) if undefined (empty or single-class labels)."""
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(s, dtype=np.float64).reshape(-1)
    if y_arr.size == 0 or s_arr.size == 0 or y_arr.shape != s_arr.shape:
        return float("nan"), float("nan")
    pos = float(np.sum(y_arr > 0.5))
    neg = float(np.sum(y_arr <= 0.5))
    if pos == 0.0 or neg == 0.0:
        return float("nan"), float("nan")
    try:
        return float(roc_auc_score(y_arr, s_arr)), float(average_precision_score(y_arr, s_arr))
    except ValueError:
        return float("nan"), float("nan")


@dataclass
class HGTTrainResult:
    train_loss: list[float]
    val_auc: list[float]
    val_ap: list[float]
    val_loss: list[float]


def neighbor_sampler_backend_available() -> bool:
    return bool(WITH_PYG_LIB or WITH_TORCH_SPARSE)


def parse_neighbor_fanouts(spec: str | None, num_layers: int, default_fanout: int = 15) -> list[int]:
    """Parse `--num-neighbors` into one fanout integer per message-passing layer."""
    if not spec or not str(spec).strip():
        return [default_fanout] * num_layers
    parts = [int(x.strip()) for x in str(spec).strip().split(",") if x.strip()]
    if len(parts) == 1:
        return [parts[0]] * num_layers
    if len(parts) != num_layers:
        raise ValueError(
            f"num_neighbors list length must be 1 or match num_layers ({num_layers}); got {parts!r}"
        )
    return parts


def build_hetero_num_neighbors(data: HeteroData, fanouts_per_layer: list[int]) -> dict:
    """One fanout list per edge type (same hop schedule for each relation)."""
    etypes = list(data.edge_types)
    return {et: list(fanouts_per_layer) for et in etypes}


def _sample_negative_edges(
    num_src: int,
    num_dst: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    src = torch.randint(0, num_src, (num_samples,), device=device)
    dst = torch.randint(0, num_dst, (num_samples,), device=device)
    return torch.stack([src, dst], dim=0)


@torch.inference_mode()
def _score_hgt_relation_fullgraph(
    model: HGTLinkPredictor,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    z_dict = model.encode(data.to(device))
    pos_scores = model.score_edges(z_dict, edge_type, pos_edge_index.to(device))
    neg_scores = model.score_edges(z_dict, edge_type, neg_edge_index.to(device))
    return pos_scores.detach().cpu(), neg_scores.detach().cpu()


@torch.inference_mode()
def _evaluate_hgt_relation_sampled(
    model: HGTLinkPredictor,
    data_cpu: HeteroData,
    edge_type: tuple[str, str, str],
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device,
    *,
    num_neighbors_dict: dict,
    eval_batch_size_pos: int,
    negatives_per_pos: int,
) -> tuple[float, float]:
    """Link-neighbor batched encode + DistMult scores (no full-graph forward)."""
    pos_scores, neg_scores = _score_hgt_relation_sampled(
        model=model,
        data_cpu=data_cpu,
        edge_type=edge_type,
        pos_edge_index=pos_edge_index,
        neg_edge_index=neg_edge_index,
        device=device,
        num_neighbors_dict=num_neighbors_dict,
        eval_batch_size_pos=eval_batch_size_pos,
        negatives_per_pos=negatives_per_pos,
    )
    y = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
    s = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    return _roc_ap_safe(y, s)


@torch.inference_mode()
def _score_hgt_relation_sampled(
    model: HGTLinkPredictor,
    data_cpu: HeteroData,
    edge_type: tuple[str, str, str],
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device,
    *,
    num_neighbors_dict: dict,
    eval_batch_size_pos: int,
    negatives_per_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive/negative logits from mini-batch neighbor-sampled evaluation."""
    from torch_geometric.loader import LinkNeighborLoader

    model.eval()
    pos_edge_index = pos_edge_index.cpu().long()
    neg_edge_index = neg_edge_index.cpu().long()
    n_pos = pos_edge_index.size(1)
    if n_pos == 0:
        return float("nan"), float("nan")

    pos_scores_parts: list[torch.Tensor] = []
    neg_scores_parts: list[torch.Tensor] = []

    for start in range(0, n_pos, eval_batch_size_pos):
        end = min(start + eval_batch_size_pos, n_pos)
        batch_pos = pos_edge_index[:, start:end]
        neg_start = start * negatives_per_pos
        neg_end = end * negatives_per_pos
        batch_neg = neg_edge_index[:, neg_start:neg_end]

        edge_label_index = torch.cat([batch_pos, batch_neg], dim=1)
        edge_label = torch.cat(
            [
                torch.ones(batch_pos.size(1), dtype=torch.float32),
                torch.zeros(batch_neg.size(1), dtype=torch.float32),
            ],
            dim=0,
        )

        loader = LinkNeighborLoader(
            data_cpu,
            num_neighbors=num_neighbors_dict,
            edge_label_index=(edge_type, edge_label_index),
            edge_label=edge_label,
            batch_size=edge_label_index.size(1),
            shuffle=False,
            neg_sampling_ratio=None,
        )
        batch = next(iter(loader)).to(device)

        z_dict = model.encode(batch)
        ei = batch[edge_type].edge_label_index
        logits = model.score_edges(z_dict, edge_type, ei)

        pos_mask = batch[edge_type].edge_label.float() > 0.5
        neg_mask = ~pos_mask
        pos_scores_parts.append(logits[pos_mask])
        neg_scores_parts.append(logits[neg_mask])

    pos_scores = torch.cat(pos_scores_parts, dim=0)
    neg_scores = torch.cat(neg_scores_parts, dim=0)
    return pos_scores.detach().cpu(), neg_scores.detach().cpu()


def _metrics_from_pos_neg_scores(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    n_pos: int,
) -> dict[str, float]:
    """Compute ROC/AP always, Hits@K when negatives can be grouped per query positive."""
    mean_bce = float(
        mean_bce_logits_link_prediction(
            pos_scores.reshape(-1).float(), neg_scores.reshape(-1).float()
        )
    )
    pos_np = pos_scores.numpy().reshape(-1)
    neg_np = neg_scores.numpy().reshape(-1)
    y = np.concatenate(
        [
            np.ones(pos_np.shape[0], dtype=np.float32),
            np.zeros(neg_np.shape[0], dtype=np.float32),
        ]
    )
    s = np.concatenate([pos_np, neg_np])
    auc, ap = _roc_ap_safe(y, s)
    metrics: dict[str, float] = {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "mean_bce_logits": mean_bce,
    }
    if n_pos > 0 and neg_np.size > 0 and neg_np.size % n_pos == 0:
        neg_matrix = neg_np.reshape(n_pos, neg_np.size // n_pos)
        grouped = grouped_link_prediction_metrics(pos_np, neg_matrix)
        metrics["hits@1"] = float(grouped.get("hits@1", float("nan")))
        metrics["hits@3"] = float(grouped.get("hits@3", float("nan")))
        metrics["hits@10"] = float(grouped.get("hits@10", float("nan")))
    else:
        metrics["hits@1"] = float("nan")
        metrics["hits@3"] = float("nan")
        metrics["hits@10"] = float("nan")
    return metrics


@torch.inference_mode()
def evaluate_hgt_relation(
    model: HGTLinkPredictor,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device,
    *,
    neighbor_sampling: bool = True,
    num_neighbors_fanouts: list[int] | None = None,
    eval_batch_size_pos: int = 1024,
    num_layers: int | None = None,
) -> tuple[float, float]:
    """Backward-compatible wrapper returning only ROC-AUC/AP."""
    metrics = evaluate_hgt_relation_metrics(
        model=model,
        data=data,
        edge_type=edge_type,
        pos_edge_index=pos_edge_index,
        neg_edge_index=neg_edge_index,
        device=device,
        neighbor_sampling=neighbor_sampling,
        num_neighbors_fanouts=num_neighbors_fanouts,
        eval_batch_size_pos=eval_batch_size_pos,
        num_layers=num_layers,
    )
    return float(metrics["roc_auc"]), float(metrics["average_precision"])


@torch.inference_mode()
def evaluate_hgt_relation_metrics(
    model: HGTLinkPredictor,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device,
    *,
    neighbor_sampling: bool = True,
    num_neighbors_fanouts: list[int] | None = None,
    eval_batch_size_pos: int = 1024,
    num_layers: int | None = None,
) -> dict[str, float]:
    """
    Link prediction metrics including ROC-AUC/AP and Hits@1/3/10 when negatives
    are provided as a fixed number per positive query edge.
    """
    num_layers_eff = num_layers if num_layers is not None else len(model.convs)
    fanouts = parse_neighbor_fanouts(
        None,
        num_layers_eff,
        default_fanout=15,
    )
    if num_neighbors_fanouts is not None:
        fanouts = num_neighbors_fanouts

    data_cpu = data.cpu()
    num_neighbors_dict = build_hetero_num_neighbors(data_cpu, fanouts)

    use_sampled = neighbor_sampling and neighbor_sampler_backend_available()
    if use_sampled:
        n_pos = pos_edge_index.size(1)
        n_neg = neg_edge_index.size(1)
        if n_pos > 0 and n_neg > 0 and n_neg % n_pos == 0:
            neg_per = n_neg // n_pos
            if neg_per > 0:
                pos_scores, neg_scores = _score_hgt_relation_sampled(
                    model,
                    data_cpu,
                    edge_type,
                    pos_edge_index,
                    neg_edge_index,
                    device,
                    num_neighbors_dict=num_neighbors_dict,
                    eval_batch_size_pos=max(1, int(eval_batch_size_pos)),
                    negatives_per_pos=neg_per,
                )
                return _metrics_from_pos_neg_scores(pos_scores, neg_scores, n_pos=n_pos)

    if neighbor_sampling and not neighbor_sampler_backend_available():
        _LOG.warning(
            "Neighbor sampling backend unavailable (install pyg_lib / torch_sparse); "
            "using full-graph HGT evaluation (may OOM on large KGs)."
        )
    elif neighbor_sampling and pos_edge_index.size(1) > 0 and neg_edge_index.size(1) % pos_edge_index.size(1) != 0:
        _LOG.warning(
            "neg_edge_index length not divisible by pos count; using full-graph HGT evaluation."
        )

    pos_scores, neg_scores = _score_hgt_relation_fullgraph(
        model, data_cpu, edge_type, pos_edge_index, neg_edge_index, device
    )
    return _metrics_from_pos_neg_scores(pos_scores, neg_scores, n_pos=pos_edge_index.size(1))


def train_hgt(
    model: HGTLinkPredictor,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    *,
    epochs: int = 20,
    batch_size: int = 1024,
    neg_samples: int = 1,
    learning_rate: float = 1e-3,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    early_stop_patience: int = 20,
    device: torch.device | None = None,
    val_pos_edge_index: torch.Tensor | None = None,
    val_neg_edge_index: torch.Tensor | None = None,
    neighbor_sampling: bool = True,
    num_neighbors_fanouts: list[int] | None = None,
    eval_batch_size_pos: int = 1024,
) -> HGTTrainResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if neg_samples < 1:
        raise ValueError("neg_samples must be >= 1 for HGT training")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    num_layers_model = len(model.convs)
    if num_neighbors_fanouts is not None:
        if len(num_neighbors_fanouts) != num_layers_model:
            raise ValueError(
                f"num_neighbors_fanouts length {len(num_neighbors_fanouts)} != model layers {num_layers_model}"
            )
        fanouts = num_neighbors_fanouts
    else:
        fanouts = parse_neighbor_fanouts(None, num_layers_model)

    use_loader = neighbor_sampling and neighbor_sampler_backend_available()
    if neighbor_sampling and not neighbor_sampler_backend_available():
        warnings.warn(
            "Neighbor sampling requested but pyg_lib/torch_sparse unavailable; "
            "falling back to full-graph HGT training (may OOM on large graphs).",
            stacklevel=2,
        )

    data_cpu = data.cpu()
    num_neighbors_dict = build_hetero_num_neighbors(data_cpu, fanouts)

    src_type, _, dst_type = edge_type
    pos_edge_index = data_cpu[edge_type].edge_index
    num_pos = pos_edge_index.size(1)
    if num_pos == 0:
        raise ValueError(
            "No training edges for the target relation on data[edge_type].edge_index "
            "(after run_hgt train split). Cannot train link prediction."
        )
    num_src = data_cpu[src_type].num_nodes
    num_dst = data_cpu[dst_type].num_nodes

    history = HGTTrainResult(train_loss=[], val_auc=[], val_ap=[], val_loss=[])

    best_val_auc = 0.0
    patience_counter = 0

    num_layers_for_eval = num_layers_model

    from torch_geometric.sampler import NegativeSampling

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0

        if use_loader:
            from torch_geometric.loader import LinkNeighborLoader

            loader = LinkNeighborLoader(
                data_cpu,
                num_neighbors=num_neighbors_dict,
                edge_label_index=(edge_type, pos_edge_index),
                edge_label=torch.ones(num_pos, dtype=torch.float32),
                neg_sampling=NegativeSampling(mode="binary", amount=float(neg_samples)),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=scaler is not None,
                ):
                    z_dict = model.encode(batch)
                    ei = batch[edge_type].edge_label_index
                    el = batch[edge_type].edge_label.float()
                    logits = model.score_edges(z_dict, edge_type, ei)
                    loss = F.binary_cross_entropy_with_logits(logits, el)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

                epoch_loss += float(loss.detach())
                steps += 1
                del loss, logits, z_dict, batch
        else:
            data_dev = data_cpu.to(device)
            perm = torch.randperm(num_pos, device=pos_edge_index.device)
            for start in range(0, num_pos, batch_size):
                idx = perm[start : start + batch_size]
                pos_batch = pos_edge_index[:, idx].to(device)
                neg_batch = _sample_negative_edges(
                    num_src,
                    num_dst,
                    num_samples=pos_batch.size(1) * neg_samples,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=scaler is not None,
                ):
                    z_dict = model.encode(data_dev)
                    pos_logits = model.score_edges(z_dict, edge_type, pos_batch)
                    neg_logits = model.score_edges(z_dict, edge_type, neg_batch)
                    loss = F.binary_cross_entropy_with_logits(
                        torch.cat([pos_logits, neg_logits]),
                        torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]),
                    )

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

                epoch_loss += float(loss.detach())
                steps += 1

                del pos_logits, neg_logits, loss

        avg_loss = epoch_loss / max(steps, 1)
        history.train_loss.append(avg_loss)

        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            val_metrics = evaluate_hgt_relation_metrics(
                model,
                data_cpu,
                edge_type,
                val_pos_edge_index,
                val_neg_edge_index,
                device,
                neighbor_sampling=use_loader,
                num_neighbors_fanouts=fanouts,
                eval_batch_size_pos=eval_batch_size_pos,
                num_layers=num_layers_for_eval,
            )
            auc = float(val_metrics["roc_auc"])
            ap = float(val_metrics["average_precision"])
            history.val_auc.append(auc)
            history.val_ap.append(ap)
            history.val_loss.append(float(val_metrics["mean_bce_logits"]))
            if not math.isnan(auc):
                scheduler.step(auc)
                if auc > best_val_auc:
                    best_val_auc = auc
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                _LOG.warning("Validation ROC-AUC is nan (single-class val set or metric undefined); skipping LR step and patience.")

        if epoch % 5 == 0 or epoch == epochs - 1:
            suffix = ""
            if history.val_auc:
                suffix = (
                    f" val_auc={history.val_auc[-1]:.4f} val_ap={history.val_ap[-1]:.4f}"
                )
                if history.val_loss:
                    suffix = f"{suffix} val_loss={history.val_loss[-1]:.4f}"
            lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}"
            print(f"hgt epoch {epoch:04d} loss={avg_loss:.4f}{suffix}{lr_str}")

        if patience_counter >= early_stop_patience:
            print(f"HGT early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    return history
