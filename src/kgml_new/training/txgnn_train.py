from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData

from kgml_new.models.txgnn import TxGNN


@dataclass
class TxGNNTrainResult:
    train_loss: list[float]
    val_auc: list[float]
    val_ap: list[float]


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
def evaluate_relation(
    model: TxGNN,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    z_dict = model.encode(data.to(device))
    pos_scores = model.score_edges(z_dict, edge_type, pos_edge_index.to(device))
    neg_scores = model.score_edges(z_dict, edge_type, neg_edge_index.to(device))
    y = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu().numpy()
    s = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    return roc_auc_score(y, s), average_precision_score(y, s)


def train_txgnn(
    model: TxGNN,
    data: HeteroData,
    edge_type: tuple[str, str, str],
    *,
    epochs: int = 20,
    batch_size: int = 1024,
    neg_samples: int = 1,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    val_pos_edge_index: torch.Tensor | None = None,
    val_neg_edge_index: torch.Tensor | None = None,
) -> TxGNNTrainResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    src_type, _, dst_type = edge_type
    pos_edge_index = data[edge_type].edge_index
    num_pos = pos_edge_index.size(1)
    num_src = data[src_type].num_nodes
    num_dst = data[dst_type].num_nodes

    history = TxGNNTrainResult(train_loss=[], val_auc=[], val_ap=[])
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_pos, device=device)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, num_pos, batch_size):
            idx = perm[start : start + batch_size]
            pos_batch = pos_edge_index[:, idx]
            neg_batch = _sample_negative_edges(
                num_src,
                num_dst,
                num_samples=pos_batch.size(1) * neg_samples,
                device=device,
            )

            optimizer.zero_grad()
            z_dict = model.encode(data)
            pos_logits = model.score_edges(z_dict, edge_type, pos_batch)
            neg_logits = model.score_edges(z_dict, edge_type, neg_batch)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_logits, neg_logits]),
                torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]),
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        history.train_loss.append(avg_loss)

        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            auc, ap = evaluate_relation(
                model,
                data,
                edge_type,
                val_pos_edge_index,
                val_neg_edge_index,
                device,
            )
            history.val_auc.append(float(auc))
            history.val_ap.append(float(ap))

        if epoch % 5 == 0 or epoch == epochs - 1:
            suffix = ""
            if history.val_auc:
                suffix = f" val_auc={history.val_auc[-1]:.4f} val_ap={history.val_ap[-1]:.4f}"
            print(f"epoch {epoch:04d} loss={avg_loss:.4f}{suffix}")

    return history
