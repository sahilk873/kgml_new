from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data

from kgml_new.config import TrainConfig
from kgml_new.training.history import TrainingHistory


def encode_model(model: nn.Module, data: Data, *, edge_aware: bool, device: torch.device) -> Tensor:
    data = data.to(device)
    if bool(getattr(model, "requires_edge_type", False)):
        return model(data.x, data.edge_index, data.edge_attr)
    if edge_aware:
        return model(data.x, data.edge_index, data.edge_attr)
    return model(data.x, data.edge_index)


def corrupt_triples(
    triples: Tensor,
    *,
    num_nodes: int,
    true_triples: set[tuple[int, int, int]],
    negatives_per_pos: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    triples_cpu = triples.detach().cpu().long()
    out = torch.empty((triples_cpu.size(0) * negatives_per_pos, 3), dtype=torch.long)
    row = 0
    for h, r, t in triples_cpu.tolist():
        for k in range(negatives_per_pos):
            corrupt_head = (k % 2) == 1
            for _ in range(100):
                cand = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
                if corrupt_head:
                    candidate = (cand, r, t)
                    if cand != h and candidate not in true_triples:
                        break
                else:
                    candidate = (h, r, cand)
                    if cand != t and candidate not in true_triples:
                        break
            out[row] = torch.tensor(candidate, dtype=torch.long)
            row += 1
    return out.to(device)


def triple_set(triples: Tensor) -> set[tuple[int, int, int]]:
    return {tuple(int(x) for x in row) for row in triples.detach().cpu().long().tolist()}


def train_encoder_triple_decoder(
    encoder: nn.Module,
    decoder: nn.Module,
    train_data: Data,
    train_triples: Tensor,
    all_true_triples: Tensor,
    config: TrainConfig,
    *,
    edge_aware: bool,
    device: torch.device,
    val_triples: Tensor | None = None,
) -> tuple[nn.Module, nn.Module, int, TrainingHistory]:
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    train_triples = train_triples.to(device)
    true_set = triple_set(all_true_triples)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(config.seed))
    history = TrainingHistory(edge_aware=edge_aware, num_neighbors=None)
    history.config = asdict(config)
    last_epoch = -1
    for epoch in range(config.epochs):
        encoder.train()
        decoder.train()
        perm = torch.randperm(train_triples.size(0), generator=gen, device=device)
        total = 0.0
        batches = 0
        for start in range(0, train_triples.size(0), config.batch_size):
            idx = perm[start : start + config.batch_size]
            pos = train_triples[idx]
            if pos.numel() == 0:
                continue
            neg = corrupt_triples(
                pos,
                num_nodes=int(train_data.num_nodes),
                true_triples=true_set,
                negatives_per_pos=int(config.neg_samples),
                generator=gen,
                device=device,
            )
            z = encode_model(encoder, train_data, edge_aware=edge_aware, device=device)
            pos_logits = decoder(z, pos)
            neg_logits = decoder(z, neg)
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        history.epoch.append(epoch)
        history.train_loss.append(total / max(batches, 1))
        history.learning_rate.append(float(config.learning_rate))
        history.batch_count.append(batches)
        last_epoch = epoch
        if val_triples is not None and val_triples.numel() > 0:
            with torch.no_grad():
                val_neg = corrupt_triples(
                    val_triples,
                    num_nodes=int(train_data.num_nodes),
                    true_triples=true_set,
                    negatives_per_pos=int(config.neg_samples),
                    generator=gen,
                    device=device,
                )
                encoder.eval()
                decoder.eval()
                z = encode_model(encoder, train_data, edge_aware=edge_aware, device=device)
                pos_l = decoder(z, val_triples.to(device))
                neg_l = decoder(z, val_neg)
                val_loss = F.binary_cross_entropy_with_logits(
                    torch.cat([pos_l, neg_l]),
                    torch.cat([torch.ones_like(pos_l), torch.zeros_like(neg_l)]),
                )
                history.val_loss.append(float(val_loss))
    return encoder, decoder, last_epoch, history


def filtered_ranking_metrics(
    encoder: nn.Module,
    decoder: nn.Module,
    data: Data,
    eval_triples: Tensor,
    all_true_triples: Tensor,
    *,
    node_types: list[str],
    edge_aware: bool,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, object]:
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    true_set = triple_set(all_true_triples)
    with torch.no_grad():
        z = encode_model(encoder, data, edge_aware=edge_aware, device=device)
    all_entities = torch.arange(int(data.num_nodes), device=device, dtype=torch.long)
    type_to_entity_ids: dict[str, list[int]] = {}
    for idx, nt in enumerate(node_types):
        type_to_entity_ids.setdefault(str(nt), []).append(idx)
    type_to_entities_t = {
        nt: torch.tensor(ids, device=device, dtype=torch.long)
        for nt, ids in type_to_entity_ids.items()
    }

    def _rank_one(h: int, r: int, t: int, *, tail: bool, candidates: Tensor) -> int:
        if tail:
            triples = torch.stack(
                [torch.full_like(candidates, h), torch.full_like(candidates, r), candidates],
                dim=1,
            )
            target = t
        else:
            triples = torch.stack(
                [candidates, torch.full_like(candidates, r), torch.full_like(candidates, t)],
                dim=1,
            )
            target = h
        scores = decoder(z, triples).detach().clone()
        cand_list = candidates.detach().cpu().tolist()
        for i, cand in enumerate(cand_list):
            tri = (h, r, int(cand)) if tail else (int(cand), r, t)
            if int(cand) != target and tri in true_set:
                scores[i] = -torch.inf
        target_pos = (candidates == target).nonzero(as_tuple=False)
        if target_pos.numel() == 0:
            return int(candidates.numel()) + 1
        target_score = scores[int(target_pos[0].item())]
        return 1 + int((scores > target_score).sum().item())

    def _metrics_for_policy(policy: str) -> dict[str, float]:
        ranks: list[int] = []
        for h, r, t in eval_triples.detach().cpu().long().tolist():
            if policy == "type_constrained":
                tail_candidates = type_to_entities_t.get(str(node_types[t]), all_entities)
                head_candidates = type_to_entities_t.get(str(node_types[h]), all_entities)
            else:
                tail_candidates = all_entities
                head_candidates = all_entities
            ranks.append(_rank_one(h, r, t, tail=True, candidates=tail_candidates))
            ranks.append(_rank_one(h, r, t, tail=False, candidates=head_candidates))
        if not ranks:
            return {"mrr": float("nan"), "hits@1": float("nan"), "hits@3": float("nan"), "hits@10": float("nan")}
        rt = torch.tensor(ranks, dtype=torch.float32)
        return {
            "mrr": float((1.0 / rt).mean().item()),
            "hits@1": float((rt <= 1).float().mean().item()),
            "hits@3": float((rt <= 3).float().mean().item()),
            "hits@10": float((rt <= 10).float().mean().item()),
        }

    return {
        "all_entities": _metrics_for_policy("all_entities"),
        "type_constrained": _metrics_for_policy("type_constrained"),
        "num_eval_triples": int(eval_triples.size(0)),
    }
