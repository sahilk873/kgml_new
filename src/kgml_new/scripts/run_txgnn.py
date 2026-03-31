from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgml_new.data.hetero import networkx_to_heterodata
from kgml_new.data.primekg import load_primekg_csv
from kgml_new.models.txgnn import TxGNN
from kgml_new.runtime import resolve_device, seed_everything, validate_positive
from kgml_new.training.splits import create_edge_split
from kgml_new.training.txgnn_train import evaluate_relation, train_txgnn


def _select_edge_type(
    edge_types: list[tuple[str, str, str]],
    requested_relation: str,
) -> tuple[tuple[str, str, str], bool]:
    requested = ("drug", requested_relation, "disease")
    if requested in edge_types:
        return requested, False

    # Prefer a forward (non-reverse) relation if available.
    forward = [et for et in edge_types if not et[1].startswith("rev_")]
    if forward:
        return forward[0], True
    if edge_types:
        return edge_types[0], True
    raise ValueError("No edge types available in hetero graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TxGNN-style hetero model on PrimeKG.")
    parser.add_argument("--csv", type=Path, default=Path("kg.csv"))
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--relation", type=str, default="indication")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--neg-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. cuda, cuda:0, cpu",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic CuDNN behavior for reproducibility",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic mixed precision on CUDA",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    validate_positive("epochs", args.epochs)
    validate_positive("batch_size", args.batch_size)
    validate_positive("neg_samples", args.neg_samples)
    validate_positive("early_stop_patience", args.early_stop_patience)
    if args.grad_clip_norm <= 0:
        raise ValueError(f"grad_clip_norm must be > 0, got {args.grad_clip_norm}")
    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)

    graph = load_primekg_csv(args.csv, max_edges=args.max_edges)
    data = networkx_to_heterodata(graph, in_dim=64, seed=args.seed, add_reverse_edges=True)
    edge_type, used_fallback_relation = _select_edge_type(list(data.edge_types), args.relation)
    if used_fallback_relation:
        print(
            f"Requested relation '{args.relation}' not found; "
            f"using available edge type {edge_type}."
        )
    src_type, relation_name, dst_type = edge_type

    split = create_edge_split(
        data[edge_type].edge_index.cpu(),
        num_src_nodes=int(data[src_type].num_nodes),
        num_dst_nodes=int(data[dst_type].num_nodes),
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed,
        undirected=False,
    )
    train_pos = split.train_pos_edge_index
    val_pos = split.val_pos_edge_index
    test_pos = split.test_pos_edge_index
    val_neg = split.val_neg_edge_index
    test_neg = split.test_neg_edge_index

    data[edge_type].edge_index = train_pos
    rev_edge_type = (dst_type, f"rev_{relation_name}", src_type)
    if rev_edge_type in data.edge_types:
        data[rev_edge_type].edge_index = train_pos.flip(0)

    model = TxGNN(
        metadata=data.metadata(),
        in_channels=64,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
        dropout=0.1,
        use_prototypes=True,
        prototype_k=5,
        prototype_alpha=0.5,
    )
    history = train_txgnn(
        model,
        data,
        edge_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_samples=args.neg_samples,
        learning_rate=1e-3,
        use_amp=args.use_amp,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        device=device,
    )

    val_auc, val_ap = evaluate_relation(model, data, edge_type, val_pos, val_neg, device)
    test_auc, test_ap = evaluate_relation(model, data, edge_type, test_pos, test_neg, device)

    output = {
        "method": "txgnn",
        "relation": relation_name,
        "requested_relation": args.relation,
        "edge_type": [src_type, relation_name, dst_type],
        "used_fallback_relation": used_fallback_relation,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_nodes": {k: int(v.num_nodes) for k, v in data.node_items()},
        "num_train_edges": int(train_pos.size(1)),
        "num_val_edges": int(val_pos.size(1)),
        "num_test_edges": int(test_pos.size(1)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "evaluation_protocol": "train-only relation graph with reverse edges updated; held-out val/test positives; true sampled bipartite non-edge negatives",
        "history": {
            "train_loss": history.train_loss,
            "val_auc": history.val_auc,
            "val_ap": history.val_ap,
        },
        "val_metrics": {"roc_auc": float(val_auc), "average_precision": float(val_ap)},
        "test_metrics": {"roc_auc": float(test_auc), "average_precision": float(test_ap)},
        "final_metrics": {"roc_auc": float(test_auc), "average_precision": float(test_ap)},
    }
    print(f"Validation: AUC={val_auc:.4f}, AP={val_ap:.4f}")
    print(f"Test set: AUC={test_auc:.4f}, AP={test_ap:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
