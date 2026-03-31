from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgml_new.data.hetero import networkx_to_heterodata
from kgml_new.data.primekg import load_primekg_csv
from kgml_new.models.txgnn import TxGNN
from kgml_new.training.txgnn_train import evaluate_relation, train_txgnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TxGNN-style hetero model on PrimeKG.")
    parser.add_argument("--csv", type=Path, default=Path("kg.csv"))
    parser.add_argument("--max-edges", type=int, default=10000)
    parser.add_argument("--relation", type=str, default="indication")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--neg-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _train_val_split(edge_index: torch.Tensor, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    num_edges = edge_index.size(1)
    val_count = max(1, int(num_edges * val_ratio))
    perm = torch.randperm(num_edges)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return edge_index[:, train_idx], edge_index[:, val_idx]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = load_primekg_csv(args.csv, max_edges=args.max_edges)
    data = networkx_to_heterodata(graph, in_dim=64, seed=args.seed, add_reverse_edges=True)
    edge_type = ("drug", args.relation, "disease")
    if edge_type not in data.edge_types:
        raise ValueError(f"Edge type {edge_type} not in hetero graph edge types: {data.edge_types}")

    train_pos, val_pos = _train_val_split(data[edge_type].edge_index, val_ratio=0.2, seed=args.seed)
    data[edge_type].edge_index = train_pos

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
        device=device,
    )

    val_neg = torch.stack(
        [
            torch.randint(0, data["drug"].num_nodes, (val_pos.size(1),)),
            torch.randint(0, data["disease"].num_nodes, (val_pos.size(1),)),
        ],
        dim=0,
    )
    val_auc, val_ap = evaluate_relation(model, data, edge_type, val_pos, val_neg, device)

    output = {
        "method": "txgnn",
        "relation": args.relation,
        "device": str(device),
        "num_nodes": {k: int(v.num_nodes) for k, v in data.node_items()},
        "num_train_edges": int(train_pos.size(1)),
        "num_val_edges": int(val_pos.size(1)),
        "history": {
            "train_loss": history.train_loss,
            "val_auc": history.val_auc,
            "val_ap": history.val_ap,
        },
        "final_metrics": {"roc_auc": float(val_auc), "average_precision": float(val_ap)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
