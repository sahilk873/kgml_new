from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgml_new.config import NodeClassificationConfig
from kgml_new.data.datasets import (
    prepare_hetero_node_classification_dataset,
    prepare_node_classification_dataset,
)
from kgml_new.data.loaders import load_pickled_graph
from kgml_new.models.baseline_sage import NEIGHBOR_AGGREGATIONS
from kgml_new.models.edge_aware_sage import EDGE_RELATION_MODES
from kgml_new.training.node_classification import (
    NODE_CLASSIFICATION_METHODS,
    train_embedding_node_classifier,
    train_native_node_classifier,
    train_txgnn_node_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run node classification on a pickled NetworkX graph.")
    parser.add_argument("--graph", type=Path, required=True, help="Pickled NetworkX graph path")
    parser.add_argument("--method", choices=tuple(NODE_CLASSIFICATION_METHODS), default="sage")
    parser.add_argument("--label-attr", type=str, default="label")
    parser.add_argument("--target-node-type", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--edge-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--semantic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-cache", type=Path, default=None)
    parser.add_argument("--strict-semantic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-relation-mode", choices=EDGE_RELATION_MODES, default="concat")
    parser.add_argument("--num-relation-bases", type=int, default=4)
    parser.add_argument(
        "--neighbor-aggr",
        choices=NEIGHBOR_AGGREGATIONS,
        default="mean",
        help="GraphSAGE neighbor aggregation for sage / edge_sage / link_mlp paths.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = load_pickled_graph(args.graph)
    spec = NODE_CLASSIFICATION_METHODS[args.method]
    if spec.kind == "hetero_native":
        if not args.target_node_type:
            raise ValueError("--target-node-type is required for txgnn node classification")
        dataset = prepare_hetero_node_classification_dataset(
            graph,
            in_dim=args.in_dim,
            target_node_type=args.target_node_type,
            label_attr=args.label_attr,
            seed=args.seed,
        )
        cfg = NodeClassificationConfig(
            in_dim=args.in_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_classes=len(dataset.label_lookup),
            epochs=args.epochs,
            classifier_epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            edge_relation_mode=args.edge_relation_mode,
            num_relation_bases=args.num_relation_bases,
            neighbor_aggr=args.neighbor_aggr,
        )
        _, _, history, val_metrics, test_metrics = train_txgnn_node_classifier(
            dataset.data,
            dataset.target_node_type,
            cfg,
            device=device,
        )
        result = {
            "method": args.method,
            "task": "node_classification",
            "kind": spec.kind,
            "target_node_type": dataset.target_node_type,
            "num_classes": len(dataset.label_lookup),
            "label_attr": args.label_attr,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": {
                "epoch": history.epoch,
                "train_loss": history.train_loss,
                "val_accuracy": history.val_accuracy,
                "val_macro_f1": history.val_macro_f1,
            },
        }
    else:
        dataset = prepare_node_classification_dataset(
            graph,
            in_dim=args.in_dim,
            label_attr=args.label_attr,
            seed=args.seed,
        )
        cfg = NodeClassificationConfig(
            in_dim=args.in_dim,
            edge_dim=args.edge_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_classes=len(dataset.label_lookup),
            epochs=args.epochs,
            classifier_epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            edge_relation_mode=args.edge_relation_mode,
            num_relation_bases=args.num_relation_bases,
            neighbor_aggr=args.neighbor_aggr,
        )
        if spec.kind == "native":
            _, history, val_metrics, test_metrics = train_native_node_classifier(
                args.method,
                dataset.data,
                cfg,
                device=device,
                graph=graph,
                relation_lookup=dataset.relation_lookup,
                use_semantic=args.semantic,
                semantic_cache=args.semantic_cache,
                strict_semantic=args.strict_semantic,
            )
            result = {
                "method": args.method,
                "task": "node_classification",
                "kind": spec.kind,
                "edge_relation_mode": args.edge_relation_mode,
                "num_relation_bases": args.num_relation_bases,
                "semantic": args.semantic,
                "semantic_cache": str(args.semantic_cache) if args.semantic_cache else None,
                "num_classes": len(dataset.label_lookup),
                "label_attr": args.label_attr,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "history": {
                    "epoch": history.epoch,
                    "train_loss": history.train_loss,
                    "val_accuracy": history.val_accuracy,
                    "val_macro_f1": history.val_macro_f1,
                },
            }
        else:
            _, _, history, val_metrics, test_metrics = train_embedding_node_classifier(
                args.method,
                dataset.data,
                cfg,
                device=device,
                relation_lookup=dataset.relation_lookup,
            )
            result = {
                "method": args.method,
                "task": "node_classification",
                "kind": spec.kind,
                "edge_relation_mode": args.edge_relation_mode,
                "num_relation_bases": args.num_relation_bases,
                "semantic": args.semantic,
                "semantic_cache": str(args.semantic_cache) if args.semantic_cache else None,
                "num_classes": len(dataset.label_lookup),
                "label_attr": args.label_attr,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "history": {
                    "epoch": history.epoch,
                    "train_loss": history.train_loss,
                    "val_accuracy": history.val_accuracy,
                    "val_macro_f1": history.val_macro_f1,
                },
            }

    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
