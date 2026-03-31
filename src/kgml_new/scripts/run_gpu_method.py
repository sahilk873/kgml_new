from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgml_new.config import LinkMLPConfig, Node2VecConfig, TrainConfig
from kgml_new.runtime import resolve_device, seed_everything, validate_positive
from kgml_new.data.graph import networkx_to_data
from kgml_new.data.primekg import load_primekg_csv
from kgml_new.embeddings.semantic import build_relation_tensor, relation_embeddings_from_graph
from kgml_new.models.baseline_gcn import BaselineGCN
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.eval import link_prediction_dot_product, link_prediction_mlp_torch, link_prediction_dot_product_with_ci, link_prediction_mlp_with_ci
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    train_unsupervised_batched,
    train_unsupervised_fullgraph,
)
from kgml_new.training.node2vec_train import (
    train_node2vec_embeddings_with_validation,
)
from kgml_new.training.splits import build_train_graph_data, create_edge_split
from kgml_new.training.train_link_mlp import train_link_mlp_with_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one KGML method on PrimeKG and save JSON results."
    )
    parser.add_argument(
        "--method",
        choices=("baseline_sage", "baseline_gcn", "edge_aware_sage", "node2vec", "link_mlp"),
        required=True,
    )
    parser.add_argument("--csv", type=Path, default=Path("kg.csv"))
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
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
        default=None,
        help="Override mixed precision behavior for all methods",
    )
    parser.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For edge_aware_sage: use semantic relation embeddings via OpenAI",
    )
    parser.add_argument(
        "--require-semantic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For edge_aware_sage: fail if semantic embeddings cannot be produced",
    )
    parser.add_argument(
        "--relation-cache",
        type=Path,
        default=None,
        help="Optional cache path for relation embeddings (edge_aware_sage)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_dataset(csv_path: Path, max_edges: int, seed: int):
    graph = load_primekg_csv(csv_path, max_edges=max_edges)
    data, relation_lookup = networkx_to_data(graph, in_dim=64, seed=seed)
    mask = data.edge_index[0] < data.edge_index[1]
    pos_edge_index = data.edge_index[:, mask]
    pos_edge_attr = data.edge_attr[mask] if hasattr(data, "edge_attr") else None
    split = create_edge_split(
        pos_edge_index,
        num_src_nodes=int(data.num_nodes),
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
        undirected=True,
    )
    train_pos_attr = None
    if pos_edge_attr is not None:
        key_to_idx = {
            (int(pos_edge_index[0, i]), int(pos_edge_index[1, i])): i
            for i in range(pos_edge_index.size(1))
        }
        train_indices = [
            key_to_idx[(int(split.train_pos_edge_index[0, i]), int(split.train_pos_edge_index[1, i]))]
            for i in range(split.train_pos_edge_index.size(1))
        ]
        train_pos_attr = pos_edge_attr[torch.tensor(train_indices, dtype=torch.long)]
    train_data = build_train_graph_data(
        data, split.train_pos_edge_index, train_pos_edge_attr=train_pos_attr
    )
    return graph, train_data, relation_lookup, split


def history_dict(history) -> dict:
    return {
        "epoch": history.epoch,
        "train_loss": history.train_loss,
        "val_auc": history.val_auc,
        "val_ap": history.val_ap,
        "batch_count": history.batch_count,
    }


def _format_metrics_with_ci(metrics: dict) -> str:
    """Format metrics with CI for display."""
    auc_str = f"AUC: {metrics['roc_auc']:.4f} [{metrics.get('roc_auc_ci_lower', 'N/A')}, {metrics.get('roc_auc_ci_upper', 'N/A')}]"
    ap_str = f"AP: {metrics['average_precision']:.4f} [{metrics.get('ap_ci_lower', 'N/A')}, {metrics.get('ap_ci_upper', 'N/A')}]"
    return f"{auc_str}, {ap_str}"


def main() -> None:
    args = parse_args()
    validate_positive("epochs", args.epochs)
    if args.batch_size is not None:
        validate_positive("batch_size", args.batch_size)
    seed_everything(args.seed, deterministic=args.deterministic)

    device = resolve_device(args.device)
    graph, train_data, relation_lookup, split = build_dataset(
        args.csv, args.max_edges, args.seed
    )
    train_pos = split.train_pos_edge_index
    val_pos = split.val_pos_edge_index
    val_neg = split.val_neg_edge_index
    test_pos = split.test_pos_edge_index
    test_neg = split.test_neg_edge_index

    output = {
        "method": args.method,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_nodes": int(train_data.num_nodes),
        "num_train_edges": int(train_pos.size(1)),
        "num_val_edges": int(val_pos.size(1)),
        "num_test_edges": int(test_pos.size(1)),
        "max_edges": args.max_edges,
        "epochs": args.epochs,
        "evaluation_protocol": "train-only graph; held-out val/test positives; true sampled non-edge negatives; direct decoder scoring",
    }

    if args.method == "baseline_sage":
        cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.use_amp is not None:
            cfg.use_amp = args.use_amp
        model = BaselineGraphSAGE(
            cfg.in_dim, cfg.hidden_dim, cfg.out_dim, num_layers=cfg.num_layers, dropout=cfg.dropout
        )
        model, last_epoch, history = train_unsupervised_batched(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=False,
            val_pos_edge_index=val_pos,
            val_neg_edge_index=val_neg,
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=False)
        val_metrics = link_prediction_dot_product_with_ci(z, val_pos, val_neg)
        test_metrics = link_prediction_dot_product_with_ci(z, test_pos, test_neg)
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics
        print(f"Test set: {_format_metrics_with_ci(test_metrics)}")

    elif args.method == "baseline_gcn":
        cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.use_amp is not None:
            cfg.use_amp = args.use_amp
        model = BaselineGCN(
            cfg.in_dim, cfg.hidden_dim, cfg.out_dim, num_layers=cfg.num_layers, dropout=cfg.dropout
        )
        model, last_epoch, history = train_unsupervised_fullgraph(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=False,
            val_pos_edge_index=val_pos,
            val_neg_edge_index=val_neg,
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=False)
        val_metrics = link_prediction_dot_product_with_ci(z, val_pos, val_neg)
        test_metrics = link_prediction_dot_product_with_ci(z, test_pos, test_neg)
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics
        print(f"Test set: {_format_metrics_with_ci(test_metrics)}")

    elif args.method == "edge_aware_sage":
        cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.use_amp is not None:
            cfg.use_amp = args.use_amp
        edge_dim = cfg.edge_dim
        rel_emb = relation_embeddings_from_graph(
            graph,
            edge_dim=edge_dim,
            cache_path=args.relation_cache,
            use_openai=args.semantic,
            strict_openai=args.require_semantic,
        )
        rel_source = "semantic_openai" if args.semantic else "random_ablation"
        relation_table = build_relation_tensor(rel_emb, relation_lookup, edge_dim, device)
        model = EdgeAwareGraphSAGE(
            cfg.in_dim,
            edge_dim,
            cfg.hidden_dim,
            cfg.out_dim,
            relation_table=relation_table,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            concat=cfg.concat,
        )
        model, last_epoch, history = train_unsupervised_batched(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=True,
            val_pos_edge_index=val_pos,
            val_neg_edge_index=val_neg,
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=True)
        val_metrics = link_prediction_dot_product_with_ci(z, val_pos, val_neg)
        test_metrics = link_prediction_dot_product_with_ci(z, test_pos, test_neg)
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["relation_embedding_source"] = rel_source
        output["semantic_enabled"] = bool(args.semantic)
        output["semantic_required"] = bool(args.require_semantic)
        output["relation_embedding_cache"] = (
            str(args.relation_cache) if args.relation_cache is not None else None
        )
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics
        print(f"Test set: {_format_metrics_with_ci(test_metrics)}")

    elif args.method == "node2vec":
        cfg = Node2VecConfig(epochs=args.epochs, seed=args.seed, embedding_dim=256)
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.use_amp is not None:
            cfg.use_amp = args.use_amp
        _, z, last_epoch, history = train_node2vec_embeddings_with_validation(
            train_data,
            cfg,
            device=device,
            val_pos_edge_index=val_pos,
            val_neg_edge_index=val_neg,
        )
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        val_metrics = link_prediction_dot_product_with_ci(z, val_pos, val_neg)
        test_metrics = link_prediction_dot_product_with_ci(z, test_pos, test_neg)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics
        print(f"Test set: {_format_metrics_with_ci(test_metrics)}")

    else:
        base_cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        if args.batch_size is not None:
            base_cfg.batch_size = args.batch_size
        if args.use_amp is not None:
            base_cfg.use_amp = args.use_amp
        base_model = BaselineGraphSAGE(
            base_cfg.in_dim,
            base_cfg.hidden_dim,
            base_cfg.out_dim,
            num_layers=base_cfg.num_layers,
            dropout=base_cfg.dropout,
        )
        base_model, base_last_epoch, base_history = train_unsupervised_batched(
            base_model,
            train_data,
            train_pos,
            base_cfg,
            device=device,
            edge_aware=False,
            val_pos_edge_index=val_pos,
            val_neg_edge_index=val_neg,
        )
        z = compute_node_embeddings(base_model, train_data, device, edge_aware=False).detach()
        mlp_cfg = LinkMLPConfig(epochs=args.epochs, seed=args.seed)
        if args.batch_size is not None:
            mlp_cfg.batch_size = args.batch_size
        if args.use_amp is not None:
            mlp_cfg.use_amp = args.use_amp
        mlp, last_epoch, history = train_link_mlp_with_validation(
            z,
            train_pos,
            mlp_cfg,
            device=device,
            val_pos_edge_index=val_pos,
            val_neg_edge_index=val_neg,
        )
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["base_embedding_history"] = history_dict(base_history)
        output["base_last_epoch"] = base_last_epoch
        val_metrics = link_prediction_mlp_with_ci(mlp, val_pos, val_neg, z)
        test_metrics = link_prediction_mlp_with_ci(mlp, test_pos, test_neg, z)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics
        print(f"Test set: {_format_metrics_with_ci(test_metrics)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
