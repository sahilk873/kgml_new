from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgml_new.config import LinkMLPConfig, Node2VecConfig, TrainConfig
from kgml_new.data.graph import networkx_to_data
from kgml_new.data.primekg import load_primekg_csv
from kgml_new.embeddings.semantic import build_relation_tensor
from kgml_new.models.baseline_gcn import BaselineGCN
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.eval import link_prediction_dot_product, link_prediction_mlp_torch
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
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
    parser.add_argument("--seed", type=int, default=42)
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        model = BaselineGraphSAGE(
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
        val_metrics = link_prediction_dot_product(z, val_pos, val_neg)
        test_metrics = link_prediction_dot_product(z, test_pos, test_neg)
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics

    elif args.method == "baseline_gcn":
        cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
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
        val_metrics = link_prediction_dot_product(z, val_pos, val_neg)
        test_metrics = link_prediction_dot_product(z, test_pos, test_neg)
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics

    elif args.method == "edge_aware_sage":
        cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        edge_dim = cfg.edge_dim
        rel_emb = {key: torch.randn(edge_dim) * 0.1 for key in relation_lookup}
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
        model, last_epoch, history = train_unsupervised_fullgraph(
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
        val_metrics = link_prediction_dot_product(z, val_pos, val_neg)
        test_metrics = link_prediction_dot_product(z, test_pos, test_neg)
        output["embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics

    elif args.method == "node2vec":
        cfg = Node2VecConfig(epochs=args.epochs, seed=args.seed, embedding_dim=64)
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
        output["val_metrics"] = link_prediction_dot_product(z, val_pos, val_neg)
        output["test_metrics"] = link_prediction_dot_product(z, test_pos, test_neg)
        output["metrics"] = output["test_metrics"]

    else:
        base_cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        base_model = BaselineGraphSAGE(
            base_cfg.in_dim,
            base_cfg.hidden_dim,
            base_cfg.out_dim,
            num_layers=base_cfg.num_layers,
            dropout=base_cfg.dropout,
        )
        base_model, base_last_epoch, base_history = train_unsupervised_fullgraph(
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
        output["val_metrics"] = link_prediction_mlp_torch(mlp, val_pos, val_neg, z)
        output["test_metrics"] = link_prediction_mlp_torch(mlp, test_pos, test_neg, z)
        output["metrics"] = output["test_metrics"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
