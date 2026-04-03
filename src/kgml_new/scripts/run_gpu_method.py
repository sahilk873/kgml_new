from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgml_new.config import LinkMLPConfig, Node2VecConfig, TrainConfig
from kgml_new.data.datasets import (
    compute_relation_diversity_buckets,
    prepare_link_prediction_dataset,
)
from kgml_new.data.loaders import GraphCSVSpec, PRIMEKG_CSV_SPEC, load_graph_csv, load_pickled_graph
from kgml_new.embeddings.semantic import (
    build_relation_tensor,
    relation_embeddings_from_graph,
)
from kgml_new.models.baseline_gcn import BaselineGCN
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.eval import evaluate_inductive_link_prediction, link_prediction_dot_product
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    train_unsupervised_batched,
    train_unsupervised_fullgraph,
)
from kgml_new.training.node2vec_train import (
    train_node2vec_embeddings_with_validation,
)
from kgml_new.training.train_link_mlp import train_link_mlp_with_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one KGML method on a generic KG and save JSON results."
    )
    parser.add_argument(
        "--method",
        choices=(
            "baseline_sage",
            "baseline_gcn",
            "edge_aware_sage",
            "node2vec",
            "link_mlp",
            "edge_aware_link_mlp",
        ),
        required=True,
    )
    parser.add_argument("--input", "--csv", dest="input_path", type=Path, default=Path("kg.csv"))
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "pickle"),
        default="auto",
    )
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--source-col", type=str, default=PRIMEKG_CSV_SPEC.source_col)
    parser.add_argument("--target-col", type=str, default=PRIMEKG_CSV_SPEC.target_col)
    parser.add_argument("--relation-col", type=str, default=PRIMEKG_CSV_SPEC.relation_col)
    parser.add_argument("--source-type-col", type=str, default=PRIMEKG_CSV_SPEC.source_type_col)
    parser.add_argument("--target-type-col", type=str, default=PRIMEKG_CSV_SPEC.target_type_col)
    parser.add_argument(
        "--split-protocol",
        choices=("node", "edge"),
        default="node",
        help="node: hold out entities from training edges for OOD evaluation; edge: legacy edge-disjoint split",
    )
    parser.add_argument(
        "--negative-sampling-mode",
        choices=("global", "type_matched"),
        default=None,
    )
    parser.add_argument("--negatives-per-pos", type=int, default=20)
    parser.add_argument("--decoder", choices=("dot", "mlp"), default="dot")
    parser.add_argument(
        "--shuffle-relations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For edge-aware methods: use OpenAI semantic relation embeddings by default.",
    )
    parser.add_argument(
        "--semantic-cache",
        type=Path,
        default=None,
        help="Optional cache path for relation embeddings used by edge-aware methods.",
    )
    parser.add_argument(
        "--strict-semantic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of falling back if OpenAI semantic embedding setup fails.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-dim", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _csv_spec_from_args(args: argparse.Namespace) -> GraphCSVSpec:
    return GraphCSVSpec(
        source_col=args.source_col,
        target_col=args.target_col,
        relation_col=args.relation_col,
        source_type_col=args.source_type_col,
        target_type_col=args.target_type_col,
        source_id_col=PRIMEKG_CSV_SPEC.source_id_col,
        target_id_col=PRIMEKG_CSV_SPEC.target_id_col,
        node_type_attr=PRIMEKG_CSV_SPEC.node_type_attr,
        relation_attr=PRIMEKG_CSV_SPEC.relation_attr,
        default_node_type=PRIMEKG_CSV_SPEC.default_node_type,
        edge_attr_cols=PRIMEKG_CSV_SPEC.edge_attr_cols,
        directed=False,
    )


def _load_graph(args: argparse.Namespace):
    input_format = args.input_format
    if input_format == "auto":
        suffix = args.input_path.suffix.lower()
        input_format = "pickle" if suffix in {".pkl", ".pickle"} else "csv"
    if input_format == "pickle":
        return load_pickled_graph(args.input_path)
    return load_graph_csv(args.input_path, spec=_csv_spec_from_args(args), max_edges=args.max_edges)


def build_dataset(args: argparse.Namespace):
    graph = _load_graph(args)
    dataset = prepare_link_prediction_dataset(
        graph,
        in_dim=args.in_dim,
        seed=args.seed,
        val_ratio=0.1,
        test_ratio=0.1,
        split_protocol=args.split_protocol,
        negative_sampling_mode=args.negative_sampling_mode,
        negatives_per_pos=args.negatives_per_pos,
        decoder=args.decoder,
        shuffle_relations=args.shuffle_relations,
    )
    return graph, dataset


def history_dict(history) -> dict:
    return {
        "epoch": history.epoch,
        "train_loss": history.train_loss,
        "val_auc": history.val_auc,
        "val_ap": history.val_ap,
        "batch_count": history.batch_count,
    }


def build_edge_aware_relation_table(
    *,
    args: argparse.Namespace,
    training_graph,
    relation_lookup: dict[str, int],
    edge_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    rel_emb = relation_embeddings_from_graph(
        training_graph,
        edge_dim=edge_dim,
        cache_path=args.semantic_cache,
        use_openai=args.semantic,
        strict_openai=args.strict_semantic,
    )
    relation_table = build_relation_tensor(rel_emb, relation_lookup, edge_dim, device)
    relation_init = "openai_semantic" if args.semantic else "random"
    return relation_table, relation_init


def maybe_train_decoder(
    *,
    decoder: str,
    z: torch.Tensor,
    train_pos: torch.Tensor,
    epochs: int,
    seed: int,
    device: torch.device,
):
    if decoder == "dot":
        return None, None, None
    mlp_cfg = LinkMLPConfig(
        epochs=epochs,
        seed=seed,
        feature_mode="concat_product",
    )
    decoder_model, decoder_last_epoch, decoder_history = train_link_mlp_with_validation(
        z.detach(),
        train_pos,
        mlp_cfg,
        device=device,
    )
    return decoder_model, decoder_last_epoch, decoder_history


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, dataset = build_dataset(args)
    train_data = dataset.train_data
    relation_lookup = dataset.relation_lookup
    split = dataset.split
    train_pos = split.train_pos_edge_index
    val_pos = split.val_pos_edge_index
    val_neg = split.val_neg_edge_index
    test_pos = split.test_pos_edge_index
    test_neg = split.test_neg_edge_index
    diversity_counts, diversity_buckets = compute_relation_diversity_buckets(
        dataset.graph,
        dataset.node_list,
    )
    val_query_buckets = [diversity_buckets[int(node)] for node in val_pos[0].tolist()]
    test_query_buckets = [diversity_buckets[int(node)] for node in test_pos[0].tolist()]

    output = {
        "method": args.method,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_nodes": int(train_data.num_nodes),
        "num_train_edges": int(train_pos.size(1)),
        "num_val_edges": int(val_pos.size(1)),
        "num_test_edges": int(test_pos.size(1)),
        "input_path": str(args.input_path),
        "input_format": args.input_format,
        "split_protocol": args.split_protocol,
        "negative_sampling_mode": dataset.negative_sampling_mode,
        "negatives_per_pos": dataset.negatives_per_pos,
        "decoder": args.decoder,
        "shuffle_relations": args.shuffle_relations,
        "semantic": args.semantic,
        "semantic_cache": str(args.semantic_cache) if args.semantic_cache else None,
        "strict_semantic": args.strict_semantic,
        "max_edges": args.max_edges,
        "in_dim": args.in_dim,
        "epochs": args.epochs,
        "evaluation_protocol": (
            "sampled inductive eval on full adjacency with scored edges removed"
            if args.split_protocol == "node"
            else "sampled eval on full adjacency with scored edges removed"
        ),
        "relation_diversity_summary": {
            "min": min(diversity_counts.values()) if diversity_counts else 0,
            "max": max(diversity_counts.values()) if diversity_counts else 0,
        },
    }

    if args.method == "baseline_sage":
        cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
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
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=False)
        decoder_model, decoder_last_epoch, decoder_history = maybe_train_decoder(
            decoder=args.decoder,
            z=z,
            train_pos=train_pos,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
        )
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
        )
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if decoder_history is not None:
            output["decoder_last_epoch"] = decoder_last_epoch
            output["decoder_history"] = history_dict(decoder_history)
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
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=False)
        decoder_model, decoder_last_epoch, decoder_history = maybe_train_decoder(
            decoder=args.decoder,
            z=z,
            train_pos=train_pos,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
        )
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
        )
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if decoder_history is not None:
            output["decoder_last_epoch"] = decoder_last_epoch
            output["decoder_history"] = history_dict(decoder_history)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics

    elif args.method == "edge_aware_sage":
        cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        edge_dim = cfg.edge_dim
        relation_table, relation_init = build_edge_aware_relation_table(
            args=args,
            training_graph=dataset.graph,
            relation_lookup=relation_lookup,
            edge_dim=edge_dim,
            device=device,
        )
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
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=True)
        decoder_model, decoder_last_epoch, decoder_history = maybe_train_decoder(
            decoder=args.decoder,
            z=z,
            train_pos=train_pos,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
        )
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
        )
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["relation_init"] = relation_init
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if decoder_history is not None:
            output["decoder_last_epoch"] = decoder_last_epoch
            output["decoder_history"] = history_dict(decoder_history)
        output["val_metrics"] = val_metrics
        output["test_metrics"] = test_metrics
        output["metrics"] = test_metrics

    elif args.method == "node2vec":
        cfg = Node2VecConfig(epochs=args.epochs, seed=args.seed, embedding_dim=64)
        _, z, last_epoch, history = train_node2vec_embeddings_with_validation(
            train_data,
            cfg,
            device=device,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if args.split_protocol == "node":
            output["warning"] = (
                "node2vec was trained on the train-node subgraph only; unseen nodes do not "
                "receive inductive embeddings, so this run is excluded from the main node-split comparison."
            )
            output["comparable_under_node_split"] = False
        else:
            output["val_metrics"] = link_prediction_dot_product(z, val_pos, val_neg)
            output["test_metrics"] = link_prediction_dot_product(z, test_pos, test_neg)
            output["metrics"] = output["test_metrics"]
            output["comparable_under_node_split"] = True

    elif args.method == "link_mlp":
        base_cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
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
        )
        z = compute_node_embeddings(base_model, train_data, device, edge_aware=False).detach()
        mlp_cfg = LinkMLPConfig(
            epochs=args.epochs,
            seed=args.seed,
            feature_mode="concat_product",
        )
        mlp, last_epoch, history = train_link_mlp_with_validation(
            z,
            train_pos,
            mlp_cfg,
            device=device,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["base_embedding_history"] = history_dict(base_history)
        output["base_last_epoch"] = base_last_epoch
        output["val_metrics"] = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=val_query_buckets,
        )
        output["test_metrics"] = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=test_query_buckets,
        )
        output["metrics"] = output["test_metrics"]

    else:
        base_cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
        edge_dim = base_cfg.edge_dim
        relation_table, relation_init = build_edge_aware_relation_table(
            args=args,
            training_graph=dataset.graph,
            relation_lookup=relation_lookup,
            edge_dim=edge_dim,
            device=device,
        )
        base_model = EdgeAwareGraphSAGE(
            base_cfg.in_dim,
            edge_dim,
            base_cfg.hidden_dim,
            base_cfg.out_dim,
            relation_table=relation_table,
            num_layers=base_cfg.num_layers,
            dropout=base_cfg.dropout,
            concat=base_cfg.concat,
        )
        base_model, base_last_epoch, base_history = train_unsupervised_batched(
            base_model,
            train_data,
            train_pos,
            base_cfg,
            device=device,
            edge_aware=True,
        )
        z = compute_node_embeddings(base_model, train_data, device, edge_aware=True).detach()
        mlp_cfg = LinkMLPConfig(
            epochs=args.epochs,
            seed=args.seed,
            feature_mode="concat_product",
        )
        mlp, last_epoch, history = train_link_mlp_with_validation(
            z,
            train_pos,
            mlp_cfg,
            device=device,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["relation_init"] = relation_init
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["base_embedding_history"] = history_dict(base_history)
        output["base_last_epoch"] = base_last_epoch
        output["val_metrics"] = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=val_query_buckets,
        )
        output["test_metrics"] = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=test_query_buckets,
        )
        output["metrics"] = output["test_metrics"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
