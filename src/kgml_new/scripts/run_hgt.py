from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import networkx as nx
import torch

from kgml_new.data.loaders import GraphCSVSpec, PRIMEKG_CSV_SPEC, load_graph_csv, load_pickled_graph
from kgml_new.data.hetero import networkx_to_heterodata
from kgml_new.models.hgt import HGTLinkPredictor
from kgml_new.training.hgt_train import (
    evaluate_hgt_relation,
    evaluate_hgt_relation_metrics,
    neighbor_sampler_backend_available,
    parse_neighbor_fanouts,
    train_hgt,
)
from kgml_new.training.splits import (
    create_bipartite_node_split,
    create_edge_split,
    create_node_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone HGT (Heterogeneous Graph Transformer) link prediction on a typed KG."
    )
    parser.add_argument("--input", "--csv", dest="input_path", type=Path, default=Path("data/kg.csv"))
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
    parser.add_argument("--relation", type=str, default="indication")
    parser.add_argument("--source-node-type", type=str, default=None)
    parser.add_argument("--target-node-type", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--neg-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--split-protocol",
        choices=("edge", "node"),
        default="edge",
        help="Link prediction split: edge (random edges) vs node (node-disjoint; bipartite-aware).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--negative-sampling-mode",
        choices=("global", "type_matched"),
        default="global",
        help="Negative sampling for val/test queries (default global matches edge-split HGT without node types).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument(
        "--optuna-metric",
        choices=("val_auc", "val_ap"),
        default="val_auc",
    )
    parser.add_argument("--optuna-storage", type=str, default=None)
    parser.add_argument("--optuna-study", type=str, default=None)
    parser.add_argument("--optuna-seed", type=int, default=None)
    parser.add_argument("--optuna-timeout", type=float, default=None)
    parser.add_argument(
        "--neighbor-sampling",
        action=argparse.BooleanOptionalAction,
        default=neighbor_sampler_backend_available(),
        help="Mini-batch LinkNeighborLoader train/eval when pyg_lib/torch_sparse available (default on iff backend present). "
        "Use --no-neighbor-sampling for full-graph encode (small graphs / debugging).",
    )
    parser.add_argument(
        "--num-neighbors",
        type=str,
        default=None,
        help="Comma-separated neighbor fanouts per layer (e.g. 15,10) or one int repeated; default 15 per layer.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1024,
        help="Positive edges per chunk for neighbor-sampled val/test evaluation.",
    )
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


def _resolve_target_edge_type(args: argparse.Namespace, edge_types: list[tuple[str, str, str]]):
    if args.source_node_type and args.target_node_type:
        edge_type = (args.source_node_type, args.relation, args.target_node_type)
        if edge_type not in edge_types:
            raise ValueError(f"Edge type {edge_type} not in hetero graph edge types: {edge_types}")
        return edge_type

    relation_matches = [edge_type for edge_type in edge_types if edge_type[1] == args.relation]
    if not relation_matches:
        raise ValueError(
            f"No edge type found for relation '{args.relation}'. Available edge types: {edge_types}"
        )
    if len(relation_matches) > 1:
        raise ValueError(
            "Relation is ambiguous across multiple typed edges. "
            "Pass --source-node-type and --target-node-type to disambiguate. "
            f"Candidates: {relation_matches}"
        )
    return relation_matches[0]


def _ordered_nodes_of_type(
    graph: nx.Graph, node_type: str, *, node_type_attr: str = "node_type"
) -> list:
    out: list = []
    for node, attrs in graph.nodes(data=True):
        if str(attrs.get(node_type_attr, "entity")) == node_type:
            out.append(node)
    return out


def _homogeneous_node_types_aligned(
    graph: nx.Graph,
    *,
    node_type: str,
    num_nodes: int,
    node_type_attr: str = "node_type",
) -> list[str] | None:
    ordered = _ordered_nodes_of_type(graph, node_type, node_type_attr=node_type_attr)
    if len(ordered) != num_nodes:
        return None
    return [str(graph.nodes[n].get(node_type_attr, "entity")) for n in ordered]


def _link_split_for_hgt(
    *,
    args: argparse.Namespace,
    data,
    edge_type: tuple[str, str, str],
    graph: nx.Graph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Returns train_pos, val_pos, val_neg, test_pos, test_neg, bipartite_node_split."""
    ei = data[edge_type].edge_index.cpu()
    num_src = int(data[edge_type[0]].num_nodes)
    num_dst = int(data[edge_type[2]].num_nodes)
    bipartite_relation = edge_type[0] != edge_type[2]
    node_types_arg: list[str] | None = None
    if args.negative_sampling_mode == "type_matched":
        if bipartite_relation:
            dst_nt = edge_type[2]
            ordered_dst = _ordered_nodes_of_type(graph, dst_nt)
            if len(ordered_dst) != num_dst:
                raise ValueError(
                    f"type_matched: dst node count mismatch for type {dst_nt!r} "
                    f"(graph {len(ordered_dst)} vs heterodata {num_dst})"
                )
            node_types_arg = [str(graph.nodes[n].get("node_type", "entity")) for n in ordered_dst]
        else:
            ntype = edge_type[0]
            node_types_arg = _homogeneous_node_types_aligned(
                graph, node_type=ntype, num_nodes=num_src
            )
            if node_types_arg is None:
                raise ValueError(
                    f"type_matched: cannot align ordered {ntype!r} nodes with heterodata (n={num_src})"
                )

    if args.split_protocol == "edge":
        split = create_edge_split(
            ei,
            num_src_nodes=num_src,
            num_dst_nodes=num_dst,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            undirected=False,
            node_types=node_types_arg,
            negative_sampling_mode=args.negative_sampling_mode,
            bipartite=bipartite_relation,
        )
        return (
            split.train_pos_edge_index,
            split.val_pos_edge_index,
            split.val_neg_edge_index,
            split.test_pos_edge_index,
            split.test_neg_edge_index,
            False,
        )

    if bipartite_relation:
        split = create_bipartite_node_split(
            ei,
            num_src_nodes=num_src,
            num_dst_nodes=num_dst,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            undirected=False,
            node_types=node_types_arg,
            negative_sampling_mode=args.negative_sampling_mode,
        )
        return (
            split.train_pos_edge_index,
            split.val_pos_edge_index,
            split.val_neg_edge_index,
            split.test_pos_edge_index,
            split.test_neg_edge_index,
            True,
        )

    nsplit = create_node_split(
        ei,
        num_src_nodes=num_src,
        num_dst_nodes=num_dst,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        undirected=False,
        node_types=node_types_arg,
        negative_sampling_mode=args.negative_sampling_mode,
    )
    return (
        nsplit.train_pos_edge_index,
        nsplit.val_pos_edge_index,
        nsplit.val_neg_edge_index,
        nsplit.test_pos_edge_index,
        nsplit.test_neg_edge_index,
        False,
    )


def run_hgt_experiment(
    args: argparse.Namespace,
    *,
    device: torch.device,
    data,
    edge_type: tuple[str, str, str],
    train_pos: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
) -> dict:
    model = HGTLinkPredictor(
        metadata=data.metadata(),
        in_channels=args.in_dim,
        hidden_channels=args.in_dim,
        out_channels=args.in_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    fanouts = (
        parse_neighbor_fanouts(args.num_neighbors, args.num_layers)
        if args.num_neighbors
        else None
    )
    history = train_hgt(
        model,
        data,
        edge_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        neg_samples=args.neg_samples,
        learning_rate=float(args.learning_rate),
        device=device,
        neighbor_sampling=args.neighbor_sampling,
        num_neighbors_fanouts=fanouts,
        eval_batch_size_pos=args.eval_batch_size,
    )

    val_metrics = evaluate_hgt_relation_metrics(
        model,
        data,
        edge_type,
        val_pos,
        val_neg,
        device,
        neighbor_sampling=args.neighbor_sampling,
        num_neighbors_fanouts=fanouts,
        eval_batch_size_pos=args.eval_batch_size,
        num_layers=args.num_layers,
    )
    test_metrics = evaluate_hgt_relation_metrics(
        model,
        data,
        edge_type,
        test_pos,
        test_neg,
        device,
        neighbor_sampling=args.neighbor_sampling,
        num_neighbors_fanouts=fanouts,
        eval_batch_size_pos=args.eval_batch_size,
        num_layers=args.num_layers,
    )

    output = {
        "method": "hgt",
        "relation": edge_type[1],
        "edge_type": list(edge_type),
        "input_path": str(args.input_path),
        "input_format": args.input_format,
        "device": str(device),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "neg_samples": int(args.neg_samples),
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "neighbor_sampling": bool(args.neighbor_sampling),
        "num_neighbors_fanouts": fanouts,
        "eval_batch_size_pos": int(args.eval_batch_size),
        "neighbor_sampler_backend": neighbor_sampler_backend_available(),
        "hidden_channels": model.hidden_channels,
        "out_channels": model.out_channels,
        "num_nodes": {k: int(v.num_nodes) for k, v in data.node_items()},
        "num_train_edges": int(train_pos.size(1)),
        "num_val_edges": int(val_pos.size(1)),
        "num_test_edges": int(test_pos.size(1)),
        "evaluation_protocol": "train-only relation graph with reverse edges updated; held-out val/test positives; true sampled bipartite non-edge negatives",
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_auc": history.val_auc,
            "val_ap": history.val_ap,
        },
        "val_metrics": {
            "roc_auc": float(val_metrics["roc_auc"]),
            "average_precision": float(val_metrics["average_precision"]),
            "hits@1": float(val_metrics["hits@1"]),
            "hits@3": float(val_metrics["hits@3"]),
            "hits@10": float(val_metrics["hits@10"]),
        },
        "test_metrics": {
            "roc_auc": float(test_metrics["roc_auc"]),
            "average_precision": float(test_metrics["average_precision"]),
            "hits@1": float(test_metrics["hits@1"]),
            "hits@3": float(test_metrics["hits@3"]),
            "hits@10": float(test_metrics["hits@10"]),
        },
        "final_metrics": {
            "roc_auc": float(test_metrics["roc_auc"]),
            "average_precision": float(test_metrics["average_precision"]),
            "hits@1": float(test_metrics["hits@1"]),
            "hits@3": float(test_metrics["hits@3"]),
            "hits@10": float(test_metrics["hits@10"]),
        },
    }
    artifacts_dir = args.output.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output.stem
    encoder_path = artifacts_dir / f"{stem}.encoder.pt"
    torch.save(
        {
            "method": "hgt",
            "model_class": model.__class__.__name__,
            "state_dict": model.state_dict(),
            "edge_type": list(edge_type),
            "in_dim": int(args.in_dim),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
        },
        encoder_path,
    )
    output["artifacts"] = {
        "enabled": True,
        "root": str(artifacts_dir),
        "encoder_state_dict": str(encoder_path),
    }
    return output


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = _load_graph(args)
    data = networkx_to_heterodata(graph, in_dim=args.in_dim, seed=args.seed, add_reverse_edges=True)
    edge_type = _resolve_target_edge_type(args, list(data.edge_types))

    train_pos, val_pos, val_neg, test_pos, test_neg, bipartite_ns = _link_split_for_hgt(
        args=args,
        data=data,
        edge_type=edge_type,
        graph=graph,
    )
    split_meta = {
        "split_protocol": args.split_protocol,
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "bipartite_node_split": bool(bipartite_ns),
        "negative_sampling_mode": args.negative_sampling_mode,
    }

    data[edge_type].edge_index = train_pos
    rev_edge_type = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
    if rev_edge_type in data.edge_types:
        data[rev_edge_type].edge_index = train_pos.flip(0)

    if args.optuna_trials > 0:
        from kgml_new.tuning.optuna_runner import require_optuna
        from kgml_new.tuning.spaces import (
            apply_param_dict_to_namespace,
            suggest_hetero_link_params,
            suggest_hgt_arch_params,
        )

        optuna = require_optuna()
        optuna_seed = int(args.optuna_seed if args.optuna_seed is not None else args.seed)
        sampler = optuna.samplers.TPESampler(seed=optuna_seed)
        if args.optuna_storage:
            study = optuna.create_study(
                study_name=args.optuna_study or "kgml_hgt",
                storage=args.optuna_storage,
                direction="maximize",
                sampler=sampler,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial: optuna.Trial) -> float:
            t_args = copy.deepcopy(args)
            suggest_hetero_link_params(trial, t_args)
            suggest_hgt_arch_params(trial, t_args)
            torch.manual_seed(optuna_seed + trial.number * 100_003)
            out = run_hgt_experiment(
                t_args,
                device=device,
                data=data,
                edge_type=edge_type,
                train_pos=train_pos,
                val_pos=val_pos,
                val_neg=val_neg,
                test_pos=test_pos,
                test_neg=test_neg,
            )
            out.update(split_meta)
            key = "roc_auc" if args.optuna_metric == "val_auc" else "average_precision"
            return float(out["val_metrics"][key])

        study.optimize(
            _objective,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
        )
        t_args = copy.deepcopy(args)
        apply_param_dict_to_namespace(t_args, study.best_params)
        torch.manual_seed(args.seed)
        output = run_hgt_experiment(
            t_args,
            device=device,
            data=data,
            edge_type=edge_type,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )
        output.update(split_meta)
        output["optuna"] = {
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "metric": args.optuna_metric,
        }
    else:
        output = run_hgt_experiment(
            args,
            device=device,
            data=data,
            edge_type=edge_type,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )
        output.update(split_meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
