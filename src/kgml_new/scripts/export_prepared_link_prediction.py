"""Build PyG link-prediction tensors + train/val/test split + negatives once; save for reuse.

This skips repeated ``prepare_link_prediction_dataset`` work on large graphs. The cache is a
Python pickle (NetworkX + PyG ``Data`` + torch tensors). Validate with
``load_prepared_link_prediction_dataset(..., expected_meta=...)`` before training.

Example (match Slurm defaults: edge split, full DRKG pickle):

  python -m kgml_new.scripts.export_prepared_link_prediction \\
    --input cache/drkg.graph.pkl \\
    --input-format pickle \\
    --split-protocol edge \\
    --output cache/drkg-prepared_edge_s42.pkl

From TSV (slow on full DRKG; prefer graph pickle as --input):

  python -m kgml_new.scripts.export_prepared_link_prediction \\
    --input drkg.tsv \\
    --output cache/drkg-prepared_edge_s42.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from kgml_new.data.datasets import prepare_link_prediction_dataset
from kgml_new.data.loaders import (
    PRIMEKG_CSV_SPEC,
    GraphCSVSpec,
    load_graph_csv,
    load_pickled_graph,
)
from kgml_new.data.prepared_dataset_cache import (
    build_cache_meta,
    save_prepared_link_prediction_dataset,
)


def _held_out_relations_from_args(args: argparse.Namespace) -> list[str] | None:
    names = [str(x).strip() for x in getattr(args, "held_out_relations", []) or []]
    names = [x for x in names if x]
    return names or None


def _held_out_node_categories_from_args(args: argparse.Namespace) -> list[str] | None:
    names = [str(x).strip() for x in getattr(args, "held_out_node_categories", []) or []]
    names = [x for x in names if x]
    return names or None


def _spec_from_args(args: argparse.Namespace) -> GraphCSVSpec:
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


def _resolve_input_format(args: argparse.Namespace) -> str:
    if args.input_format != "auto":
        return args.input_format
    suffix = args.input.suffix.lower()
    return "pickle" if suffix in {".pkl", ".pickle"} else "csv"


def _load_graph(args: argparse.Namespace):
    fmt = _resolve_input_format(args)
    if fmt == "pickle":
        return load_pickled_graph(args.input), None, "pickle"
    spec = _spec_from_args(args)
    g = load_graph_csv(args.input, spec=spec, max_edges=args.max_edges)
    return g, spec, "csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a cached LinkPredictionDataset (PyG + split + negatives)."
    )
    p.add_argument("--input", type=Path, required=True, help="Graph CSV/TSV or .pkl from export_graph_pickle")
    p.add_argument("--input-format", choices=("auto", "csv", "pickle"), default="auto")
    p.add_argument("--max-edges", type=int, default=None)
    p.add_argument("--output", type=Path, required=True, help="Output .pkl path")
    p.add_argument("--source-col", type=str, default=PRIMEKG_CSV_SPEC.source_col)
    p.add_argument("--target-col", type=str, default=PRIMEKG_CSV_SPEC.target_col)
    p.add_argument("--relation-col", type=str, default=PRIMEKG_CSV_SPEC.relation_col)
    p.add_argument("--source-type-col", type=str, default=PRIMEKG_CSV_SPEC.source_type_col)
    p.add_argument("--target-type-col", type=str, default=PRIMEKG_CSV_SPEC.target_type_col)
    p.add_argument(
        "--split-protocol",
        choices=("node", "node_category", "edge"),
        default="edge",
        help="Must match future training runs (default edge, same as drkg_full_experiments_array.slurm).",
    )
    p.add_argument(
        "--negative-sampling-mode",
        choices=("global", "type_matched"),
        default=None,
        help="Omit to use the same default as run_gpu_method (node→type_matched, edge→global).",
    )
    p.add_argument("--negatives-per-pos", type=int, default=20)
    p.add_argument("--decoder", choices=("dot", "mlp"), default="dot")
    p.add_argument(
        "--shuffle-relations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--held-out-relations",
        nargs="*",
        default=[],
        metavar="REL",
        help="Same as run_gpu_method: relation names excluded from training positives.",
    )
    p.add_argument(
        "--held-out-node-categories",
        nargs="*",
        default=[],
        metavar="TYPE",
        help="Same as run_gpu_method: required for split-protocol node_category.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in-dim", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument(
        "--add-self-loops",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    def _log(msg: str) -> None:
        print(f"[export_prepared] ({time.perf_counter() - t0:,.1f}s) {msg}", file=sys.stderr, flush=True)

    resolved_fmt = _resolve_input_format(args)
    _log(f"Loading graph from {args.input} (format={resolved_fmt}) …")
    graph, csv_spec, _ = _load_graph(args)
    _log(
        f"Graph in memory: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges"
    )

    _log(
        "Running prepare_link_prediction_dataset (PyG + split + negatives; can take a long time) …"
    )
    dataset = prepare_link_prediction_dataset(
        graph,
        in_dim=args.in_dim,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_protocol=args.split_protocol,
        negative_sampling_mode=args.negative_sampling_mode,
        negatives_per_pos=args.negatives_per_pos,
        decoder=args.decoder,
        shuffle_relations=args.shuffle_relations,
        add_self_loops=args.add_self_loops,
        held_out_relations=_held_out_relations_from_args(args),
        held_out_node_categories=_held_out_node_categories_from_args(args),
    )
    _log(
        f"Prepared: split={dataset.split_protocol} neg_mode={dataset.negative_sampling_mode} "
        f"train_pos_edges={dataset.split.train_pos_edge_index.size(1):,}"
    )

    meta = build_cache_meta(
        input_path=args.input.resolve(),
        resolved_input_format=resolved_fmt,
        max_edges=args.max_edges,
        csv_spec=csv_spec,
        dataset=dataset,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        in_dim=args.in_dim,
        add_self_loops=args.add_self_loops,
        negative_sampling_mode_cli=args.negative_sampling_mode,
    )
    _log(f"Writing cache -> {args.output} …")
    save_prepared_link_prediction_dataset(args.output, dataset, meta=meta)
    _log(f"Done. Load later with kgml_new.data.prepared_dataset_cache.load_prepared_link_prediction_dataset")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
