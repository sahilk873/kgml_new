"""Compare embedding methods via 2D UMAP projections (faceted and/or joint)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kgml_new.embeddings.umap_loading import (
    LoadedEmbeddings,
    joint_standardize,
    load_node_cache_kind,
    load_primekg_node_types_from_kg_csv,
    load_row_labels_from_prepared,
    load_tensor_kind,
    num_nodes_from_result_json,
    optional_row_l2_normalize,
    parse_methods_manifest,
    subsample_indices,
    validate_same_row_space,
)
from kgml_new.models.edge_aware_sage import build_edge_aware_model
from kgml_new.training.link_unsupervised import compute_node_embeddings


def _require_viz_stack():
    try:
        import matplotlib.pyplot as plt
        import umap
    except ImportError as e:
        raise SystemExit(
            "umap-learn and matplotlib are required. Install with:\n"
            "  pip install 'kgml-new[viz]'"
        ) from e
    return plt, umap


def _load_one_method(
    entry: dict[str, Any],
    *,
    max_nodes: int | None,
    seed: int,
    embeddings_key: str | None,
    tensor_row_labels: list[str] | None,
) -> LoadedEmbeddings:
    name = str(entry["name"])
    path = Path(entry["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name}: embedding file not found: {path}")
    kind = str(entry.get("kind", "tensor")).lower()
    expected: int | None = None
    if entry.get("expected_num_nodes") is not None:
        expected = int(entry["expected_num_nodes"])
    elif entry.get("result_json"):
        rj = Path(entry["result_json"]).expanduser().resolve()
        expected = num_nodes_from_result_json(rj)
    key = entry.get("embeddings_key")
    if embeddings_key is not None:
        key = embeddings_key
    key_str = str(key) if key is not None else None

    if kind == "tensor":
        return load_tensor_kind(
            name=name,
            path=path,
            embeddings_key=key_str,
            expected_num_nodes=expected,
            max_nodes=max_nodes,
            seed=seed,
            row_labels=tensor_row_labels,
        )
    if kind == "node_cache":
        return load_node_cache_kind(
            name=name,
            path=path,
            embeddings_key=key_str,
            max_nodes=max_nodes,
            seed=seed,
        )
    raise ValueError(f"{name}: unknown kind {kind!r} (use tensor, node_cache, or edge_aware_encoder).")


def _resolve_relation_table_path(entry: dict[str, Any], result_json: dict[str, Any] | None) -> Path | None:
    if entry.get("relation_table_path"):
        return Path(entry["relation_table_path"]).expanduser().resolve()
    if not isinstance(result_json, dict):
        return None
    artifacts = result_json.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    extra = artifacts.get("extra_tensors")
    if not isinstance(extra, dict):
        return None
    rel = extra.get("relation_table")
    if not isinstance(rel, str):
        return None
    return Path(rel).expanduser().resolve()


def _infer_edge_aware_dims_from_state(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    layer_indices: list[int] = []
    for key in state_dict:
        if key.startswith("layers.") and key.endswith(".lin_root.weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                layer_indices.append(int(parts[1]))
    if not layer_indices:
        raise ValueError("Unable to infer layer count from checkpoint state_dict")
    num_layers = max(layer_indices) + 1
    first = state_dict["layers.0.lin_root.weight"]
    last = state_dict[f"layers.{num_layers - 1}.lin_root.weight"]
    in_dim = int(first.shape[1])
    out_dim = int(last.shape[0])
    hidden_dim = out_dim if num_layers == 1 else int(first.shape[0])
    return in_dim, hidden_dim, out_dim


def _infer_edge_dim_from_state(
    state_dict: dict[str, torch.Tensor],
    *,
    in_dim: int,
    edge_mode: str,
) -> int:
    """Infer projected edge_dim used by edge-aware encoder checkpoint."""
    proj_w = state_dict.get("relation_projection.weight")
    if isinstance(proj_w, torch.Tensor):
        return int(proj_w.shape[0])
    if edge_mode == "concat" and "layers.0.lin.weight" in state_dict:
        msg_in = int(state_dict["layers.0.lin.weight"].shape[1])
        edge_dim = msg_in - in_dim
        if edge_dim <= 0:
            raise ValueError(f"Invalid inferred edge_dim from msg_in={msg_in}, in_dim={in_dim}")
        return edge_dim
    raise ValueError(
        "Unable to infer edge_dim from checkpoint. "
        "Expected relation_projection.weight or concat layer dimensions."
    )


def _load_edge_aware_encoder_kind(
    *,
    name: str,
    encoder_path: Path,
    result_json_path: Path | None,
    prepared_cache_path: Path,
    expected_num_nodes: int | None,
    max_nodes: int | None,
    seed: int,
    row_labels: list[str] | None,
    device: torch.device,
    entry: dict[str, Any],
) -> LoadedEmbeddings:
    from kgml_new.data.prepared_dataset_cache import load_prepared_link_prediction_dataset

    if not prepared_cache_path.is_file():
        raise FileNotFoundError(f"{name}: prepared cache not found: {prepared_cache_path}")
    dataset, _meta = load_prepared_link_prediction_dataset(prepared_cache_path)
    train_data = dataset.train_data

    result_json: dict[str, Any] | None = None
    if result_json_path is not None and result_json_path.is_file():
        result_json = json.loads(result_json_path.read_text())

    checkpoint = torch.load(encoder_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"{name}: unsupported encoder checkpoint payload {type(checkpoint)}")

    relation_table_path = _resolve_relation_table_path(entry, result_json)
    if relation_table_path is None or not relation_table_path.is_file():
        raise FileNotFoundError(
            f"{name}: relation_table artifact missing. Set relation_table_path in manifest entry."
        )
    relation_table = torch.load(relation_table_path, map_location="cpu")
    if not isinstance(relation_table, torch.Tensor):
        raise TypeError(f"{name}: relation_table payload is not a tensor")

    in_dim, hidden_dim, out_dim = _infer_edge_aware_dims_from_state(state_dict)
    edge_mode = "concat"
    num_relation_bases = 4
    neighbor_aggr = "mean"
    concat = True
    if isinstance(result_json, dict):
        edge_mode = str(result_json.get("edge_relation_mode", edge_mode))
        num_relation_bases = int(result_json.get("num_relation_bases", num_relation_bases))
        neighbor_aggr = str(result_json.get("neighbor_aggr", neighbor_aggr))
        concat = bool(result_json.get("edge_message_concat", True))

    edge_dim = _infer_edge_dim_from_state(
        state_dict,
        in_dim=in_dim,
        edge_mode=edge_mode,
    )
    model = build_edge_aware_model(
        edge_relation_mode=edge_mode,
        in_channels=in_dim,
        edge_dim=edge_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim,
        relation_table=relation_table.to(device),
        num_layers=max(1, len({k.split(".")[1] for k in state_dict if k.startswith("layers.") and k.split(".")[1].isdigit()})),
        dropout=0.0,
        concat=concat,
        normalize_output=True,
        num_relation_bases=num_relation_bases,
        neighbor_aggr=neighbor_aggr,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    z = compute_node_embeddings(model, train_data, device, edge_aware=True).detach().cpu()
    n = int(z.size(0))
    if expected_num_nodes is not None and n != expected_num_nodes:
        raise ValueError(f"{name}: embedding rows {n} != expected_num_nodes {expected_num_nodes}")
    if row_labels is not None and len(row_labels) != n:
        raise ValueError(f"{name}: row_labels length {len(row_labels)} != embedding rows {n}")

    idx = subsample_indices(n, max_nodes, seed)
    mat = z.numpy().astype(np.float32)[idx]
    labels = [row_labels[i] for i in idx.tolist()] if row_labels is not None else None
    return LoadedEmbeddings(name=name, matrix=mat, row_indices=idx, node_types=labels)


def _plot_faceted(
    loads: list[LoadedEmbeddings],
    *,
    coords_list: list[np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    _require_viz_stack()
    import matplotlib.pyplot as plt_module

    n = len(loads)
    fig, axes = plt_module.subplots(
        1,
        n,
        figsize=(4 * n, 4),
        squeeze=False,
    )
    row = axes[0]
    for ax, le, xy in zip(row, loads, coords_list, strict=True):
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.6, c="#2563eb")
        ax.set_title(le.name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt_module.close(fig)


def _plot_faceted_colored_by_type(
    loads: list[LoadedEmbeddings],
    *,
    coords_list: list[np.ndarray],
    out_path: Path,
    title: str,
    legend_suffix: str = "",
) -> None:
    _require_viz_stack()
    import matplotlib.pyplot as plt_module

    n = len(loads)
    fig, axes = plt_module.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    cmap = plt_module.colormaps["tab20"].resampled(20)
    row = axes[0]
    for ax, le, xy in zip(row, loads, coords_list, strict=True):
        if le.node_types is None:
            ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.6, c="#64748b")
        else:
            cats = sorted(set(le.node_types))
            cat_to_i = {c: i for i, c in enumerate(cats)}
            colors = [cmap(cat_to_i[t] % 20) for t in le.node_types]
            ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.6, c=colors)
        ax.set_title(le.name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title + legend_suffix)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt_module.close(fig)


def _plot_joint(
    *,
    stacked_coords: np.ndarray,
    method_labels: np.ndarray,
    method_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    _require_viz_stack()
    import matplotlib.pyplot as plt_module

    fig, ax = plt_module.subplots(figsize=(7, 6))
    uniq = method_names
    for i, name in enumerate(uniq):
        mask = method_labels == i
        ax.scatter(
            stacked_coords[mask, 0],
            stacked_coords[mask, 1],
            s=4,
            alpha=0.55,
            label=name,
            color=f"C{i % 10}",
        )
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt_module.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UMAP comparison of method embeddings (install optional [viz])."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="YAML or JSON list of {name, path, kind?, result_json?, ...}",
    )
    p.add_argument(
        "--method",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        default=None,
        help="Repeatable: method label and path to .pt (implies kind=tensor).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("figures/umap"))
    p.add_argument("--stem", type=str, default="umap_compare", help="Output filename stem.")
    p.add_argument("--title", type=str, default="Method embedding UMAP")
    p.add_argument(
        "--layout",
        choices=("faceted", "joint", "both"),
        default="both",
    )
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument(
        "--prepared-dataset-cache",
        type=Path,
        default=None,
        help="Prepared dataset cache .pkl for encoder artifact kinds (row/order source).",
    )
    p.add_argument(
        "--encoder-device",
        type=str,
        default=None,
        help="Device for encoder->embedding conversion (default: cuda if available).",
    )
    p.add_argument(
        "--joint-l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize rows before per-method z-score (joint layout only).",
    )
    p.add_argument(
        "--color-by",
        choices=("none", "labels"),
        default="none",
        help="Color faceted points from per-node labels (node cache, or --labels-from-prepared for tensors).",
    )
    p.add_argument(
        "--labels-from-prepared",
        type=Path,
        default=None,
        help="Pickle from save_prepared_link_prediction_dataset; label order matches train_embeddings rows.",
    )
    p.add_argument(
        "--label-mode",
        choices=("node_type", "relation_diversity"),
        default="node_type",
        help="With --labels-from-prepared: which per-node label to use (relation_diversity: low/medium/high).",
    )
    p.add_argument(
        "--labels-from-kg-csv",
        type=Path,
        default=None,
        help="PrimeKG `kg.csv` path. Uses x_name/y_name -> x_type/y_type to build node labels.",
    )
    p.add_argument(
        "--embeddings-key",
        type=str,
        default=None,
        help="Override tensor key inside dict payloads for all methods.",
    )
    p.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save coordinates and labels under output-dir.",
    )
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _require_viz_stack()
    import umap as umap_lib

    methods_list: list[dict[str, Any]]
    if args.manifest:
        methods_list = parse_methods_manifest(Path(args.manifest).resolve())
    elif args.method:
        methods_list = [
            {"name": name, "path": Path(path), "kind": "tensor"} for name, path in args.method
        ]
    else:
        raise SystemExit("Provide --manifest or one or more --method NAME PATH")

    prepared_labels: list[str] | None = None
    kg_csv_labels: list[str] | None = None
    if args.labels_from_prepared is not None:
        pkl = Path(args.labels_from_prepared).expanduser().resolve()
        if not pkl.is_file():
            raise SystemExit(f"--labels-from-prepared not found: {pkl}")
        prepared_labels = load_row_labels_from_prepared(
            pkl, mode=str(args.label_mode)
        )
    if args.labels_from_kg_csv is not None:
        if args.prepared_dataset_cache is None:
            raise SystemExit(
                "--labels-from-kg-csv requires --prepared-dataset-cache for row alignment."
            )
        kg_csv = Path(args.labels_from_kg_csv).expanduser().resolve()
        prepared = Path(args.prepared_dataset_cache).expanduser().resolve()
        kg_csv_labels = load_primekg_node_types_from_kg_csv(
            prepared_pkl=prepared,
            kg_csv_path=kg_csv,
        )

    loads: list[LoadedEmbeddings] = []
    encoder_device = torch.device(
        args.encoder_device if args.encoder_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    for e in methods_list:
        kind = str(e.get("kind", "tensor")).lower()
        labels_for_tensor = prepared_labels
        if labels_for_tensor is None and kg_csv_labels is not None:
            labels_for_tensor = kg_csv_labels
        tlab = labels_for_tensor if kind == "tensor" else None
        if kind in ("tensor", "node_cache"):
            loads.append(
                _load_one_method(
                    e,
                    max_nodes=args.max_nodes,
                    seed=args.seed,
                    embeddings_key=args.embeddings_key,
                    tensor_row_labels=tlab,
                )
            )
            continue
        if kind == "edge_aware_encoder":
            name = str(e["name"])
            path = Path(e["path"]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{name}: encoder file not found: {path}")
            result_json_path = (
                Path(e["result_json"]).expanduser().resolve()
                if e.get("result_json") is not None
                else None
            )
            expected = None
            if e.get("expected_num_nodes") is not None:
                expected = int(e["expected_num_nodes"])
            elif result_json_path is not None:
                expected = num_nodes_from_result_json(result_json_path)
            prepared = (
                Path(e["prepared_dataset_cache"]).expanduser().resolve()
                if e.get("prepared_dataset_cache") is not None
                else (
                    Path(args.prepared_dataset_cache).expanduser().resolve()
                    if args.prepared_dataset_cache is not None
                    else None
                )
            )
            if prepared is None:
                raise SystemExit(
                    f"{name}: edge_aware_encoder requires prepared_dataset_cache "
                    "(entry field or --prepared-dataset-cache)."
                )
            encoder_labels = prepared_labels
            if encoder_labels is None and kg_csv_labels is not None:
                encoder_labels = kg_csv_labels
            loads.append(
                _load_edge_aware_encoder_kind(
                    name=name,
                    encoder_path=path,
                    result_json_path=result_json_path,
                    prepared_cache_path=prepared,
                    expected_num_nodes=expected,
                    max_nodes=args.max_nodes,
                    seed=args.seed,
                    row_labels=encoder_labels,
                    device=encoder_device,
                    entry=e,
                )
            )
            continue
        raise ValueError(f"Unknown method kind: {kind}")
    validate_same_row_space(loads)

    kinds = {str(e.get("kind", "tensor")).lower() for e in methods_list}
    joint_requested = args.layout in ("joint", "both")
    tensor_like = any(k in {"tensor", "edge_aware_encoder"} for k in kinds)
    if joint_requested and "node_cache" in kinds and tensor_like:
        raise SystemExit(
            "Joint UMAP mixes tensor (graph row order) with node_cache (sorted node_ids); "
            "run separate manifests or use faceted-only (--layout faceted)."
        )

    n_rows = loads[0].matrix.shape[0]
    nn_fac = max(2, min(args.n_neighbors, max(1, n_rows - 1)))
    reducer = umap_lib.UMAP(
        n_neighbors=nn_fac,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        verbose=False,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.layout in ("faceted", "both"):
        coords_fac: list[np.ndarray] = []
        for le in loads:
            xy = reducer.fit_transform(le.matrix)
            coords_fac.append(xy.astype(np.float32))
        stem_fac = out_dir / f"{args.stem}_faceted"
        if args.color_by == "labels":
            leg = ""
            if args.labels_from_prepared is not None:
                leg = f" (labels: {args.label_mode} from prepared cache)"
            _plot_faceted_colored_by_type(
                loads,
                coords_list=coords_fac,
                out_path=stem_fac.with_suffix(".png"),
                title=args.title,
                legend_suffix=leg,
            )
        else:
            _plot_faceted(
                loads,
                coords_list=coords_fac,
                out_path=stem_fac.with_suffix(".png"),
                title=args.title,
            )
        if args.save_npz:
            np.savez(
                out_dir / f"{args.stem}_faceted.npz",
                **{
                    f"{le.name}_xy": coords_fac[i]
                    for i, le in enumerate(loads)
                },
            )

    if args.layout in ("joint", "both"):
        mats = [optional_row_l2_normalize(le.matrix, args.joint_l2_normalize) for le in loads]
        mats = joint_standardize(mats)
        stacked = np.vstack(mats).astype(np.float32)
        labels = np.concatenate(
            [np.full(le.matrix.shape[0], i, dtype=np.int32) for i, le in enumerate(loads)]
        )
        nn_joint = max(2, min(args.n_neighbors, max(1, stacked.shape[0] - 1)))
        reducer_j = umap_lib.UMAP(
            n_neighbors=nn_joint,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed + 1,
            verbose=False,
        )
        xy_joint = reducer_j.fit_transform(stacked)
        joint_path = out_dir / f"{args.stem}_joint.png"
        _plot_joint(
            stacked_coords=xy_joint.astype(np.float32),
            method_labels=labels,
            method_names=[le.name for le in loads],
            out_path=joint_path,
            title=args.title + " (joint; standardized per method)",
        )
        if args.save_npz:
            np.savez(
                out_dir / f"{args.stem}_joint.npz",
                xy=xy_joint.astype(np.float32),
                method_index=labels,
                method_names=np.array([le.name for le in loads]),
            )

    meta = {
        "methods": [m["name"] for m in methods_list],
        "paths": [str(Path(m["path"]).resolve()) for m in methods_list],
        "layout": args.layout,
        "seed": args.seed,
        "color_by": args.color_by,
        "labels_from_prepared": str(args.labels_from_prepared)
        if args.labels_from_prepared
        else None,
        "labels_from_kg_csv": str(args.labels_from_kg_csv)
        if args.labels_from_kg_csv
        else None,
        "label_mode": args.label_mode,
        "umap": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
        },
        "joint_l2_normalize": args.joint_l2_normalize,
        "output_dir": str(out_dir),
    }
    (out_dir / f"{args.stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"ok": True, **meta}, indent=2))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
