"""Save/load a full :class:`LinkPredictionDataset` (PyG tensors + splits + negatives) for reuse."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from kgml_new.data.datasets import LinkPredictionDataset
from kgml_new.data.loaders import GraphCSVSpec


PREPARED_LINK_PREDICTION_CACHE_VERSION = 1


def graph_csv_spec_to_dict(spec: GraphCSVSpec) -> dict[str, Any]:
    return {
        "source_col": spec.source_col,
        "target_col": spec.target_col,
        "relation_col": spec.relation_col,
        "source_type_col": spec.source_type_col,
        "target_type_col": spec.target_type_col,
        "source_id_col": spec.source_id_col,
        "target_id_col": spec.target_id_col,
        "node_type_attr": spec.node_type_attr,
        "relation_attr": spec.relation_attr,
        "default_node_type": spec.default_node_type,
        "edge_attr_cols": list(spec.edge_attr_cols),
        "directed": spec.directed,
        "delimiter": spec.delimiter,
    }


def input_fingerprint(path: Path) -> dict[str, Any] | None:
    path = Path(path).resolve()
    if not path.is_file():
        return None
    st = path.stat()
    return {"path": str(path), "st_size": st.st_size, "st_mtime_ns": st.st_mtime_ns}


def build_cache_meta(
    *,
    input_path: Path,
    resolved_input_format: str,
    max_edges: int | None,
    csv_spec: GraphCSVSpec | None,
    dataset: LinkPredictionDataset,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    in_dim: int,
    add_self_loops: bool,
    negative_sampling_mode_cli: str | None,
) -> dict[str, Any]:
    """Metadata stored next to the pickled dataset for safe reload validation."""
    g = dataset.graph
    meta = {
        "format_version": PREPARED_LINK_PREDICTION_CACHE_VERSION,
        "input": input_fingerprint(input_path),
        "resolved_input_format": resolved_input_format,
        "max_edges": max_edges,
        "graph_csv_spec": graph_csv_spec_to_dict(csv_spec) if csv_spec is not None else None,
        "num_graph_nodes": int(g.number_of_nodes()),
        "num_graph_edges": int(g.number_of_edges()),
        "in_dim": int(in_dim),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "split_protocol": str(dataset.split_protocol),
        "negative_sampling_mode_resolved": str(dataset.negative_sampling_mode),
        "negative_sampling_mode_cli": negative_sampling_mode_cli,
        "negatives_per_pos": int(dataset.negatives_per_pos),
        "decoder": str(dataset.decoder),
        "shuffle_relations": bool(dataset.shuffle_relations),
        "add_self_loops": bool(add_self_loops),
        "split_class": type(dataset.split).__name__,
    }
    if getattr(dataset, "held_out_relation_ids", None):
        meta["held_out_relations"] = sorted(dataset.held_out_relations or [])
        meta["held_out_relation_ids"] = sorted(dataset.held_out_relation_ids)
    hoc = getattr(dataset, "held_out_node_categories", None)
    if hoc:
        meta["held_out_node_categories"] = sorted(hoc)
    return meta


def save_prepared_link_prediction_dataset(
    path: Path,
    dataset: LinkPredictionDataset,
    *,
    meta: dict[str, Any],
) -> None:
    """Pickle ``LinkPredictionDataset`` plus ``meta`` (version + build fingerprint)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta.get("format_version") != PREPARED_LINK_PREDICTION_CACHE_VERSION:
        raise ValueError(
            f"meta format_version must be {PREPARED_LINK_PREDICTION_CACHE_VERSION}, got {meta.get('format_version')!r}"
        )
    payload = {"format_version": PREPARED_LINK_PREDICTION_CACHE_VERSION, "meta": meta, "dataset": dataset}
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_prepared_link_prediction_dataset(
    path: Path,
    *,
    expected_meta: dict[str, Any] | None = None,
) -> tuple[LinkPredictionDataset, dict[str, Any]]:
    """
    Load cache written by :func:`save_prepared_link_prediction_dataset`.

    If ``expected_meta`` is set, every key present in ``expected_meta`` must equal the
    stored meta (for safe reuse with ``run_gpu_method`` or other runners).
    """
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    version = int(payload.get("format_version", 0))
    if version != PREPARED_LINK_PREDICTION_CACHE_VERSION:
        raise ValueError(
            f"Unsupported prepared dataset cache version {version!r} in {path} "
            f"(expected {PREPARED_LINK_PREDICTION_CACHE_VERSION})"
        )
    meta = payload.get("meta")
    dataset = payload.get("dataset")
    if not isinstance(meta, dict) or not isinstance(dataset, LinkPredictionDataset):
        raise ValueError(f"Invalid prepared dataset cache contents in {path}")

    if expected_meta is not None:
        _assert_meta_subset(expected_meta, meta, path)

    return dataset, meta


def _input_fingerprint_compatible_for_reload(
    expected: dict[str, Any], stored: dict[str, Any]
) -> bool:
    """Same graph file as when the cache was built: path + byte size must match.

    ``st_mtime_ns`` is ignored — it changes on touch/rsync/metadata refresh without
    changing file contents, which would otherwise invalidate a valid cache.
    """
    if expected.get("path") != stored.get("path"):
        return False
    if expected.get("st_size") != stored.get("st_size"):
        return False
    return True


def _assert_meta_subset(expected: dict[str, Any], actual: dict[str, Any], path: Path) -> None:
    mismatches: list[str] = []
    for key, exp_val in expected.items():
        if key not in actual:
            mismatches.append(f"{key!r}: missing in cache")
            continue
        act_val = actual[key]
        if (
            key == "input"
            and isinstance(exp_val, dict)
            and isinstance(act_val, dict)
            and _input_fingerprint_compatible_for_reload(exp_val, act_val)
        ):
            continue
        if exp_val != act_val:
            mismatches.append(f"{key!r}: cache has {act_val!r}, expected {exp_val!r}")
    if mismatches:
        raise ValueError(
            f"Prepared dataset cache meta mismatch for {path}:\n  " + "\n  ".join(mismatches)
        )
