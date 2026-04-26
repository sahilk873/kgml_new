"""Load embedding tensors from .pt artifacts for UMAP visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class LoadedEmbeddings:
    """Embeddings matrix and optional per-row labels for plotting."""

    name: str
    matrix: np.ndarray  # float32 [n, d]
    row_indices: np.ndarray  # int64 — global row ids used after subsampling (graph index or cache row index)
    node_types: list[str] | None = None


def extract_tensor_from_payload(payload: object, *, key: str | None = None) -> torch.Tensor:
    """Match run_gpu_method node embedding resolution."""
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        if key is not None:
            value = payload.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Node embedding key {key!r} not found or not a tensor.")
            return value
        for candidate in ("embeddings", "node_embeddings", "x", "features"):
            value = payload.get(candidate)
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError(
            "Dict payload has no tensor under embeddings, node_embeddings, x, or features."
        )
    raise TypeError(f"Unsupported payload type: {type(payload)}")


def load_payload(path: Path, *, map_location: str | torch.device = "cpu") -> object:
    return torch.load(path, map_location=map_location)


def num_nodes_from_result_json(path: Path) -> int | None:
    """Read num_nodes from run_gpu_method JSON (single int) or skip if hetero."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    nn = data.get("num_nodes")
    if isinstance(nn, int):
        return nn
    if isinstance(nn, dict) and len(nn) == 1:
        return int(next(iter(nn.values())))
    return None


def subsample_indices(num_rows: int, max_nodes: int | None, seed: int) -> np.ndarray:
    """Deterministic subset of row indices in [0, num_rows)."""
    if max_nodes is None or max_nodes >= num_rows:
        return np.arange(num_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_rows, size=max_nodes, replace=False)
    return np.sort(idx.astype(np.int64))


def load_row_labels_from_prepared(
    prepared_pkl: Path,
    *,
    mode: str,
) -> list[str]:
    """Load per-node labels aligned with LinkPredictionDataset / train_embeddings rows."""
    from kgml_new.data.datasets import compute_relation_diversity_buckets
    from kgml_new.data.prepared_dataset_cache import load_prepared_link_prediction_dataset

    ds, _ = load_prepared_link_prediction_dataset(Path(prepared_pkl))
    if mode == "node_type":
        return [str(t) for t in ds.node_types]
    if mode == "relation_diversity":
        _counts, buckets = compute_relation_diversity_buckets(ds.graph, ds.node_list)
        n = len(ds.node_list)
        return [str(buckets[i]) for i in range(n)]
    raise ValueError(f"Unknown label mode {mode!r} (use node_type or relation_diversity).")


def load_primekg_node_types_from_kg_csv(
    *,
    prepared_pkl: Path,
    kg_csv_path: Path,
    unknown_label: str = "unknown",
) -> list[str]:
    """
    Build PrimeKG node-type labels from `kg.csv` and align to prepared dataset node order.

    Alignment target is `dataset.node_list` from prepared cache, which matches train embedding
    row order for tensor artifacts and reconstructed edge-aware encoders.
    """
    from kgml_new.data.prepared_dataset_cache import load_prepared_link_prediction_dataset

    pkl = Path(prepared_pkl)
    csv = Path(kg_csv_path)
    if not pkl.is_file():
        raise FileNotFoundError(f"Prepared cache not found: {pkl}")
    if not csv.is_file():
        raise FileNotFoundError(f"kg.csv not found: {csv}")

    ds, _ = load_prepared_link_prediction_dataset(pkl)
    node_list = [str(n) for n in ds.node_list]
    needed = set(node_list)

    node_to_type: dict[str, str] = {}
    usecols = ["x_name", "x_type", "y_name", "y_type"]
    chunks = pd.read_csv(csv, usecols=usecols, chunksize=500_000)
    for chunk in chunks:
        for name_col, type_col in (("x_name", "x_type"), ("y_name", "y_type")):
            names = chunk[name_col].astype(str).tolist()
            types = chunk[type_col].astype(str).tolist()
            for n, t in zip(names, types, strict=False):
                if n not in needed:
                    continue
                prev = node_to_type.get(n)
                if prev is None:
                    node_to_type[n] = t
                elif prev != t:
                    # Keep first observed type for determinism; mismatch is unexpected.
                    continue
        if len(node_to_type) >= len(needed):
            break

    labels = [node_to_type.get(n, unknown_label) for n in node_list]
    return labels


def load_tensor_kind(
    *,
    name: str,
    path: Path,
    embeddings_key: str | None,
    expected_num_nodes: int | None,
    max_nodes: int | None,
    seed: int,
    row_labels: list[str] | None = None,
) -> LoadedEmbeddings:
    payload = load_payload(path)
    tensor = extract_tensor_from_payload(payload, key=embeddings_key)
    if tensor.dim() != 2:
        raise ValueError(f"{name}: expected rank-2 tensor, got shape {tuple(tensor.shape)}")
    n, _d = int(tensor.size(0)), int(tensor.size(1))
    if expected_num_nodes is not None and n != expected_num_nodes:
        raise ValueError(
            f"{name}: embedding rows {n} != expected_num_nodes {expected_num_nodes}"
        )
    if row_labels is not None and len(row_labels) != n:
        raise ValueError(
            f"{name}: row_labels length {len(row_labels)} != embedding rows {n}"
        )
    idx = subsample_indices(n, max_nodes, seed)
    mat = tensor.detach().cpu().numpy().astype(np.float32)[idx]
    node_types: list[str] | None = None
    if row_labels is not None:
        node_types = [row_labels[i] for i in idx.tolist()]
    return LoadedEmbeddings(name=name, matrix=mat, row_indices=idx, node_types=node_types)


def load_node_cache_kind(
    *,
    name: str,
    path: Path,
    embeddings_key: str | None,
    max_nodes: int | None,
    seed: int,
) -> LoadedEmbeddings:
    payload = load_payload(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{name}: node_cache expects a dict payload, got {type(payload)}")
    tensor = extract_tensor_from_payload(payload, key=embeddings_key)
    node_types_raw = payload.get("node_types")
    n = int(tensor.size(0))
    idx = subsample_indices(n, max_nodes, seed)
    mat = tensor.detach().cpu().numpy().astype(np.float32)[idx]
    node_types: list[str] | None = None
    if isinstance(node_types_raw, list) and len(node_types_raw) == n:
        node_types = [str(node_types_raw[i]) for i in idx]
    elif node_types_raw is not None:
        raise ValueError(
            f"{name}: node_types length {len(node_types_raw)} != embeddings rows {n}"
        )
    return LoadedEmbeddings(name=name, matrix=mat, row_indices=idx, node_types=node_types)


def validate_same_row_space(loads: list[LoadedEmbeddings]) -> None:
    """Ensure same subsample size (different methods had same N after subsample)."""
    if len(loads) < 2:
        return
    n0 = loads[0].matrix.shape[0]
    for le in loads[1:]:
        if le.matrix.shape[0] != n0:
            raise ValueError(
                f"Subsampled row counts differ: {loads[0].name} has {n0}, "
                f"{le.name} has {le.matrix.shape[0]} — use same max-nodes or matching manifests."
            )


def joint_standardize(matrices: list[np.ndarray]) -> list[np.ndarray]:
    """Per-matrix z-score columns (after stacking alignment), applied independently per method."""
    out: list[np.ndarray] = []
    for m in matrices:
        mu = m.mean(axis=0, keepdims=True)
        sig = m.std(axis=0, keepdims=True)
        sig = np.where(sig < 1e-8, 1.0, sig)
        out.append(((m - mu) / sig).astype(np.float32))
    return out


def optional_row_l2_normalize(m: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (m / norms).astype(np.float32)


def parse_methods_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    raw = path.read_text()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise SystemExit(
                "PyYAML is required for YAML manifests. "
                "Install with: pip install 'kgml-new[viz]'"
            ) from e
        parsed = yaml.safe_load(raw)
    elif suffix == ".json":
        parsed = json.loads(raw)
    else:
        raise ValueError(f"Unsupported manifest extension: {suffix}")

    if not isinstance(parsed, list):
        raise ValueError("Manifest root must be a list of method entries.")
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {i} must be an object.")
        if "name" not in entry or "path" not in entry:
            raise ValueError(f"Manifest entry {i} requires 'name' and 'path'.")
        entry = dict(entry)
        entry["path"] = Path(entry["path"])
        out.append(entry)
    return out
