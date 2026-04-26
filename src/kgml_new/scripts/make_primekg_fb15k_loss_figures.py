#!/usr/bin/env python3
"""
Build train/validation loss comparison figures for PrimeKG and FB15k-237 from saved JSON runs.

Reads ``results/slurm/*.json`` produces by ``run_gpu_method`` / ``run_hgt`` on PrimeKG / FB15k-237,
deduplicates by (dataset, split, method variant), keeps the newest file per key, and writes overlay
figures under ``figures/loss_curves/``.

Validation loss is plotted only when present in JSON (runs produced with ``kgml_new`` that records
``val_loss``). Older artifacts may only have ``train_loss``; those still appear in the train panel.

Usage::

  python -m kgml_new.scripts.make_primekg_fb15k_loss_figures
  python -m kgml_new.scripts.make_primekg_fb15k_loss_figures \\
      --results-dir results/slurm --out-dir figures/loss_curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kgml_new.training.plot_curves import (
    dataset_key_from_payload,
    method_series_label,
    plot_methods_comparison,
    primary_training_history,
)


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _legend_sort_key(label: str) -> tuple[int, str]:
    """Stable ordering for legend (baselines first, then node2vec / hgt, then edge-aware)."""
    pri = 99
    if label.startswith("baseline"):
        pri = 0
    elif label.startswith("node2vec"):
        pri = 1
    elif label.startswith("hgt"):
        pri = 2
    elif "edge_aware" in label:
        pri = 10
    return (pri, label.lower())


def collect_best_runs(results_dir: Path) -> dict[tuple[str, str, str], Path]:
    """
    Map (dataset, split_protocol, series_label) -> newest JSON path by mtime.
    """
    best: dict[tuple[str, str, str], tuple[float, Path]] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = _load_payload(path)
        except (json.JSONDecodeError, OSError):
            continue
        ds = dataset_key_from_payload(payload)
        if ds not in ("primekg", "fb15k237"):
            continue
        split = str(payload.get("split_protocol") or "").strip()
        if split not in ("edge", "node"):
            continue
        hist = primary_training_history(payload)
        if hist is None:
            continue
        label = method_series_label(payload)
        key = (ds, split, label)
        mtime = path.stat().st_mtime
        prev = best.get(key)
        if prev is None or mtime > prev[0]:
            best[key] = (mtime, path)
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # .../src/kgml_new/scripts/this_file.py -> repo root is parents[3]
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo / "results" / "slurm",
        help="Directory containing run JSON artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "figures" / "loss_curves",
        help="Output directory for PNG figures.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"results dir not found: {results_dir}")

    best_paths = collect_best_runs(results_dir)
    by_panel: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for (ds, split, label), path in best_paths.items():
        payload = _load_payload(path)
        hist = primary_training_history(payload)
        if hist is None:
            continue
        by_panel.setdefault((ds, split), []).append((label, hist))

    if not by_panel:
        raise SystemExit(
            f"No PrimeKG / FB15k-237 runs with train_loss found under {results_dir}"
        )

    written: list[Path] = []
    for (ds, split), series in sorted(by_panel.items()):
        series_sorted = sorted(series, key=lambda item: _legend_sort_key(item[0]))
        title = f"{ds} · {split}-split · training loss"
        note = (
            "Validation loss appears in the lower panel when JSON includes val_loss "
            "(rerun sweeps with current kgml_new to populate)."
        )
        any_val = any(bool(h.get("val_loss")) for _, h in series_sorted)
        suptitle = note if not any_val else None
        out_path = out_dir / f"{ds}_{split}_methods_train_val_loss.png"
        plot_methods_comparison(
            series_sorted,
            out_path,
            title=title,
            suptitle=suptitle,
        )
        written.append(out_path)

    print(f"Wrote {len(written)} figure(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
