from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_KEYS = ("mrr", "hits@1", "hits@3", "hits@10", "roc_auc", "average_precision")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate kgml_new JSON metrics across seeds.")
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--metric-block",
        default="metrics",
        help="JSON metric block to aggregate. For KG completion use metrics (default), "
        "test_metrics.type_constrained, or test_metrics.all_entities.",
    )
    return p.parse_args()


def _get_block(payload: dict, dotted: str) -> dict:
    cur = payload
    for part in dotted.split("."):
        cur = cur.get(part, {})
        if not isinstance(cur, dict):
            return {}
    return cur


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        payload = json.loads(path.read_text())
        block = _get_block(payload, args.metric_block)
        rows.append({"path": str(path), "seed": payload.get("seed"), "metrics": block})
    summary: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        vals = [float(row["metrics"][key]) for row in rows if key in row["metrics"]]
        vals = [v for v in vals if v == v]
        if vals:
            arr = np.asarray(vals, dtype=float)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n": int(arr.size),
            }
    out = {"metric_block": args.metric_block, "summary": summary, "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
