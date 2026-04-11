"""
Aggregate OOD difficulty metrics from multiple run JSON files (seeds / models).

Example:
  python -m kgml_new.scripts.aggregate_ood_results \\
    --inputs run1.json run2.json \\
    --tag drkg_edge \\
    --group-keys input_path split_protocol method edge_relation_mode embedding_model_resolved
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None and v == v]  # noqa: PLR1714
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return float(statistics.mean(clean)), float(statistics.pstdev(clean))


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate OOD difficulty JSON outputs.")
    p.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Result JSON files from run_gpu_method.",
    )
    p.add_argument("--tag", type=str, default="aggregate", help="Output filename tag.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/ood"),
        help="Directory for aggregate JSON/CSV.",
    )
    p.add_argument(
        "--group-keys",
        nargs="+",
        default=[
            "input_path",
            "split_protocol",
            "method",
            "edge_relation_mode",
            "embedding_model_resolved",
        ],
        help="Record fields used to group runs before averaging.",
    )
    p.add_argument(
        "--baseline-embedding",
        type=str,
        default="random",
        help="embedding_model_resolved value treated as baseline for gap columns.",
    )
    p.add_argument(
        "--target-embedding",
        type=str,
        default="openai",
        help="embedding_model_resolved value treated as target for gap columns.",
    )
    args = p.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.inputs:
        data = _load(path)
        if "ood_difficulty" not in data:
            continue
        rec = {
            "_path": str(path),
            "ood_difficulty": data["ood_difficulty"],
            "seed": data.get("seed"),
        }
        for k in args.group_keys:
            rec[k] = data.get(k)
        records.append(rec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by tuple of group key values
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = tuple(rec.get(k) for k in args.group_keys)
        groups[key].append(rec)

    summary: dict[str, Any] = {
        "tag": args.tag,
        "group_keys": args.group_keys,
        "num_files": len(args.inputs),
        "num_with_ood": len(records),
        "groups": {},
    }

    flat_rows: list[dict[str, Any]] = []

    for gkey, grecs in groups.items():
        gdict = {
            k: v for k, v in zip(args.group_keys, gkey, strict=False)
        }
        gdict["n_runs"] = len(grecs)
        key_str = "_".join(str(x) for x in gkey)[:200]

        # Collect per (split, difficulty, bucket) lists of roc_auc
        metric_cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)

        for rec in grecs:
            ood = rec["ood_difficulty"]
            for split in ("val", "test"):
                sp = ood.get(split)
                if not sp or "metrics" not in sp:
                    continue
                for diff_type, block in sp["metrics"].items():
                    if not isinstance(block, dict):
                        continue
                    for bname, bmet in block.get("buckets", {}).items():
                        if not isinstance(bmet, dict):
                            continue
                        auc = bmet.get("roc_auc")
                        if auc is not None and auc == auc:
                            metric_cells[(split, diff_type, str(bname))].append(float(auc))

        agg_buckets: dict[str, Any] = {}
        for (split, diff_type, bname), aucs in metric_cells.items():
            m, s = _mean_std(aucs)
            agg_buckets.setdefault(split, {}).setdefault(diff_type, {})[bname] = {
                "roc_auc_mean": m,
                "roc_auc_std": s,
                "n": len(aucs),
            }

        gdict["aggregated_metrics"] = agg_buckets
        summary["groups"][key_str] = gdict

        for split, diff_type, bname in metric_cells:
            m, st = _mean_std(metric_cells[(split, diff_type, bname)])
            flat_rows.append(
                {
                    **{k: v for k, v in zip(args.group_keys, gkey, strict=False)},
                    "split": split,
                    "difficulty_type": diff_type,
                    "bucket": bname,
                    "roc_auc_mean": m,
                    "roc_auc_std": st,
                    "n": len(metric_cells[(split, diff_type, bname)]),
                }
            )

        # Pair by seed: semantic minus random gap per bucket (test split).
        by_emb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in grecs:
            emb = rec.get("embedding_model_resolved")
            by_emb[str(emb)].append(rec)

        gaps: list[dict[str, Any]] = []
        base_key, tgt_key = args.baseline_embedding, args.target_embedding
        if base_key in by_emb and tgt_key in by_emb:
            by_seed_base = {rec.get("seed"): rec for rec in by_emb[base_key]}
            for rec_t in by_emb[tgt_key]:
                s = rec_t.get("seed")
                rec_b = by_seed_base.get(s)
                if rec_b is None:
                    continue
                ood_t = rec_t["ood_difficulty"].get("test", {}).get("metrics", {})
                ood_b = rec_b["ood_difficulty"].get("test", {}).get("metrics", {})
                for diff_type in set(ood_t) & set(ood_b):
                    bt = ood_t.get(diff_type, {}).get("buckets", {})
                    bb = ood_b.get(diff_type, {}).get("buckets", {})
                    for bname in set(bt) & set(bb):
                        a_t = bt[bname].get("roc_auc")
                        a_b = bb[bname].get("roc_auc")
                        if (
                            a_t is not None
                            and a_b is not None
                            and a_t == a_t
                            and a_b == a_b
                        ):
                            gaps.append(
                                {
                                    "seed": s,
                                    "difficulty_type": diff_type,
                                    "bucket": bname,
                                    "roc_auc_gap": float(a_t) - float(a_b),
                                }
                            )
        gdict["semantic_minus_random_gaps_by_seed_test"] = gaps

    json_path = out_dir / f"aggregate_{args.tag}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    csv_path = out_dir / f"aggregate_{args.tag}.csv"
    if flat_rows:
        import csv

        keys = sorted({k for row in flat_rows for k in row})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat_rows)
    else:
        csv_path.write_text("")

    print(json.dumps({"wrote_json": str(json_path), "wrote_csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
