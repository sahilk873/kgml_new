from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


SPLIT_CANDIDATES = {
    "train": ("train.txt", "train.tsv", "train.csv"),
    "valid": (
        "valid.txt",
        "validation.txt",
        "valid.tsv",
        "valid.csv",
        "dev.txt",
        "dev.tsv",
        "dev.csv",
    ),
    "test": ("test.txt", "test.tsv", "test.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FB15k-237 from Hugging Face and export a kgml_new-compatible "
            "headerless TSV edge list (source relation target)."
        )
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="KGraph/FB15k-237",
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face repo revision, tag, or commit.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/fb15k-237/raw"),
        help="Directory where raw snapshot files are downloaded.",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("data/fb15k-237/fb15k-237.tsv"),
        help="Output headerless TSV in source-relation-target order.",
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=Path("data/fb15k-237/metadata.json"),
        help="Metadata JSON file summarizing split counts and source files.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for private or gated datasets.",
    )
    return parser.parse_args()


def _locate_split_file(raw_root: Path, candidates: tuple[str, ...]) -> Path:
    # Hugging Face dataset repos often nest splits under data/ (e.g. KGraph/FB15k-237).
    search_roots: list[Path] = [raw_root]
    data_dir = raw_root / "data"
    if data_dir.is_dir():
        search_roots.append(data_dir)
    for root in search_roots:
        for candidate in candidates:
            path = root / candidate
            if path.is_file():
                return path
    tried = ", ".join(f"{r}/{c}" for r in search_roots for c in candidates)
    raise FileNotFoundError(
        f"Could not locate split file under {raw_root}. Tried: {tried}"
    )


def _parse_triple_line(line: str, source_path: Path) -> tuple[str, str, str]:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"Encountered empty line in {source_path}")

    if "\t" in stripped:
        parts = [p.strip() for p in stripped.split("\t")]
    elif "," in stripped:
        parts = [p.strip() for p in stripped.split(",")]
    else:
        parts = stripped.split()

    if len(parts) != 3:
        raise ValueError(
            f"Expected 3 fields in {source_path}, got {len(parts)} for line: {stripped!r}"
        )
    head, relation, tail = parts
    return head, relation, tail


def _read_split(path: Path) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            triples.append(_parse_triple_line(line, path))
    return triples


def _write_combined_tsv(
    out_path: Path,
    splits: dict[str, list[tuple[str, str, str]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for split_name in ("train", "valid", "test"):
            for head, relation, tail in splits[split_name]:
                handle.write(f"{head}\t{relation}\t{tail}\n")


def main() -> None:
    args = parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    download_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.raw_dir),
        token=args.hf_token,
    )
    raw_root = Path(download_dir)

    split_paths = {
        split: _locate_split_file(raw_root, candidates)
        for split, candidates in SPLIT_CANDIDATES.items()
    }
    splits = {split: _read_split(path) for split, path in split_paths.items()}

    _write_combined_tsv(args.output_tsv, splits)

    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "raw_dir": str(raw_root),
        "split_files": {k: str(v) for k, v in split_paths.items()},
        "split_counts": {k: len(v) for k, v in splits.items()},
        "num_total_edges": sum(len(v) for v in splits.values()),
        "output_tsv": str(args.output_tsv),
    }
    args.meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print("")
    print("FB15k-237 prepared for kgml_new.")
    print(
        "Example run: python -m kgml_new.scripts.run_gpu_method "
        "--method baseline_sage --input data/fb15k-237/fb15k-237.tsv "
        "--split-protocol node --epochs 20 --output results/fb15k237-baseline-sage.json"
    )


if __name__ == "__main__":
    main()
