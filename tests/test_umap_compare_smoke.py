"""Smoke tests for UMAP comparison (requires optional [viz] deps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


pytest.importorskip("umap")
pytest.importorskip("matplotlib.pyplot")


from kgml_new.scripts.umap_compare_methods import run as umap_compare_run


def test_umap_compare_faceted_smoke(tmp_path: Path) -> None:
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    torch.manual_seed(0)
    torch.save(torch.randn(80, 16), a)
    torch.save(torch.randn(80, 32), b)

    out_dir = tmp_path / "out"
    umap_compare_run(
        [
            "--method",
            "m_a",
            str(a),
            "--method",
            "m_b",
            str(b),
            "--output-dir",
            str(out_dir),
            "--stem",
            "smoke",
            "--layout",
            "faceted",
            "--max-nodes",
            "40",
            "--seed",
            "1",
            "--n-neighbors",
            "5",
            "--min-dist",
            "0.2",
        ]
    )

    png = out_dir / "smoke_faceted.png"
    pdf = out_dir / "smoke_faceted.pdf"
    meta = out_dir / "smoke_meta.json"
    assert png.is_file(), "PNG should be written"
    assert pdf.is_file(), "PDF should be written"
    assert meta.is_file(), "meta JSON should be written"


def test_umap_compare_joint_smoke(tmp_path: Path) -> None:
    a = tmp_path / "x.pt"
    torch.manual_seed(1)
    torch.save(torch.randn(50, 8), a)

    out_dir = tmp_path / "out2"
    umap_compare_run(
        [
            "--method",
            "only",
            str(a),
            "--output-dir",
            str(out_dir),
            "--stem",
            "one",
            "--layout",
            "joint",
            "--max-nodes",
            "50",
            "--seed",
            "2",
            "--n-neighbors",
            "4",
        ]
    )
    assert (out_dir / "one_joint.png").is_file()


def test_mixed_kind_joint_error(tmp_path: Path) -> None:
    """Joint layout must reject mixing tensor rows with node_cache ordering."""
    import kgml_new.scripts.umap_compare_methods as mod

    tensor_p = tmp_path / "t.pt"
    torch.save(torch.randn(30, 4), tensor_p)

    cache_p = tmp_path / "cache.pt"
    torch.save(
        {
            "embeddings": torch.randn(30, 4),
            "node_types": ["a"] * 30,
            "num_nodes": 30,
            "artifact_type": "node_embeddings",
            "format_version": 2,
            "node_ids": [str(i) for i in range(30)],
        },
        cache_p,
    )

    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            [
                {"name": "t", "path": str(tensor_p), "kind": "tensor"},
                {"name": "c", "path": str(cache_p), "kind": "node_cache"},
            ]
        )
    )

    with pytest.raises(SystemExit):
        mod.run(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "ox"),
                "--layout",
                "joint",
            ]
        )
