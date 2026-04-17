from __future__ import annotations

from pathlib import Path

import torch

from kgml_new.scripts import build_node_embedding_caches as script


def _fake_vectors(texts: list[str], dim: int) -> list[list[float]]:
    return [[float(i + j) for j in range(dim)] for i, _ in enumerate(texts)]


def test_build_node_embedding_caches_smoke(tmp_path: Path, monkeypatch):
    drkg_path = tmp_path / "drkg.tsv"
    drkg_path.write_text(
        "Gene::1\trelA\tDisease::2\n"
        "Gene::3\trelB\tCompound::4\n",
        encoding="utf-8",
    )

    primekg_dir = tmp_path / "data"
    primekg_dir.mkdir(parents=True, exist_ok=True)
    primekg_path = primekg_dir / "kg.csv"
    primekg_path.write_text(
        "x_name,y_name,relation,x_type,y_type\n"
        "Aspirin,PTGS2,drug_protein,drug,gene\n"
        "PTGS2,Inflammation,gene_disease,gene,disease\n",
        encoding="utf-8",
    )

    fb_dir = tmp_path / "data" / "fb15k-237"
    fb_dir.mkdir(parents=True, exist_ok=True)
    fb_path = fb_dir / "fb15k-237.tsv"
    fb_path.write_text(
        "/m/abc\t/r/likes\t/m/def\n"
        "/m/ghi\t/r/part_of\t/m/jkl\n",
        encoding="utf-8",
    )

    monkeypatch.setitem(
        script.DEFAULT_DATASETS,
        "drkg",
        script.DatasetSpec(
            name="drkg",
            input_path=drkg_path,
            input_format="tsv",
            source_col="source",
            target_col="target",
            source_type_col=None,
            target_type_col=None,
            has_header=False,
        ),
    )
    monkeypatch.setitem(
        script.DEFAULT_DATASETS,
        "primekg",
        script.DatasetSpec(
            name="primekg",
            input_path=primekg_path,
            input_format="csv",
            source_col="x_name",
            target_col="y_name",
            source_type_col="x_type",
            target_type_col="y_type",
            has_header=True,
        ),
    )
    monkeypatch.setitem(
        script.DEFAULT_DATASETS,
        "fb15k237",
        script.DatasetSpec(
            name="fb15k237",
            input_path=fb_path,
            input_format="tsv",
            source_col="source",
            target_col="target",
            source_type_col=None,
            target_type_col=None,
            has_header=False,
        ),
    )

    monkeypatch.setattr(
        script,
        "_embed_openai",
        lambda texts, model_name: _fake_vectors(texts, dim=3),
    )
    monkeypatch.setattr(
        script,
        "_embed_e5",
        lambda texts, model_name: _fake_vectors(texts, dim=4),
    )
    monkeypatch.setattr(
        script,
        "_embed_gemini",
        lambda texts, model_name: _fake_vectors(texts, dim=5),
    )

    out_dir = tmp_path / "cache"
    args = script.parse_args(
        [
            "--output-dir",
            str(out_dir),
            "--datasets",
            "drkg",
            "primekg",
            "fb15k237",
            "--embedding-models",
            "openai",
            "e5",
            "gemini",
            "--overwrite",
        ]
    )
    written = script.run(args)

    assert len(written) == 9
    for dataset in ("drkg", "primekg", "fb15k237"):
        for model, expected_dim in (("openai", 3), ("e5", 4), ("gemini", 5)):
            path = out_dir / f"{dataset}-node-embeddings-{model}.pt"
            assert path.is_file()
            payload = torch.load(path)
            assert payload["artifact_type"] == "node_embeddings"
            assert payload["embedding_model"] == model
            assert payload["node_text_mode"] == "human_readable_id_plus_type"
            assert payload["num_nodes"] > 0
            assert payload["embedding_dim"] == expected_dim
            assert payload["embeddings"].shape == (
                payload["num_nodes"],
                expected_dim,
            )
            assert len(payload["node_ids"]) == payload["num_nodes"]
            assert len(payload["node_types"]) == payload["num_nodes"]
            assert len(payload["node_texts"]) == payload["num_nodes"]
