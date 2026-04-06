from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from kgml_new.data.datasets import prepare_link_prediction_dataset
from kgml_new.data.loaders import GraphCSVSpec, load_graph_csv
from kgml_new.data.prepared_dataset_cache import (
    build_cache_meta,
    input_fingerprint,
    load_prepared_link_prediction_dataset,
    save_prepared_link_prediction_dataset,
)


def test_prepared_link_prediction_cache_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "mini.csv"
    pd.DataFrame(
        [
            {"src": "a", "dst": "b", "rel": "r1", "st": "t", "dt": "t"},
            {"src": "b", "dst": "c", "rel": "r2", "st": "t", "dt": "t"},
            {"src": "c", "dst": "d", "rel": "r1", "st": "t", "dt": "t"},
            {"src": "d", "dst": "e", "rel": "r2", "st": "t", "dt": "t"},
            {"src": "e", "dst": "a", "rel": "r1", "st": "t", "dt": "t"},
        ]
    ).to_csv(csv_path, index=False)

    spec = GraphCSVSpec(
        source_col="src",
        target_col="dst",
        relation_col="rel",
        source_type_col="st",
        target_type_col="dt",
    )
    graph = load_graph_csv(csv_path, spec=spec)
    dataset = prepare_link_prediction_dataset(
        graph,
        in_dim=8,
        seed=0,
        val_ratio=0.1,
        test_ratio=0.1,
        split_protocol="node",
        negative_sampling_mode="type_matched",
        negatives_per_pos=2,
        decoder="dot",
        shuffle_relations=False,
    )
    meta = build_cache_meta(
        input_path=csv_path.resolve(),
        resolved_input_format="csv",
        max_edges=None,
        csv_spec=spec,
        dataset=dataset,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=0,
        in_dim=8,
        add_self_loops=False,
        negative_sampling_mode_cli="type_matched",
    )
    out = tmp_path / "prep.pkl"
    save_prepared_link_prediction_dataset(out, dataset, meta=meta)

    loaded, meta2 = load_prepared_link_prediction_dataset(
        out,
        expected_meta={
            "split_protocol": "node",
            "seed": 0,
            "in_dim": 8,
            "negatives_per_pos": 2,
        },
    )
    assert meta2["format_version"] == meta["format_version"]
    assert loaded.train_data.num_nodes == dataset.train_data.num_nodes
    assert loaded.split.train_pos_edge_index.shape == dataset.split.train_pos_edge_index.shape


def test_prepared_cache_load_ignores_input_mtime_when_size_unchanged(tmp_path: Path):
    csv_path = tmp_path / "mini.csv"
    pd.DataFrame(
        [
            {"src": "a", "dst": "b", "rel": "r1", "st": "t", "dt": "t"},
            {"src": "b", "dst": "c", "rel": "r2", "st": "t", "dt": "t"},
        ]
    ).to_csv(csv_path, index=False)

    spec = GraphCSVSpec(
        source_col="src",
        target_col="dst",
        relation_col="rel",
        source_type_col="st",
        target_type_col="dt",
    )
    graph = load_graph_csv(csv_path, spec=spec)
    dataset = prepare_link_prediction_dataset(
        graph,
        in_dim=8,
        seed=0,
        val_ratio=0.1,
        test_ratio=0.1,
        split_protocol="node",
        negative_sampling_mode="type_matched",
        negatives_per_pos=2,
        decoder="dot",
        shuffle_relations=False,
    )
    meta = build_cache_meta(
        input_path=csv_path.resolve(),
        resolved_input_format="csv",
        max_edges=None,
        csv_spec=spec,
        dataset=dataset,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=0,
        in_dim=8,
        add_self_loops=False,
        negative_sampling_mode_cli="type_matched",
    )
    out = tmp_path / "prep.pkl"
    save_prepared_link_prediction_dataset(out, dataset, meta=meta)

    now = time.time()
    os.utime(csv_path, (now, now))
    fp_after_touch = input_fingerprint(csv_path.resolve())
    assert fp_after_touch is not None
    assert fp_after_touch["st_mtime_ns"] != meta["input"]["st_mtime_ns"]

    loaded, _ = load_prepared_link_prediction_dataset(
        out,
        expected_meta={"input": fp_after_touch},
    )
    assert loaded.train_data.num_nodes == dataset.train_data.num_nodes
