from __future__ import annotations

from types import SimpleNamespace

from kgml_new.config import (
    train_config_from_link_prediction_args,
    train_config_from_run_gpu_method_args,
)


def _gpu_ns(**kw):
    base = dict(
        in_dim=32,
        epochs=5,
        seed=3,
        neighbor_aggr="mean",
        edge_relation_mode="concat",
        num_relation_bases=4,
        concat=True,
        learning_rate=None,
        train_batch_size=None,
        dropout=None,
        num_layers=None,
        hidden_dim=None,
        edge_dim=None,
        num_neighbors_spec=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_train_config_from_run_gpu_method_args_defaults():
    cfg = train_config_from_run_gpu_method_args(_gpu_ns())
    assert cfg.in_dim == 32
    assert cfg.epochs == 5
    assert cfg.learning_rate == 1e-3


def test_train_config_from_run_gpu_method_args_overrides():
    cfg = train_config_from_run_gpu_method_args(
        _gpu_ns(learning_rate=0.05, dropout=0.2, num_neighbors_spec="8,8")
    )
    assert cfg.learning_rate == 0.05
    assert cfg.dropout == 0.2
    assert cfg.num_neighbors == [8, 8]


def test_train_config_from_link_prediction_args():
    ns = SimpleNamespace(
        epochs=4,
        seed=9,
        neighbor_aggr="max",
        edge_relation_mode="concat",
        num_relation_bases=4,
        learning_rate=0.02,
    )
    cfg = train_config_from_link_prediction_args(ns)
    assert cfg.epochs == 4
    assert cfg.neighbor_aggr == "max"
    assert cfg.learning_rate == 0.02
