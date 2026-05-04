from __future__ import annotations

from types import SimpleNamespace

from kgml_new.config import (
    rotate_config_from_run_gpu_method_args,
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


def test_rotate_config_from_run_gpu_method_args_defaults():
    ns = SimpleNamespace(
        in_dim=128,
        epochs=50,
        seed=7,
        learning_rate=None,
        train_batch_size=None,
        rotate_embedding_dim=None,
        rotate_batch_size=None,
        rotate_gamma=10.0,
        rotate_weight_decay=1e-5,
        rotate_neg_samples=3,
        rotate_eval_batch_size=2048,
        rotate_grad_clip=0.5,
        rotate_early_stop_patience=15,
    )
    cfg = rotate_config_from_run_gpu_method_args(ns)
    assert cfg.embedding_dim == 128
    assert cfg.epochs == 50
    assert cfg.batch_size == 1024
    assert cfg.gamma == 10.0
    assert cfg.weight_decay == 1e-5
    assert cfg.neg_samples == 3
    assert cfg.eval_batch_size == 2048
    assert cfg.grad_clip_norm == 0.5
    assert cfg.early_stop_patience == 15


def test_rotate_config_from_run_gpu_method_args_overrides():
    ns = SimpleNamespace(
        in_dim=64,
        epochs=10,
        seed=1,
        learning_rate=0.001,
        train_batch_size=512,
        rotate_embedding_dim=32,
        rotate_batch_size=256,
        rotate_gamma=12.0,
        rotate_weight_decay=0.0,
        rotate_neg_samples=5,
        rotate_eval_batch_size=4096,
        rotate_grad_clip=1.0,
        rotate_early_stop_patience=20,
    )
    cfg = rotate_config_from_run_gpu_method_args(ns)
    assert cfg.embedding_dim == 32
    assert cfg.batch_size == 256
    assert cfg.learning_rate == 0.001


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
