from __future__ import annotations

import argparse
from typing import Any


def apply_param_dict_to_namespace(ns: argparse.Namespace, params: dict[str, Any]) -> None:
    """Set attributes from an Optuna ``trial.params`` or ``study.best_params`` mapping."""
    for k, v in params.items():
        setattr(ns, k, v)


def suggest_run_gpu_method_params(trial: Any, args: argparse.Namespace) -> None:
    """Sample tunable hyperparameters for ``run_gpu_method`` (mutates ``args``)."""
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    args.train_batch_size = trial.suggest_categorical("train_batch_size", [128, 256, 512, 1024])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    args.num_layers = trial.suggest_int("num_layers", 1, 3)
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    args.edge_dim = trial.suggest_categorical("edge_dim", [16, 32, 64])
    args.num_neighbors_spec = trial.suggest_categorical(
        "num_neighbors_spec", ["5,5", "10,10", "15,10", "20,10"]
    )

    args.link_mlp_learning_rate = trial.suggest_float(
        "link_mlp_learning_rate", 1e-5, 1e-2, log=True
    )
    args.link_mlp_dropout = trial.suggest_float("link_mlp_dropout", 0.0, 0.5)
    args.link_mlp_batch_size = trial.suggest_categorical(
        "link_mlp_batch_size", [256, 512, 1024]
    )
    hidden_choice = trial.suggest_categorical(
        "link_mlp_hidden_dims",
        ["128,64", "256,128", "256,128,64"],
    )
    args.link_mlp_hidden_dims = hidden_choice

    # Suggest every trial (constant search space for Optuna); ignored unless method == node2vec.
    args.n2v_lr = trial.suggest_float("n2v_lr", 1e-4, 0.1, log=True)
    args.n2v_batch_size = trial.suggest_categorical("n2v_batch_size", [256, 512, 1024])
    args.n2v_walk_length = trial.suggest_int("n2v_walk_length", 5, 20)
    args.n2v_context_size = trial.suggest_int("n2v_context_size", 5, 15)
    args.n2v_embedding_dim = trial.suggest_categorical("n2v_embedding_dim", [32, 64, 128])


def suggest_hetero_link_params(trial: Any, args: argparse.Namespace) -> None:
    """Hyperparameters for ``run_hgt`` / ``run_txgnn`` training loops."""
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    args.neg_samples = trial.suggest_int("neg_samples", 1, 5)
    args.epochs = trial.suggest_int("epochs", 10, 60)


def suggest_hgt_arch_params(trial: Any, args: argparse.Namespace) -> None:
    args.dropout = trial.suggest_float("dropout", 0.0, 0.4)
    args.num_layers = trial.suggest_int("num_layers", 1, 4)
    args.num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])


def suggest_txgnn_arch_params(trial: Any, args: argparse.Namespace) -> None:
    args.dropout = trial.suggest_float("dropout", 0.0, 0.4)
    args.num_layers = trial.suggest_int("num_layers", 1, 4)
    args.prototype_k = trial.suggest_int("prototype_k", 3, 10)
    args.prototype_alpha = trial.suggest_float("prototype_alpha", 0.1, 0.9)


def suggest_node_classification_params(trial: Any, args: argparse.Namespace) -> None:
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    args.classifier_learning_rate = trial.suggest_float(
        "classifier_learning_rate", 1e-5, 1e-2, log=True
    )
    args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    args.weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3)
    args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    args.embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    args.num_layers = trial.suggest_int("num_layers", 1, 4)
    args.num_neighbors_spec = trial.suggest_categorical(
        "num_neighbors_spec", ["5,5", "10,10", "15,10"]
    )


def suggest_link_prediction_legacy_params(trial: Any, args: argparse.Namespace) -> None:
    """Tuning for ``run_link_prediction`` (pickle graph runner)."""
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    args.train_batch_size = trial.suggest_categorical("train_batch_size", [128, 256, 512])
    args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    args.num_layers = trial.suggest_int("num_layers", 1, 3)
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    args.edge_dim = trial.suggest_categorical("edge_dim", [16, 32, 64])
    args.num_neighbors_spec = trial.suggest_categorical(
        "num_neighbors_spec", ["5,5", "10,10", "15,10"]
    )
