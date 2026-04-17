from __future__ import annotations


def require_optuna():
    try:
        import optuna  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Optuna is required for hyperparameter tuning. Install with: "
            "pip install 'kgml-new[tuning]' or pip install optuna>=3.0"
        ) from e
    return optuna
