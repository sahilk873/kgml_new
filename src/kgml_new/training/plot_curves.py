"""Plot train/val loss curves from ``TrainingHistory`` dicts or ``run_gpu_method`` JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install 'kgml-new[plots]' "
            "or 'kgml-new[viz]'."
        ) from e
    return plt


def plot_history_curves(
    history: dict[str, Any],
    out_path: Path,
    *,
    title: str | None = None,
) -> None:
    """
    Plot ``train_loss`` and ``val_loss`` (if present and aligned) vs ``epoch``.

    ``history`` matches ``TrainingHistory.to_dict()`` / ``run_gpu_method`` ``history`` JSON.
    """
    plt = _require_pyplot()
    train = history.get("train_loss") or []
    val_loss = history.get("val_loss") or []
    epoch = history.get("epoch")
    if not train:
        raise ValueError("history has no train_loss entries")
    if epoch is None or len(epoch) != len(train):
        epoch = list(range(len(train)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(epoch, train, label="train_loss", color="C0")
    if val_loss:
        n = min(len(val_loss), len(train))
        if n > 0:
            ax.plot(epoch[:n], val_loss[:n], label="val_loss", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_run_json(
    path: Path,
    out_path: Path,
    *,
    include_encoder: bool = True,
    include_decoder: bool = True,
) -> None:
    """
    Load a JSON run artifact (e.g. from ``run_gpu_method``) and save loss curve figure(s).

    Writes a single figure with one subplot per history block:
    ``base_embedding_history`` (encoder), main ``history`` (often MLP or primary),
    ``decoder_history`` (optional MLP decoder phase).
    """
    plt = _require_pyplot()
    payload = json.loads(Path(path).read_text())
    method = str(payload.get("method", "run"))

    blocks: list[tuple[str, dict[str, Any]]] = []
    if include_encoder and payload.get("base_embedding_history"):
        blocks.append(("encoder (base)", payload["base_embedding_history"]))
    if payload.get("history"):
        blocks.append((method or "main", payload["history"]))
    if include_decoder and payload.get("decoder_history"):
        blocks.append(("decoder (MLP)", payload["decoder_history"]))

    if not blocks:
        raise ValueError("JSON has no history, base_embedding_history, or decoder_history")

    n = len(blocks)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 3.8 * n), squeeze=False)
    for ax_row, (label, hist) in zip(axes[:, 0], blocks, strict=True):
        train = hist.get("train_loss") or []
        if not train:
            ax_row.text(0.5, 0.5, f"{label}: no train_loss", ha="center", va="center")
            ax_row.set_axis_off()
            continue
        epoch = hist.get("epoch")
        if epoch is None or len(epoch) != len(train):
            epoch = list(range(len(train)))
        val_loss = hist.get("val_loss") or []
        ax_row.plot(epoch, train, label="train_loss", color="C0")
        if val_loss:
            m = min(len(val_loss), len(train))
            if m > 0:
                ax_row.plot(epoch[:m], val_loss[:m], label="val_loss", color="C1")
        ax_row.set_title(label)
        ax_row.set_xlabel("epoch")
        ax_row.set_ylabel("loss")
        ax_row.legend()
        ax_row.grid(True, alpha=0.3)

    fig.suptitle(f"{method} training curves")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def method_series_label(payload: dict[str, Any]) -> str:
    """Human-readable curve name from a ``run_*`` JSON payload."""
    m = str(payload.get("method", "unknown"))
    if m == "edge_aware_sage":
        emb = payload.get("embedding_model") or payload.get("embedding_model_resolved") or ""
        emb = str(emb).strip()
        return f"{m} ({emb})" if emb else m
    if m == "edge_aware_sage_node_emb":
        emb = payload.get("embedding_model") or payload.get("embedding_model_resolved") or ""
        emb = str(emb).strip()
        return f"{m} ({emb})" if emb else m
    rel = payload.get("relation")
    if m == "hgt" and rel:
        return f"hgt ({rel})"
    return m


def dataset_key_from_payload(payload: dict[str, Any]) -> str | None:
    ip = str(payload.get("input_path") or "").lower()
    if "primekg" in ip or "/primekg." in ip:
        return "primekg"
    if "fb15k" in ip or "fb15k-237" in ip:
        return "fb15k237"
    return None


def primary_training_history(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Training history block used for link-pred sweeps (``run_gpu_method``, ``run_hgt``, ``run_txgnn``)."""
    h = payload.get("history")
    if isinstance(h, dict) and (h.get("train_loss") or []):
        return h
    return None


def plot_methods_comparison(
    labeled_histories: list[tuple[str, dict[str, Any]]],
    out_path: Path,
    *,
    title: str,
    suptitle: str | None = None,
) -> None:
    """
    Overlay multiple methods on train loss; optional second axis for validation loss when present.

    ``labeled_histories`` is a list of ``(legend_label, history_dict)`` where each history has
    ``train_loss`` and optionally ``val_loss`` / ``epoch`` (same schema as ``TrainingHistory.to_dict()``).
    """
    plt = _require_pyplot()
    if not labeled_histories:
        raise ValueError("no series to plot")

    any_val = any(bool(h.get("val_loss")) for _, h in labeled_histories)
    nrows = 2 if any_val else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9.5, 4.2 * nrows), squeeze=False)
    ax_train = axes[0, 0]
    ax_val = axes[1, 0] if any_val else None

    for i, (label, hist) in enumerate(labeled_histories):
        train = hist.get("train_loss") or []
        if not train:
            continue
        epoch = hist.get("epoch")
        if epoch is None or len(epoch) != len(train):
            epoch = list(range(len(train)))
        color = f"C{i % 10}"
        ax_train.plot(epoch, train, label=label, color=color, linewidth=1.6, alpha=0.9)

    ax_train.set_ylabel("train loss")
    ax_train.set_xlabel("epoch")
    ax_train.set_title(title)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc="upper right", fontsize=8, ncol=2)

    if ax_val is not None:
        for i, (label, hist) in enumerate(labeled_histories):
            val_loss = hist.get("val_loss") or []
            train = hist.get("train_loss") or []
            if not val_loss or not train:
                continue
            epoch = hist.get("epoch")
            if epoch is None or len(epoch) != len(train):
                epoch = list(range(len(train)))
            n = min(len(val_loss), len(train))
            color = f"C{i % 10}"
            ax_val.plot(
                epoch[:n],
                val_loss[:n],
                label=label,
                color=color,
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
            )
        ax_val.set_ylabel("validation loss")
        ax_val.set_xlabel("epoch")
        ax_val.grid(True, alpha=0.3)
        ax_val.legend(loc="upper right", fontsize=8, ncol=2)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
