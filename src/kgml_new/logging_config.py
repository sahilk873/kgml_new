"""Configure loggers for CLI entry points (e.g. ``run_gpu_method``)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_KGML = "kgml_new"
_configured = False


def parse_log_level(name: str) -> int:
    """Map a level name to :mod:`logging` constants (default INFO on unknown)."""
    n = (name or "INFO").upper()
    if n in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return int(getattr(logging, n))
    return logging.INFO


def setup_kgml_logging(
    *,
    log_file: Path | str | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Attach stderr (and optionally file) handlers to the ``kgml_new`` logger tree.

    Idempotent: only the first call in a process adds handlers.
    """
    global _configured
    if _configured:
        return
    _configured = True

    pkg = logging.getLogger(_KGML)
    pkg.setLevel(level)
    pkg.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    err = logging.StreamHandler(sys.stderr)
    err.setLevel(level)
    err.setFormatter(fmt)
    pkg.addHandler(err)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        pkg.addHandler(fh)
