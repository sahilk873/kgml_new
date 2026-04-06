"""Central logging setup for CLI runs (file + stderr, structured phases)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

KGML_LOGGER_NAME = "kgml_new"


def setup_kgml_logging(
    *,
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the ``kgml_new`` logger with stderr + optional file handlers.
    Clears prior handlers on that logger to avoid duplicate lines on re-entry.
    """
    log = logging.getLogger(KGML_LOGGER_NAME)
    log.handlers.clear()
    log.setLevel(level)
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(fmt)
    log.addHandler(stderr)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def get_train_logger() -> logging.Logger:
    """Training loop messages (neighbor sampling, epochs)."""
    return logging.getLogger(f"{KGML_LOGGER_NAME}.training")


def parse_log_level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)
