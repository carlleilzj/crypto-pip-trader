"""Logging setup for crypto-pip-trader.

Provides a single ``get_logger(name)`` that returns a logger pre-configured
with a console handler and (optionally) a rotating file handler under
``reports/logs/``.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """Return a logger with console + file handlers.

    Safe to call multiple times — only adds handlers once per name.
    File handler writes to ``log_dir / {name}.log``, rotating at midnight.
    Directory resolution: explicit arg > PIP_LOG_DIR env (tests set this so
    logs stay out of reports/) > reports/logs.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (DEBUG+) — optional
    env_dir = os.environ.get("PIP_LOG_DIR")
    log_dir = Path(log_dir) if log_dir else (Path(env_dir) if env_dir else REPORTS / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / f"{name}.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    return logger
