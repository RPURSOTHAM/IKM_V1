"""Append pipeline failures to ``data/logs/failures.log`` and keep going.

Help
----
Extract, filter, group, analysis, rewrite, and draft generation catch
recoverable errors, record them here, and continue with the rest of
the work. The log is a dedicated file so operators can review skips
without scraping console output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .errors import RECOVERABLE

ROOT = Path(__file__).resolve().parents[1]
FAILURE_LOG_PATH = ROOT / "data" / "logs" / "failures.log"

_LOGGER_NAME = "pdf_reader.failures"
_failure_logger = logging.getLogger(_LOGGER_NAME)

T = TypeVar("T")


def log_failure(stage: str, message: str, exc: BaseException | None = None) -> None:
    """Write one failure line (and traceback) to ``FAILURE_LOG_PATH``."""

    _ensure_handler()
    if exc is None:
        _failure_logger.error("%s | %s", stage, message)
        return
    _failure_logger.error(
        "%s | %s | %s: %s",
        stage,
        message,
        type(exc).__name__,
        exc,
        exc_info=exc,
    )


def run_continuing(
    stage: str,
    action: Callable[[], T],
    fallback: T,
    message: str | None = None,
) -> T:
    """Run ``action``; on a recoverable error log it and return ``fallback``."""

    try:
        return action()
    except RECOVERABLE as exc:
        log_failure(stage, message or stage, exc)
        return fallback


def _ensure_handler() -> None:
    """Attach a UTF-8 file handler once so every process shares one log."""

    if _failure_logger.handlers:
        return
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(FAILURE_LOG_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    _failure_logger.addHandler(handler)
    _failure_logger.setLevel(logging.ERROR)
    _failure_logger.propagate = False
