"""Watch configs/processors.yaml and reload deployment processors on change."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_WATCHER_LOCK = threading.RLock()
_WATCHER: "ProcessorConfigWatcher | None" = None


def _watch_enabled() -> bool:
    raw = (os.getenv("PROCESSORS_CONFIG_WATCH") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _poll_interval_seconds() -> float:
    raw = (os.getenv("PROCESSORS_CONFIG_WATCH_INTERVAL") or "").strip()
    if not raw:
        return 1.0
    try:
        return max(0.25, float(raw))
    except ValueError:
        return 1.0


class ProcessorConfigWatcher:
    """Poll ``processors.yaml`` mtime and invoke a reload callback on change.

    Polling avoids an extra watchdog dependency and works with Docker bind mounts.
    """

    def __init__(
        self,
        config_path: Path,
        on_change: Callable[[], None],
        *,
        interval_seconds: float = 1.0,
    ) -> None:
        self._config_path = config_path
        self._on_change = on_change
        self._interval = max(0.25, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_mtime: float | None = self._read_mtime()
        self._last_signature: str | None = self._read_signature()

    def _read_mtime(self) -> float | None:
        try:
            if not self._config_path.is_file():
                return None
            return self._config_path.stat().st_mtime
        except OSError:
            return None

    def _read_signature(self) -> str | None:
        try:
            if not self._config_path.is_file():
                return None
            # Content hash-ish via size+mtime is enough; also read text for editors
            # that rewrite with identical mtime on some filesystems.
            stat = self._config_path.stat()
            text = self._config_path.read_text(encoding="utf-8")
            return f"{stat.st_size}:{stat.st_mtime_ns}:{hash(text)}"
        except OSError:
            return None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="processor-config-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Watching processor config for changes: %s (interval=%.2fs)",
            self._config_path,
            self._interval,
        )

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                signature = self._read_signature()
                mtime = self._read_mtime()
            except Exception:
                logger.exception("Processor config watcher failed reading %s", self._config_path)
                continue

            if signature == self._last_signature and mtime == self._last_mtime:
                continue

            # File may be mid-write; brief settle delay then re-read.
            time.sleep(0.15)
            try:
                signature = self._read_signature()
                mtime = self._read_mtime()
            except Exception:
                continue

            if signature == self._last_signature and mtime == self._last_mtime:
                continue

            self._last_signature = signature
            self._last_mtime = mtime
            logger.info("Detected change in processor config: %s", self._config_path)
            try:
                self._on_change()
            except Exception:
                logger.exception("Processor config reload callback failed.")


def start_processor_config_watcher(
    *,
    initialize: bool = True,
    log: logging.Logger | None = None,
) -> ProcessorConfigWatcher | None:
    """Start the singleton file watcher when ``PROCESSORS_CONFIG_WATCH`` is enabled."""
    if not _watch_enabled():
        (log or logger).info("Processor config file watcher disabled (PROCESSORS_CONFIG_WATCH=false).")
        return None

    from src.features.document_processing.shared_processor.deployment.config_provider import ProcessorConfigurationProvider
    from src.features.document_processing.shared_processor.deployment.reload import reload_deployment_processors

    provider = ProcessorConfigurationProvider.instance()
    path = provider.config_path

    def _on_change() -> None:
        reload_deployment_processors(
            initialize=initialize,
            keep_previous_on_error=True,
            log=log or logger,
            reason="file_watch",
        )

    with _WATCHER_LOCK:
        global _WATCHER
        if _WATCHER is not None:
            _WATCHER.stop()
        _WATCHER = ProcessorConfigWatcher(
            path,
            _on_change,
            interval_seconds=_poll_interval_seconds(),
        )
        _WATCHER.start()
        return _WATCHER


def stop_processor_config_watcher() -> None:
    with _WATCHER_LOCK:
        global _WATCHER
        if _WATCHER is not None:
            _WATCHER.stop()
            _WATCHER = None
