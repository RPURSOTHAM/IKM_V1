"""Structured JSON logging for enterprise services."""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def __init__(self, *, service: str | None = None, extra_fields: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.service = service or os.getenv("SERVICE_NAME") or os.getenv("HOSTNAME") or "rag-builder"
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service,
        }
        payload.update(self.extra_fields)
        for key in (
            "request_id",
            "trace_id",
            "span_id",
            "document_id",
            "repository_id",
            "component",
            "event",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info)).strip()
        # Include any non-standard extras passed via logger.bind-style adapters
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in payload:
                continue
            if key in {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "asctime",
            }:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_json_logging(
    *,
    service: str | None = None,
    level: str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging for structured JSON output.

    Safe to call multiple times; when force=False and handlers already exist,
    only the level is updated unless INFRA_JSON_LOGGING is explicitly enabled.
    """
    enabled = os.getenv("INFRA_JSON_LOGGING", "true").strip().lower() in {"1", "true", "yes", "on"}
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(log_level)

    if not enabled:
        return logging.getLogger(service or "rag-builder")

    formatter = JsonFormatter(service=service)
    if force or not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        root.handlers.clear()
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)

    return logging.getLogger(service or "rag-builder")


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Attach request/document context fields to every log record."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.pop("extra", {}) or {})
        kwargs["extra"] = extra
        return msg, kwargs
