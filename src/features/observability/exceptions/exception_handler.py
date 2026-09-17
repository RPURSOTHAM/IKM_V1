

from __future__ import annotations

import logging
from typing import Any

from src.features.observability.audit.application.audit_service import get_audit_service
from src.features.observability.metrics.application.metrics_service import get_metrics_service

logger = logging.getLogger(__name__)


def record_http_exception(
    exc: Exception,
    *,
    endpoint: str | None = None,
    status_code: int | None = None,
    component: str = "api",
    operation: str = "request",
    duration_ms: float | None = None,
) -> None:
    try:
        get_audit_service().record_exception(
            "HTTP_EXCEPTION",
            exc,
            entity_type="http_request",
            entity_id=endpoint,
            metadata={"status_code": status_code},
        )
        get_metrics_service().record(
            "failure_count",
            duration_ms=duration_ms,
            status="failure",
            metadata={"endpoint": endpoint, "status_code": status_code},
        )
        get_metrics_service().record_error(
            component=component,
            operation=operation,
            error=str(exc),
            exception_type=type(exc).__name__,
            duration_ms=duration_ms,
            metadata={"endpoint": endpoint, "status_code": status_code},
        )
    except Exception:
        logger.debug("Failed to record HTTP exception observability", exc_info=True)
