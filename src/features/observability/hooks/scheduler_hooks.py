"""Scheduler observability hooks with synthetic request context."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from src.features.observability.audit.application.audit_service import get_audit_service
from src.features.observability.metrics.application.metrics_service import get_metrics_service
from src.features.observability.middleware.request_context import apply_synthetic_context, reset_context

logger = logging.getLogger(__name__)


def record_document_job_dispatched(
    document_id: str,
    *,
    repository_id: str | None = None,
    document_name: str | None = None,
    repository_name: str | None = None,
    processor_type: str | None = None,
) -> None:
    """Record that a document job was successfully assigned to a processor (business event)."""
    rid = f"sched-{uuid.uuid4()}"
    tokens = apply_synthetic_context(
        request_id=rid,
        correlation_id=rid,
        repository_id=repository_id,
        document_id=document_id,
        metadata={
            "component": "scheduler",
            "operation": "dispatch",
            "username": "scheduler",
            "display_name": "System Scheduler",
            "role": "System",
            "authentication_provider": "Internal",
            "source": "Scheduler",
        },
    )
    try:
        get_audit_service().record(
            "DOCUMENT_JOB_DISPATCHED",
            category="processing",
            action="dispatch",
            source="Scheduler",
            entity_type="document",
            entity_id=document_id,
            document_name=document_name,
            repository_name=repository_name,
            user_id="scheduler",
            metadata={
                "repository_id": repository_id,
                "processor_type": processor_type,
            },
        )
    except Exception:
        pass
    finally:
        reset_context(tokens)


def record_document_job_failed(
    document_id: str,
    error: str,
    *,
    repository_id: str | None = None,
    document_name: str | None = None,
    repository_name: str | None = None,
    processor_type: str | None = None,
) -> None:
    """Record that a document job dispatch failed (business event)."""
    rid = f"sched-{uuid.uuid4()}"
    tokens = apply_synthetic_context(
        request_id=rid,
        correlation_id=rid,
        repository_id=repository_id,
        document_id=document_id,
        metadata={
            "component": "scheduler",
            "operation": "dispatch",
            "username": "scheduler",
            "display_name": "System Scheduler",
            "role": "System",
            "authentication_provider": "Internal",
            "source": "Scheduler",
        },
    )
    try:
        get_audit_service().record_failure(
            "DOCUMENT_JOB_FAILED",
            category="processing",
            action="dispatch",
            source="Scheduler",
            entity_type="document",
            entity_id=document_id,
            document_name=document_name,
            repository_name=repository_name,
            user_id="scheduler",
            error=error,
            metadata={
                "repository_id": repository_id,
                "processor_type": processor_type,
            },
        )
        get_metrics_service().record(
            "scheduler_job",
            status="failure",
            repository_id=repository_id,
            document_id=document_id,
            metadata={
                "category": "scheduler",
                "operation": "dispatch_failed",
                "processor_type": processor_type,
                "repository_name": repository_name,
                "document_name": document_name,
                "error": error,
                "exception_type": "DispatchError",
            },
        )
        get_metrics_service().record_error(
            component="scheduler",
            operation="dispatch",
            error=error,
            exception_type="DispatchError",
            repository_id=repository_id,
            document_id=document_id,
            metadata={
                "processor_type": processor_type,
                "repository_name": repository_name,
                "document_name": document_name,
            },
        )
    except Exception:
        pass
    finally:
        reset_context(tokens)


def record_queue_depth(depth: int) -> None:
    try:
        get_metrics_service().record_gauge("queue_depth", depth)
    except Exception:
        pass


def record_active_jobs(count: int) -> None:
    try:
        get_metrics_service().record_gauge("active_jobs", count)
    except Exception:
        pass


def record_active_processors(count: int) -> None:
    try:
        get_metrics_service().record_gauge("active_processors", count)
    except Exception:
        pass


_last_aggregation_at: float = 0.0


def maybe_run_daily_metrics_aggregation(*, interval_sec: float = 3600.0) -> None:
    """Run daily metrics aggregation at most once per interval (scheduler poll hook)."""
    global _last_aggregation_at
    now = time.time()
    if now - _last_aggregation_at < interval_sec:
        return
    try:
        from src.features.observability.metrics.application.aggregation_service import get_aggregation_service

        get_aggregation_service().run_daily_aggregation(days_back=7)
        _last_aggregation_at = now
    except Exception:
        logger.warning("Scheduled daily metrics aggregation failed", exc_info=True)
