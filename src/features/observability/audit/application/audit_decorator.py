"""Audit event decorator."""

from __future__ import annotations

import functools
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from src.features.observability.audit.application.audit_service import get_audit_service
from src.features.observability.middleware.request_context import (
    current_context,
    extract_call_context,
    timing_metadata,
)

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _audit_metadata(
    func: Callable[..., Any],
    start: float,
    end: float,
    ctx: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    status: str,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    ids = extract_call_context(args, kwargs)
    meta = timing_metadata(start, end, func_name=func.__qualname__)
    meta.update(
        {
            "status": status,
            "request_id": ctx.request_id,
            "correlation_id": ctx.correlation_id,
            "repository_id": ids["repository_id"] or ctx.repository_id,
            "document_id": ids["document_id"] or ctx.document_id,
        }
    )
    if exc is not None:
        meta["exception"] = str(exc)
    return meta


def audit_event(
    event_type: str,
    *,
    entity_type: str | None = None,
    entity_id_param: str | None = None,
    category: str | None = None,
    action: str | None = None,
    source: str | None = None,
    on_failure: bool = True,
) -> Callable[[F], F]:
    """Record audit events around a function call."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            audit = get_audit_service()
            ctx = current_context()
            ids = extract_call_context(args, kwargs)
            entity_id = kwargs.get(entity_id_param) if entity_id_param else None
            if entity_id is None:
                entity_id = ids["document_id"] or ids["repository_id"]
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end = time.perf_counter()
                duration_ms = (end - start) * 1000.0
                try:
                    audit.record(
                        event_type,
                        category=category,
                        action=action,
                        source=source,
                        entity_type=entity_type,
                        entity_id=str(entity_id) if entity_id else None,
                        status="success",
                        duration_ms=duration_ms,
                        metadata=_audit_metadata(func, start, end, ctx, args, kwargs, status="success"),
                    )
                except Exception:
                    logger.debug("Audit decorator record failed for %s", event_type, exc_info=True)
                return result
            except Exception as exc:
                end = time.perf_counter()
                duration_ms = (end - start) * 1000.0
                if on_failure:
                    try:
                        audit.record_exception(
                            event_type,
                            exc,
                            category=category,
                            action=action,
                            source=source,
                            entity_type=entity_type,
                            entity_id=str(entity_id) if entity_id else None,
                            duration_ms=duration_ms,
                            metadata=_audit_metadata(func, start, end, ctx, args, kwargs, status="failure", exc=exc),
                        )
                    except Exception:
                        logger.debug("Audit decorator record_exception failed for %s", event_type, exc_info=True)
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            audit = get_audit_service()
            ctx = current_context()
            ids = extract_call_context(args, kwargs)
            entity_id = kwargs.get(entity_id_param) if entity_id_param else None
            if entity_id is None:
                entity_id = ids["document_id"] or ids["repository_id"]
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                end = time.perf_counter()
                duration_ms = (end - start) * 1000.0
                try:
                    audit.record(
                        event_type,
                        category=category,
                        action=action,
                        source=source,
                        entity_type=entity_type,
                        entity_id=str(entity_id) if entity_id else None,
                        status="success",
                        duration_ms=duration_ms,
                        metadata=_audit_metadata(func, start, end, ctx, args, kwargs, status="success"),
                    )
                except Exception:
                    logger.debug("Audit decorator record failed for %s", event_type, exc_info=True)
                return result
            except Exception as exc:
                end = time.perf_counter()
                duration_ms = (end - start) * 1000.0
                if on_failure:
                    try:
                        audit.record_exception(
                            event_type,
                            exc,
                            category=category,
                            action=action,
                            source=source,
                            entity_type=entity_type,
                            entity_id=str(entity_id) if entity_id else None,
                            duration_ms=duration_ms,
                            metadata=_audit_metadata(func, start, end, ctx, args, kwargs, status="failure", exc=exc),
                        )
                    except Exception:
                        logger.debug("Audit decorator record_exception failed for %s", event_type, exc_info=True)
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator
