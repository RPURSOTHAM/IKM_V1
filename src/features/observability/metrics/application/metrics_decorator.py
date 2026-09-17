"""Metrics capture decorator."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

from src.features.observability.metrics.application.metrics_service import enrich_metric_metadata, get_metrics_service
from src.features.observability.middleware.request_context import (
    current_context,
    extract_call_context,
    timing_metadata,
)

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _metric_record(
    metric_name: str,
    func: Callable[..., Any],
    start: float,
    end: float,
    ctx: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    status: str,
) -> None:
    ids = extract_call_context(args, kwargs)
    meta = timing_metadata(start, end, func_name=func.__qualname__)
    meta.update(
        {
            "status": status,
            "request_id": ctx.request_id,
            "correlation_id": ctx.correlation_id,
        }
    )
    meta = enrich_metric_metadata(
        metric_name,
        meta,
        status=status,
        duration_ms=meta.get("duration_ms"),
        ctx_metadata=ctx.metadata,
    )
    get_metrics_service().record(
        metric_name,
        duration_ms=meta["duration_ms"],
        status=status,
        repository_id=ids["repository_id"] or ctx.repository_id,
        document_id=ids["document_id"] or ctx.document_id,
        metadata=meta,
    )


def _safe_metric_record(*args: Any, **kwargs: Any) -> None:
    try:
        _metric_record(*args, **kwargs)
    except Exception:
        metric = args[0] if args else "?"
        logger.debug("Metrics decorator record failed for %s", metric, exc_info=True)


def capture_metric(metric_name: str) -> Callable[[F], F]:
    """Record duration and status for a function invocation."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = current_context()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                _safe_metric_record(metric_name, func, start, time.perf_counter(), ctx, args, kwargs, status="success")
                return result
            except Exception:
                _safe_metric_record(metric_name, func, start, time.perf_counter(), ctx, args, kwargs, status="failure")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = current_context()
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                _safe_metric_record(metric_name, func, start, time.perf_counter(), ctx, args, kwargs, status="success")
                return result
            except Exception:
                _safe_metric_record(metric_name, func, start, time.perf_counter(), ctx, args, kwargs, status="failure")
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator
