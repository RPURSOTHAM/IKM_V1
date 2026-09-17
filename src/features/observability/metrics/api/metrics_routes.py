'''Metrics API routes.'''

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.features.observability.audit.infrastructure.audit_repository import get_audit_repository
from src.features.observability.metrics.application.aggregation_service import get_aggregation_service
from src.features.observability.metrics.application.benchmark_metrics import benchmark_catalog, benchmark_summary
from src.features.observability.metrics.application.dashboard_service import get_dashboard_service
from src.features.observability.metrics.domain.metrics_models import MetricsQuery
from src.features.observability.metrics.infrastructure.metrics_repository import get_metrics_repository

router = APIRouter(prefix="/metrics", tags=["Observability — Metrics"])


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid datetime: {value}") from exc


@router.get("/")
def list_metrics(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    metric_name: str | None = None,
    metric_names: str | None = Query(None, description="Comma-separated list of metric names to include"),
    exclude_metrics: str | None = Query(None, description="Comma-separated list of metric names to exclude"),
    date_from: str | None = None,
    date_to: str | None = None,
    request_id: str | None = None,
    repository_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    repo = get_metrics_repository()
    if repo is None:
        return {"items": [], "total": 0, "page": page, "page_size": page_size}
    parsed_names = [n.strip() for n in metric_names.split(",") if n.strip()] if metric_names else None
    parsed_excludes = [n.strip() for n in exclude_metrics.split(",") if n.strip()] if exclude_metrics else None
    query = MetricsQuery(
        page=page,
        page_size=page_size,
        metric_name=metric_name,
        metric_names=parsed_names,
        exclude_metric_names=parsed_excludes,
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        request_id=request_id,
        repository_id=repository_id,
        status=status,
    )
    items, total = repo.list_events(query)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/summary")
def metrics_summary(
    date_from: str | None = None,
    date_to: str | None = None,
    repository_id: str | None = None,
) -> dict[str, Any]:
    agg = get_aggregation_service()
    try:
        summary = agg.summary(
            date_from=_parse_dt(date_from),
            date_to=_parse_dt(date_to),
            repository_id=repository_id,
        )
    except Exception:
        summary = {"total": 0, "success": 0, "failure": 0, "success_rate": 0.0, "failure_rate": 0.0}
    repo = get_metrics_repository()
    daily = []
    if repo:
        try:
            daily = repo.list_daily()
        except Exception:
            daily = []
    return {"summary": summary, "daily": daily}


@router.get("/applications")
def application_usage_metrics(
    date_from: str | None = None,
    date_to: str | None = None,
    application_name: str | None = None,
) -> dict[str, Any]:
    """Usage counters grouped by integrated application."""
    repo = get_audit_repository()
    if repo is None:
        return {"applications": [], "totals": {}}
    applications = repo.application_usage(
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        application_name=application_name,
    )
    counter_fields = (
        "total_requests",
        "documents_uploaded",
        "search_requests",
        "ai_queries",
        "documents_deleted",
        "failed_requests",
    )
    totals = {
        field: sum(int(row.get(field) or 0) for row in applications)
        for field in counter_fields
    }
    totals["active_users"] = sum(int(row.get("active_users") or 0) for row in applications)
    totals["application_count"] = len(applications)
    return {"applications": applications, "totals": totals}


@router.get("/latency")
def metrics_latency(
    metric_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    agg = get_aggregation_service()
    repo = get_metrics_repository()
    latency = agg.latency_summary(
        metric_name=metric_name,
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
    )
    top_slow = repo.top_slow_operations(date_from=_parse_dt(date_from), date_to=_parse_dt(date_to)) if repo else []
    return {"latency": latency, "top_slow_operations": top_slow}


@router.get("/benchmark")
def metrics_benchmark() -> dict[str, Any]:
    """Official RAG benchmark metric catalog (RAG_Benchmark_Metrics.xlsx)."""
    return {"summary": benchmark_summary(), "metrics": benchmark_catalog()}


@router.get("/dashboard")
def metrics_dashboard(
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Aggregated observability dashboard data sourced entirely from metrics_events."""
    return get_dashboard_service().build_dashboard(
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
    )


@router.get("/failures")
def metrics_failures(
    date_from: str | None = None,
    date_to: str | None = None,
    repository_id: str | None = None,
) -> dict[str, Any]:
    agg = get_aggregation_service()
    return agg.failure_summary(
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        repository_id=repository_id,
    )
