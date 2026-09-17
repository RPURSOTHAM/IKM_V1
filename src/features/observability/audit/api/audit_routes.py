'''Audit query HTTP routes.'''

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.features.observability.audit.domain.audit_models import AuditQuery
from src.features.observability.audit.infrastructure.audit_repository import get_audit_repository

router = APIRouter(prefix="/audit", tags=["Observability — Audit"])


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid datetime: {value}") from exc


@router.get("/events")
def list_audit_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    request_id: str | None = None,
    correlation_id: str | None = None,
    application_name: str | None = None,
    repository_id: str | None = None,
    status: str | None = None,
    event_type: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    category: str | None = None,
    action: str | None = None,
    source: str | None = None,
    business_only: bool = False,
) -> dict[str, Any]:
    repo = get_audit_repository()
    if repo is None:
        return {"items": [], "total": 0, "page": page, "page_size": page_size}
    query = AuditQuery(
        page=page,
        page_size=page_size,
        search=search,
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        request_id=request_id,
        correlation_id=correlation_id,
        application_name=application_name,
        repository_id=repository_id,
        status=status,
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        category=category,
        action=action,
        source=source,
        business_only=business_only,
    )
    items, total = repo.list_events(query)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/changes")
def list_audit_changes(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    request_id: str | None = None,
    repository_id: str | None = None,
    status: str | None = None,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Return business change audit rows (events that include a non-empty changes payload)."""
    repo = get_audit_repository()
    if repo is None:
        raise HTTPException(status_code=503, detail="Observability store unavailable")
    query = AuditQuery(
        page=page,
        page_size=page_size,
        search=search,
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        request_id=request_id,
        repository_id=repository_id,
        status=status,
        event_type=event_type,
        has_changes=True,
    )
    items, total = repo.list_changes(query)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/{event_id}")
def get_audit_event(event_id: int) -> dict[str, Any]:
    repo = get_audit_repository()
    if repo is None:
        raise HTTPException(status_code=503, detail="Observability store unavailable")
    row = repo.get_by_id(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Audit event not found")
    return row
