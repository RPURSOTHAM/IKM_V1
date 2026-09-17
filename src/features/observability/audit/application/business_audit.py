"""Helpers for persisting business change audits into ``audit_events.changes``."""

from __future__ import annotations

import logging
from typing import Any

from src.features.observability.audit.application.audit_service import get_audit_service

logger = logging.getLogger(__name__)


def audit_business_change(
    event_type: str,
    *,
    entity_type: str,
    entity_id: str,
    old: Any,
    new: Any,
    action: str | None = None,
    category: str | None = None,
    source: str = "API",
    repository_name: str | None = None,
    document_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    deep: bool = True,
    **kwargs: Any,
) -> None:
    """Record a before/after business change. No-ops when nothing changed. Never raises."""
    try:
        get_audit_service().record_change(
            event_type,
            entity_type=entity_type,
            entity_id=str(entity_id),
            old=old,
            new=new,
            category=category,
            action=action,
            source=source,
            repository_name=repository_name,
            document_name=document_name,
            metadata=metadata,
            deep=deep,
            **kwargs,
        )
    except Exception:
        logger.debug("Business change audit failed for %s", event_type, exc_info=True)
