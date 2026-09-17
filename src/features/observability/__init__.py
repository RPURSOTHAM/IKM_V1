"""Observability feature — metrics, audit trail, middleware, hooks."""

from __future__ import annotations

from src.features.observability.audit.application.audit_service import get_audit_service
from src.features.observability.infrastructure.db import get_observability_db
from src.features.observability.metrics.application.aggregation_service import get_aggregation_service
from src.features.observability.metrics.application.metrics_service import get_metrics_service

__all__ = [
    "get_audit_service",
    "get_metrics_service",
    "get_aggregation_service",
    "get_observability_db",
    "initialize_observability",
]


def initialize_observability() -> bool:
    """Ensure PostgreSQL schema exists. Returns True when store is available."""
    db = get_observability_db()
    return db is not None and db.ping()
