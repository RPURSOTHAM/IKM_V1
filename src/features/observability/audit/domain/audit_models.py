"""Audit event data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Event types shown in the default business audit view (Streamlit + business_only queries).
BUSINESS_EVENT_TYPE_PREFIXES: tuple[str, ...] = (
    "REPOSITORY_",
    "DOCUMENT_",
    "SEARCH_",
    "CITATION_",
    "PERMISSION_",
    "USER_LOGIN",
    "USER_PASSWORD_",
)


@dataclass
class AuditEventRecord:
    id: int | None = None
    timestamp: datetime | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    application_name: str | None = None
    user_id: str | None = None
    event_type: str = ""
    # Business classification fields
    category: str | None = None          # repository / document / retrieval / processing / scheduler
    action: str | None = None            # create / update / delete / upload / search / submit / process
    source: str | None = None            # API / Scheduler / Processor / UI
    # Entity fields
    entity_type: str | None = None
    entity_id: str | None = None
    repository_name: str | None = None   # human-readable repository name
    document_name: str | None = None     # human-readable document name
    # Outcome fields
    status: str = "success"
    duration_ms: float | None = None     # operation duration
    # Change tracking
    changes: list[dict[str, Any]] | None = None
    # Error fields (only on failure)
    metadata: dict[str, Any] | None = None
    exception: str | None = None
    exception_type: str | None = None    # exception class name
    stacktrace: str | None = None


@dataclass
class AuditQuery:
    page: int = 1
    page_size: int = 50
    search: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    application_name: str | None = None
    repository_id: str | None = None
    status: str | None = None
    event_type: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    has_changes: bool | None = None
    # New structured filters
    category: str | None = None
    action: str | None = None
    source: str | None = None
    business_only: bool = False
