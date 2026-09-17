"""Audit recording service — failures never propagate to callers."""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Any

from src.features.observability.audit.domain.audit_models import AuditEventRecord
from src.features.observability.audit.infrastructure.audit_repository import get_audit_repository
from src.features.observability.audit.application.change_tracker import build_change_metadata, get_changes, get_deep_changes
from src.features.observability.audit.application.user_context import enrich_audit_metadata
from src.features.observability.middleware.request_context import current_context

logger = logging.getLogger(__name__)


class AuditService:
    def _persist(self, event: AuditEventRecord) -> int | None:
        repo = get_audit_repository()
        if repo is None:
            return None
        try:
            return repo.insert(event)
        except Exception:
            logger.debug("Audit persist failed for %s", event.event_type, exc_info=True)
            return None

    def _async_persist(self, event: AuditEventRecord) -> None:
        threading.Thread(target=self._persist, args=(event,), daemon=True).start()

    def _base_event(
        self,
        *,
        event_type: str,
        status: str,
        # Business classification
        category: str | None = None,
        action: str | None = None,
        source: str | None = None,
        # Entity
        entity_type: str | None = None,
        entity_id: str | None = None,
        repository_name: str | None = None,
        document_name: str | None = None,
        # Outcome
        duration_ms: float | None = None,
        # Change tracking
        changes: list[dict[str, Any]] | None = None,
        # Error fields
        metadata: dict[str, Any] | None = None,
        exception: str | None = None,
        exception_type: str | None = None,
        stacktrace: str | None = None,
        # Context overrides
        request_id: str | None = None,
        correlation_id: str | None = None,
        application_name: str | None = None,
        user_id: str | None = None,
    ) -> AuditEventRecord:
        ctx = current_context()
        merged_metadata = enrich_audit_metadata(metadata)
        resolved_user_id = user_id or merged_metadata.get("user_id") or ctx.user_id
        resolved_application = (
            application_name
            or merged_metadata.get("application_name")
            or ctx.application_name
            or "unknown"
        )
        return AuditEventRecord(
            timestamp=datetime.now(timezone.utc),
            request_id=request_id or merged_metadata.get("request_id") or ctx.request_id,
            correlation_id=correlation_id or merged_metadata.get("correlation_id") or ctx.correlation_id,
            application_name=str(resolved_application),
            user_id=str(resolved_user_id) if resolved_user_id else None,
            event_type=event_type,
            category=category,
            action=action,
            source=source,
            entity_type=entity_type,
            entity_id=entity_id,
            repository_name=repository_name,
            document_name=document_name,
            status=status,
            duration_ms=duration_ms,
            changes=changes,
            metadata=merged_metadata or None,
            exception=exception,
            exception_type=exception_type,
            stacktrace=stacktrace,
        )

    def record(
        self,
        event_type: str,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        event = self._base_event(
            event_type=event_type,
            status=status,
            category=kwargs.get("category"),
            action=kwargs.get("action"),
            source=kwargs.get("source"),
            entity_type=entity_type,
            entity_id=entity_id,
            repository_name=kwargs.get("repository_name"),
            document_name=kwargs.get("document_name"),
            duration_ms=kwargs.get("duration_ms"),
            metadata=metadata,
            request_id=kwargs.get("request_id"),
            correlation_id=kwargs.get("correlation_id"),
            application_name=kwargs.get("application_name"),
            user_id=kwargs.get("user_id"),
        )
        self._async_persist(event)

    def record_failure(
        self,
        event_type: str,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        error: str | Exception | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        exc_obj = error if isinstance(error, BaseException) else None
        msg = str(error) if error is not None else "failure"
        exc_type = type(exc_obj).__name__ if exc_obj else kwargs.get("exception_type")
        event = self._base_event(
            event_type=event_type,
            status="failure",
            category=kwargs.get("category"),
            action=kwargs.get("action"),
            source=kwargs.get("source"),
            entity_type=entity_type,
            entity_id=entity_id,
            repository_name=kwargs.get("repository_name"),
            document_name=kwargs.get("document_name"),
            duration_ms=kwargs.get("duration_ms"),
            metadata=metadata,
            exception=msg,
            exception_type=exc_type,
            request_id=kwargs.get("request_id"),
            correlation_id=kwargs.get("correlation_id"),
            application_name=kwargs.get("application_name"),
            user_id=kwargs.get("user_id"),
        )
        self._async_persist(event)

    def record_change(
        self,
        event_type: str,
        *,
        entity_type: str,
        entity_id: str,
        old: Any,
        new: Any,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
        deep: bool = True,
        **kwargs: Any,
    ) -> None:
        changes = get_deep_changes(old, new) if deep else get_changes(old, new)
        if not changes:
            return
        meta = dict(metadata or {})
        ctx = current_context()
        meta.setdefault("repository_id", ctx.repository_id or entity_id)
        meta.update(build_change_metadata(changes))
        event = self._base_event(
            event_type=event_type,
            status=status,
            category=kwargs.get("category"),
            action=kwargs.get("action"),
            source=kwargs.get("source"),
            entity_type=entity_type,
            entity_id=entity_id,
            repository_name=kwargs.get("repository_name"),
            document_name=kwargs.get("document_name"),
            duration_ms=kwargs.get("duration_ms"),
            changes=changes,
            metadata=meta or None,
            request_id=kwargs.get("request_id"),
            correlation_id=kwargs.get("correlation_id"),
            application_name=kwargs.get("application_name"),
            user_id=kwargs.get("user_id"),
        )
        # Business changes are persisted synchronously so /audit/changes can read them
        # immediately after the mutation completes.
        self._persist(event)

    def record_state_transition(
        self,
        event_type: str,
        *,
        entity_type: str,
        entity_id: str,
        old_state: str,
        new_state: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        changes = [{"field": "state", "old": old_state, "new": new_state}]
        event = self._base_event(
            event_type=event_type,
            status="success",
            category=kwargs.get("category"),
            action=kwargs.get("action"),
            source=kwargs.get("source"),
            entity_type=entity_type,
            entity_id=entity_id,
            repository_name=kwargs.get("repository_name"),
            document_name=kwargs.get("document_name"),
            changes=changes,
            metadata=metadata,
            request_id=kwargs.get("request_id"),
            correlation_id=kwargs.get("correlation_id"),
            application_name=kwargs.get("application_name"),
            user_id=kwargs.get("user_id"),
        )
        self._persist(event)

    def record_exception(
        self,
        event_type: str,
        exc: BaseException,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        event = self._base_event(
            event_type=event_type,
            status="failure",
            category=kwargs.get("category"),
            action=kwargs.get("action"),
            source=kwargs.get("source"),
            entity_type=entity_type,
            entity_id=entity_id,
            repository_name=kwargs.get("repository_name"),
            document_name=kwargs.get("document_name"),
            duration_ms=kwargs.get("duration_ms"),
            metadata=metadata,
            exception=str(exc) or repr(exc),
            exception_type=type(exc).__name__,
            stacktrace=traceback.format_exc(),
            request_id=kwargs.get("request_id"),
            correlation_id=kwargs.get("correlation_id"),
            application_name=kwargs.get("application_name"),
            user_id=kwargs.get("user_id"),
        )
        self._async_persist(event)


_service: AuditService | None = None


def get_audit_service() -> AuditService:
    global _service
    if _service is None:
        _service = AuditService()
    return _service
