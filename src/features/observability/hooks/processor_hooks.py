"""Processor pipeline observability hooks."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from src.features.observability.audit.application.audit_service import get_audit_service
from src.features.observability.middleware.request_context import apply_synthetic_context, reset_context


@contextmanager
def processor_job_context(
    *,
    document_id: str | None = None,
    repository_id: str | None = None,
    correlation_id: str | None = None,
    request_id: str | None = None,
) -> Iterator[None]:
    rid = request_id or f"proc-{uuid.uuid4()}"
    corr = correlation_id or rid
    tokens = apply_synthetic_context(
        request_id=rid,
        correlation_id=corr,
        repository_id=repository_id,
        document_id=document_id,
        metadata={
            "component": "processor",
            "operation": "process",
            "username": "processor",
            "display_name": "Document Processor",
            "role": "System",
            "authentication_provider": "Internal",
            "source": "Processor",
        },
    )
    try:
        yield
    finally:
        reset_context(tokens)


def record_document_submitted(
    document_id: str,
    *,
    repository_id: str | None = None,
    document_name: str | None = None,
    repository_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record that a user submitted a document for processing."""
    get_audit_service().record(
        "DOCUMENT_SUBMITTED",
        category="document",
        action="submit",
        source="API",
        entity_type="document",
        entity_id=document_id,
        document_name=document_name,
        repository_name=repository_name,
        metadata={"repository_id": repository_id, **(metadata or {})},
    )


def record_document_reprocessed(
    document_id: str,
    *,
    repository_id: str | None = None,
    document_name: str | None = None,
    repository_name: str | None = None,
) -> None:
    """Record that a user triggered reprocessing of a document."""
    get_audit_service().record(
        "DOCUMENT_REPROCESSED",
        category="document",
        action="reprocess",
        source="API",
        entity_type="document",
        entity_id=document_id,
        document_name=document_name,
        repository_name=repository_name,
        metadata={"repository_id": repository_id},
    )


def record_processing_failure(document_id: str, error: str, *, processor_type: str | None = None) -> None:
    get_audit_service().record_failure(
        "PROCESSOR_FAILURE",
        category="processing",
        action="process",
        source="Processor",
        entity_type="document",
        entity_id=document_id,
        user_id="processor",
        error=error,
        metadata={
            "processor_type": processor_type,
            "component": "processor",
            "operation": "process",
            "username": "processor",
            "display_name": "Document Processor",
            "role": "System",
            "authentication_provider": "Internal",
        },
    )
