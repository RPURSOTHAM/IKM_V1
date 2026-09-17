"""Retrieval and citation observability hooks."""

from __future__ import annotations

from typing import Any

from src.features.observability.audit.application.audit_service import get_audit_service
from src.features.observability.metrics.application.metrics_service import get_metrics_service


def record_search_executed(
    query: str,
    *,
    repository_id: str | None = None,
    repository_name: str | None = None,
    collection_name: str | None = None,
    search_mode: str | None = None,
    user_id: str | None = None,
) -> None:
    """Record that a user executed a search/retrieval query."""
    get_audit_service().record(
        "SEARCH_EXECUTED",
        category="retrieval",
        action="search",
        source="API",
        entity_type="repository",
        entity_id=repository_id,
        repository_name=repository_name,
        user_id=user_id,
        metadata={
            "query_preview": query[:200] if query else None,
            "repository_id": repository_id,
            "collection_name": collection_name,
            "search_mode": search_mode,
        },
    )


def record_search_completed(
    *,
    repository_id: str | None = None,
    repository_name: str | None = None,
    duration_ms: float | None = None,
    result_count: int | None = None,
    search_mode: str | None = None,
    user_id: str | None = None,
    retrieved_chunks: int | None = None,
    retrieved_documents: int | None = None,
) -> None:
    """Record successful completion of a search/retrieval query."""
    get_metrics_service().record(
        "retrieval_duration",
        duration_ms=duration_ms,
        status="success",
        repository_id=repository_id,
        metadata={
            "category": "retrieval",
            "component": "retrieval_service",
            "operation": "search",
            "search_mode": search_mode,
            "result_count": result_count,
            "retrieved_chunks": retrieved_chunks,
            "retrieved_documents": retrieved_documents,
        },
    )
    no_results = result_count is not None and result_count == 0
    get_audit_service().record(
        "SEARCH_NO_RESULTS" if no_results else "SEARCH_COMPLETED",
        category="retrieval",
        action="search",
        source="API",
        entity_type="repository",
        entity_id=repository_id,
        repository_name=repository_name,
        duration_ms=duration_ms,
        user_id=user_id,
        metadata={
            "result_count": result_count,
            "repository_id": repository_id,
            "search_mode": search_mode,
        },
    )


def record_retrieval_failure(
    repository_id: str | None,
    error: str,
    *,
    repository_name: str | None = None,
    duration_ms: float | None = None,
    user_id: str | None = None,
) -> None:
    get_metrics_service().record(
        "retrieval_duration",
        duration_ms=duration_ms,
        status="failure",
        repository_id=repository_id,
        metadata={
            "category": "retrieval",
            "component": "retrieval_service",
            "operation": "search",
            "error": error,
        },
    )
    get_audit_service().record_failure(
        "SEARCH_FAILED",
        category="retrieval",
        action="search",
        source="API",
        entity_type="repository",
        entity_id=repository_id,
        repository_name=repository_name,
        duration_ms=duration_ms,
        user_id=user_id,
        error=error,
        metadata={"repository_id": repository_id},
    )


def record_citation_validated(
    *,
    document_id: str | None = None,
    repository_id: str | None = None,
    repository_name: str | None = None,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_metrics_service().record(
        "citation_duration",
        duration_ms=duration_ms,
        status="success",
        repository_id=repository_id,
        document_id=document_id,
    )
    get_audit_service().record(
        "CITATION_GENERATED",
        category="retrieval",
        action="cite",
        source="API",
        entity_type="document",
        entity_id=document_id,
        repository_name=repository_name,
        metadata={"repository_id": repository_id, **(metadata or {})},
    )
