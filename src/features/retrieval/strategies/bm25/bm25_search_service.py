"""Standalone BM25 search and index lifecycle service."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.shared.errors import COMPONENT_RETRIEVAL, raise_client_error, raise_service_error
from src.features.retrieval.strategies.bm25.executor import bm25_score_from_object, execute_bm25
from src.features.retrieval.strategies.bm25.schemas import (
    Bm25SearchMetrics,
    Bm25SearchResponse,
    Bm25SearchResultItem,
)
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut
from src.infrastructure.vector_store.shared_weaviate.tenancy import resolve_collection

_logger = logging.getLogger(__name__)


def _apply_bm25_retrieval_dlp(results: list[Bm25SearchResultItem]) -> list[Bm25SearchResultItem]:
    """Reuse the /retrieve DLP pipeline so BM25 responses expose the same protected content."""
    if not results:
        return results
    from src.features.retrieval.application.retrieval_service import _apply_retrieval_dlp

    chunks = [
        ChunkOut(
            chunk_id=item.chunk_id,
            id=item.chunk_id,
            text=item.text,
            doc_name=str(item.metadata.get("doc_name") or item.document_id or ""),
            section_name=item.section or "",
            page=item.page,
            score=item.bm25_score,
            source="bm25",
            graph_context=[],
            highlight_spans=[],
            metadata=item.metadata,
        )
        for item in results
    ]
    redacted = _apply_retrieval_dlp(chunks)
    return [
        item.model_copy(update={"text": chunk.text}) if chunk.text != item.text else item
        for item, chunk in zip(results, redacted)
    ]


def _record_bm25_search_audit(
    *,
    actor: Any,
    repository_id: str,
    query: str,
    result_count: int,
    latency_ms: float,
) -> None:
    """Mirror retrieval.search_executed audit events from POST /api/v1/retrieve."""
    if not actor:
        return
    try:
        from src.features.audit.application.security_audit_service import AuditService
        from src.features.users.infrastructure.user_repository import get_platform_security_store

        sec_store = get_platform_security_store()
        if sec_store:
            AuditService(sec_store).record(
                event_category="retrieval",
                event_type="retrieval.search_executed",
                action="execute",
                outcome="success",
                actor=actor,
                repository_id=repository_id,
                new_value={
                    "search_mode": "bm25",
                    "query": query,
                    "result_count": result_count,
                    "latency_ms": round(latency_ms, 1),
                },
            )
    except Exception:
        pass

_TOP_LEVEL_METADATA_KEYS = frozenset(
    {
        "chunk_id",
        "document_id",
        "doc_name",
        "page",
        "section",
        "section_name",
        "text",
        "repository_id",
    }
)


def _metadata_from_props(props: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in props.items():
        if key in _TOP_LEVEL_METADATA_KEYS:
            continue
        if key in {"semantic_metadata", "extraction_metadata", "retrieval_metadata"}:
            if isinstance(value, str):
                try:
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    metadata[key] = value
            else:
                metadata[key] = value
        else:
            metadata[key] = value
    return metadata


def _format_bm25_result(obj: Any, *, index: int) -> Bm25SearchResultItem:
    props = dict(getattr(obj, "properties", None) or {})
    chunk_id = str(props.get("chunk_id") or getattr(obj, "uuid", "") or "")
    document_id = props.get("document_id")
    page_raw = props.get("page")
    page = int(page_raw) if isinstance(page_raw, int) else None
    section = props.get("section") or props.get("section_name") or props.get("heading")
    return Bm25SearchResultItem(
        chunk_id=chunk_id,
        document_id=str(document_id) if document_id else None,
        bm25_score=round(bm25_score_from_object(obj, fallback_index=index), 4),
        page=page,
        section=str(section) if section else None,
        text=str(props.get("text") or ""),
        metadata=_metadata_from_props(props),
    )


class Bm25SearchService:
    """BM25 lexical retrieval and index lifecycle facade."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def _require_client(self) -> Any:
        if self._client is not None:
            return self._client
        from src.infrastructure.vector_store.dms_weaviate_client.client import connect_weaviate

        client = connect_weaviate()
        if not client.is_ready():
            raise_service_error(
                COMPONENT_RETRIEVAL,
                code="weaviate_unavailable",
                http_status=503,
                user_message="Search is temporarily unavailable because the index is not ready.",
                reason="Weaviate client is not ready for BM25 search",
            )
        self._client = client
        return client

    @staticmethod
    def _resolve_repository(repository_id: str) -> dict[str, Any]:
        from src.features.retrieval.application.repository_gate import load_active_repository_context

        return load_active_repository_context(repository_id)

    def _collection_handle(
        self,
        repository_id: str,
        *,
        tenant_id: str | None = None,
    ) -> tuple[Any, dict[str, Any], str | None]:
        repo_context = self._resolve_repository(repository_id)
        collection_name = str(repo_context.get("weaviate_collection") or "").strip()
        if not collection_name:
            raise_client_error(
                COMPONENT_RETRIEVAL,
                code="repository_collection_missing",
                http_status=422,
                user_message="Repository is not bound to a search index.",
                reason=f"Missing weaviate_collection for repository_id={repository_id}",
            )
        tenant = tenant_id if tenant_id is not None else repo_context.get("default_tenant_id")
        client = self._require_client()
        if not client.collections.exists(collection_name):
            raise_client_error(
                COMPONENT_RETRIEVAL,
                code="collection_not_found",
                http_status=404,
                user_message="Repository search index was not found.",
                reason=f"Weaviate collection missing: {collection_name}",
                details={"repository_id": repository_id, "collection_name": collection_name},
            )
        collection = client.collections.get(collection_name)
        collection = resolve_collection(collection, str(tenant) if tenant else None)
        return collection, repo_context, str(tenant) if tenant else None

    def execute_bm25_on_collection(
        self,
        collection: Any,
        query_text: str,
        *,
        limit: int,
        filters: Any | None = None,
        explain_score: bool = False,
    ) -> Any:
        """Central BM25 entry point used by RetrievalService and this service."""
        return execute_bm25(
            collection,
            query_text,
            limit=limit,
            filters=filters,
            explain_score=explain_score,
        )

    def search_bm25(
        self,
        repository_id: str,
        query: str,
        top_k: int = 20,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query_text = (query or "").strip()
        if not query_text:
            raise_client_error(
                COMPONENT_RETRIEVAL,
                code="invalid_query",
                http_status=422,
                user_message="Query text is required.",
                reason="Empty BM25 query",
            )

        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.users.application.user_service import get_platform_security_service

        actor = get_current_user_from_context()
        if actor and repository_id:
            get_platform_security_service().check_retrieval_access(actor, repository_id)

        collection, _repo_context, _tenant = self._collection_handle(repository_id)
        search_started = time.perf_counter()
        try:
            result = self.execute_bm25_on_collection(
                collection,
                query_text,
                limit=top_k,
            )
        except Exception as exc:
            _logger.error(
                "bm25_search_failed repository_id=%s query=%r top_k=%d error=%s",
                repository_id,
                query_text,
                top_k,
                exc,
                exc_info=exc,
            )
            raise_service_error(
                COMPONENT_RETRIEVAL,
                code="bm25_search_failed",
                http_status=500,
                user_message="BM25 search could not be completed. Please try again later.",
                reason=f"BM25 search failed for repository_id={repository_id}",
                cause=exc,
            )

        bm25_search_ms = (time.perf_counter() - search_started) * 1000.0
        objects = list(getattr(result, "objects", None) or [])
        results = _apply_bm25_retrieval_dlp(
            [_format_bm25_result(obj, index=idx) for idx, obj in enumerate(objects)]
        )
        total_ms = (time.perf_counter() - started) * 1000.0

        _record_bm25_search_audit(
            actor=actor,
            repository_id=repository_id,
            query=query_text,
            result_count=len(results),
            latency_ms=total_ms,
        )

        _logger.info(
            "bm25_search repository_id=%s query=%r top_k=%d result_count=%d bm25_search_ms=%.1f total_ms=%.1f",
            repository_id,
            query_text,
            top_k,
            len(results),
            bm25_search_ms,
            total_ms,
        )

        response = Bm25SearchResponse(
            query=query_text,
            repository_id=repository_id,
            result_count=len(results),
            results=results,
            metrics=Bm25SearchMetrics(
                bm25_search_ms=round(bm25_search_ms, 1),
                total_ms=round(total_ms, 1),
                chunks_examined=len(objects),
                chunks_returned=len(results),
            ),
        )
        return response.model_dump()

    def delete_document(
        self,
        *,
        repository_id: str,
        document_id: str | None = None,
        document_name: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Remove BM25/vector chunks for a document from the repository collection."""
        from src.infrastructure.vector_store.shared_weaviate.chunk_index_sync import purge_document_chunks_from_collection

        if not document_id and not document_name:
            raise ValueError("document_id or document_name is required")

        collection, repo_context, tenant = self._collection_handle(
            repository_id,
            tenant_id=tenant_id,
        )
        payload = purge_document_chunks_from_collection(
            collection,
            document_name=document_name,
            document_id=document_id,
        )
        _logger.info(
            "bm25_delete_document repository_id=%s document_id=%s document_name=%s matches=%s",
            repository_id,
            document_id,
            document_name,
            payload.get("matches"),
        )
        return {
            "repository_id": repository_id,
            "collection_name": repo_context.get("weaviate_collection"),
            "tenant": tenant,
            **payload,
        }

    def index_document(
        self,
        *,
        repository_id: str,
        document_id: str,
        document_name: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Synchronize BM25 index before (re)ingestion by deleting stale chunks."""
        purge = self.delete_document(
            repository_id=repository_id,
            document_id=document_id,
            document_name=document_name,
            tenant_id=tenant_id,
        )
        _logger.info(
            "bm25_index_document_prepared repository_id=%s document_id=%s document_name=%s",
            repository_id,
            document_id,
            document_name,
        )
        return {
            "repository_id": repository_id,
            "document_id": document_id,
            "document_name": document_name,
            "prepared_for_indexing": True,
            "purge": purge,
        }

    def rebuild_repository(self, repository_id: str) -> dict[str, Any]:
        """Clear all chunks in a repository collection (documents must be reprocessed)."""
        from src.features.repositories.infrastructure import weaviate_admin

        repo_context = self._resolve_repository(repository_id)
        collection_name = str(repo_context.get("weaviate_collection") or "")
        tenant = repo_context.get("default_tenant_id")
        started = time.perf_counter()
        try:
            listing = weaviate_admin.list_unique_documents(
                collection_name,
                tenant=str(tenant) if tenant else None,
                limit=5000,
            )
        except Exception as exc:
            raise_service_error(
                COMPONENT_RETRIEVAL,
                code="bm25_rebuild_failed",
                http_status=500,
                user_message="Repository index rebuild failed.",
                reason=f"Could not list documents for rebuild repository_id={repository_id}",
                cause=exc,
            )

        purged = 0
        for document_name in listing.get("documents") or []:
            weaviate_admin.purge_document_chunks(
                collection_name,
                document_name,
                tenant=str(tenant) if tenant else None,
            )
            purged += 1

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _logger.info(
            "bm25_rebuild_repository repository_id=%s collection=%s documents_purged=%d elapsed_ms=%.1f",
            repository_id,
            collection_name,
            purged,
            elapsed_ms,
        )
        return {
            "repository_id": repository_id,
            "collection_name": collection_name,
            "tenant": tenant,
            "rebuilt": True,
            "documents_purged": purged,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def delete_repository(self, repository_id: str) -> dict[str, Any]:
        """Remove the repository Weaviate collection (BM25 + vector index)."""
        from src.features.repositories.infrastructure import weaviate_admin

        repo_context = self._resolve_repository(repository_id)
        collection_name = str(repo_context.get("weaviate_collection") or "").strip()
        if not collection_name:
            return {
                "repository_id": repository_id,
                "collection_name": None,
                "deleted": False,
                "reason": "no_collection_configured",
            }

        client = self._require_client()
        deleted = False
        if client.collections.exists(collection_name):
            try:
                weaviate_admin.delete_collection(collection_name)
                deleted = True
            except Exception as exc:
                _logger.error(
                    "bm25_delete_repository_failed repository_id=%s collection=%s error=%s",
                    repository_id,
                    collection_name,
                    exc,
                    exc_info=exc,
                )
                raise_service_error(
                    COMPONENT_RETRIEVAL,
                    code="bm25_delete_repository_failed",
                    http_status=502,
                    user_message="Repository search index could not be removed.",
                    reason=f"Failed to delete Weaviate collection {collection_name}",
                    cause=exc,
                )

        _logger.info(
            "bm25_delete_repository repository_id=%s collection=%s deleted=%s",
            repository_id,
            collection_name,
            deleted,
        )
        return {
            "repository_id": repository_id,
            "collection_name": collection_name,
            "deleted": deleted,
        }


_default_bm25_service: Bm25SearchService | None = None


def get_bm25_search_service() -> Bm25SearchService:
    global _default_bm25_service
    if _default_bm25_service is None:
        _default_bm25_service = Bm25SearchService()
    return _default_bm25_service
