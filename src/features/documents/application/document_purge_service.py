"""Cascade purge of document artifacts across storage backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def purge_weaviate_chunks(
    *,
    collection_name: str | None,
    document_name: str | None = None,
    document_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Remove vector/BM25 chunks for a document from Weaviate (by ``doc_name`` and/or ``document_id``)."""
    if not collection_name or (not document_name and not document_id):
        return {"attempted": False, "reason": "missing_collection_or_document_identifiers"}

    from src.features.repositories.infrastructure.weaviate_backend import backend

    if not backend.ready:
        return {
            "attempted": False,
            "reason": "weaviate_unavailable",
            "error": backend.last_error,
        }

    try:
        from src.features.repositories.infrastructure.weaviate_admin import purge_document_chunks

        result = purge_document_chunks(
            collection_name,
            document_name,
            document_id=document_id,
            tenant=tenant_id,
        )
        return {"attempted": True, **result}
    except Exception as exc:
        logger.exception(
            "Weaviate purge failed for collection=%s document_name=%s document_id=%s",
            collection_name,
            document_name,
            document_id,
        )
        return {"attempted": True, "error": str(exc)}


def purge_neo4j_document(document_id: str) -> dict[str, Any]:
    """Remove Document and DocumentArtifact nodes for a document."""
    try:
        from src.infrastructure.document_databases.neo4j_store import delete_document_graph

        return {"attempted": True, **delete_document_graph(document_id)}
    except Exception as exc:
        logger.exception("Neo4j purge failed for document_id=%s", document_id)
        return {"attempted": True, "error": str(exc)}


def purge_document_type_mysql(document_id: str) -> dict[str, Any]:
    """Remove document_instance and document_metadata_value rows."""
    try:
        from src.features.document_types.infrastructure.document_type_repository import get_document_type_store

        store = get_document_type_store()
        if store is None:
            return {"attempted": False, "reason": "document_type_store_unavailable"}
        return {"attempted": True, **store.delete_document_data(document_id)}
    except Exception as exc:
        logger.exception("Document type MySQL purge failed for document_id=%s", document_id)
        return {"attempted": True, "error": str(exc)}


def purge_repository_link(document_id: str) -> dict[str, Any]:
    try:
        from src.features.repositories.application.repository_service import get_repository_service

        removed = get_repository_service().unlink_document_global(document_id)
        return {"attempted": True, "removed": removed}
    except Exception as exc:
        logger.exception("Repository link purge failed for document_id=%s", document_id)
        return {"attempted": True, "error": str(exc)}


def purge_source_file(repository_path: str | None, *, repository_type: str | None) -> dict[str, Any]:
    if repository_type != "local":
        return {"attempted": False, "reason": "non_local_repository"}
    path = Path(repository_path or "")
    if not path.is_file():
        return {"attempted": True, "deleted": False, "path": str(path) or None}
    try:
        path.unlink()
        return {"attempted": True, "deleted": True, "path": str(path)}
    except OSError as exc:
        logger.exception("Failed to delete source file %s", path)
        return {"attempted": True, "deleted": False, "path": str(path), "error": str(exc)}


def purge_document_job(document_id: str) -> dict[str, Any]:
    try:
        from src.infrastructure.database.document_jobs import get_document_job_store

        store = get_document_job_store()
        if store is None:
            return {"attempted": False, "reason": "document_job_store_unavailable"}
        store.delete_by_document_id(document_id)
        return {"attempted": True, "deleted": True}
    except Exception as exc:
        logger.exception("document_job purge failed for document_id=%s", document_id)
        return {"attempted": True, "error": str(exc)}


def purge_redis_render_cache(document_id: str) -> dict[str, Any]:
    try:
        from src.features.documents.infrastructure.content.render_cache import delete_render_payload

        return {"attempted": True, **delete_render_payload(document_id=document_id)}
    except Exception as exc:
        logger.exception("Redis render-cache purge failed for document_id=%s", document_id)
        return {"attempted": True, "error": str(exc)}


def purge_all_artifacts(record: dict[str, Any], *, delete_file: bool = True) -> dict[str, Any]:
    """Purge external stores and links before intake metadata is removed."""
    document_id = str(record["document_id"])
    result = {
        "weaviate": purge_weaviate_chunks(
            collection_name=record.get("collection_name"),
            document_name=record.get("document_name"),
            document_id=document_id,
            tenant_id=record.get("tenant_id"),
        ),
        "neo4j": purge_neo4j_document(document_id),
        "document_type_mysql": purge_document_type_mysql(document_id),
        "repository_link": purge_repository_link(document_id),
        "document_job": purge_document_job(document_id),
        "redis_render_cache": purge_redis_render_cache(document_id),
    }
    if delete_file:
        result["source_file"] = purge_source_file(
            str(record.get("repository_path") or ""),
            repository_type=str(record.get("repository_type") or "local"),
        )
    else:
        result["source_file"] = {"attempted": False, "skipped": True}
    return result
