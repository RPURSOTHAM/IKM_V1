"""Shared Weaviate chunk purge helpers for BM25/vector index synchronization."""

from __future__ import annotations

from typing import Any


def purge_document_chunks_from_collection(
    collection: Any,
    *,
    document_name: str | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Remove all Weaviate objects for a document before re-indexing."""
    from weaviate.classes.query import Filter
    from weaviate.connect import executor as wexec

    clauses: list[Any] = []
    if document_name:
        clauses.append(Filter.by_property("doc_name").equal(document_name))
    if document_id:
        clauses.append(Filter.by_property("document_id").equal(document_id))
    if not clauses:
        return {"attempted": False, "reason": "missing_document_identifiers"}

    combined = clauses[0]
    for clause in clauses[1:]:
        combined = combined | clause

    result = wexec.result(collection.data.delete_many(where=combined, verbose=False, dry_run=False))
    return {
        "attempted": True,
        "document_name": document_name,
        "document_id": document_id,
        "successful": result.successful,
        "matches": result.matches,
        "failed": result.failed,
    }
