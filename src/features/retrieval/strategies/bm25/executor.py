"""Centralized Weaviate BM25 query execution."""

from __future__ import annotations

from typing import Any


def execute_bm25(
    collection: Any,
    query_text: str,
    *,
    limit: int,
    filters: Any | None = None,
    return_metadata: Any | None = None,
    explain_score: bool = False,
) -> Any:
    """Run a BM25 query against a Weaviate collection handle."""
    from weaviate.classes.query import MetadataQuery

    kwargs: dict[str, Any] = {
        "query": query_text,
        "limit": max(1, limit),
        "return_metadata": return_metadata
        or MetadataQuery(score=True, explain_score=explain_score),
    }
    if filters is not None:
        kwargs["filters"] = filters
    return collection.query.bm25(**kwargs)


def bm25_score_from_object(obj: Any, *, fallback_index: int = 0) -> float:
    """Extract raw BM25 score from a Weaviate result object."""
    metadata = getattr(obj, "metadata", None)
    score = getattr(metadata, "score", None) if metadata else None
    if score is not None:
        return float(score)
    distance = getattr(metadata, "distance", None) if metadata else None
    if distance is not None:
        return max(0.0, 1.0 - float(distance))
    return max(0.01, 1.0 - (fallback_index * 0.025))
