from __future__ import annotations

from typing import Any

from src.features.retrieval.schemas.retrieval_schemas import DocumentCatalogItem, IndexedDocumentMetadata

INTAKE_FILTER_KEYS = frozenset(
    {
        "status",
        "tenant_id",
        "collection_name",
        "repository_id",
        "document_type",
        "original_file_name",
    }
)
INDEXED_FILTER_KEYS = frozenset({"category", "is_duplicate", "indexed"})


def split_catalog_filters(filters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split filters into intake store, custom metadata, and indexed post-filters."""
    intake: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    indexed: dict[str, Any] = {}
    for key, value in filters.items():
        if value in (None, ""):
            continue
        if key in INTAKE_FILTER_KEYS:
            intake[key] = value
        elif key in INDEXED_FILTER_KEYS:
            indexed[key] = value
        elif key.startswith("metadata."):
            metadata[key] = value
        else:
            metadata[f"metadata.{key}"] = value
    return intake, metadata, indexed


def _canonical_document_keys(props: dict[str, Any]) -> list[str]:
    """Return lookup keys for a Weaviate chunk object, preferring stable document_id."""
    keys: list[str] = []
    document_id = str(props.get("document_id") or "").strip()
    if document_id:
        keys.append(document_id)
    for field in ("doc_name", "document_name", "original_file_name"):
        value = str(props.get(field) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _empty_indexed_row() -> dict[str, Any]:
    return {
        "chunk_count": 0,
        "embedded_chunk_count": 0,
        "category": None,
        "category_summary": None,
        "category_confidence": None,
        "is_duplicate": False,
        "file_size_bytes": None,
    }


def aggregate_indexed_metadata(
    objects: list[Any],
) -> dict[str, dict[str, Any]]:
    """Build per-document aggregate stats from Weaviate chunk objects.

    Keys are canonical document identifiers (document_id when present, else doc_name).
    """
    stats: dict[str, dict[str, Any]] = {}
    for obj in objects:
        props = obj.properties or {}
        keys = _canonical_document_keys(props)
        if not keys:
            continue
        primary_key = keys[0]
        row = stats.setdefault(primary_key, _empty_indexed_row())
        row["chunk_count"] += 1
        vector = getattr(obj, "vector", None)
        if vector is not None:
            row["embedded_chunk_count"] += 1
        if row["file_size_bytes"] is None and props.get("file_size_bytes") is not None:
            try:
                row["file_size_bytes"] = int(props.get("file_size_bytes"))
            except (TypeError, ValueError):
                pass
        if row["category"] is None and props.get("category"):
            row["category"] = str(props.get("category"))
        if row["category_summary"] is None and props.get("category_summary"):
            row["category_summary"] = str(props.get("category_summary"))
        if row["category_confidence"] is None and props.get("category_confidence") is not None:
            row["category_confidence"] = float(props.get("category_confidence"))
        if props.get("is_duplicate"):
            row["is_duplicate"] = True
        snapshot = dict(row)
        for key in keys:
            stats[key] = dict(snapshot)
    return stats


def _lookup_indexed_stats(
    record: dict[str, Any],
    indexed_stats: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not indexed_stats:
        return _empty_indexed_row()
    document_id = str(record.get("document_id") or "").strip()
    if document_id and document_id in indexed_stats:
        return indexed_stats[document_id]
    doc_name = str(record.get("document_name") or "").strip()
    if doc_name and doc_name in indexed_stats:
        return indexed_stats[doc_name]
    original = str(record.get("original_file_name") or "").strip()
    if original and original in indexed_stats:
        return indexed_stats[original]
    return _empty_indexed_row()


def record_to_catalog_item(
    record: dict[str, Any],
    *,
    indexed_stats: dict[str, dict[str, Any]] | None = None,
    job: dict[str, Any] | None = None,
    relevance_score: float | None = None,
) -> DocumentCatalogItem:
    doc_name = str(record.get("document_name") or "")
    stats = _lookup_indexed_stats(record, indexed_stats)
    chunk_count = int(stats.get("chunk_count") or 0)
    embedded_chunk_count = int(stats.get("embedded_chunk_count") or 0)
    meta = dict(record.get("metadata") or {})
    job_id = meta.get("job_id") or (job or {}).get("job_id")
    batch_id = meta.get("batch_id") or (job or {}).get("batch_id")
    status = str(record.get("status") or "unknown")
    if job and job.get("status"):
        from src.features.documents.application.document_service import DocumentReceiverService

        status = DocumentReceiverService.document_status_from_job(str(job.get("status")))

    return DocumentCatalogItem(
        document_id=str(record["document_id"]),
        document_name=doc_name,
        original_file_name=str(record.get("original_file_name") or doc_name),
        document_type=str(record.get("document_type") or ""),
        status=status,
        repository_id=record.get("repository_id"),
        collection_name=str(record.get("collection_name") or ""),
        tenant_id=record.get("tenant_id"),
        upload_timestamp=record.get("upload_timestamp"),
        queue_submission_timestamp=record.get("queue_submission_timestamp"),
        processing_completion_timestamp=record.get("processing_completion_timestamp"),
        job_id=str(job_id) if job_id else None,
        batch_id=str(batch_id) if batch_id else None,
        metadata={k: v for k, v in meta.items() if k not in {"job_id", "batch_id"}},
        indexed=IndexedDocumentMetadata(
            indexed=chunk_count > 0,
            chunk_count=chunk_count,
            embedded_chunk_count=embedded_chunk_count,
            category=stats.get("category"),
            category_summary=stats.get("category_summary"),
            category_confidence=stats.get("category_confidence"),
            is_duplicate=stats.get("is_duplicate") if chunk_count else None,
            file_size_bytes=stats.get("file_size_bytes"),
        ),
        relevance_score=relevance_score,
    )


def matches_indexed_filters(item: DocumentCatalogItem, indexed_filters: dict[str, Any]) -> bool:
    if not indexed_filters:
        return True
    for key, expected in indexed_filters.items():
        if key == "indexed":
            want = str(expected).lower() in {"true", "1", "yes"}
            if item.indexed.indexed != want:
                return False
        elif key == "category":
            if str(item.indexed.category or "") != str(expected):
                return False
        elif key == "is_duplicate":
            want = str(expected).lower() in {"true", "1", "yes"}
            if bool(item.indexed.is_duplicate) != want:
                return False
    return True


def rank_documents_by_chunk_scores(
    chunk_scores: dict[str, float],
    records_by_key: dict[str, dict[str, Any]],
    *,
    top_k: int,
    min_score: float,
) -> list[tuple[dict[str, Any], float]]:
    ranked: list[tuple[dict[str, Any], float]] = []
    for doc_key, score in sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True):
        if score < min_score:
            continue
        record = records_by_key.get(doc_key)
        if record is None:
            continue
        ranked.append((record, score))
        if len(ranked) >= top_k:
            break
    return ranked


def catalog_record_keys(record: dict[str, Any]) -> list[str]:
    """Return all keys that may identify a catalog record in chunk-score maps."""
    keys: list[str] = []
    document_id = str(record.get("document_id") or "").strip()
    if document_id:
        keys.append(document_id)
    for field in ("document_name", "original_file_name"):
        value = str(record.get(field) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def build_catalog_records_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map document_id / document_name / original_file_name to intake records."""
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        for key in catalog_record_keys(record):
            index[key] = record
    return index
