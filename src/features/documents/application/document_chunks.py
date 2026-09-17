"""Document-scoped Weaviate chunk listing for consumer inspection."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.features.repositories.infrastructure import weaviate_admin as repository_admin


def _flatten_chunk(entry: dict[str, Any], *, include_text: bool, include_vector: bool) -> dict[str, Any]:
    props = dict(entry.get("properties") or {})
    meta = dict(entry.get("metadata") or {})
    row: dict[str, Any] = {
        "uuid": entry.get("uuid"),
        "chunk_id": props.get("chunk_id"),
        "doc_name": props.get("doc_name"),
        "page": props.get("page"),
        "section_name": props.get("section_name"),
        "section_path": props.get("section_path"),
        "category": props.get("category"),
        "is_duplicate": props.get("is_duplicate"),
        "cluster_id": props.get("cluster_id"),
        "match_pct": props.get("match_pct"),
        "file_size_bytes": props.get("file_size_bytes"),
        "score": meta.get("score"),
        "distance": meta.get("distance"),
    }
    if include_text:
        row["text"] = props.get("text") or props.get("raw_text") or ""
    if include_vector:
        row["vector"] = entry.get("vector")
    row["properties"] = props if include_text else {k: v for k, v in props.items() if k != "text"}
    return row


def _weaviate_doc_name_candidates(record: dict[str, Any]) -> list[str]:
    """Names that may appear on Weaviate ``doc_name`` / ``document_name`` properties.

    Processors store the original upload filename as ``doc_name``. Intake
    ``document_name`` is often ``{document_id}{ext}`` on disk and does not match
    Weaviate — prefer ``original_file_name``, then fall back to stored names.
    """
    candidates: list[str] = []
    for key in ("original_file_name", "document_name"):
        value = str(record.get(key) or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for key in ("original_file_name", "document_name"):
        value = str(meta.get(key) or "").strip()
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def list_document_chunks(
    record: dict[str, Any],
    *,
    limit: int = 500,
    offset: int = 0,
    include_text: bool = True,
    include_vector: bool = True,
) -> dict[str, Any]:
    """Return indexed chunks for a document from its Weaviate collection.

    Lookup prefers ``document_id`` (authoritative property written by the
    processor). ``doc_name`` filters use the original filename when available,
    because intake ``document_name`` is frequently ``{uuid}.ext`` and does not
    match stored chunk metadata.
    """
    collection_name = str(record.get("collection_name") or "").strip()
    if not collection_name:
        raise HTTPException(status_code=422, detail="Document is not bound to a Weaviate collection.")
    document_id = str(record.get("document_id") or "").strip() or None
    name_candidates = _weaviate_doc_name_candidates(record)
    document_name = name_candidates[0] if name_candidates else str(record.get("document_name") or "").strip()
    if not document_id and not document_name:
        raise HTTPException(
            status_code=422,
            detail="Document has no document_id or document_name for chunk lookup.",
        )

    tenant = record.get("tenant_id")
    payload = repository_admin.list_chunks_metadata(
        collection_name,
        tenant=str(tenant) if tenant else None,
        document_id=document_id,
        # Pass original filename (not uuid disk name) so OR-filter covers legacy
        # objects that lack document_id while still matching current writes.
        document_name=document_name or None,
        limit=limit,
        offset=offset,
        include_text=include_text,
        include_vector=include_vector,
    )
    raw_chunks = payload.get("chunks") or []
    chunks = [
        _flatten_chunk(item, include_text=include_text, include_vector=include_vector)
        for item in raw_chunks
    ]
    return {
        "document_id": record.get("document_id"),
        "document_name": document_name or record.get("document_name"),
        "original_file_name": record.get("original_file_name"),
        "collection_name": collection_name,
        "tenant": tenant,
        "offset": offset,
        "limit": limit,
        "returned": len(chunks),
        "include_text": include_text,
        "include_vector": include_vector,
        "chunks": chunks,
    }
