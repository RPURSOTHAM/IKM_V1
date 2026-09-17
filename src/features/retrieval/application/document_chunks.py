"""Repository-scoped document chunk listing for inspection and debugging."""

from __future__ import annotations

import json
from typing import Any

from src.features.repositories.infrastructure import weaviate_admin

_TOP_LEVEL_PROPS = frozenset(
    {
        "chunk_id",
        "document_id",
        "page",
        "section_name",
        "text",
    }
)


def _estimate_token_count(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _chunk_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    props = dict(entry.get("properties") or {})
    page_raw = props.get("page")
    try:
        page = int(page_raw) if page_raw is not None else 0
    except (TypeError, ValueError):
        page = 0
    section_path = str(props.get("section_path") or props.get("section_name") or "")
    line_start = props.get("line_start")
    try:
        chunk_index = int(line_start) if line_start is not None else 0
    except (TypeError, ValueError):
        chunk_index = 0
    chunk_id = str(props.get("chunk_id") or entry.get("uuid") or "")
    return (page, section_path, chunk_index, chunk_id)


def sort_document_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order chunks by page, section path, then line/chunk position."""
    return sorted(chunks, key=_chunk_sort_key)


def format_document_chunk(
    entry: dict[str, Any],
    *,
    document_id: str,
    chunk_index: int,
) -> dict[str, Any]:
    props = dict(entry.get("properties") or {})
    text = str(props.get("text") or props.get("raw_text") or "")
    section = props.get("section_name") or props.get("heading") or props.get("section_path")
    page_raw = props.get("page")
    try:
        page = int(page_raw) if page_raw is not None else None
    except (TypeError, ValueError):
        page = None

    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    page_end = _optional_int(props.get("page_end") or props.get("end_page"))
    line_start = _optional_int(props.get("line_start"))
    line_end = _optional_int(props.get("line_end"))
    source_blocks = _parse_json_field(props.get("source_blocks"))
    if not isinstance(source_blocks, list):
        source_blocks = []
    citation_anchor = _parse_json_field(props.get("citation_anchor"))
    if not isinstance(citation_anchor, dict):
        citation_anchor = None

    embedding_version = str(props.get("embedding_version") or "").strip()
    metadata: dict[str, Any] = {}
    for key, value in props.items():
        if key in _TOP_LEVEL_PROPS:
            continue
        if key in {"semantic_metadata", "extraction_metadata", "source_blocks", "citation_anchor"}:
            metadata[key] = _parse_json_field(value)
        else:
            metadata[key] = value

    return {
        "chunk_id": str(props.get("chunk_id") or entry.get("uuid") or ""),
        "document_id": str(props.get("document_id") or document_id),
        "page": page,
        "page_end": page_end,
        "line_start": line_start,
        "line_end": line_end,
        "section": section,
        "section_path": props.get("section_path") or section,
        "source_blocks": source_blocks,
        "citation_anchor": citation_anchor,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": _estimate_token_count(text),
        "embedding_exists": bool(embedding_version),
        "metadata": metadata,
    }


def fetch_repository_document_chunks(
    *,
    collection_name: str,
    tenant: str | None,
    document_id: str,
    document_name: str | None,
) -> list[dict[str, Any]]:
    """Load all Weaviate chunks for a document (metadata + text only, no vectors)."""
    raw = weaviate_admin.fetch_all_document_chunks(
        collection_name,
        tenant=tenant,
        document_id=document_id,
        document_name=document_name,
        include_text=True,
    )
    ordered = sort_document_chunks(raw)
    return [
        format_document_chunk(entry, document_id=document_id, chunk_index=index)
        for index, entry in enumerate(ordered)
    ]
