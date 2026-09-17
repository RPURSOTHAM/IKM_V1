"""Citation anchors — stable page/line references stored at ingestion, not reconstructed at retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CitationAnchor:
    """Immutable citation reference for a single chunk."""

    chunk_id: str
    start_page: int | None = None
    end_page: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    bbox: list[float] | None = None
    text_hash: str | None = None
    section_path: str = ""
    section_name: str = ""
    parent_section: str = ""
    title: str = ""
    table_name: str = ""
    chunk_type: str = "paragraph"
    strategy_name: str = ""
    document_id: str = ""
    document_name: str = ""
    table_index: int | None = None
    source_blocks: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_start": self.start_page,
            "page_end": self.end_page,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "section_path": self.section_path,
            "section_name": self.section_name,
            "parent_section": self.parent_section,
            "title": self.title,
            "table_name": self.table_name,
            "chunk_type": self.chunk_type,
            "strategy_name": self.strategy_name,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "source_blocks": self.source_blocks,
        }
        if self.bbox is not None:
            payload["bbox"] = self.bbox
        if self.text_hash:
            payload["text_hash"] = self.text_hash
        if self.table_index is not None:
            payload["table_index"] = self.table_index
        if self.extra:
            payload.update(self.extra)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CitationAnchor | None:
        if not data:
            return None
        chunk_id = str(data.get("chunk_id") or "").strip()
        if not chunk_id:
            return None
        start_page = _optional_int(
            data.get("start_page") if data.get("start_page") is not None else data.get("page_start")
        )
        if start_page is None:
            start_page = _optional_int(data.get("page"))
        end_page = _optional_int(
            data.get("end_page") if data.get("end_page") is not None else data.get("page_end")
        )
        if end_page is None:
            end_page = start_page
        bbox = data.get("bbox")
        if isinstance(bbox, (list, tuple)):
            bbox = [float(v) for v in bbox]
        else:
            bbox = None
        source_blocks = data.get("source_blocks")
        if not isinstance(source_blocks, list):
            source_blocks = []
        known = {
            "chunk_id",
            "start_page",
            "end_page",
            "page_start",
            "page_end",
            "start_line",
            "end_line",
            "bbox",
            "text_hash",
            "section_path",
            "section_name",
            "parent_section",
            "title",
            "table_name",
            "chunk_type",
            "strategy_name",
            "document_id",
            "document_name",
            "table_index",
            "source_blocks",
            "page",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            chunk_id=chunk_id,
            start_page=start_page,
            end_page=end_page,
            start_line=_optional_int(data.get("start_line")),
            end_line=_optional_int(data.get("end_line")),
            bbox=bbox,
            text_hash=str(data["text_hash"]) if data.get("text_hash") else None,
            section_path=str(data.get("section_path") or ""),
            section_name=str(data.get("section_name") or ""),
            parent_section=str(data.get("parent_section") or ""),
            title=str(data.get("title") or ""),
            table_name=str(data.get("table_name") or ""),
            chunk_type=str(data.get("chunk_type") or "paragraph"),
            strategy_name=str(data.get("strategy_name") or ""),
            document_id=str(data.get("document_id") or ""),
            document_name=str(data.get("document_name") or ""),
            table_index=_optional_int(data.get("table_index")),
            source_blocks=source_blocks,
            extra=extra,
        )

    @classmethod
    def from_chunk(cls, chunk: Any) -> CitationAnchor:
        page_start = _optional_int(getattr(chunk, "page_start", None))
        if page_start is None:
            page_start = _optional_int(getattr(chunk, "page", None))
        page_end = _optional_int(getattr(chunk, "page_end", None))
        if page_end is None:
            page_end = _optional_int(getattr(chunk, "end_page", None))
        if page_end is None:
            page_end = page_start
        extraction = getattr(chunk, "extraction_metadata", None) or {}
        bbox: list[float] | None = None
        table_index: int | None = None
        tables = extraction.get("tables") or []
        if tables and isinstance(tables[0], dict):
            table_meta = tables[0]
            raw_bbox = table_meta.get("bbox")
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                bbox = [float(v) for v in raw_bbox]
            table_index = _optional_int(table_meta.get("table_index"))

        raw_text = str(getattr(chunk, "raw_text", None) or getattr(chunk, "text", "") or "")
        text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16] if raw_text.strip() else None

        return cls(
            chunk_id=str(getattr(chunk, "id", "")),
            start_page=page_start,
            end_page=page_end,
            start_line=_optional_int(getattr(chunk, "line_start", None)),
            end_line=_optional_int(getattr(chunk, "line_end", None)),
            bbox=bbox,
            text_hash=text_hash,
            section_path=str(getattr(chunk, "section_path", "") or getattr(chunk, "section_name", "") or ""),
            section_name=str(getattr(chunk, "section_name", "") or ""),
            parent_section=str(getattr(chunk, "parent_section", "") or ""),
            title=str(getattr(chunk, "title", "") or ""),
            table_name=str(getattr(chunk, "table_name", "") or ""),
            chunk_type=str(getattr(chunk, "chunk_type", "") or "paragraph"),
            document_name=str(getattr(chunk, "document_name", "") or getattr(chunk, "doc_name", "") or ""),
            strategy_name=str(getattr(chunk, "strategy_name", "") or ""),
            document_id=str(getattr(chunk, "document_id", "") or ""),
            table_index=table_index,
            source_blocks=list(getattr(chunk, "source_blocks", None) or []),
        )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_citation_anchors(chunks: list[Any]) -> None:
    """Attach serialized citation anchors to each chunk before indexing."""
    for chunk in chunks:
        page_start = _optional_int(getattr(chunk, "page_start", None))
        if page_start is None:
            page_start = _optional_int(getattr(chunk, "page", None))
        page_end = _optional_int(getattr(chunk, "page_end", None))
        if page_end is None:
            page_end = _optional_int(getattr(chunk, "end_page", None))
        if page_end is None:
            page_end = page_start
        # Only write page fields when extraction/enrichment produced them.
        if page_start is not None:
            chunk.page = page_start
            chunk.page_start = page_start
            chunk.end_page = page_end
            chunk.page_end = page_end
        if not getattr(chunk, "document_name", ""):
            chunk.document_name = getattr(chunk, "doc_name", "") or ""
        anchor = CitationAnchor.from_chunk(chunk)
        chunk.citation_anchor = anchor.to_dict()


def parse_citation_anchor(raw: Any) -> dict[str, Any] | None:
    """Parse citation anchor from Weaviate property (dict or JSON string)."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return CitationAnchor.from_dict(raw).to_dict() if CitationAnchor.from_dict(raw) else None
    if isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
        except Exception:
            return None
        return CitationAnchor.from_dict(data).to_dict() if CitationAnchor.from_dict(data) else None
    return None
