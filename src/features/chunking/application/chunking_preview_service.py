"""Preview chunking for arbitrary text without document ingestion."""

from __future__ import annotations

import re
from typing import Any

from src.features.chunking.domain.chunking_strategy import resolve_chunking_strategy
from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy
from src.features.document_processing.loaders.loader import Block
from src.features.repositories.configuration.chunking_strategy_catalog import (
    default_chunking_config,
    default_settings_params,
    get_chunking_strategy,
)

_HEADING_STYLE_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?)\s+)?[A-Z][A-Za-z0-9 /,&()\-]{2,80}$"
)
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")


def text_to_preview_blocks(text: str) -> list[Block]:
    """Convert plain text into Block units suitable for all chunking strategies."""
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    blocks: list[Block] = []
    line_number = 0
    page = 1
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line_number += 1
        style = ""
        heading_level = None
        bold = False
        if _looks_like_preview_heading(line):
            style = "Heading 1"
            heading_level = 1
            bold = True
        blocks.append(
            Block(
                text=line,
                page=page,
                line_number=line_number,
                style=style,
                bold=bold,
                heading_level=heading_level,
                block_type="text",
                component_type="paragraph",
            )
        )
    return blocks


def _looks_like_preview_heading(line: str) -> bool:
    if len(line) > 120 or len(line.split()) > 14:
        return False
    if line.endswith((".", "?", "!")):
        return False
    if _NUMBERED_HEADING_RE.match(line):
        return True
    if line.isupper() and 2 <= len(line.split()) <= 8:
        return True
    return bool(_HEADING_STYLE_RE.match(line)) and line == line.title()


def _resolve_preview_parameters(
    strategy_id: str,
    *,
    chunk_size: int | None,
    chunk_overlap: int | None,
    min_content_words: int | None,
    chunking_config: dict[str, Any] | None,
) -> tuple[int, int, int, dict[str, Any]]:
    defaults = default_settings_params(strategy_id)
    config = dict(default_chunking_config(strategy_id))
    if chunking_config:
        config.update({k: v for k, v in chunking_config.items() if v is not None})

    resolved_size = int(chunk_size if chunk_size is not None else defaults.get("chunk_size", 512))
    resolved_overlap = int(
        chunk_overlap if chunk_overlap is not None else defaults.get("chunk_overlap", 0)
    )
    if min_content_words is not None:
        resolved_min = int(min_content_words)
    else:
        resolved_min = int(config.get("min_content_words") or defaults.get("min_content_words") or 20)
    # Explicit resolved min always wins over catalog defaults inside chunking_config.
    config["min_content_words"] = resolved_min
    return resolved_size, resolved_overlap, resolved_min, config


def preview_chunking(
    *,
    strategy: str,
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_content_words: int | None = None,
    chunking_config: dict[str, Any] | None = None,
    document_name: str = "preview.txt",
    citation_retainment: bool = True,
) -> dict[str, Any]:
    """Chunk custom text with a catalog strategy and return a JSON-serializable preview."""
    strategy_id = resolve_chunking_strategy(strategy)
    # Validate against catalog (raises ValueError for unknown ids).
    get_chunking_strategy(strategy_id)

    size, overlap, min_words, config = _resolve_preview_parameters(
        strategy_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_content_words=min_content_words,
        chunking_config=chunking_config,
    )
    blocks = text_to_preview_blocks(text)
    if not blocks:
        return {
            "strategy_id": strategy_id,
            "document_name": document_name or "preview.txt",
            "chunk_count": 0,
            "parameters": {
                "chunk_size": size,
                "chunk_overlap": overlap,
                "min_content_words": min_words,
                "chunking_config": config,
                "citation_retainment": citation_retainment,
            },
            "chunks": [],
        }

    chunks = chunk_document_with_strategy(
        blocks,
        document_name or "preview.txt",
        strategy=strategy_id,
        chunk_size=size,
        overlap_sentences=overlap,
        min_content_words=min_words,
        chunking_config=config,
        document_id="preview",
        citation_retainment=citation_retainment,
    )

    preview_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text_value = str(getattr(chunk, "text", "") or "")
        preview_chunks.append(
            {
                "index": index,
                "chunk_id": str(getattr(chunk, "id", "") or ""),
                "text": text_value,
                "word_count": len(text_value.split()),
                "page": getattr(chunk, "page", None),
                "page_end": getattr(chunk, "end_page", None) or getattr(chunk, "page_end", None),
                "section_name": str(getattr(chunk, "section_name", "") or ""),
                "section_path": str(getattr(chunk, "section_path", "") or ""),
                "parent_section": str(getattr(chunk, "parent_section", "") or ""),
                "title": str(getattr(chunk, "title", "") or ""),
                "chunk_type": str(getattr(chunk, "chunk_type", "") or "paragraph"),
                "table_name": str(getattr(chunk, "table_name", "") or ""),
                "line_start": getattr(chunk, "line_start", None),
                "line_end": getattr(chunk, "line_end", None),
                "strategy_name": str(getattr(chunk, "strategy_name", "") or strategy_id),
                "source_blocks": list(getattr(chunk, "source_blocks", None) or []),
            }
        )

    return {
        "strategy_id": strategy_id,
        "document_name": document_name or "preview.txt",
        "chunk_count": len(preview_chunks),
        "parameters": {
            "chunk_size": size,
            "chunk_overlap": overlap,
            "min_content_words": min_words,
            "chunking_config": config,
            "citation_retainment": citation_retainment,
        },
        "chunks": preview_chunks,
    }
