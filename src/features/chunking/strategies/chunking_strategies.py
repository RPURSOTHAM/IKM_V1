"""Apply repository chunking strategies to loaded document blocks."""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from src.features.document_processing.core.logger import log
from src.features.citations.application.citation_metadata import finalize_chunks_for_citation
from src.features.document_processing.loaders.component_classification import (
    COMPONENT_FOOTER,
    COMPONENT_HEADER,
    COMPONENT_IMAGE,
    COMPONENT_SUBTITLE,
    COMPONENT_TABLE,
    COMPONENT_TITLE,
    block_component_type,
    blocks_for_chunking,
    image_block_chunk_text,
    image_block_content_types,
    is_image_block,
)
from src.features.document_processing.loaders.pdf_visual_content import image_block_extraction_metadata
from src.features.chunking.application.chunking_service import (
    Chunk,
    _prepare_blocks_for_section_detection,
    _append_chunks_from_blocks,
    _strip_image_placeholders,
    _table_row_looks_like_header,
    chunk_section_text,
    chunk_document_blocks,
)
from src.features.document_processing.utilities.text_utils import (
    auto_detect_sections,
    is_skip_heading,
    split_into_paragraphs,
    split_into_sentences,
)
from src.features.chunking.domain.chunking_strategy import resolve_chunking_strategy
from src.shared.semantic.models import Coordinates


def _blocks_to_text(blocks: list[Any]) -> str:
    cleaned_blocks: list[str] = []
    for block in blocks:
        if is_image_block(block):
            text = image_block_chunk_text(block)
            if text:
                cleaned_blocks.append(text)
            continue
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        block_type = str(getattr(block, "block_type", "text") or "text").lower()
        component_type = str(getattr(block, "component_type", "") or "").lower()
        if block_type == "table" or component_type == "table":
            rows = [line.strip() for line in text.splitlines() if line.strip()]
            if rows and rows[0].lower().startswith("table"):
                rows = rows[1:]
            if rows and _table_row_looks_like_header(rows[0]):
                rows = rows[1:]
            text = "\n".join(rows).strip()
        text, _ = _strip_image_placeholders(text)
        if text:
            cleaned_blocks.append(text)
    return "\n".join(cleaned_blocks)


def _words(text: str) -> list[str]:
    return text.split()


def _make_chunk(doc_name: str, page: int | None, section_name: str, section_path: str, text: str) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        doc_name=doc_name,
        page=page,
        section_name=section_name,
        section_path=section_path or section_name,
        text=text,
    )


def _make_chunk_from_image_block(
    block: Any,
    doc_name: str,
    text: str,
    *,
    strategy_name: str,
    document_id: str = "",
) -> Chunk:
    metadata = image_block_extraction_metadata(block)
    content_types = image_block_content_types(block)
    chunk_type = "table" if "image_table" in content_types else "image_caption"
    page = int(getattr(block, "page", 0) or metadata.get("page_number") or 0) or None
    chunk = Chunk(
        id=str(uuid.uuid4()),
        doc_name=doc_name,
        document_id=document_id,
        page=page,
        section_name="",
        section_path="",
        text=text,
        strategy_name=strategy_name,
        chunk_type=chunk_type,
        content_types=content_types,
        image_count=1,
        extraction_metadata=metadata,
        coordinates=_block_coordinates(block),
        source_blocks=[
            {
                "page": page,
                "block_type": "image",
                "image_index": metadata.get("image_index"),
                "content_type": metadata.get("content_type"),
                "content_source": metadata.get("content_source"),
            }
        ],
    )
    return chunk


def _page_for_blocks(blocks: list[Any]) -> int | None:
    """Use the first available extracted page; never invent page 1."""
    for block in blocks:
        raw = getattr(block, "page", None)
        if raw is None or raw == "":
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def chunk_fixed_overlap_based(
    blocks: list[Any],
    doc_name: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_content_words: int,
) -> list[Chunk]:
    text = _blocks_to_text(blocks)
    page = _page_for_blocks(blocks)
    chunks: list[Chunk] = []
    for raw in chunk_section_text(
        text,
        chunk_size=chunk_size,
        min_words=min_content_words,
        overlap_sentences=max(chunk_overlap, 0),
    ):
        if raw.strip():
            chunks.append(_make_chunk(doc_name, page, "", "", raw))
    return chunks


def chunk_sentence_based(
    blocks: list[Any],
    doc_name: str,
    *,
    max_sentences_per_chunk: int,
    overlap_sentences: int,
    min_content_words: int,
) -> list[Chunk]:
    text = _blocks_to_text(blocks)
    page = _page_for_blocks(blocks)
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    chunks: list[Chunk] = []
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences_per_chunk, len(sentences))
        candidate = " ".join(sentences[start:end]).strip()
        if len(candidate.split()) >= min_content_words:
            chunks.append(_make_chunk(doc_name, page, "", "", candidate))
        if end >= len(sentences):
            break
        overlap = min(overlap_sentences, max(end - start - 1, 0))
        start = end - overlap if overlap else end
    return chunks


def chunk_paragraph_based(
    blocks: list[Any],
    doc_name: str,
    *,
    max_paragraphs_per_chunk: int,
    chunk_size: int,
    min_content_words: int,
    document_id: str = "",
) -> list[Chunk]:
    """Merge whole paragraph blocks; never split a paragraph and never group by heading."""
    max_paras = max(1, int(max_paragraphs_per_chunk))
    word_budget = max(1, int(chunk_size))
    min_words = max(1, int(min_content_words))
    heading_types = {
        COMPONENT_TITLE,
        COMPONENT_SUBTITLE,
        COMPONENT_HEADER,
        COMPONENT_FOOTER,
        COMPONENT_IMAGE,
    }
    chunks: list[Chunk] = []
    batch: list[Any] = []
    batch_words = 0

    def _flush() -> None:
        nonlocal batch, batch_words
        if not batch:
            return
        text = "\n\n".join(
            str(getattr(block, "text", "") or "").strip()
            for block in batch
            if str(getattr(block, "text", "") or "").strip()
        ).strip()
        if text and len(text.split()) >= min_words:
            chunk = _make_chunk(doc_name, _page_for_blocks(batch), "", "", text)
            chunk.strategy_name = "paragraph-based"
            chunk.chunk_type = "paragraph"
            chunks.append(chunk)
        batch = []
        batch_words = 0

    for block in blocks:
        if is_image_block(block):
            text = image_block_chunk_text(block)
            if text:
                image_chunk = _make_chunk_from_image_block(
                    block,
                    doc_name,
                    text,
                    strategy_name="paragraph-based",
                    document_id=document_id,
                )
                if len(text.split()) >= 1:
                    chunks.append(image_chunk)
            continue
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        ctype = block_component_type(block)
        if ctype in heading_types:
            continue
        if ctype == COMPONENT_TABLE:
            _flush()
            table_chunk = _make_chunk(doc_name, _page_for_blocks([block]), "", "", text)
            table_chunk.strategy_name = "paragraph-based"
            table_chunk.chunk_type = "table"
            if len(text.split()) >= min_words:
                chunks.append(table_chunk)
            continue
        words = len(text.split())
        if batch and (len(batch) >= max_paras or batch_words + words > word_budget):
            _flush()
        batch.append(block)
        batch_words += words
    _flush()
    if chunks:
        return chunks
    # Fallback for documents whose blocks are not classified as paragraphs.
    text = _blocks_to_text(blocks)
    page = _page_for_blocks(blocks)
    paragraphs = [p.strip() for p in split_into_paragraphs(text) if p.strip()]
    batch_text: list[str] = []
    batch_words = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if batch_text and (len(batch_text) >= max_paras or batch_words + words > word_budget):
            candidate = "\n\n".join(batch_text).strip()
            if len(candidate.split()) >= min_words:
                chunk = _make_chunk(doc_name, page, "", "", candidate)
                chunk.strategy_name = "paragraph-based"
                chunk.chunk_type = "paragraph"
                chunks.append(chunk)
            batch_text = []
            batch_words = 0
        batch_text.append(paragraph)
        batch_words += words
    if batch_text:
        candidate = "\n\n".join(batch_text).strip()
        if len(candidate.split()) >= min_words:
            chunk = _make_chunk(doc_name, page, "", "", candidate)
            chunk.strategy_name = "paragraph-based"
            chunk.chunk_type = "paragraph"
            chunks.append(chunk)
    return chunks


def chunk_sliding_window(
    blocks: list[Any],
    doc_name: str,
    *,
    window_size: int,
    step_size: int,
    min_content_words: int,
) -> list[Chunk]:
    words = _words(_blocks_to_text(blocks))
    page = _page_for_blocks(blocks)
    if not words:
        return []
    step = max(1, min(step_size, window_size - 1))
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + window_size]
        if not window:
            break
        candidate = " ".join(window).strip()
        if len(candidate.split()) >= min_content_words:
            chunks.append(_make_chunk(doc_name, page, "", "", candidate))
        if start + window_size >= len(words):
            break
    return chunks


def chunk_hierarchical(
    blocks: list[Any],
    doc_name: str,
    *,
    chunk_size: int,
    overlap_sentences: int,
    min_content_words: int,
    max_heading_depth: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = auto_detect_sections(blocks)
    for sec_name, _, sec_blocks in sections:
        # A document without detectable headings still has valid body content.
        # Only discard an explicitly detected skip-heading, not the empty path
        # returned for ordinary unheaded paragraphs.
        if sec_name and is_skip_heading(sec_name):
            continue
        path_parts = [part.strip() for part in sec_name.split(">") if part.strip()]
        if len(path_parts) > max_heading_depth:
            path_parts = path_parts[:max_heading_depth]
        section_path = " > ".join(path_parts) if path_parts else sec_name
        _append_chunks_from_blocks(
            chunks=chunks,
            doc_name=doc_name,
            section_name=path_parts[-1] if path_parts else sec_name,
            section_path=section_path,
            sec_blocks=sec_blocks,
            chunk_size=chunk_size,
            min_content_words=min_content_words,
            overlap_sentences=overlap_sentences,
        )
    return chunks


def chunk_semantic(
    blocks: list[Any],
    doc_name: str,
    *,
    chunk_size: int,
    similarity_threshold: float,
    max_sentences_per_chunk: int,
    min_content_words: int,
    embed_fn: Callable[[list[str]], np.ndarray] | None,
) -> list[Chunk]:
    text = _blocks_to_text(blocks)
    page = _page_for_blocks(blocks)
    sentences = [s.strip() for s in split_into_sentences(text) if s.strip()]
    if not sentences:
        return []
    if embed_fn is None:
        log.warning("Semantic chunking requested without embeddings; falling back to sentence-based chunking.")
        return chunk_sentence_based(
            blocks,
            doc_name,
            max_sentences_per_chunk=max_sentences_per_chunk,
            overlap_sentences=0,
            min_content_words=min_content_words,
        )

    vectors = embed_fn(sentences)
    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]
    for idx in range(1, len(sentences)):
        prev = vectors[idx - 1]
        cur = vectors[idx]
        denom = float(np.linalg.norm(prev) * np.linalg.norm(cur))
        similarity = float(np.dot(prev, cur) / denom) if denom else 0.0
        current_words = sum(len(s.split()) for s in current)
        if (
            similarity >= similarity_threshold
            and len(current) < max_sentences_per_chunk
            and current_words < chunk_size
        ):
            current.append(sentences[idx])
        else:
            groups.append(current)
            current = [sentences[idx]]
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    for group in groups:
        candidate = " ".join(group).strip()
        if len(candidate.split()) >= min_content_words:
            chunks.append(_make_chunk(doc_name, page, "", "", candidate))
    return chunks


def _semantic_chunk_id(
    *,
    document_id: str,
    doc_name: str,
    ordinal: int,
    section_path: str,
    page: int,
    text: str,
) -> str:
    identity = document_id or doc_name
    value = f"{identity}|{ordinal}|{section_path}|{page}|{text.strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def _block_page(block: Any) -> int | None:
    raw = getattr(block, "page", None)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _block_coordinates(block: Any) -> list[Coordinates]:
    metadata = dict(getattr(block, "metadata", {}) or {})
    bounds = metadata.get("bounds") or metadata.get("bbox")
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
        try:
            return [
                Coordinates(
                    page=_block_page(block),
                    x0=float(bounds[0]),
                    y0=float(bounds[1]),
                    x1=float(bounds[2]),
                    y1=float(bounds[3]),
                )
            ]
        except (TypeError, ValueError):
            return []
    try:
        x0 = metadata.get("x0")
        top = metadata.get("top")
        x1 = metadata.get("x1")
        bottom = metadata.get("bottom")
        if None not in (x0, top, x1, bottom):
            return [
                Coordinates(
                    page=_block_page(block),
                    x0=float(x0),
                    y0=float(top),
                    x1=float(x1),
                    y1=float(bottom),
                )
            ]
    except (TypeError, ValueError):
        return []
    return []


@lru_cache(maxsize=1)
def _semantic_enrichment_services() -> tuple[Any, Any, Any]:
    from src.features.security.dlp.ner_detector import NERDetector
    from src.features.security.dlp.sensitive_data_detector import ComplianceScanner
    from src.features.security.classification.topic_classifier import TopicClassifier

    return TopicClassifier(), NERDetector(), ComplianceScanner()


def _enrich_semantic_chunk(chunk: Chunk) -> None:
    """Reuse existing enrichment implementations for required metadata fields."""
    try:
        topic_classifier, ner_detector, compliance_scanner = _semantic_enrichment_services()
        topics = topic_classifier.classify_topics(chunk.text).get("topics") or []
        if topics:
            chunk.topic = str(topics[0].get("topic") or "General")
    except Exception:
        ner_detector = None
        compliance_scanner = None
        chunk.topic = chunk.topic or "General"

    try:
        detections = ner_detector.detect_entities(chunk.text) if ner_detector is not None else []
        chunk.entities = list(
            dict.fromkeys(
                str(item.get("matched_value") or "").strip()
                for item in detections
                if str(item.get("matched_value") or "").strip()
            )
        )
    except Exception:
        chunk.entities = []

    try:
        detections = compliance_scanner.scan_text(chunk.text) if compliance_scanner is not None else []
        rank = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        chunk.sensitivity = max(
            (str(item.get("severity") or "low").lower() for item in detections),
            key=lambda value: rank.get(value, 0),
            default="low",
        )
    except Exception:
        chunk.sensitivity = "unknown"


def chunk_semantic_hierarchy(
    blocks: list[Any],
    doc_name: str,
    *,
    document_id: str = "",
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    similarity_threshold: float = 0.72,
) -> list[Chunk]:
    """Chunk semantic elements without crossing structural hierarchy boundaries.

    The primary units are paragraphs, tables, and image captions nested under
    section/subsection headings. Fixed token windows are deliberately not used.
    """
    chunks: list[Chunk] = []
    heading_stack: list[str] = []

    for block in blocks:
        text = str(getattr(block, "text", "") or "").strip()
        component = block_component_type(block)
        metadata = dict(getattr(block, "metadata", {}) or {})
        page = _block_page(block)
        line = int(getattr(block, "line_number", 0) or 0) or None

        if component in {COMPONENT_HEADER, COMPONENT_FOOTER} or bool(
            getattr(block, "is_toc_entry", False)
        ):
            continue

        if component in {COMPONENT_TITLE, COMPONENT_SUBTITLE} or getattr(block, "heading_level", None):
            if not text:
                continue
            level = int(
                getattr(block, "heading_level", 0)
                or (1 if component == COMPONENT_TITLE else 2)
            )
            level = max(1, level)
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(text)
            continue

        element_type = "paragraph"
        content_types = ["text"]
        table_count = 0
        image_count = 0
        candidate = text

        if component == COMPONENT_TABLE or str(getattr(block, "block_type", "")).lower() == "table":
            element_type = "table"
            content_types = ["table"]
            table_count = 1
        elif component == COMPONENT_IMAGE or is_image_block(block):
            candidate = image_block_chunk_text(block)
            if not candidate:
                candidate, _ = _strip_image_placeholders(text)
            if not candidate:
                continue
            content_types = image_block_content_types(block)
            element_type = "image_table" if "image_table" in content_types else "image_caption"
            image_count = 1
        else:
            candidate, _ = _strip_image_placeholders(candidate)
            if not candidate:
                continue

        section_path = " > ".join(heading_stack) if heading_stack else ""
        heading = heading_stack[-1] if heading_stack else ""
        ordinal = len(chunks)
        chunk = Chunk(
            id=_semantic_chunk_id(
                document_id=document_id,
                doc_name=doc_name,
                ordinal=ordinal,
                section_path=section_path,
                page=page,
                text=candidate,
            ),
            document_id=document_id,
            doc_name=doc_name,
            page=page,
            section_name=heading,
            heading=heading,
            section_path=section_path,
            text=candidate,
            line_start=line,
            line_end=line,
            content_types=content_types,
            table_count=table_count,
            image_count=image_count,
            coordinates=_block_coordinates(block),
            source=doc_name,
            extraction_metadata={
                **(
                    image_block_extraction_metadata(block)
                    if is_image_block(block)
                    else metadata
                ),
                "semantic_element_type": element_type,
                "hierarchy": list(heading_stack),
                "similarity_threshold": similarity_threshold,
                "semantic_embeddings_available": embed_fn is not None,
            },
        )
        if is_image_block(block):
            chunk.chunk_type = "table" if "image_table" in content_types else "image_caption"
        _enrich_semantic_chunk(chunk)
        chunks.append(chunk)

    return chunks


def chunk_document_with_strategy(
    blocks: list[Any],
    doc_name: str,
    *,
    strategy: str | None,
    chunk_size: int,
    overlap_sentences: int,
    min_content_words: int,
    chunking_config: dict[str, Any] | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    document_id: str = "",
    citation_retainment: bool = True,
) -> list[Chunk]:
    normalized = resolve_chunking_strategy(strategy)
    config = dict(chunking_config or {})
    min_words = int(config.get("min_content_words", min_content_words))

    if normalized == "semantic-hierarchy":
        # Unlike section-based chunking, this strategy maintains its heading
        # stack directly.  Clean PDF artifacts before a block can enter that
        # stack, otherwise a title fragment or running header contaminates all
        # following paragraph and table chunks.
        blocks = _prepare_blocks_for_section_detection(blocks_for_chunking(blocks))
        if not blocks:
            return []
        chunks = chunk_semantic_hierarchy(
            blocks,
            doc_name,
            document_id=document_id,
            embed_fn=embed_fn,
            similarity_threshold=float(config.get("similarity_threshold", 0.72)),
        )
        if not citation_retainment:
            return chunks
        return finalize_chunks_for_citation(
            chunks,
            blocks,
            strategy_name=normalized,
            document_id=document_id,
            min_words=min_words,
            chunk_size=chunk_size,
        )

    blocks = blocks_for_chunking(blocks)
    if not blocks:
        return []

    if normalized == "section-based":
        chunks = chunk_document_blocks(
            blocks,
            doc_name=doc_name,
            chunk_size=chunk_size,
            min_content_words=min_words,
            overlap_sentences=overlap_sentences,
        )
    elif normalized == "fixed-overlap-based":
        chunks = chunk_fixed_overlap_based(
            blocks,
            doc_name,
            chunk_size=chunk_size,
            chunk_overlap=overlap_sentences,
            min_content_words=min_words,
        )
    elif normalized == "sentence-based":
        chunks = chunk_sentence_based(
            blocks,
            doc_name,
            max_sentences_per_chunk=int(config.get("max_sentences_per_chunk", 8)),
            overlap_sentences=overlap_sentences,
            min_content_words=min_words,
        )
    elif normalized == "paragraph-based":
        chunks = chunk_paragraph_based(
            blocks,
            doc_name,
            max_paragraphs_per_chunk=int(config.get("max_paragraphs_per_chunk", 3)),
            chunk_size=chunk_size,
            min_content_words=min_words,
            document_id=document_id,
        )
    elif normalized == "hierarchical":
        # Hierarchical chunking derives every later section path from its
        # heading stack.  Apply the same PDF margin/title-fragment cleaning
        # used by the other hierarchy-aware strategies before a block can
        # become a heading.  Otherwise text such as "5 OF TREND)" or a
        # running header can be inherited by subsequent table chunks.
        blocks = _prepare_blocks_for_section_detection(blocks)
        if not blocks:
            return []
        chunks = chunk_hierarchical(
            blocks,
            doc_name,
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences,
            min_content_words=min_words,
            max_heading_depth=int(config.get("max_heading_depth", 4)),
        )
    elif normalized == "semantic":
        chunks = chunk_semantic(
            blocks,
            doc_name,
            chunk_size=chunk_size,
            similarity_threshold=float(config.get("similarity_threshold", 0.72)),
            max_sentences_per_chunk=int(config.get("max_sentences_per_chunk", 20)),
            min_content_words=min_words,
            embed_fn=embed_fn,
        )
    elif normalized == "sliding-window":
        chunks = chunk_sliding_window(
            blocks,
            doc_name,
            window_size=chunk_size,
            step_size=overlap_sentences or max(1, chunk_size // 4),
            min_content_words=min_words,
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {normalized}")

    if not citation_retainment:
        for chunk in chunks:
            chunk.strategy_name = normalized
            chunk.document_id = document_id
            if not chunk.document_name:
                chunk.document_name = doc_name
        return chunks

    # All strategies (including section-based) need citation finalize so
    # source_blocks, table metadata, and page/line mapping stay consistent.
    return finalize_chunks_for_citation(
        chunks,
        blocks,
        strategy_name=normalized,
        document_id=document_id,
        min_words=min_words,
        chunk_size=chunk_size,
    )
