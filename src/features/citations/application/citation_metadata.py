"""Citation metadata helpers shared across all chunking strategies."""

from __future__ import annotations

import re
from typing import Any

from src.features.chunking.application.chunking_service import (
    Chunk,
    _line_is_substantial_for_match,
    _locate_text_span,
    _lines_matching_chunk_text,
    _lines_overlapping_span,
    _merge_undersized_chunk_objects,
    _page_range_from_lines,
    _resolve_chunk_line_range,
    _section_leaf_label,
)
from src.features.document_processing.loaders.component_classification import COMPONENT_TABLE
from src.features.document_processing.utilities.text_utils import (
    _heading_body_from_numbered_text,
    _is_plausible_section_heading,
    _looks_like_instructional_heading,
    _looks_like_spaced_numbered_heading,
    _strip_leading_outline_number,
    auto_detect_sections,
    is_skip_heading,
)

_GENERIC_SECTIONS = frozenset({"document", "unknown", "", "introduction", "none"})
_MIN_ORPHAN_WORDS = 30
_NUMBERED_SECTION_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)*\.?)\s+(?P<title>.+)$")
_TABLE_CAPTION_RE = re.compile(
    r"^\s*table\s*(?P<num>\d+[A-Za-z]?)?\s*(?P<sep>[:.\-])?\s*(?P<caption>.*)$",
    re.IGNORECASE,
)
# QA / form tables often wrap "Test Case | Title" across lines; match collapsed text.
_TABLE_PIPE_TITLE_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:test\s*)?case\s*\|\s*(?P<title>[^\n|]+?)(?=\s*(?:\n|$)|(?:\s+(?:descr(?:iption)?|expected|actual|status)\b))",
)
_TABLE_LABEL_PIPE_TITLE_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?P<label>[A-Za-z][A-Za-z0-9 /_-]{0,40}?)\s*\|\s*(?P<title>[A-Z][^\n|]{2,120})",
)


def split_section_number_and_title(text: str) -> tuple[str, str]:
    """Split '5.5.3.1 Working Standard Qualification' into number and title."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "", ""
    match = _NUMBERED_SECTION_RE.match(cleaned)
    if match:
        return match.group("number").rstrip("."), match.group("title").strip()
    return cleaned, ""


def sanitize_section_label(value: str) -> str:
    """Reject generic section placeholders; recover structural headings from noisy OCR."""
    label = (value or "").strip()
    if not label:
        return ""
    if label.lower() in _GENERIC_SECTIONS:
        return ""
    # OCR often glues stamps around real headings (e.g. "B/Y REVISION HISTORY: gis y").
    structural = re.search(
        r"(?i)\b("
        r"revision\s+history|change\s+history|document\s+history|"
        r"table\s+of\s+contents|purpose|scope|references|"
        r"instructions?\s+for\s+[a-z][a-z\s]{2,40}"
        r")\b",
        label,
    )
    if structural:
        phrase = re.sub(r"\s+", " ", structural.group(1).strip())
        return phrase.title() if phrase.lower() != phrase.title().lower() else phrase.title()
    return label


def leaf_section_labels(path: str) -> tuple[str, str, str]:
    """Derive section_name, full path, and title from a hierarchical section path."""
    parts = [part.strip() for part in re.split(r"\s*>\s*", path or "") if part.strip()]
    if not parts:
        return "", "", ""
    leaf = parts[-1]
    number, title = split_section_number_and_title(leaf)
    section_path = " > ".join(parts)
    if number and title:
        # Hierarchical SOP ids (5.1.4) keep the number; a bare form digit joined to a
        # long instructional title should cite the title, not the digit alone.
        if "." in number.rstrip("."):
            section_name = number.rstrip(".")
        elif len(title.split()) >= 4:
            section_name = title
            parts = parts[:-1] + [title]
            section_path = " > ".join(parts)
        else:
            section_name = number.rstrip(".")
    else:
        section_name = number if number else leaf
    if not title and len(parts) > 1:
        _, parent_title = split_section_number_and_title(parts[-2])
        title = parent_title
    return section_name, section_path, title


def compute_page_range(blocks: list[Any]) -> tuple[int | None, int | None]:
    """Return start/end pages from block provenance only (never invent page 1)."""
    pages: set[int] = set()
    for block in blocks:
        raw = getattr(block, "page", None) if hasattr(block, "page") else None
        if raw is None or raw == "":
            continue
        try:
            pages.add(int(raw))
        except (TypeError, ValueError):
            continue
    if not pages:
        return None, None
    return min(pages), max(pages)


def infer_parent_section(section_path: str) -> str:
    parts = [part.strip() for part in re.split(r"\s*>\s*", section_path or "") if part.strip()]
    if len(parts) > 1:
        return " > ".join(parts[:-1])
    leaf = parts[0] if parts else (section_path or "").strip()
    match = re.match(r"^(?P<number>\d+(?:\.\d+)*)\.?\s+", leaf)
    if match:
        bits = [bit for bit in match.group("number").split(".") if bit]
        if len(bits) > 1:
            return ".".join(bits[:-1])
    return ""


def infer_title(blocks: list[Any], *, section_path: str = "") -> str:
    for block in blocks:
        component = str(getattr(block, "component_type", "") or "").lower()
        style = str(getattr(block, "style", "") or "").lower()
        if component in {"title", "subtitle"} or "title" in style or "heading 1" in style:
            text = str(getattr(block, "text", "") or "").strip()
            if text:
                return text
    parts = [part.strip() for part in re.split(r"\s*>\s*", section_path or "") if part.strip()]
    return parts[0] if parts else ""


def _table_caption_from_text(text: str) -> tuple[str, str]:
    """Return (table_number, caption_title) from table block text when present."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "", ""
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    num = ""
    caption = ""
    match = _TABLE_CAPTION_RE.match(first_line)
    if match:
        num = (match.group("num") or "").strip()
        caption = (match.group("caption") or "").strip()

    if not caption:
        # Prefer pipe titles such as "Test Case | Chunking Strategy Selection",
        # including when DOCX extraction wraps "Test" / "Case |" onto separate lines.
        collapsed = re.sub(r"[ \t]+", " ", cleaned)
        pipe = _TABLE_PIPE_TITLE_RE.search(collapsed)
        if pipe:
            caption = re.sub(r"\s+", " ", pipe.group("title")).strip(" :.-")
        else:
            for line in lines[1:8]:
                label_pipe = _TABLE_LABEL_PIPE_TITLE_RE.match(line)
                if not label_pipe:
                    continue
                label = re.sub(r"\s+", " ", label_pipe.group("label")).strip().lower()
                title = re.sub(r"\s+", " ", label_pipe.group("title")).strip(" :.-")
                if label in {"description", "expected result", "actual result", "status", "screenshot"}:
                    continue
                if len(title.split()) >= 2 or (title[:1].isupper() and len(title) >= 8):
                    caption = title
                    break
    return num, caption


def infer_table_name(block: Any, *, fallback_index: int = 1) -> str:
    text = str(getattr(block, "text", "") or "").strip()
    num, caption = _table_caption_from_text(text)
    if num and caption:
        return f"Table {num}: {caption}"
    if num:
        return f"Table {num}"
    if caption:
        return f"Table: {caption}"
    metadata = getattr(block, "metadata", {}) or {}
    table_index = metadata.get("table_index")
    if table_index is not None:
        return f"Table {table_index}"
    return f"Table {fallback_index}"


def table_caption_title(table_name: str) -> str:
    """Strip a leading 'Table N:' prefix, leaving the human caption when present."""
    cleaned = (table_name or "").strip()
    if not cleaned:
        return ""
    match = _TABLE_CAPTION_RE.match(cleaned)
    if match:
        caption = (match.group("caption") or "").strip(" :.-")
        if caption:
            return caption
    return ""


def _is_generic_section_label(value: str) -> bool:
    return (value or "").strip().lower() in _GENERIC_SECTIONS


def infer_section(
    blocks: list[Any],
    *,
    section_name: str | None = None,
    section_path: str | None = None,
) -> tuple[str, str, str, str]:
    """Infer section_name, section_path, parent_section, and title from blocks."""
    path = (section_path or section_name or "").strip()
    name = (section_name or "").strip()

    if _is_generic_section_label(path):
        path = ""
    if _is_generic_section_label(name):
        name = ""

    if not path and blocks:
        for block in blocks:
            text = str(getattr(block, "text", "") or "").strip()
            if re.match(r"^\d+(?:\.\d+)*\.?\s+\S", text):
                # Prefer instructional title body over a bare leading form number.
                if _looks_like_instructional_heading(text):
                    path = _strip_leading_outline_number(text)
                    name = path
                    break
                # Only adopt numbered text when it is a genuine heading candidate.
                if _heading_body_from_numbered_text(text) or _looks_like_spaced_numbered_heading(text):
                    path = text
                    name = sanitize_section_label(_section_leaf_label(text)) or text
                    break
                continue
            # Generic heading cues (not document-specific titles).
            if re.match(
                r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:instructions\s+for|procedure\s+for|note|revision\s+history)\b",
                text,
                re.IGNORECASE,
            ):
                path = _strip_leading_outline_number(text)
                name = path
                break
        if not path:
            sections = auto_detect_sections(blocks)
            for sec_name, _, sec_blocks in sections:
                if is_skip_heading(sec_name) or _is_generic_section_label(sec_name):
                    continue
                leaf = sec_name.split(">")[-1].strip() if ">" in sec_name else sec_name
                if _is_generic_section_label(leaf):
                    continue
                if sec_blocks is blocks or set(id(b) for b in sec_blocks) <= set(id(b) for b in blocks):
                    path_parts = [part.strip() for part in sec_name.split(">") if part.strip()]
                    path = " > ".join(path_parts) if path_parts else sec_name
                    name = path_parts[-1] if path_parts else sec_name
                    break
            if not path and len(sections) == 1:
                sec_name, _, _ = sections[0]
                if not _is_generic_section_label(sec_name):
                    path_parts = [part.strip() for part in sec_name.split(">") if part.strip()]
                    path = " > ".join(path_parts) if path_parts else sec_name
                    name = path_parts[-1] if path_parts else sec_name

    if not path:
        for block in blocks:
            component = str(getattr(block, "component_type", "") or "").lower()
            style = str(getattr(block, "style", "") or "").lower()
            text = str(getattr(block, "text", "") or "").strip()
            if not text or _is_generic_section_label(text):
                continue
            if component in {"title", "subtitle"} or "heading" in style:
                path = text
                name = sanitize_section_label(_section_leaf_label(text)) or text
                break
            if getattr(block, "bold", False) and _looks_like_instructional_heading(text):
                path = _strip_leading_outline_number(text)
                name = path
                break

    # Never invent section labels. Leave empty when extraction cannot determine one.
    name = (
        sanitize_section_label(name)
        or sanitize_section_label(_section_leaf_label(path))
        or sanitize_section_label(path)
    )
    if not name:
        return "", "", "", ""
    path = path if not _is_generic_section_label(path) else name
    section_name, section_path, inferred_title = leaf_section_labels(path if " > " in path else name)
    if sanitize_section_label(section_name):
        name = section_name
        path = section_path or path
    if not inferred_title and name and not re.match(r"^\d+(?:\.\d+)*\.?$", name):
        inferred_title = name
    return name, path, infer_parent_section(path), inferred_title


def build_source_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    """Summarize contributing blocks for citation traceability."""
    refs: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        line_number = getattr(block, "line_number", None)
        metadata = getattr(block, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        line_end = metadata.get("source_line_end") or line_number
        raw_page = getattr(block, "page", None)
        page: int | None
        try:
            page = int(raw_page) if raw_page is not None and raw_page != "" else None
        except (TypeError, ValueError):
            page = None
        full_text = str(getattr(block, "text", "") or "")
        ref: dict[str, Any] = {
            "index": index,
            "page": page,
            "line_number": line_number,
            "line_start": int(line_number) if line_number is not None else None,
            "line_end": int(line_end) if line_end is not None else None,
            "block_type": getattr(block, "block_type", "text"),
            "component_type": getattr(block, "component_type", "paragraph"),
            "text_preview": full_text[:240],
            # Longer text retained for precise supporting-block matching only.
            "text": full_text[:4000] if full_text else "",
        }
        for key in ("section_name", "section_path", "heading", "table_name", "source_block_id"):
            value = metadata.get(key) or getattr(block, key, None)
            if value not in (None, ""):
                ref[key] = value
        refs.append(ref)
    return refs


def build_source_blocks_from_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build source_blocks only from lines that actually contribute to the chunk."""
    refs: list[dict[str, Any]] = []
    substantial_present = any(
        _line_is_substantial_for_match(str(line.get("text") or "")) for line in lines
    )
    for index, line in enumerate(lines):
        full_text = str(line.get("text") or "")
        preview = full_text[:240]
        if not preview.strip():
            continue
        # Drop digit/noise tokens when real body lines are also present.
        if substantial_present and not _line_is_substantial_for_match(preview):
            continue
        line_number = line.get("line_number")
        line_end = line.get("line_end")
        if line_end is None:
            line_end = line_number
        raw_page = line.get("page")
        try:
            page = int(raw_page) if raw_page is not None and raw_page != "" else None
        except (TypeError, ValueError):
            page = None
        metadata = line.get("metadata") if isinstance(line.get("metadata"), dict) else {}
        ref: dict[str, Any] = {
            "index": index,
            "page": page,
            "line_number": line_number,
            "line_start": int(line_number) if line_number is not None else None,
            "line_end": int(line_end) if line_end is not None else None,
            "block_type": line.get("block_type", "text"),
            "component_type": line.get("component_type", "paragraph"),
            "text_preview": preview,
            "text": full_text[:4000] if full_text else "",
        }
        for key in ("section_name", "section_path", "heading", "table_name", "source_block_id"):
            value = metadata.get(key) or line.get(key)
            if value not in (None, ""):
                ref[key] = value
        refs.append(ref)
    for index, ref in enumerate(refs):
        ref["index"] = index
    return refs


def _blocks_to_lines_info(blocks: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    lines_info: list[dict[str, Any]] = []
    cursor = 0
    for block in blocks:
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        start = cursor
        end = start + len(text)
        raw_page = getattr(block, "page", None)
        try:
            page = int(raw_page) if raw_page is not None and raw_page != "" else None
        except (TypeError, ValueError):
            page = None
        lines_info.append(
            {
                "text": text,
                "page": page,
                "line_number": getattr(block, "line_number", None),
                "line_end": (getattr(block, "metadata", {}) or {}).get("source_line_end")
                or getattr(block, "line_number", None),
                "start": start,
                "end": end,
                "block_type": getattr(block, "block_type", "text"),
                "component_type": getattr(block, "component_type", "paragraph"),
                "metadata": dict(getattr(block, "metadata", {}) or {}),
            }
        )
        cursor = end + 1
    section_text = "\n".join(line["text"] for line in lines_info).strip()
    return section_text, lines_info


def map_chunk_text_to_blocks(
    chunk_text: str,
    blocks: list[Any],
    *,
    prefer_page: int | None = None,
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    """Map chunk text to source lines using offset matching first."""
    section_text, lines_info = _blocks_to_lines_info(blocks)
    if not section_text or not lines_info:
        return [], *compute_page_range(blocks)

    overlapping: list[dict[str, Any]] = []
    span = _locate_text_span(section_text, chunk_text)
    if span is not None:
        start_pos, end_pos = span
        overlapping = _lines_overlapping_span(lines_info, start_pos, end_pos)
    if not overlapping:
        overlapping = _lines_matching_chunk_text(
            chunk_text,
            lines_info,
            prefer_page=prefer_page,
        )
    if overlapping:
        return overlapping, *_page_range_from_lines(overlapping)
    return [], *compute_page_range(blocks)


def enrich_chunk_metadata(
    chunk: Chunk,
    blocks: list[Any],
    *,
    strategy_name: str,
    document_id: str = "",
    chunk_type: str | None = None,
    table_name: str | None = None,
    section_name: str | None = None,
    section_path: str | None = None,
) -> Chunk:
    """Attach deterministic citation metadata to a chunk from extraction provenance only."""
    sec_name, sec_path, parent, inferred_title = infer_section(
        blocks,
        section_name=section_name or chunk.section_name,
        section_path=section_path or chunk.section_path,
    )
    # Prefer content-based mapping over any provisional page on the chunk.
    overlapping, mapped_start, mapped_end = map_chunk_text_to_blocks(
        chunk.text or chunk.raw_text or "",
        blocks,
        prefer_page=None,
    )
    if overlapping:
        page_start, page_end = mapped_start, mapped_end
        if page_start is not None:
            line_start, line_end = _resolve_chunk_line_range(overlapping, chunk_page=page_start)
            if line_start is not None:
                chunk.line_start = line_start
            if line_end is not None:
                chunk.line_end = line_end
        chunk.source_blocks = build_source_blocks_from_lines(overlapping)
        chunk.page = page_start
        chunk.end_page = page_end
        chunk.page_start = page_start
        chunk.page_end = page_end
    else:
        # Mapping failed — keep creation-time provenance if present; never invent
        # page/line/source_blocks from unrelated section-wide blocks.
        existing_start = chunk.page_start if chunk.page_start is not None else chunk.page
        existing_end = chunk.page_end if chunk.page_end is not None else chunk.end_page
        if existing_start is not None:
            try:
                page_start = int(existing_start)
                page_end = int(existing_end if existing_end is not None else page_start)
            except (TypeError, ValueError):
                page_start, page_end = None, None
            chunk.page = page_start
            chunk.end_page = page_end
            chunk.page_start = page_start
            chunk.page_end = page_end
        else:
            # Leave unknown pages unset (null) rather than inventing page 1.
            chunk.page = mapped_start
            chunk.end_page = mapped_end
            chunk.page_start = mapped_start
            chunk.page_end = mapped_end
        chunk.source_blocks = []

    chunk.section_name = sec_name
    chunk.section_path = sec_path
    chunk.parent_section = parent
    chunk.strategy_name = strategy_name
    chunk.document_id = document_id or chunk.document_id
    chunk.document_name = chunk.document_name or chunk.doc_name
    chunk.title = (chunk.title or inferred_title or infer_title(blocks, section_path=sec_path) or sec_name).strip()

    resolved_type = (chunk_type or chunk.chunk_type or "paragraph").strip().lower() or "paragraph"
    # Default chunk_type is "paragraph"; allow upgrade when provenance shows a table.
    provisional_paragraph = resolved_type == "paragraph"
    has_table_signal = bool(
        int(chunk.table_count or 0) > 0
        or COMPONENT_TABLE in (chunk.content_types or [])
        or str(chunk.category or "").lower() == COMPONENT_TABLE
        or any(
            str(getattr(block, "component_type", "") or "").lower() == COMPONENT_TABLE
            or str(getattr(block, "block_type", "") or "").lower() == "table"
            for block in blocks
        )
    )
    if provisional_paragraph and has_table_signal:
        resolved_type = "table"
    chunk.chunk_type = resolved_type

    if resolved_type == "table" and blocks:
        table_block = next(
            (
                block
                for block in blocks
                if str(getattr(block, "component_type", "") or "").lower() == COMPONENT_TABLE
                or str(getattr(block, "block_type", "") or "").lower() == "table"
            ),
            blocks[0],
        )
        chunk.table_name = table_name or chunk.table_name or infer_table_name(table_block)
        chunk.table_count = max(int(chunk.table_count or 0), 1)
        if COMPONENT_TABLE not in (chunk.content_types or []):
            chunk.content_types = sorted(set(chunk.content_types or []) | {COMPONENT_TABLE})
        caption = table_caption_title(chunk.table_name)
        if caption:
            if not sanitize_section_label(chunk.section_name):
                chunk.section_name = caption
                if not sanitize_section_label(chunk.section_path):
                    chunk.section_path = caption
            if not (chunk.title or "").strip() or _is_generic_section_label(chunk.title):
                chunk.title = caption

    return chunk


def clamp_chunk_pages_to_document(chunk: Chunk, max_page: int | None) -> None:
    """Ensure citation pages never exceed the document's declared page count."""
    if max_page is None or max_page < 1:
        return
    raw_start = chunk.page_start if chunk.page_start is not None else chunk.page
    raw_end = chunk.page_end if chunk.page_end is not None else chunk.end_page
    if raw_start is None and raw_end is None:
        return
    page_start = int(raw_start if raw_start is not None else raw_end)
    page_end = int(raw_end if raw_end is not None else page_start)
    page_start = min(max(page_start, 1), max_page)
    page_end = min(max(page_end, 1), max_page)
    if page_end < page_start:
        page_end = page_start
    chunk.page = page_start
    chunk.page_start = page_start
    chunk.end_page = page_end
    chunk.page_end = page_end


def clamp_chunks_to_document_page_count(chunks: list[Chunk], max_page: int | None) -> None:
    for chunk in chunks:
        clamp_chunk_pages_to_document(chunk, max_page)


def merge_orphan_chunks(
    chunks: list[Chunk],
    *,
    min_words: int = _MIN_ORPHAN_WORDS,
    chunk_size: int = 150,
) -> list[Chunk]:
    """Merge undersized chunks into neighbors within the same section."""
    return _merge_undersized_chunk_objects(chunks, min_words, chunk_size)


def _section_path_label(sec_name: str) -> str:
    path_parts = [part.strip() for part in sec_name.split(">") if part.strip()]
    return " > ".join(path_parts) if path_parts else (sec_name or "").strip()


def _resolve_section_context_for_chunk(
    chunk: Chunk,
    blocks: list[Any],
    sections: list[tuple[str, int, list[Any]]],
    section_blocks: dict[str, list[Any]],
) -> tuple[list[Any], str | None, str | None]:
    """Pick the section that actually owns the chunk text (data-driven, not first heading)."""
    path = (chunk.section_path or chunk.section_name or "").strip()
    if path and path.lower() not in _GENERIC_SECTIONS and path in section_blocks:
        return section_blocks[path], path, path
    if path and path.lower() not in _GENERIC_SECTIONS:
        for key, sec_blocks in section_blocks.items():
            if path in key or key.endswith(path):
                return sec_blocks, key, key

    overlapping, _, _ = map_chunk_text_to_blocks(
        chunk.text or chunk.raw_text or "",
        blocks,
        prefer_page=None,
    )
    if not overlapping:
        return blocks, None, None

    overlap_keys = {
        (
            int(line["page"]) if line.get("page") is not None else None,
            int(line["line_number"]) if line.get("line_number") is not None else None,
            " ".join(str(line.get("text") or "").split())[:80],
        )
        for line in overlapping
    }

    best_path = ""
    best_blocks = blocks
    best_score = -1
    for sec_name, _, sec_blocks in sections:
        if is_skip_heading(sec_name) or _is_generic_section_label(sec_name):
            continue
        sec_path = _section_path_label(sec_name)
        leaf = sec_path.split(">")[-1].strip() if sec_path else ""
        if not _looks_like_instructional_heading(leaf) and not _is_plausible_section_heading(leaf):
            continue
        hits = 0
        for block in sec_blocks:
            key = (
                int(block.page) if getattr(block, "page", None) is not None else None,
                int(block.line_number) if getattr(block, "line_number", None) is not None else None,
                " ".join(str(getattr(block, "text", "") or "").split())[:80],
            )
            if key in overlap_keys:
                hits += 1
        if hits <= 0:
            continue
        score = hits * 100
        if _looks_like_instructional_heading(leaf):
            score += 1000
        score += min(len(leaf), 80)
        if score > best_score:
            best_score = score
            best_path = sec_path
            best_blocks = sec_blocks

    if best_score < 0:
        # Prefer original extraction blocks (keeps table/image component types + full text)
        # over synthetic line stubs when page/line provenance matches.
        matched_blocks: list[Any] = []
        seen: set[int] = set()
        for line in overlapping:
            page = line.get("page")
            line_number = line.get("line_number")
            for block in blocks:
                block_page = getattr(block, "page", None)
                block_line = getattr(block, "line_number", None)
                try:
                    same_page = (
                        page is None
                        or block_page is None
                        or int(page) == int(block_page)
                    )
                except (TypeError, ValueError):
                    same_page = page == block_page
                try:
                    same_line = (
                        line_number is None
                        or block_line is None
                        or int(line_number) == int(block_line)
                    )
                except (TypeError, ValueError):
                    same_line = line_number == block_line
                if not (same_page and same_line):
                    continue
                marker = id(block)
                if marker in seen:
                    continue
                seen.add(marker)
                matched_blocks.append(block)
                break
        if matched_blocks:
            return matched_blocks, None, None

        # Fall back to inferring from the overlapping source lines only.
        line_blocks = []
        for line in overlapping:
            line_blocks.append(
                type("TmpBlock", (), {
                    "text": line.get("text") or "",
                    "page": line.get("page"),
                    "line_number": line.get("line_number"),
                    "block_type": line.get("block_type", "text"),
                    "component_type": line.get("component_type", "paragraph"),
                    "metadata": dict(line.get("metadata") or {}),
                    "bold": False,
                    "style": "",
                    "heading_level": None,
                })()
            )
        return line_blocks or blocks, None, None

    leaf = best_path.split(">")[-1].strip() if best_path else ""
    return best_blocks, leaf or best_path or None, best_path or None


def finalize_chunks_for_citation(
    chunks: list[Chunk],
    blocks: list[Any],
    *,
    strategy_name: str,
    document_id: str = "",
    min_words: int = _MIN_ORPHAN_WORDS,
    chunk_size: int = 150,
) -> list[Chunk]:
    """Post-process chunks from any strategy with consistent citation metadata."""
    if not chunks:
        return []

    section_blocks: dict[str, list[Any]] = {}
    sections = auto_detect_sections(blocks)
    for sec_name, _, sec_blocks in sections:
        if is_skip_heading(sec_name) or _is_generic_section_label(sec_name):
            continue
        section_path = _section_path_label(sec_name)
        if section_path:
            section_blocks[section_path] = sec_blocks

    enriched: list[Chunk] = []
    for chunk in chunks:
        context_blocks, section_name, section_path = _resolve_section_context_for_chunk(
            chunk,
            blocks,
            sections,
            section_blocks,
        )
        enrich_chunk_metadata(
            chunk,
            context_blocks,
            strategy_name=strategy_name,
            document_id=document_id,
            chunk_type=chunk.chunk_type or None,
            table_name=chunk.table_name or None,
            section_name=section_name or chunk.section_name or None,
            section_path=section_path or chunk.section_path or None,
        )
        enriched.append(chunk)

    return merge_orphan_chunks(enriched, min_words=min_words, chunk_size=chunk_size)
