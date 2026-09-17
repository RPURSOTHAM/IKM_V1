"""Resolve deterministic citations from retrieved chunk metadata only."""

from __future__ import annotations

import re
from typing import Any

from src.features.citations.application.citation_anchor_builder import parse_citation_anchor
from src.features.citations.application.citation_metadata import (
    sanitize_section_label,
    split_section_number_and_title,
)
from src.features.observability.audit.application.audit_decorator import audit_event
from src.features.observability.metrics.application.metrics_decorator import capture_metric


_GENERIC_SECTIONS = frozenset({"", "document", "unknown", "introduction", "none"})
_HEADING_HINT_RE = re.compile(
    r"(?i)^\s*(?:"
    r"\d+(?:\.\d+)*\.?\s+\S|"
    r"revision\s+history|change\s+history|document\s+history|"
    r"table\s+of\s+contents|purpose|scope|references|annex(?:ure)?|"
    r"procedure|instructions?\s+for"
    r")"
)
_FILENAME_TOKEN_RE = re.compile(
    r"(?i)\b([A-Za-z0-9][\w.\-]{0,240}\.(?:pdf|docx?|pptx?|txt|html?|xlsx?|csv))\b"
)


def citation_quality_score(props: dict[str, Any]) -> float:
    """Score how citation-complete a chunk is (0.0–1.0)."""
    anchor = parse_citation_anchor(props.get("citation_anchor")) or {}
    score = 0.0
    if props.get("page") or anchor.get("start_page"):
        score += 0.25
    section = _resolved_section(props, anchor)
    if section:
        score += 0.2
    parent = str(props.get("parent_section") or anchor.get("parent_section") or "").strip()
    if parent:
        score += 0.15
    title = _resolved_title(props, anchor, section)
    if title and title.lower() not in _GENERIC_SECTIONS:
        score += 0.15
    if props.get("line_start") is not None or anchor.get("start_line") is not None:
        score += 0.1
    if str(props.get("chunk_type") or anchor.get("chunk_type") or "") == "table":
        if props.get("table_name") or anchor.get("table_name"):
            score += 0.15
    elif props.get("table_name") or anchor.get("table_name"):
        score += 0.05

    text = str(props.get("text") or "")
    if len(text.split()) < 30:
        score *= 0.5
    if section.lower() in _GENERIC_SECTIONS:
        score *= 0.25
    return min(score, 1.0)


def _resolved_section(props: dict[str, Any], anchor: dict[str, Any]) -> str:
    section = sanitize_section_label(str(props.get("section_name") or anchor.get("section_name") or ""))
    if section:
        return section
    section_path = str(props.get("section_path") or anchor.get("section_path") or "").strip()
    if section_path:
        leaf = section_path.split(">")[-1].strip()
        number, _ = split_section_number_and_title(leaf)
        return sanitize_section_label(number or leaf)
    return ""


def _resolved_title(props: dict[str, Any], anchor: dict[str, Any], section: str) -> str:
    title = str(props.get("title") or anchor.get("title") or "").strip()
    if title and title.lower() not in _GENERIC_SECTIONS and title != section:
        return title
    section_path = str(props.get("section_path") or anchor.get("section_path") or "").strip()
    if section_path:
        leaf = section_path.split(">")[-1].strip()
        _, leaf_title = split_section_number_and_title(leaf)
        if leaf_title:
            return leaf_title
    return ""


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _block_search_text(block: dict[str, Any]) -> str:
    parts = [
        str(block.get("text") or ""),
        str(block.get("text_preview") or ""),
        str(block.get("section_name") or ""),
        str(block.get("heading") or ""),
        str(block.get("table_name") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _evidence_needles(
    *,
    evidence_terms: list[str] | None = None,
    query_text: str | None = None,
    answer_text: str | None = None,
) -> list[str]:
    """Build match needles from query/answer/evidence — never invent document-specific tokens."""
    needles: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        cleaned = str(token or "").strip().lower()
        if not cleaned or cleaned in seen:
            return
        if cleaned.isdigit():
            if len(cleaned) < 4:
                return
        elif len(cleaned) < 3:
            return
        seen.add(cleaned)
        needles.append(cleaned)

    for term in evidence_terms or []:
        _add(term)

    for blob in (answer_text, query_text):
        if not blob:
            continue
        # Strip filename mentions so they don't bias block selection.
        scrubbed = _FILENAME_TOKEN_RE.sub(" ", str(blob))
        for token in re.findall(r"[A-Za-z0-9]+", scrubbed.lower()):
            _add(token)
    return needles


def _score_block_against_needles(block: dict[str, Any], needles: list[str]) -> float:
    haystack = _block_search_text(block).lower()
    if not haystack:
        return 0.0
    score = 0.0
    for needle in needles:
        if needle in haystack:
            weight = 3.0 if needle.isdigit() or len(needle) >= 8 else 1.0
            score += weight
    return score


def _score_block_for_query(
    block: dict[str, Any],
    *,
    needles: list[str],
    query_text: str | None = None,
) -> float:
    """Score a source block for citation selection (needles + generic number/code cues)."""
    score = _score_block_against_needles(block, needles)
    haystack = _block_search_text(block)
    query = str(query_text or "").lower()
    # Questions asking for a number/code/id often match a digit cell whose OCR
    # omitted the column header words — boost long digit tokens generically.
    if re.search(r"\b(number|code|identifier|id|control)\b", query):
        if re.search(r"\b\d{4,}\b", haystack):
            score += 5.0
        if re.search(r"(?i)\brevision\s+history\b", haystack):
            score += 2.0
    return score


def _is_heading_like_block(block: dict[str, Any]) -> bool:
    text = str(block.get("heading") or block.get("text_preview") or block.get("text") or "").strip()
    if not text:
        return False
    component = str(block.get("component_type") or "").lower()
    if component in {"title", "subtitle", "heading"} or "heading" in component:
        return True
    # Table/body cells are never treated as section headings.
    block_type = str(block.get("block_type") or "").lower()
    if block_type == "table" or component == "table":
        return False
    if len(text) > 160:
        return False
    # Digit-heavy rows (e.g. control numbers) are not headings.
    if sum(ch.isdigit() for ch in text) >= 4:
        return False
    if _HEADING_HINT_RE.match(text):
        return True
    words = text.split()
    if 1 <= len(words) <= 8 and (text.isupper() or text[:1].isupper()):
        return True
    return False


def select_supporting_source_blocks(
    source_blocks: list[dict[str, Any]] | None,
    *,
    evidence_terms: list[str] | None = None,
    query_text: str | None = None,
    answer_text: str | None = None,
) -> list[dict[str, Any]]:
    """Pick the smallest reliable source block(s) that support the answer/query.

    Uses only provenance already stored on source_blocks — never invents locations.
    Returns [] when no block can be matched, so callers keep chunk-level ranges.
    """
    blocks = [b for b in (source_blocks or []) if isinstance(b, dict)]
    if not blocks:
        return []
    needles = _evidence_needles(
        evidence_terms=evidence_terms,
        query_text=query_text,
        answer_text=answer_text,
    )
    # Still allow number/code cue scoring when token needles are sparse.
    scored: list[tuple[float, dict[str, Any]]] = []
    for block in blocks:
        score = _score_block_for_query(block, needles=needles, query_text=query_text)
        if score <= 0 and answer_text:
            score = _score_block_for_query(block, needles=needles, query_text=answer_text)
        if score > 0:
            scored.append((score, block))
    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    selected = [block for score, block in scored if score >= best_score]
    selected.sort(
        key=lambda b: (
            _coerce_int(b.get("page")) is None,
            _coerce_int(b.get("page")) or 0,
            _coerce_int(b.get("line_start") or b.get("line_number")) or 0,
        )
    )
    return selected


def _line_bounds_from_blocks(blocks: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    starts: list[int] = []
    ends: list[int] = []
    for block in blocks:
        start = _coerce_int(block.get("line_start") or block.get("line_number"))
        end = _coerce_int(block.get("line_end") or block.get("line_start") or block.get("line_number"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts:
        return None, None
    return min(starts), max(ends) if ends else min(starts)


def _page_bounds_from_blocks(blocks: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    pages = [_coerce_int(b.get("page")) for b in blocks]
    pages = [p for p in pages if p is not None]
    if not pages:
        return None, None
    return min(pages), max(pages)


def _section_from_supporting_blocks(
    all_blocks: list[dict[str, Any]],
    supporting: list[dict[str, Any]],
    *,
    fallback_section: str,
    fallback_path: str,
    fallback_title: str,
) -> tuple[str, str, str]:
    """Derive section labels from supporting block provenance / nearest heading."""
    for block in supporting:
        for key in ("section_name", "heading", "table_name"):
            label = sanitize_section_label(str(block.get(key) or ""))
            if label:
                path = str(block.get("section_path") or label).strip() or label
                return label, path, label

    support_page = _coerce_int(supporting[0].get("page")) if supporting else None
    support_line = (
        _coerce_int(supporting[0].get("line_start") or supporting[0].get("line_number"))
        if supporting
        else None
    )
    supporting_ids = {id(b) for b in supporting}
    best_heading = ""
    best_line = -1
    for block in all_blocks:
        # Prefer preceding structural headings; skip the supporting evidence cell itself
        # unless it is explicitly classified as a heading.
        if block in supporting or id(block) in supporting_ids:
            component = str(block.get("component_type") or "").lower()
            if component not in {"title", "subtitle", "heading"} and "heading" not in component:
                continue
        if not _is_heading_like_block(block):
            continue
        page = _coerce_int(block.get("page"))
        line = _coerce_int(block.get("line_start") or block.get("line_number"))
        if support_page is not None and page is not None and page != support_page:
            continue
        if support_line is not None and line is not None and line > support_line:
            continue
        if line is None:
            continue
        if line >= best_line:
            text = str(
                block.get("heading")
                or block.get("section_name")
                or block.get("text_preview")
                or block.get("text")
                or ""
            ).strip()
            text = sanitize_section_label(text.split("\n")[0].strip())
            if text:
                best_heading = text
                best_line = line
    if best_heading:
        return best_heading, best_heading, best_heading

    return fallback_section, fallback_path, fallback_title


def format_citation_line(citation: dict[str, Any]) -> str:
    """Format a human-readable citation line for UI / LLM context."""
    document = str(citation.get("document") or citation.get("document_name") or "").strip()
    section = str(citation.get("section") or "").strip()
    page_start = citation.get("page_start")
    table_name = str(citation.get("table_name") or "").strip()

    parts: list[str] = []
    if document:
        parts.append(document)
    if section:
        parts.append(f"Section {section}")
    if table_name:
        parts.append(table_name)
    if page_start is not None:
        page_end = citation.get("page_end")
        if page_end is not None and page_end != page_start:
            parts.append(f"Pages {page_start}-{page_end}")
        else:
            parts.append(f"Page {page_start}")
    line_range = citation.get("line_range")
    if line_range:
        parts.append(f"Lines {line_range}")
    return " | ".join(parts)


@audit_event("CITATION_VALIDATED", entity_type="document", category="retrieval", action="validate", source="API")
@capture_metric("citation_duration")
def resolve_citation(
    props: dict[str, Any],
    *,
    document_display_name: str | None = None,
    evidence_terms: list[str] | None = None,
    query_text: str | None = None,
    answer_text: str | None = None,
) -> dict[str, Any]:
    """Build a citation payload strictly from stored chunk metadata.

    When ``source_blocks`` are present and evidence/query/answer terms match a
    subset of blocks, narrow page/line/section to those supporting blocks only.
    Never invents provenance.
    """
    anchor = parse_citation_anchor(props.get("citation_anchor")) or {}
    page_start = _coerce_int(props.get("page")) or _coerce_int(anchor.get("start_page"))
    page_end = _coerce_int(props.get("page_end")) or _coerce_int(anchor.get("end_page")) or page_start
    line_start = _coerce_int(props.get("line_start")) or _coerce_int(anchor.get("start_line"))
    line_end = _coerce_int(props.get("line_end")) or _coerce_int(anchor.get("end_line"))

    section = _resolved_section(props, anchor)
    section_path = str(props.get("section_path") or anchor.get("section_path") or section).strip()
    parent = str(props.get("parent_section") or anchor.get("parent_section") or "").strip()
    title = _resolved_title(props, anchor, section)
    table_name = str(props.get("table_name") or anchor.get("table_name") or "").strip()
    chunk_type = str(props.get("chunk_type") or anchor.get("chunk_type") or "paragraph").strip()
    strategy_name = str(props.get("strategy_name") or anchor.get("strategy_name") or "").strip()
    document_id = str(props.get("document_id") or anchor.get("document_id") or "").strip()
    document_name = (
        document_display_name
        or str(props.get("document_name") or "").strip()
        or str(props.get("doc_name") or "").strip()
    )

    source_blocks = props.get("source_blocks")
    if not isinstance(source_blocks, list):
        source_blocks = anchor.get("source_blocks") if isinstance(anchor.get("source_blocks"), list) else []
    source_blocks = [b for b in source_blocks if isinstance(b, dict)]

    supporting = select_supporting_source_blocks(
        source_blocks,
        evidence_terms=evidence_terms,
        query_text=query_text,
        answer_text=answer_text,
    )
    citation_source_blocks = source_blocks
    if supporting:
        citation_source_blocks = supporting
        narrow_page_start, narrow_page_end = _page_bounds_from_blocks(supporting)
        narrow_line_start, narrow_line_end = _line_bounds_from_blocks(supporting)
        if narrow_page_start is not None:
            page_start = narrow_page_start
            page_end = narrow_page_end if narrow_page_end is not None else narrow_page_start
        if narrow_line_start is not None:
            line_start = narrow_line_start
            line_end = narrow_line_end if narrow_line_end is not None else narrow_line_start
        section, section_path, title = _section_from_supporting_blocks(
            source_blocks,
            supporting,
            fallback_section=section,
            fallback_path=section_path,
            fallback_title=title,
        )
        for block in supporting:
            block_table = str(block.get("table_name") or "").strip()
            if block_table:
                table_name = block_table
                break
        if any(
            str(b.get("block_type") or "").lower() == "table"
            or str(b.get("component_type") or "").lower() == "table"
            for b in supporting
        ):
            chunk_type = "table"

    citation: dict[str, Any] = {
        "chunk_id": str(props.get("chunk_id") or anchor.get("chunk_id") or ""),
        "document_id": document_id,
        "document": document_name,
        "document_name": document_name,
        "section": section,
        "section_path": section_path,
        "parent_section": parent,
        "title": title,
        "page": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "line_start": line_start,
        "line_end": line_end,
        "chunk_type": chunk_type,
        "strategy": strategy_name,
        "strategy_name": strategy_name,
    }
    if line_start is not None and line_end is not None:
        citation["line_range"] = f"{line_start}-{line_end}"
    if table_name:
        citation["table_name"] = table_name
    if citation_source_blocks:
        citation["source_blocks"] = citation_source_blocks
    if supporting:
        citation["supporting_source_blocks"] = supporting
    if anchor:
        citation["citation_anchor"] = anchor
    citation["citation_text"] = format_citation_line(citation)
    return citation


def build_llm_citation_context(citations: list[dict[str, Any]]) -> str:
    """Build a prompt appendix that constrains the LLM to use only supplied citations."""
    if not citations:
        return "No source citations were retrieved. Do not invent page numbers or section references."
    lines = [
        "Use ONLY the citations below. Do not invent page numbers, section numbers, or document names.",
        "",
    ]
    for index, citation in enumerate(citations, start=1):
        lines.append(f"[{index}] {citation.get('citation_text') or format_citation_line(citation)}")
        if citation.get("title"):
            lines.append(f"    Title: {citation['title']}")
    return "\n".join(lines)
