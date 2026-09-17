"""DOCX extract helper module (split from docx_extract)."""

from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from .docx_common import (
    A_NS,
    R_NS,
    _CORRUPTED_NUM_ONLY_RE,
    _EMU_PER_INCH,
    _SECTION_INLINE_RE,
    _SECTION_NUM_ONLY_RE,
    _TOC_TITLE_RE,
    _TWIPS_PER_INCH,
    _apply_numbering_label,
    _clean_outline_title,
    _is_banner_chrome_title,
    _is_heading_noise,
    _is_sane_outline_number,
    _is_valid_outline_title,
    _iter_paragraphs_with_meta,
    _length_to_twips,
    _load_numbering_resolver,
    _normalize_number,
    _number_sort_key,
    _outline_level,
    _parse_outline,
    _style_font,
)
from .docx_fonts import _paragraph_line_spacing_sample

_PROCEDURE_SPACING_PROSE_MIN = 40
_PROCEDURE_SPACING_SAMPLE_LIMIT = 12


def _is_procedure_section_title(title: str) -> bool:
    """True only for the major PROCEDURE heading (not '…procedure…' body titles)."""
    return " ".join(str(title or "").casefold().split()) == "procedure"


def _finalize_line_spacing(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Typical PROCEDURE body line spacing (modal multiple), exposed as minimum_line_spacing.

    Short SINGLE-spaced list stubs under PROCEDURE were previously sampled first and
    incorrectly reported as 1.00x when body prose is ~1.5x.
    """
    multiples = [
        s
        for s in samples
        if s.get("kind") == "multiple" and s.get("display") and s.get("twips") is not None
    ]
    pool = multiples or [
        s for s in samples if s.get("display") and s.get("twips") is not None
    ]
    if not pool:
        return {"minimum": "", "minimum_twips": None, "rule": None, "samples": []}
    counts: Counter[str] = Counter(str(s.get("display") or "") for s in pool)
    best_display = counts.most_common(1)[0][0]
    best = next(s for s in pool if str(s.get("display") or "") == best_display)
    return {
        "minimum": str(best.get("display") or ""),
        "minimum_twips": best.get("twips"),
        "rule": best.get("rule"),
        "samples": [s for s in pool if str(s.get("display") or "") == best_display][:2],
    }


def _extract_toc(doc: Document) -> dict[str, Any]:
    present = False
    title = ""
    has_field = False
    entries: list[dict[str, Any]] = []

    for paragraph in doc.paragraphs:
        text = " ".join((paragraph.text or "").split()).strip()
        style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
        if _TOC_TITLE_RE.match(text):
            present = True
            title = text
        if "toc" in style_name.casefold():
            present = True
            if text:
                entries.append({"level": _toc_level_from_style(style_name), "text": text, "page": None})
        try:
            for node in paragraph._element.iter(qn("w:instrText")):
                instr = str(node.text or "")
                if re.search(r"\bTOC\b", instr, re.IGNORECASE):
                    has_field = True
                    present = True
                    if not title:
                        title = text or "Table of Contents"
        except Exception:
            pass
        if re.search(r"\.{2,}\s*\d+\s*$", text) or "…" in text:
            # Dotted leader TOC line
            if len(text) < 120:
                present = True
                entries.append({"level": 1, "text": text, "page": None})

    if present and not title:
        title = "Table of Contents"
    # Cap noisy leader lines
    if len(entries) > 100:
        entries = entries[:100]
    return {
        "present": present,
        "title": title,
        "has_toc_field": has_field,
        "entries": entries,
    }


def _toc_level_from_style(style_name: str) -> int:
    match = re.search(r"(\d+)$", style_name.strip())
    if match:
        return max(1, min(2, int(match.group(1))))
    return 1


def _extract_sections(
    doc: Document,
    *,
    scan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scan = scan or scan_document_body(doc)
    stream = list(scan.get("outline_events") or [])

    heads: list[dict[str, Any]] = []
    pending_number: str | None = None
    pending_indent: dict[str, Any] | None = None
    pending_meta: dict[str, Any] | None = None
    title_buffer: list[str] = []
    pending_skips = 0
    pending_bold = False

    def _append_head(number: str, title: str, level: int, item: dict[str, Any]) -> None:
        # Level-2 subsections only when the heading is bold (run or heading style).
        if level == 2 and not bool(item.get("is_bold")):
            return
        indent = item.get("indent") or {}
        heads.append(
            {
                "number": number,
                "title": title,
                "level": level,
                "indent_twips": indent.get("left_twips"),
                "first_line_indent_twips": indent.get("first_line_twips"),
                "indent_inches": indent.get("left_inches"),
                "first_line_indent_inches": indent.get("first_line_inches"),
                "alignment": str(indent.get("alignment") or ""),
                "style_name": str(item.get("style_name") or ""),
                "in_table": bool(item.get("in_table")),
                "stream_index": int(item.get("stream_index") or 0),
                "is_bold": bool(item.get("is_bold")),
            }
        )

    def flush_pending() -> None:
        nonlocal pending_number, pending_indent, pending_meta, title_buffer, pending_skips, pending_bold
        if not pending_number:
            title_buffer = []
            pending_skips = 0
            pending_bold = False
            return
        title = _clean_outline_title(" ".join(title_buffer))
        level = _outline_level(pending_number)
        if (
            title
            and level in {1, 2}
            and _is_valid_outline_title(title, level)
            and _is_sane_outline_number(pending_number, level)
        ):
            indent = pending_indent or {}
            _append_head(
                pending_number,
                title,
                level,
                {
                    "indent": indent,
                    "style_name": str((pending_meta or {}).get("style_name") or ""),
                    "in_table": bool((pending_meta or {}).get("in_table")),
                    "stream_index": int((pending_meta or {}).get("stream_index") or 0),
                    "is_bold": pending_bold or bool((pending_meta or {}).get("is_bold")),
                },
            )
        pending_number = None
        pending_indent = None
        pending_meta = None
        title_buffer = []
        pending_skips = 0
        pending_bold = False

    def clear_pending() -> None:
        nonlocal pending_number, pending_indent, pending_meta, title_buffer, pending_skips, pending_bold
        pending_number = None
        pending_indent = None
        pending_meta = None
        title_buffer = []
        pending_skips = 0
        pending_bold = False

    for item in stream:
        kind = item["kind"]
        if kind == "outline_inline":
            number = item["number"]
            title = _clean_outline_title(item.get("title") or "")
            level = _outline_level(number)
            if level not in {1, 2} or not _is_sane_outline_number(number, level):
                # Ignore junk markers; do not break an open title capture.
                continue
            flush_pending()
            if title and _is_valid_outline_title(title, level):
                _append_head(number, title, level, item)
            continue

        if kind == "outline_number":
            level = _outline_level(item["number"])
            if level not in {1, 2} or not _is_sane_outline_number(item["number"], level):
                continue
            flush_pending()
            # Prefer same-row title from table cells when present.
            row_title = _clean_outline_title(item.get("row_title") or "")
            row_bold = bool(item.get("row_title_bold")) or bool(item.get("is_bold"))
            if row_title and _is_valid_outline_title(row_title, level):
                merged = dict(item)
                merged["is_bold"] = row_bold
                _append_head(item["number"], row_title, level, merged)
                continue
            pending_number = item["number"]
            pending_indent = item.get("indent") or {}
            pending_meta = item
            title_buffer = []
            pending_skips = 0
            pending_bold = bool(item.get("is_bold"))
            continue

        if kind == "text":
            text = _clean_outline_title(item.get("text") or "")
            if not text:
                continue
            if not pending_number:
                continue
            level = _outline_level(pending_number) or 2
            # Skip form / OCR debris while still looking for the real heading.
            if _is_heading_noise(text):
                pending_skips += 1
                if pending_skips >= 40:
                    clear_pending()
                continue
            if bool(item.get("is_bold")):
                pending_bold = True
            if not title_buffer:
                title_buffer.append(text)
                joined = _clean_outline_title(text)
                # Flush majors when ready; keep short L2 open for split titles.
                if level == 1 and _is_valid_outline_title(joined, 1):
                    flush_pending()
                elif (
                    level == 2
                    and _is_valid_outline_title(joined, 2)
                    and not _looks_incomplete_title(joined)
                ):
                    flush_pending()
                elif pending_skips + len(title_buffer) >= 40:
                    clear_pending()
                continue
            if _is_title_continuation(title_buffer[-1], text):
                title_buffer.append(text)
                joined = _clean_outline_title(" ".join(title_buffer))
                if _is_valid_outline_title(joined, level) and not _looks_incomplete_title(joined):
                    flush_pending()
                elif len(title_buffer) >= 4:
                    clear_pending()
                continue
            # New line is not a continuation — accept current buffer if valid.
            flush_pending()
            continue
    flush_pending()

    # Prefer higher-quality heading when the same number appears multiple times.
    best_by_number: dict[str, dict[str, Any]] = {}
    for head in heads:
        number = str(head.get("number") or "")
        prev = best_by_number.get(number)
        if prev is None or _heading_quality(head) > _heading_quality(prev):
            best_by_number[number] = head
    unique = list(best_by_number.values())
    unique.sort(key=lambda h: int(h.get("stream_index") or 0))

    # Truncate outline after the revision-history major (changelog tables reuse numbers).
    revision_at = None
    for head in unique:
        words = set(" ".join(str(head.get("title") or "").casefold().split()).split())
        if int(head.get("level") or 0) == 1 and "revision" in words and "history" in words:
            revision_at = int(head.get("stream_index") or 0)
            break
    if revision_at is not None:
        unique = [h for h in unique if int(h.get("stream_index") or 0) <= revision_at]

    # Bare ALL-CAPS majors (e.g. REFERENCES without "7.0") before orphan L2 children.
    unique = _synthesize_missing_majors(unique, stream)
    unique = _fill_major_gaps_from_bare_titles(unique, stream)

    stream_texts = [
        (int(i.get("stream_index") or 0), str(i.get("text") or i.get("title") or ""))
        for i in stream
    ]
    head_indexes = {h["number"]: int(h.get("stream_index") or 0) for h in unique}
    all_numbers = set(head_indexes)

    sections_out: list[dict[str, Any]] = []
    indents_out: list[dict[str, Any]] = []
    level1 = [h for h in unique if h["level"] == 1]
    level2 = [h for h in unique if h["level"] == 2]

    for head in level1:
        children = sorted(
            [c for c in level2 if _is_child_number(head["number"], c["number"])],
            key=lambda c: _number_sort_key(str(c.get("number") or "")),
        )
        section = {
            "number": head["number"],
            "title": head["title"],
            "level": 1,
            "indent_twips": head["indent_twips"],
            "indent_inches": head["indent_inches"],
            "first_line_indent_twips": head["first_line_indent_twips"],
            "first_line_indent_inches": head["first_line_indent_inches"],
            "alignment": head.get("alignment") or "",
            "content_preview": _content_preview_from_stream(
                head["number"], stream_texts, head_indexes, all_numbers
            ),
            "children": [
                {
                    "number": child["number"],
                    "title": child["title"],
                    "level": 2,
                    "indent_twips": child["indent_twips"],
                    "indent_inches": child["indent_inches"],
                    "first_line_indent_twips": child["first_line_indent_twips"],
                    "first_line_indent_inches": child["first_line_indent_inches"],
                    "alignment": child.get("alignment") or "",
                    "content_preview": _content_preview_from_stream(
                        child["number"], stream_texts, head_indexes, all_numbers
                    ),
                }
                for child in children
            ],
        }
        sections_out.append(section)
        indents_out.append(
            {
                "number": head["number"],
                "title": head["title"],
                "level": 1,
                "left_twips": head["indent_twips"],
                "left_inches": head["indent_inches"],
                "first_line_twips": head["first_line_indent_twips"],
                "first_line_inches": head["first_line_indent_inches"],
                "alignment": head.get("alignment") or "",
            }
        )
        for child in children:
            indents_out.append(
                {
                    "number": child["number"],
                    "title": child["title"],
                    "level": 2,
                    "left_twips": child["indent_twips"],
                    "left_inches": child["indent_inches"],
                    "first_line_twips": child["first_line_indent_twips"],
                    "first_line_inches": child["first_line_indent_inches"],
                    "alignment": child.get("alignment") or "",
                }
            )

    # Do not promote orphan L2 as top-level sections (avoids post-procedure list noise).
    sections_out.sort(key=lambda s: _number_sort_key(s["number"]))
    indents_out.sort(key=lambda s: _number_sort_key(s["number"]))
    return sections_out, indents_out


def _synthesize_missing_majors(
    heads: list[dict[str, Any]], stream: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach ALL-CAPS bare titles to orphan L2 groups missing a major parent."""
    level1_nums = {str(h.get("number") or "") for h in heads if int(h.get("level") or 0) == 1}
    orphans: dict[str, list[dict[str, Any]]] = {}
    for head in heads:
        if int(head.get("level") or 0) != 2:
            continue
        parts = [p for p in str(head.get("number") or "").split(".") if p]
        if len(parts) != 2:
            continue
        parent = f"{parts[0]}.0"
        if parent in level1_nums:
            continue
        orphans.setdefault(parts[0], []).append(head)

    extra: list[dict[str, Any]] = []
    for major, children in orphans.items():
        children = sorted(children, key=lambda c: int(c.get("stream_index") or 0))
        first_idx = int(children[0].get("stream_index") or 0)
        prev_idx = max(
            (
                int(h.get("stream_index") or 0)
                for h in heads
                if int(h.get("level") or 0) == 1 and int(h.get("stream_index") or 0) < first_idx
            ),
            default=-1,
        )
        best_title = ""
        best_idx = first_idx
        for item in stream:
            idx = int(item.get("stream_index") or 0)
            if idx <= prev_idx:
                continue
            if idx >= first_idx:
                break
            if item.get("kind") != "text":
                continue
            title = _clean_outline_title(item.get("text") or "")
            if _is_banner_chrome_title(title):
                continue
            if _is_valid_outline_title(title, 1):
                best_title = title
                best_idx = idx
        if not best_title:
            continue
        parent = f"{major}.0"
        extra.append(
            {
                "number": parent,
                "title": best_title,
                "level": 1,
                "indent_twips": None,
                "first_line_indent_twips": None,
                "indent_inches": None,
                "first_line_indent_inches": None,
                "alignment": "",
                "style_name": "",
                "in_table": False,
                "stream_index": best_idx,
            }
        )
        level1_nums.add(parent)
    if not extra:
        return heads
    merged = list(heads) + extra
    merged.sort(key=lambda h: int(h.get("stream_index") or 0))
    return merged


def _fill_major_gaps_from_bare_titles(
    heads: list[dict[str, Any]], stream: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Fill missing majors when a bare ALL-CAPS title sits between numbered majors
    (common when '7.0 REFERENCES' is lost but 'REFERENCES' text remains).
    Prefer the title immediately before numbered children of the missing major.
    """
    majors = [h for h in heads if int(h.get("level") or 0) == 1]
    majors.sort(key=lambda h: int(h.get("stream_index") or 0))
    if len(majors) < 2:
        return heads
    existing_titles = {
        " ".join(str(h.get("title") or "").casefold().split()) for h in majors
    }
    existing_nums = {
        int(str(h.get("number") or "0").split(".")[0])
        for h in majors
        if str(h.get("number") or "").split(".")[0].isdigit()
    }
    # Precompute child markers in the stream for structural title pairing.
    child_at: dict[int, int] = {}
    for item in stream:
        kind = item.get("kind")
        if kind not in {"outline_inline", "outline_number"}:
            continue
        number = str(item.get("number") or "")
        parts = [p for p in number.split(".") if p]
        if len(parts) == 2 and parts[1] != "0" and parts[0].isdigit():
            child_at[int(item.get("stream_index") or 0)] = int(parts[0])

    extra: list[dict[str, Any]] = []
    for left, right in zip(majors, majors[1:]):
        try:
            left_n = int(str(left.get("number") or "0").split(".")[0])
            right_n = int(str(right.get("number") or "0").split(".")[0])
        except ValueError:
            continue
        if right_n <= left_n + 1:
            continue
        left_idx = int(left.get("stream_index") or 0)
        right_idx = int(right.get("stream_index") or 0)
        missing = [n for n in range(left_n + 1, right_n) if n not in existing_nums]
        for maj in missing:
            # Prefer bare title that is immediately followed by maj.x children.
            paired: list[tuple[int, int, str]] = []
            for item in stream:
                idx = int(item.get("stream_index") or 0)
                if idx <= left_idx or idx >= right_idx:
                    continue
                if item.get("kind") != "text":
                    continue
                title = _clean_outline_title(item.get("text") or "")
                title_key = " ".join(title.casefold().split())
                if not title or title_key in existing_titles:
                    continue
                if not _is_valid_outline_title(title, 1):
                    continue
                words = title.split()
                if any(len("".join(ch for ch in w if ch.isalpha())) <= 1 for w in words):
                    continue
                # Distance to next child numbered under this major.
                dist = None
                for c_idx, c_maj in child_at.items():
                    if c_idx <= idx or c_maj != maj:
                        continue
                    dist = c_idx - idx
                    if dist <= 40:
                        paired.append((dist, idx, title))
                    break
            if paired:
                paired.sort(key=lambda row: (row[0], row[1]))
                _dist, idx, title = paired[0]
            else:
                continue
            parent = f"{maj}.0"
            extra.append(
                {
                    "number": parent,
                    "title": title,
                    "level": 1,
                    "indent_twips": None,
                    "first_line_indent_twips": None,
                    "indent_inches": None,
                    "first_line_indent_inches": None,
                    "alignment": "",
                    "style_name": "",
                    "in_table": False,
                    "stream_index": idx,
                    "is_bold": False,
                }
            )
            existing_nums.add(maj)
            existing_titles.add(" ".join(title.casefold().split()))
    if not extra:
        return heads
    merged = list(heads) + extra
    merged.sort(key=lambda h: int(h.get("stream_index") or 0))
    return merged


def scan_document_body(doc: Document) -> dict[str, Any]:
    """
    Single body walk collecting outline events, fonts, TOC, line-spacing samples,
    media attribution rows, and repeated-field body notes.
    """
    numbering = _load_numbering_resolver(doc)
    events: list[dict[str, Any]] = []
    row_cells: dict[tuple[int, int], list[tuple[int, str, bool]]] = {}
    raw_items: list[dict[str, Any]] = []

    body_fonts: Counter[str] = Counter()
    body_sizes: Counter[float] = Counter()
    heading_fonts: Counter[str] = Counter()
    heading_sizes: Counter[float] = Counter()
    all_fonts: Counter[str] = Counter()
    all_sizes: Counter[float] = Counter()

    toc_present = False
    toc_title = ""
    toc_has_field = False
    toc_entries: list[dict[str, Any]] = []

    # minimum_line_spacing = modal prose multiple under PROCEDURE (not short list stubs)
    spacing_in_procedure = False
    spacing_procedure_done = False
    spacing_procedure_samples: list[dict[str, Any]] = []

    media_rows: list[dict[str, Any]] = []
    repeated_body: list[dict[str, Any]] = []

    for stream_index, (paragraph, meta) in enumerate(_iter_paragraphs_with_meta(doc)):
        raw = " ".join((paragraph.text or "").split()).strip()
        label = ""
        if numbering is not None:
            try:
                label = str(numbering.label_for_paragraph(paragraph) or "").strip()
            except Exception:
                label = ""
        text = _apply_numbering_label(raw, label)
        indent = _paragraph_indent(paragraph)
        style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
        is_bold = _paragraph_has_bold_emphasis(paragraph)
        # Heading styles are outline titles even when runs are not explicitly bold.
        if style_name.casefold().startswith("heading"):
            is_bold = True
        in_table = bool(meta.get("in_table"))
        table_index = int(meta.get("table_index") or -1)
        row_index = int(meta.get("row_index") or -1)
        cell_index = int(meta.get("cell_index") or -1)
        item = {
            "stream_index": stream_index,
            "text": text,
            "raw": raw,
            "label": label,
            "indent": indent,
            "style_name": style_name,
            "is_bold": is_bold,
            "in_table": in_table,
            "table_index": table_index,
            "row_index": row_index,
            "cell_index": cell_index,
        }
        raw_items.append(item)
        if in_table and table_index >= 0 and row_index >= 0 and text:
            key = (table_index, row_index)
            row_cells.setdefault(key, []).append((cell_index, text, is_bold))

        # --- fonts ---
        if raw:
            is_heading = style_name.casefold().startswith("heading")
            if not is_heading:
                number_h, title_h = _parse_outline(raw)
                if (
                    number_h
                    and title_h
                    and _outline_level(number_h) == 1
                    and _is_valid_outline_title(title_h, 1)
                ):
                    is_heading = True
            style_font, style_size = _style_font(paragraph)
            runs = list(paragraph.runs)
            if not runs:
                if style_font:
                    (heading_fonts if is_heading else body_fonts)[style_font] += len(raw)
                    all_fonts[style_font] += len(raw)
                if style_size:
                    (heading_sizes if is_heading else body_sizes)[style_size] += len(raw)
                    all_sizes[style_size] += len(raw)
            else:
                for run in runs:
                    run_text = (run.text or "").strip()
                    if not run_text:
                        continue
                    weight = max(1, len(run_text))
                    name = ""
                    try:
                        name = str(run.font.name or "").strip()
                    except Exception:
                        name = ""
                    if not name:
                        name = style_font
                    size = None
                    try:
                        if run.font.size is not None:
                            size = round(float(run.font.size.pt), 1)
                    except Exception:
                        size = None
                    if size is None:
                        size = style_size
                    if name:
                        (heading_fonts if is_heading else body_fonts)[name] += weight
                        all_fonts[name] += weight
                    if size:
                        (heading_sizes if is_heading else body_sizes)[size] += weight
                        all_sizes[size] += weight

        # --- TOC ---
        if raw:
            if _TOC_TITLE_RE.match(raw):
                toc_present = True
                toc_title = raw
            if "toc" in style_name.casefold():
                toc_present = True
                toc_entries.append(
                    {"level": _toc_level_from_style(style_name), "text": raw, "page": None}
                )
            try:
                for node in paragraph._element.iter(qn("w:instrText")):
                    instr = str(node.text or "")
                    if re.search(r"\bTOC\b", instr, re.IGNORECASE):
                        toc_has_field = True
                        toc_present = True
                        if not toc_title:
                            toc_title = raw or "Table of Contents"
            except Exception:
                pass
            if (re.search(r"\.{2,}\s*\d+\s*$", raw) or "…" in raw) and len(raw) < 120:
                toc_present = True
                toc_entries.append({"level": 1, "text": raw, "page": None})

        # --- line spacing: modal prose multiple under PROCEDURE ---
        if not spacing_procedure_done:
            outline_line = text or raw
            number_s, title_s = _parse_outline(outline_line) if outline_line else (None, None)
            level_s = _outline_level(number_s) if number_s else None
            if number_s and title_s and level_s == 1 and _is_valid_outline_title(title_s, 1):
                if spacing_in_procedure:
                    spacing_procedure_done = True
                    spacing_in_procedure = False
                elif _is_procedure_section_title(title_s):
                    spacing_in_procedure = True
            elif (
                spacing_in_procedure
                and raw
                and not in_table
                and len(raw) >= _PROCEDURE_SPACING_PROSE_MIN
                and any(ch.isalpha() for ch in raw)
            ):
                sample = _paragraph_line_spacing_sample(paragraph, raw)
                if sample:
                    spacing_procedure_samples.append(sample)
                    if len(spacing_procedure_samples) >= _PROCEDURE_SPACING_SAMPLE_LIMIT:
                        spacing_procedure_done = True
                        spacing_in_procedure = False

        # --- media attribution row ---
        image_ids = _paragraph_image_rids(paragraph)
        media_rows.append(
            {
                "text": text,
                "in_table": in_table,
                "table_index": table_index,
                "image_ids": image_ids,
            }
        )

        # --- repeated-field body notes ---
        bold_pairs = _bold_label_value_pairs_from_paragraph(paragraph)
        if raw or bold_pairs:
            repeated_body.append({"text": raw, "bold_pairs": bold_pairs})

    # Build outline events (same logic as legacy _outline_stream).
    for item in raw_items:
        text = item["text"]
        if not text:
            continue
        if re.match(r"^\d+\.\d+\.\d+", text):
            continue

        number_only = _SECTION_NUM_ONLY_RE.match(text)
        corrupt = None if number_only else _CORRUPTED_NUM_ONLY_RE.match(text)
        if corrupt:
            candidate = _normalize_number(corrupt.group(1))
            if _outline_level(candidate) == 2 and _is_sane_outline_number(candidate, 2):
                number_only = corrupt
        inline = _SECTION_INLINE_RE.match(text)
        if not number_only and not inline and item["label"]:
            label_only = _SECTION_NUM_ONLY_RE.match(item["label"].rstrip("."))
            if label_only and not item["raw"]:
                number_only = label_only
                text = item["label"]

        if number_only:
            number = _normalize_number(number_only.group(1))
            level = _outline_level(number)
            if level not in {1, 2}:
                continue
            row_title = ""
            row_title_bold = False
            if item["in_table"]:
                key = (item["table_index"], item["row_index"])
                peers = [
                    (t, b)
                    for idx, t, b in sorted(row_cells.get(key, []), key=lambda x: x[0])
                    if idx != item["cell_index"]
                ]
                for peer, peer_bold in peers:
                    if _SECTION_NUM_ONLY_RE.match(peer) or re.match(r"^\d+\.\d+\.\d+", peer):
                        continue
                    cleaned = _clean_outline_title(peer)
                    if cleaned and not _is_heading_noise(cleaned):
                        row_title = cleaned
                        row_title_bold = bool(peer_bold)
                        break
            events.append(
                {
                    "kind": "outline_number",
                    "number": number,
                    "row_title": row_title,
                    "row_title_bold": row_title_bold,
                    "indent": item["indent"],
                    "style_name": item["style_name"],
                    "is_bold": bool(item["is_bold"]),
                    "in_table": item["in_table"],
                    "stream_index": item["stream_index"],
                    "text": text,
                }
            )
            continue

        if inline:
            number = _normalize_number(inline.group(1))
            title = _clean_outline_title(inline.group(2))
            level = _outline_level(number)
            if level not in {1, 2}:
                continue
            if re.search(r"\.{2,}\s*\d+\s*$", text) or "…" in text:
                events.append(
                    {
                        "kind": "text",
                        "text": text,
                        "is_bold": bool(item["is_bold"]),
                        "stream_index": item["stream_index"],
                    }
                )
                continue
            events.append(
                {
                    "kind": "outline_inline",
                    "number": number,
                    "title": title,
                    "indent": item["indent"],
                    "style_name": item["style_name"],
                    "is_bold": bool(item["is_bold"]),
                    "in_table": item["in_table"],
                    "stream_index": item["stream_index"],
                    "text": text,
                }
            )
            continue

        if item["label"] and item["raw"]:
            number = _normalize_number(item["label"].rstrip("."))
            level = _outline_level(number)
            if level in {1, 2} and not re.match(r"^\d+\.\d+\.\d+", item["label"]):
                title = _clean_outline_title(item["raw"])
                if title:
                    events.append(
                        {
                            "kind": "outline_inline",
                            "number": number,
                            "title": title,
                            "indent": item["indent"],
                            "style_name": item["style_name"],
                            "is_bold": bool(item["is_bold"]),
                            "in_table": item["in_table"],
                            "stream_index": item["stream_index"],
                            "text": text,
                        }
                    )
                    continue

        events.append(
            {
                "kind": "text",
                "text": text,
                "is_bold": bool(item["is_bold"]),
                "stream_index": item["stream_index"],
                "in_table": item["in_table"],
            }
        )

    def _top_name(counter: Counter[str]) -> str | None:
        return counter.most_common(1)[0][0] if counter else None

    def _top_size(counter: Counter[float]) -> float | None:
        return counter.most_common(1)[0][0] if counter else None

    fonts = {
        "body_font": _top_name(body_fonts) or _top_name(all_fonts),
        "body_size_pt": _top_size(body_sizes) or _top_size(all_sizes),
        "heading_font": _top_name(heading_fonts) or _top_name(all_fonts),
        "heading_size_pt": _top_size(heading_sizes) or _top_size(all_sizes),
        "all_fonts": [name for name, _ in all_fonts.most_common()],
        "all_sizes_pt": [size for size, _ in all_sizes.most_common()],
    }

    if toc_present and not toc_title:
        toc_title = "Table of Contents"
    if len(toc_entries) > 100:
        toc_entries = toc_entries[:100]
    toc = {
        "present": toc_present,
        "title": toc_title,
        "has_toc_field": toc_has_field,
        "entries": toc_entries,
    }

    line_spacing = _finalize_line_spacing(spacing_procedure_samples)

    return {
        "outline_events": events,
        "fonts": fonts,
        "toc": toc,
        "line_spacing": line_spacing,
        "media_rows": media_rows,
        "repeated_body": repeated_body,
    }


def _paragraph_image_rids(paragraph: Paragraph) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    try:
        element = paragraph._element
    except Exception:
        return ids
    try:
        for blip in element.iter(f"{{{A_NS}}}blip"):
            rid = blip.get(qn("r:embed")) or blip.get(f"{{{R_NS}}}embed")
            if not rid:
                for key, value in blip.attrib.items():
                    if str(key).endswith("embed"):
                        rid = value
                        break
            rid_s = str(rid or "").strip()
            if rid_s and rid_s not in seen:
                seen.add(rid_s)
                ids.append(rid_s)
        for imagedata in element.iter(
            "{urn:schemas-microsoft-com:vml}imagedata"
        ):
            rid = imagedata.get(qn("r:id")) or imagedata.get(f"{{{R_NS}}}id")
            if not rid:
                for key, value in imagedata.attrib.items():
                    if str(key).endswith("id") and value:
                        rid = value
                        break
            rid_s = str(rid or "").strip()
            if rid_s and rid_s not in seen:
                seen.add(rid_s)
                ids.append(rid_s)
    except Exception:
        return ids
    return ids


def _outline_stream(doc: Document, numbering) -> list[dict[str, Any]]:
    """Backward-compatible outline events (uses unified body scan)."""
    del numbering  # scan loads numbering internally
    return list(scan_document_body(doc).get("outline_events") or [])


def _content_preview_from_stream(
    number: str,
    stream_texts: list[tuple[int, str]],
    head_indexes: dict[str, int],
    all_numbers: set[str],
) -> str:
    start = head_indexes.get(number)
    if start is None:
        return ""
    parts: list[str] = []
    for idx, text in stream_texts:
        if idx <= start:
            continue
        if not text:
            continue
        other_num, _title = _parse_outline(text)
        if other_num and other_num in all_numbers:
            break
        if _SECTION_NUM_ONLY_RE.match(text) or re.match(r"^\d+\.\d+\.\d+", text):
            # Keep first depth-3 preview line under this leaf.
            if re.match(r"^\d+\.\d+\.\d+", text):
                parts.append(text)
                if len(parts) >= 2:
                    break
            continue
        parts.append(text)
        if len(parts) >= 2:
            break
    return "\n".join(parts)


def _paragraph_indent(paragraph: Paragraph) -> dict[str, Any]:
    pf = paragraph.paragraph_format
    left_twips = _length_to_twips(pf.left_indent)
    first_twips = _length_to_twips(pf.first_line_indent)
    # Many SOP headings store indent on Heading 1/2 style, not direct pPr.
    if left_twips is None or first_twips is None:
        style_left, style_first = _indent_from_style_chain(paragraph)
        if left_twips is None:
            left_twips = style_left
        if first_twips is None:
            first_twips = style_first
    if left_twips is None and first_twips is None:
        left_twips, first_twips = _indent_from_ppr(paragraph)
    if left_twips is None:
        num_left = _indent_from_numbering_level(paragraph)
        if num_left is not None:
            left_twips = num_left
    return {
        "left_twips": left_twips,
        "first_line_twips": first_twips,
        "left_inches": round(left_twips / _TWIPS_PER_INCH, 3) if left_twips is not None else None,
        "first_line_inches": round(first_twips / _TWIPS_PER_INCH, 3) if first_twips is not None else None,
        "alignment": _paragraph_alignment(paragraph),
    }


def _indent_from_style_chain(paragraph: Paragraph) -> tuple[int | None, int | None]:
    left_twips = None
    first_twips = None
    style = getattr(paragraph, "style", None)
    depth = 0
    while style is not None and depth < 8:
        try:
            spf = style.paragraph_format
            if left_twips is None:
                left_twips = _length_to_twips(spf.left_indent)
            if first_twips is None:
                first_twips = _length_to_twips(spf.first_line_indent)
        except Exception:
            pass
        if left_twips is not None and first_twips is not None:
            break
        # Also read style XML w:ind when python-docx Length is unset.
        if left_twips is None or first_twips is None:
            try:
                ppr = style._element.find(qn("w:pPr"))
                if ppr is not None:
                    ind = ppr.find(qn("w:ind"))
                    if ind is not None:
                        if left_twips is None:
                            left = ind.get(qn("w:left")) or ind.get("left")
                            if left and str(left).lstrip("-").isdigit():
                                left_twips = int(left)
                        if first_twips is None:
                            first = ind.get(qn("w:firstLine")) or ind.get("firstLine")
                            hanging = ind.get(qn("w:hanging")) or ind.get("hanging")
                            if first and str(first).lstrip("-").isdigit():
                                first_twips = int(first)
                            elif hanging and str(hanging).lstrip("-").isdigit():
                                first_twips = -int(hanging)
            except Exception:
                pass
        style = getattr(style, "base_style", None)
        depth += 1
    return left_twips, first_twips


def _indent_from_numbering_level(paragraph: Paragraph) -> int | None:
    """Read left indent from the active numbering level definition, if present."""
    try:
        ppr = paragraph._element.find(qn("w:pPr"))
        if ppr is None:
            return None
        num_pr = ppr.find(qn("w:numPr"))
        if num_pr is None:
            return None
        ilvl_el = num_pr.find(qn("w:ilvl"))
        num_id_el = num_pr.find(qn("w:numId"))
        if ilvl_el is None or num_id_el is None:
            return None
        ilvl = int(ilvl_el.get(qn("w:val")) or ilvl_el.get("val") or 0)
        num_id = int(num_id_el.get(qn("w:val")) or num_id_el.get("val") or 0)
    except Exception:
        return None
    try:
        numbering_part = paragraph.part.numbering_part
        numbering_elm = numbering_part._element
    except Exception:
        return None
    try:
        abstract_id = None
        for num in numbering_elm.findall(qn("w:num")):
            if int(num.get(qn("w:numId")) or num.get("numId") or -1) != num_id:
                continue
            abs_el = num.find(qn("w:abstractNumId"))
            if abs_el is not None:
                abstract_id = int(abs_el.get(qn("w:val")) or abs_el.get("val") or -1)
            break
        if abstract_id is None:
            return None
        for abs_num in numbering_elm.findall(qn("w:abstractNum")):
            if int(abs_num.get(qn("w:abstractNumId")) or abs_num.get("abstractNumId") or -1) != abstract_id:
                continue
            for lvl in abs_num.findall(qn("w:lvl")):
                if int(lvl.get(qn("w:ilvl")) or lvl.get("ilvl") or -1) != ilvl:
                    continue
                ppr = lvl.find(qn("w:pPr"))
                if ppr is None:
                    return None
                ind = ppr.find(qn("w:ind"))
                if ind is None:
                    return None
                left = ind.get(qn("w:left")) or ind.get("left")
                if left and str(left).lstrip("-").isdigit():
                    return int(left)
                return None
    except Exception:
        return None
    return None


def _paragraph_alignment(paragraph: Paragraph) -> str:
    """Return Left / Right / Center / Justified from paragraph, style chain, or pPr/jc."""
    try:
        mapped = _alignment_label(paragraph.paragraph_format.alignment)
        if mapped:
            return mapped
    except Exception:
        pass
    style = getattr(paragraph, "style", None)
    depth = 0
    while style is not None and depth < 8:
        try:
            mapped = _alignment_label(style.paragraph_format.alignment)
            if mapped:
                return mapped
        except Exception:
            pass
        try:
            ppr = style._element.find(qn("w:pPr"))
            if ppr is not None:
                jc = ppr.find(qn("w:jc"))
                if jc is not None:
                    raw = str(jc.get(qn("w:val")) or jc.get("val") or "").strip().casefold()
                    mapped = _alignment_from_jc_value(raw)
                    if mapped:
                        return mapped
        except Exception:
            pass
        style = getattr(style, "base_style", None)
        depth += 1
    mapped = _alignment_from_ppr(paragraph)
    if mapped:
        return mapped
    # Word LTR default when no explicit jc is Left.
    return "Left"


def _alignment_label(align: Any) -> str:
    if align is None:
        return ""
    try:
        if align == WD_ALIGN_PARAGRAPH.LEFT:
            return "Left"
        if align == WD_ALIGN_PARAGRAPH.RIGHT:
            return "Right"
        if align == WD_ALIGN_PARAGRAPH.CENTER:
            return "Center"
        if align == WD_ALIGN_PARAGRAPH.JUSTIFY:
            return "Justified"
    except Exception:
        pass
    name = str(getattr(align, "name", "") or align).casefold()
    if "left" in name:
        return "Left"
    if "right" in name:
        return "Right"
    if "center" in name:
        return "Center"
    if "justify" in name or name == "both":
        return "Justified"
    return ""


def _alignment_from_jc_value(raw: str) -> str:
    if raw in {"left", "start"}:
        return "Left"
    if raw in {"right", "end"}:
        return "Right"
    if raw == "center":
        return "Center"
    if raw in {"both", "justify", "distribute"}:
        return "Justified"
    return ""


def _alignment_from_ppr(paragraph: Paragraph) -> str:
    try:
        ppr = paragraph._element.find(qn("w:pPr"))
        if ppr is None:
            return ""
        jc = ppr.find(qn("w:jc"))
        if jc is None:
            return ""
        raw = str(jc.get(qn("w:val")) or jc.get("val") or "").strip().casefold()
    except Exception:
        return ""
    return _alignment_from_jc_value(raw)


def _indent_from_ppr(paragraph: Paragraph) -> tuple[int | None, int | None]:
    try:
        ppr = paragraph._element.find(qn("w:pPr"))
        if ppr is None:
            return None, None
        ind = ppr.find(qn("w:ind"))
        if ind is None:
            return None, None
        left = ind.get(qn("w:left")) or ind.get("left")
        first = ind.get(qn("w:firstLine")) or ind.get("firstLine")
        hanging = ind.get(qn("w:hanging")) or ind.get("hanging")
        left_twips = int(left) if left and str(left).lstrip("-").isdigit() else None
        first_twips = None
        if first and str(first).lstrip("-").isdigit():
            first_twips = int(first)
        elif hanging and str(hanging).lstrip("-").isdigit():
            first_twips = -int(hanging)
        return left_twips, first_twips
    except Exception:
        return None, None


def _run_is_bold(run) -> bool:
    if run.bold is True:
        return True
    try:
        r_pr = run._element.find(qn("w:rPr"))
        if r_pr is not None:
            b = r_pr.find(qn("w:b"))
            if b is not None:
                val = b.get(qn("w:val"))
                if val in (None, "true", "1", "on"):
                    return True
    except Exception:
        pass
    return False


def _rpr_is_bold(r_pr) -> bool:
    if r_pr is None:
        return False
    b = r_pr.find(qn("w:b"))
    if b is None:
        return False
    val = b.get(qn("w:val"))
    return val in (None, "true", "1", "on")


def _style_resolves_bold(style) -> bool:
    seen: set[int] = set()
    current = style
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            if getattr(current.font, "bold", None) is True:
                return True
            r_pr = current.element.find(qn("w:rPr"))
            if _rpr_is_bold(r_pr):
                return True
        except Exception:
            pass
        current = getattr(current, "base_style", None)
    return False


def _paragraph_has_bold_emphasis(paragraph) -> bool:
    """True when paragraph text is bold via run, paragraph mark, or style (e.g. Heading)."""
    try:
        p_pr = paragraph._element.find(qn("w:pPr"))
        if p_pr is not None and _rpr_is_bold(p_pr.find(qn("w:rPr"))):
            return True
    except Exception:
        pass
    try:
        if _style_resolves_bold(getattr(paragraph, "style", None)):
            return True
    except Exception:
        pass
    bold_letters = 0
    total_letters = 0
    for run in paragraph.runs:
        letters = sum(1 for ch in (run.text or "") if ch.isalpha())
        if not letters:
            continue
        total_letters += letters
        if _run_is_bold(run):
            bold_letters += letters
    if total_letters and bold_letters * 2 >= total_letters:
        return True
    return False


def _bold_label_value_pairs_from_paragraph(paragraph) -> list[tuple[str, str]]:
    """
    Discover Key/Value pairs from bold label runs followed by non-bold value runs.

    Matches repeating chrome rows like:
      Facility<TAB>BMS1  Department  Quality Control
    No hard-coded field names and no regex.
    """
    from .page_meta import is_valid_meta_key

    key_parts: list[str] = []
    value_parts: list[str] = []
    pairs: list[tuple[str, str]] = []
    mode = "key"  # key | value

    def _flush() -> None:
        nonlocal key_parts, value_parts, mode
        key = " ".join(" ".join(key_parts).split()).strip().rstrip(":")
        value = " ".join(" ".join(value_parts).split()).strip()
        # Rejoin OCR hyphen breaks: "Contro-" + "l" -> "Contro-l"
        glued: list[str] = []
        for token in value.split():
            if glued and glued[-1].endswith("-"):
                glued[-1] = glued[-1] + token
            else:
                glued.append(token)
        value = " ".join(glued)
        if key and value and is_valid_meta_key(key):
            # Reject values that are almost non-alphanumeric glyph noise.
            alnum = sum(1 for ch in value if ch.isalnum())
            if alnum >= 2:
                pairs.append((key, value))
        key_parts = []
        value_parts = []
        mode = "key"

    for run in paragraph.runs:
        raw = str(run.text or "")
        if not raw:
            continue
        chunks = raw.split("\t")
        for index, chunk in enumerate(chunks):
            if index > 0 and mode == "value" and value_parts and key_parts:
                # Tab after a value usually starts the next bold label group.
                # Keep value open until bold arrives (flush happens on bold).
                pass
            text = " ".join(chunk.split())
            if not text:
                continue
            bold = _run_is_bold(run)
            if bold:
                if mode == "value" and (key_parts or value_parts):
                    _flush()
                mode = "key"
                key_parts.append(text)
            else:
                if not key_parts:
                    continue
                mode = "value"
                value_parts.append(text)
    if key_parts and value_parts:
        _flush()
    return pairs


def _looks_incomplete_title(title: str) -> bool:
    """True when a heading likely continues on the next line."""
    text = _clean_outline_title(title)
    words = text.split()
    if not words:
        return True
    # Single content word (e.g. "Material") often continues with "and …".
    return len(words) == 1 and words[0][:1].isupper() and not words[0].isupper()


def _is_title_continuation(prev: str, nxt: str) -> bool:
    """Join split titles when the next fragment continues the previous heading."""
    left = _clean_outline_title(prev)
    right = _clean_outline_title(nxt)
    if not left or not right:
        return False
    if _is_heading_noise(right):
        return False
    # Continuation lines typically start lowercase ("and Apparatus").
    return right[:1].islower()


def _heading_quality(head: dict[str, Any]) -> int:
    """Score competing headings that share a number; higher wins."""
    title = _clean_outline_title(str(head.get("title") or ""))
    level = int(head.get("level") or 0)
    score = 0
    if not bool(head.get("in_table")):
        score += 120
    style = str(head.get("style_name") or "").casefold()
    if style.startswith("heading"):
        score += 40
    if _is_valid_outline_title(title, level or 2):
        score += 30
    words = title.split()
    if level == 1:
        # Prefer short majors (SCOPE) over header banners (STANDARD OPERATING PROCEDURE).
        score += max(0, 60 - len(words) * 15)
        if any(ch.isdigit() for ch in title):
            score -= 100
    else:
        score += min(40, len(words) * 8)
    letters = re.sub(r"[^A-Za-z]", "", title)
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if level == 1:
            score += int(upper_ratio * 20)
        elif upper_ratio > 0.85 and len(words) == 1:
            score -= 50
    if _is_heading_noise(title):
        score -= 200
    # Prefer earlier body outline over later header/footer echoes.
    score -= min(80, int(head.get("stream_index") or 0) // 100)
    return score


def _is_child_number(parent: str, child: str) -> bool:
    parent_parts = [p for p in parent.split(".") if p]
    child_parts = [p for p in child.split(".") if p]
    if len(parent_parts) == 2 and parent_parts[1] == "0":
        return (
            len(child_parts) == 2
            and child_parts[0] == parent_parts[0]
            and child_parts[1] != "0"
        )
    if len(parent_parts) == 1:
        return len(child_parts) == 2 and child_parts[0] == parent_parts[0]
    return False
