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
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from .docx_common import (
    _EMU_PER_INCH,
    _apply_numbering_label,
    _is_heading_paragraph,
    _is_valid_outline_title,
    _iter_paragraphs,
    _iter_paragraphs_with_meta,
    _load_numbering_resolver,
    _outline_level,
    _parse_outline,
    _style_font,
)

def _package_hf_font(path: Path, *, role: str = "header") -> dict[str, Any]:
    """
    Dominant font name/size from OOXML header* or footer* parts.

    Ignores tiny placeholder sizes (< 6pt). Falls back to styles.xml
    document defaults when parts omit rFonts (common in table chrome).
    """
    names: Counter[str] = Counter()
    sizes: Counter[float] = Counter()
    role_lc = str(role or "header").casefold()
    pattern = re.compile(rf"word/{re.escape(role_lc)}\d*\.xml$", re.IGNORECASE)
    try:
        with zipfile.ZipFile(path) as archive:
            part_names = sorted(n for n in archive.namelist() if pattern.fullmatch(n))
            for name in part_names:
                xml = archive.read(name).decode("utf-8", errors="ignore")
                for match in re.finditer(
                    r"<w:rFonts\b[^>]*(?:w:ascii|w:hAnsi)=\"([^\"]+)\"",
                    xml,
                    flags=re.IGNORECASE,
                ):
                    font = match.group(1).strip()
                    if font and not font.lower().endswith("theme"):
                        names[font] += 1
                for match in re.finditer(
                    r"<w:sz(?:Cs)?\b[^>]*w:val=\"(\d+)\"",
                    xml,
                    flags=re.IGNORECASE,
                ):
                    try:
                        size_pt = round(int(match.group(1)) / 2.0, 1)
                    except Exception:
                        continue
                    # Skip half-point stubs / border glyphs.
                    if 6.0 <= size_pt <= 72.0:
                        sizes[size_pt] += 1
            if not names and "word/styles.xml" in archive.namelist():
                styles = archive.read("word/styles.xml").decode("utf-8", errors="ignore")
                for match in re.finditer(
                    r"<w:rFonts\b[^>]*(?:w:ascii|w:hAnsi)=\"([^\"]+)\"",
                    styles,
                    flags=re.IGNORECASE,
                ):
                    font = match.group(1).strip()
                    if font:
                        names[font] += 1
                        break
    except Exception:
        return {"name": "", "size": None}
    return {
        "name": names.most_common(1)[0][0] if names else "",
        "size": sizes.most_common(1)[0][0] if sizes else None,
    }


def _extract_fonts(doc: Document) -> dict[str, Any]:
    body_fonts: Counter[str] = Counter()
    body_sizes: Counter[float] = Counter()
    heading_fonts: Counter[str] = Counter()
    heading_sizes: Counter[float] = Counter()
    all_fonts: Counter[str] = Counter()
    all_sizes: Counter[float] = Counter()

    for paragraph in _iter_paragraphs(doc):
        text = (paragraph.text or "").strip()
        if not text:
            continue
        is_heading = _is_heading_paragraph(paragraph)
        style_font, style_size = _style_font(paragraph)
        runs = list(paragraph.runs)
        if not runs:
            if style_font:
                (heading_fonts if is_heading else body_fonts)[style_font] += len(text)
                all_fonts[style_font] += len(text)
            if style_size:
                (heading_sizes if is_heading else body_sizes)[style_size] += len(text)
                all_sizes[style_size] += len(text)
            continue
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

    def _top_name(counter: Counter[str]) -> str | None:
        return counter.most_common(1)[0][0] if counter else None

    def _top_size(counter: Counter[float]) -> float | None:
        return counter.most_common(1)[0][0] if counter else None

    return {
        "body_font": _top_name(body_fonts) or _top_name(all_fonts),
        "body_size_pt": _top_size(body_sizes) or _top_size(all_sizes),
        "heading_font": _top_name(heading_fonts) or _top_name(all_fonts),
        "heading_size_pt": _top_size(heading_sizes) or _top_size(all_sizes),
        "all_fonts": [name for name, _ in all_fonts.most_common()],
        "all_sizes_pt": [size for size, _ in all_sizes.most_common()],
    }


def _extract_line_spacing(doc: Document) -> dict[str, Any]:
    """
    Typical PROCEDURE body line spacing (modal multiple among prose lines).

    If PROCEDURE is missing or has no usable spacing, return empty.
    """
    samples = _procedure_first_line_spacing_samples(doc, limit=12)
    return _finalize_line_spacing_pool(samples)


def _finalize_line_spacing_pool(samples: list[dict[str, Any]]) -> dict[str, Any]:
    multiples = [
        s
        for s in samples
        if s.get("kind") == "multiple" and s.get("display") and s.get("twips") is not None
    ]
    usable = multiples or [
        s for s in samples if s.get("display") and s.get("twips") is not None
    ]
    if not usable:
        return {"minimum": "", "minimum_twips": None, "rule": None, "samples": []}
    counts: Counter[str] = Counter(str(s.get("display") or "") for s in usable)
    best_display = counts.most_common(1)[0][0]
    best = next(s for s in usable if str(s.get("display") or "") == best_display)
    return {
        "minimum": str(best.get("display") or ""),
        "minimum_twips": best.get("twips"),
        "rule": best.get("rule"),
        "samples": [s for s in usable if str(s.get("display") or "") == best_display][:2],
    }


def _procedure_first_line_spacing_samples(
    doc: Document,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Collect spacing from prose body lines under the PROCEDURE major heading."""
    in_procedure = False
    samples: list[dict[str, Any]] = []
    prose_min = 40
    numbering = _load_numbering_resolver(doc)

    for paragraph, meta in _iter_paragraphs_with_meta(doc):
        raw = " ".join((paragraph.text or "").split()).strip()
        if not raw:
            continue
        label = ""
        if numbering is not None:
            try:
                label = str(numbering.label_for_paragraph(paragraph) or "").strip()
            except Exception:
                label = ""
        text = _apply_numbering_label(raw, label)
        outline = text or raw
        number, title = _parse_outline(outline)
        level = _outline_level(number) if number else None
        if number and title and level == 1 and _is_valid_outline_title(title, 1):
            title_norm = " ".join(title.casefold().split())
            if in_procedure:
                break
            in_procedure = title_norm == "procedure"
            continue
        if not in_procedure:
            continue
        if bool((meta or {}).get("in_table")):
            continue
        if len(raw) < prose_min or not any(ch.isalpha() for ch in raw):
            continue
        sample = _paragraph_line_spacing_sample(paragraph, raw)
        if sample:
            samples.append(sample)
            if len(samples) >= limit:
                break
    return samples


def _line_spacing_samples_from_section_body(
    doc: Document,
    *,
    prefer_title: str,
) -> list[dict[str, Any]]:
    """Backward-compatible helper; PROCEDURE-first path is preferred."""
    want = " ".join(str(prefer_title or "").casefold().split())
    if want and "procedure" not in want:
        # Legacy callers asking for another section still get that section's body.
        in_target = False
        samples: list[dict[str, Any]] = []
        for paragraph, meta in _iter_paragraphs_with_meta(doc):
            text = " ".join((paragraph.text or "").split()).strip()
            if not text:
                continue
            number, title = _parse_outline(text)
            level = _outline_level(number) if number else None
            if number and title and level == 1 and _is_valid_outline_title(title, 1):
                title_norm = " ".join(title.casefold().split())
                if in_target:
                    break
                in_target = want in title_norm
                continue
            if not in_target or bool((meta or {}).get("in_table")):
                continue
            if len(text) < 12 or not any(ch.isalpha() for ch in text):
                continue
            sample = _paragraph_line_spacing_sample(paragraph, text)
            if sample:
                samples.append(sample)
                if len(samples) >= 2:
                    break
        return samples
    return _procedure_first_line_spacing_samples(doc, limit=2)


def _paragraph_line_spacing_sample(paragraph: Paragraph, text: str) -> dict[str, Any] | None:
    pf = paragraph.paragraph_format
    rule = pf.line_spacing_rule
    spacing = pf.line_spacing
    # Fall back to paragraph style when direct formatting is unset.
    if spacing is None and rule is None:
        try:
            style_pf = paragraph.style.paragraph_format if paragraph.style is not None else None
        except Exception:
            style_pf = None
        if style_pf is not None:
            rule = style_pf.line_spacing_rule
            spacing = style_pf.line_spacing
    if spacing is None and rule is None:
        return None
    twips = None
    display = None
    kind = "other"
    rule_name = getattr(rule, "name", None) or (str(rule) if rule is not None else None)
    try:
        if rule == WD_LINE_SPACING.EXACTLY or rule == WD_LINE_SPACING.AT_LEAST:
            if spacing is not None:
                pt = float(spacing) / _EMU_PER_INCH * 72.0
                # Tiny exact values are table/OCR noise (e.g. 4pt), not body standard.
                if pt < 8.0 or pt > 48.0:
                    return None
                twips = int(round(float(spacing) / 635.0))
                display = f"{round(pt, 1)} pt"
                kind = "exact"
        else:
            multiple = None
            if rule == WD_LINE_SPACING.SINGLE:
                multiple = 1.0
            elif rule == WD_LINE_SPACING.ONE_POINT_FIVE:
                multiple = 1.5
            elif rule == WD_LINE_SPACING.DOUBLE:
                multiple = 2.0
            elif isinstance(spacing, (int, float)):
                multiple = float(spacing)
            elif spacing is not None:
                pt = float(spacing) / _EMU_PER_INCH * 72.0
                if pt < 8.0 or pt > 48.0:
                    return None
                twips = int(round(float(spacing) / 635.0))
                display = f"{round(pt, 1)} pt"
                kind = "exact"
            if multiple is not None:
                # Body standard is typically ~1.0x–2.0x; drop absurd multiples.
                if multiple < 0.85 or multiple > 3.0:
                    return None
                # Snap near common Word presets (e.g. 1.55 → 1.50) for stable UI labels.
                for preset in (1.0, 1.5, 2.0):
                    if abs(multiple - preset) <= 0.08:
                        multiple = preset
                        break
                display = f"{multiple:.2f}x"
                twips = int(round(220 * multiple))
                kind = "multiple"
    except Exception:
        return None
    if twips is None or not display:
        return None
    return {
        "text_preview": text[:80],
        "twips": twips,
        "display": display,
        "rule": rule_name,
        "kind": kind,
    }
