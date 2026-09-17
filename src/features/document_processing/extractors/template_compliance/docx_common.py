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

from .page_meta import (
    colon_fields_from_page_lines,
    column_key,
    is_valid_meta_key,
    metadata_from_colon_fields,
    metadata_slot_for_key,
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


V_NS = "urn:schemas-microsoft-com:vml"


CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"


DC_NS = "http://purl.org/dc/elements/1.1/"


EP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"


_EMU_PER_INCH = 914400.0


_TWIPS_PER_INCH = 1440.0


_SECTION_INLINE_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")


_SECTION_NUM_ONLY_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?$")


_OUTLINE_START_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?(?:\s+(.*))?$")


_TOC_TITLE_RE = re.compile(r"^(table\s+of\s+contents|contents|toc)\b", re.IGNORECASE)


_NUMBERING_ALREADY_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*|[A-Za-z]|[ivxlcdm]+)[\.)]\s+\S",
    re.IGNORECASE,
)


_CORRUPTED_NUM_ONLY_RE = re.compile(r"^[^\d]{1,4}(\d+\.\d+(?:\.\d+)*)\.?\s*$")


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _document_type_from_chrome(body_lines: list[str], header_lines: list[str]) -> str:
    """
    Best-effort document type label: short non Key:Value line from header/body
    chrome (e.g. 'Standard Operating Procedure'). No fixed phrase list.
    """
    candidates = list(header_lines or []) + list(body_lines or [])
    for raw in candidates:
        line = " ".join(str(raw or "").split()).strip()
        if not line or ":" in line:
            continue
        words = line.split()
        if 2 <= len(words) <= 6 and line[0].isalpha() and len(line) <= 80:
            return line
    return ""


def _colon_fields_from_lines(lines: list[str]) -> list[dict[str, str]]:
    """Colon Key: Value finder. Returns label/value for legacy callers."""
    out: list[dict[str, str]] = []
    for item in colon_fields_from_page_lines(list(lines or [])):
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if name and value:
            out.append({"label": name, "name": name, "value": value})
    return out


def _xml_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter(f"{{{W_NS}}}t"):
        if node.text:
            parts.append(node.text)
    return " ".join("".join(parts).split()).strip()


def hashlib_sha1(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _rels_map(archive: zipfile.ZipFile, rels_name: str) -> dict[str, str]:
    if rels_name not in archive.namelist():
        return {}
    xml = archive.read(rels_name).decode("utf-8", errors="ignore")
    mapping: dict[str, str] = {}
    for match in re.finditer(
        r'<Relationship\b[^>]*\bId="([^"]+)"[^>]*\bTarget="([^"]+)"[^>]*/?>',
        xml,
        flags=re.IGNORECASE,
    ):
        mapping[match.group(1)] = match.group(2)
    for match in re.finditer(
        r'<Relationship\b[^>]*\bTarget="([^"]+)"[^>]*\bId="([^"]+)"[^>]*/?>',
        xml,
        flags=re.IGNORECASE,
    ):
        mapping.setdefault(match.group(2), match.group(1))
    return mapping


def _target_looks_like_image(target: str) -> bool:
    value = str(target or "").replace("\\", "/").casefold()
    if not value:
        return False
    if "/media/" in value or value.startswith("media/"):
        return True
    return bool(re.search(r"\.(png|jpe?g|gif|bmp|tiff?|emf|wmf|svg)(?:\?|$)", value))


def _resolve_media_part(target: str) -> str:
    value = str(target or "").replace("\\", "/").lstrip("/")
    if value.startswith("../"):
        value = value[3:]
    if not value.startswith("word/"):
        value = f"word/{value}"
    return value


def _length_to_twips(value) -> int | None:
    if value is None:
        return None
    try:
        # python-docx Length is EMU
        return int(round(float(value) / 635.0))
    except Exception:
        return None


def _iter_paragraphs(doc: Document):
    for paragraph, _meta in _iter_paragraphs_with_meta(doc):
        yield paragraph


def _iter_paragraphs_with_meta(doc: Document):
    try:
        from rewrite_engine.docx_numbering import iter_paragraphs_with_table_meta

        yield from iter_paragraphs_with_table_meta(doc)
        return
    except Exception:
        pass
    for paragraph in doc.paragraphs:
        yield paragraph, {"in_table": False}


def _load_numbering_resolver(doc: Document):
    try:
        optimizer_dir = Path(__file__).resolve().parents[1]
        if str(optimizer_dir) not in sys.path:
            sys.path.insert(0, str(optimizer_dir))
        from rewrite_engine.docx_numbering import DocxNumberingResolver

        return DocxNumberingResolver(doc)
    except Exception:
        return None


def _full_para_text(paragraph: Paragraph, numbering) -> str:
    raw = " ".join((paragraph.text or "").split()).strip()
    label = ""
    if numbering is not None:
        try:
            label = str(numbering.label_for_paragraph(paragraph) or "").strip()
        except Exception:
            label = ""
    return _apply_numbering_label(raw, label)


def _apply_numbering_label(text: str, label: str) -> str:
    body = " ".join(str(text or "").split()).strip()
    marker = str(label or "").strip()
    if not body:
        return marker
    if not marker:
        return body
    if body.lower().startswith(marker.lower()):
        return body
    if _NUMBERING_ALREADY_RE.match(body):
        return body
    return f"{marker} {body}".strip()


def _parse_outline(text: str) -> tuple[str | None, str | None]:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return None, None
    # Skip dotted-leader TOC rows.
    if re.search(r"\.{2,}\s*\d+\s*$", cleaned) or "…" in cleaned:
        return None, None
    match = _SECTION_INLINE_RE.match(cleaned)
    if not match:
        return None, None
    number = _normalize_number(match.group(1))
    title = _clean_outline_title(match.group(2))
    return number, title


def _normalize_number(number: str) -> str:
    value = str(number or "").strip().rstrip(".")
    parts = [p for p in value.split(".") if p != ""]
    if len(parts) == 1 and parts[0].isdigit():
        return f"{int(parts[0])}.0"
    if len(parts) >= 1 and all(p.isdigit() for p in parts):
        # Drop leading zeros: 0284 -> 284 (later rejected if not a real major).
        normalized = ".".join(str(int(p)) for p in parts)
        if len(parts) == 1:
            return f"{normalized}.0"
        return normalized
    return value


def _clean_outline_title(text: str) -> str:
    title = re.sub(r"[\u0000-\u001f\ufffd]+", " ", str(text or ""))
    title = re.sub(r"\s+", " ", title).strip(" \t-:;")
    title = re.sub(r"^[.:)\-]+\s*", "", title).strip()
    # Strip trailing TOC page numbers: "PURPOSE 3"
    title = re.sub(r"[\t ]+\d{1,4}$", "", title).strip()
    # Drop annotation markers tacked onto titles
    title = re.sub(r"\s+NOTE:?\s*$", "", title, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"[\d/\-.\s:\"',]+", title or ""):
        return ""
    return title


def _outline_level(number: str) -> int | None:
    parts = [p for p in str(number).split(".") if p != ""]
    if not parts or not all(p.isdigit() for p in parts):
        return None
    if len(parts) == 1:
        return 1
    if len(parts) == 2 and parts[1] == "0":
        return 1
    if len(parts) == 2:
        return 2
    return None


def _is_sane_outline_number(number: str, level: int) -> bool:
    parts = [p for p in str(number).split(".") if p != ""]
    if not parts or not all(p.isdigit() for p in parts):
        return False
    major = int(parts[0])
    # Reject OCR/page-noise majors and zero.
    if major <= 0 or major > 40:
        return False
    if level == 1:
        return len(parts) == 2 and parts[1] == "0"
    if level == 2:
        return len(parts) == 2 and parts[1] != "0"
    return False


def _is_heading_noise(text: str) -> bool:
    """True for form chrome / OCR debris that must not become section titles."""
    cleaned = _clean_outline_title(text)
    if not cleaned:
        return True
    # Strip wrapping quotes often left on OCR fragments: `"C Ql C`
    cleaned = cleaned.strip(" \"'`“”‘’")
    if not cleaned:
        return True
    letters = re.sub(r"[^A-Za-z]", "", cleaned)
    if len(letters) < 3:
        return True
    words = [w.strip(" \"'`“”‘’.,;:") for w in cleaned.split() if w.strip(" \"'`“”‘’.,;:")]
    if not words:
        return True
    # Short ALL-CAPS single tokens are form residues (PASS/NOTE), not headings.
    # Keep 5+ letter majors such as SCOPE.
    if len(words) == 1 and words[0].isupper() and len(letters) <= 4:
        return True
    # Junk glyph runs with almost no word structure.
    if len(words) == 1 and not words[0][:1].isalpha():
        return True
    # OCR shards like `"C Ql C` — no real word (≥3 letters).
    significant = [w for w in words if len(re.sub(r"[^A-Za-z]", "", w)) >= 3]
    if not significant:
        return True
    # Mostly single-letter tokens (glyph soup).
    tiny = sum(1 for w in words if len(re.sub(r"[^A-Za-z]", "", w)) <= 1)
    if tiny >= 2 and tiny >= len(words) - 1:
        return True
    # Almost no vowels → glyph/OCR noise (keep short acronyms with a real word elsewhere).
    vowels = sum(1 for ch in letters.casefold() if ch in "aeiou")
    if len(letters) >= 4 and vowels == 0 and not significant:
        return True
    if len(letters) >= 4 and vowels == 0 and max(len(re.sub(r"[^A-Za-z]", "", w)) for w in words) <= 3:
        return True
    return False


def _is_banner_chrome_title(title: str) -> bool:
    """
    True for watermark / confidentiality banners mistaken for majors.

    Structural only: long mostly-uppercase multi-word lines, not short SOP majors
    like PURPOSE / REVISION HISTORY / LIST OF ANNEXURES.
    """
    text = _clean_outline_title(title)
    if not text:
        return True
    words = text.split()
    letters = re.sub(r"[^A-Za-z]", "", text)
    if not letters:
        return True
    low = text.casefold()
    banner_hints = (
        "confidential",
        "for internal use",
        "internal use only",
        "proprietary",
        "do not copy",
        "do not distribute",
        "watermark",
    )
    if any(hint in low for hint in banner_hints):
        return True
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    # Very long ALL-CAPS chrome lines only — keep 4-word SOP majors like
    # "LIST OF ANNEXURES/ ATTACHMENTS".
    if upper_ratio >= 0.8 and len(words) >= 6 and len(letters) >= 40:
        return True
    return False


def _is_title_case_heading(text: str) -> bool:
    """True when most alphabetic words start with an uppercase letter (Title Case)."""
    words = [w for w in text.split() if any(ch.isalpha() for ch in w)]
    if not words:
        return False
    titled = 0
    for word in words:
        first = next((ch for ch in word if ch.isalpha()), "")
        if first and first.isupper():
            titled += 1
    return titled >= max(1, int(round(len(words) * 0.8)))


def _is_valid_outline_title(title: str, level: int) -> bool:
    text = _clean_outline_title(title)
    if not text or len(text) < 2:
        return False
    if _is_heading_noise(text):
        return False
    if len(text) > (90 if level == 1 else 70):
        return False
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < 3:
        return False
    words = text.split()
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    if level == 1:
        if text[:1].islower():
            return False
        if text.endswith("."):
            return False
        # Reject date/doc-no / glyph noise that is not a section name.
        if any(ch.isdigit() for ch in text) or ":" in text:
            return False
        if len(words) > 10:
            return False
        # Watermark / confidentiality banners are not outline majors.
        if _is_banner_chrome_title(text):
            return False
        # Accept ALL-CAPS majors (PURPOSE) and Title Case majors (Objective).
        if upper_ratio >= 0.7:
            return True
        return _is_title_case_heading(text)
    # Level 2: reject sentence-like body clauses; allow Title Case headings.
    if text[:1].islower():
        return False
    # Long clause ending with period is body text, not a subsection title.
    if text.endswith(".") and len(words) >= 8:
        return False
    if len(words) >= 12:
        return False
    if text.count(",") >= 2 and upper_ratio < 0.5:
        return False
    return True


def _number_sort_key(number: str) -> tuple[int, ...]:
    parts = [p for p in str(number).split(".") if p]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return (9999,)


def _is_heading_paragraph(paragraph: Paragraph) -> bool:
    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
    if style_name.lower().startswith("heading"):
        return True
    text = " ".join((paragraph.text or "").split()).strip()
    number, title = _parse_outline(text)
    if number and title and _outline_level(number) == 1:
        return True
    return False


def _style_font(paragraph: Paragraph) -> tuple[str, float | None]:
    style = getattr(paragraph, "style", None)
    name = ""
    size = None
    try:
        if style is not None and style.font is not None:
            name = str(style.font.name or "").strip()
            if style.font.size is not None:
                size = round(float(style.font.size.pt), 1)
    except Exception:
        pass
    return name, size


def _unique_fields(fields: list[dict[str, str]]) -> list[dict[str, str]]:
    """Dedupe by label; last value wins."""
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for field in fields:
        label = str(field.get("label") or field.get("name") or "").strip()
        value = str(field.get("value") or "").strip()
        if not label or not value:
            continue
        key = label.casefold()
        if key not in best:
            order.append(key)
        best[key] = {"label": label, "value": value}
    return [best[key] for key in order]
