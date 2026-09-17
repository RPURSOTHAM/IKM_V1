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
from .docx_common import (
    W_NS,
    _SECTION_INLINE_RE,
    _SECTION_NUM_ONLY_RE,
    _colon_fields_from_lines,
    _iter_paragraphs,
    _parse_outline,
    _rels_map,
    _resolve_media_part,
    _target_looks_like_image,
    _unique_fields,
    _unique_strings,
    _xml_paragraph_text,
    hashlib_sha1,
)
from .docx_images import (
    _image_refs,
    _images_from_paragraphs,
)
from .docx_sections import (
    _bold_label_value_pairs_from_paragraph,
)
from .docx_tables import (
    _extract_tables_from_container,
    _fields_from_table_cells,
)

def _detect_page_field_present(
    path: Path,
    headers: list[dict[str, Any]] | None,
    footers: list[dict[str, Any]] | None,
) -> bool:
    """
    True when page numbering is present in the first 5 pages.

    Checks header/footer/body/text-box XML (not footer-only). Joins split
    Word runs so 'Page' '1' 'of' '67' is detected as one string.
    """
    del headers, footers
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            ordered: list[str] = []
            ordered.extend(sorted(n for n in names if n.lower().startswith("word/footer")))
            ordered.extend(sorted(n for n in names if n.lower().startswith("word/header")))
            if "word/document.xml" in names:
                ordered.append("word/document.xml")

            early_pages: set[int] = set()
            has_page_field = False
            for name in ordered:
                xml = archive.read(name).decode("utf-8", errors="ignore")
                if _xml_has_page_field(xml):
                    has_page_field = True
                early_pages |= {
                    n for n in _page_numbers_from_xml_text(xml) if 1 <= n <= 5
                }
                # Found page chrome on any of the first 5 pages.
                if early_pages:
                    return True
                # PAGE field in HF means it prints on early pages too.
                if has_page_field:
                    return True
            return False
    except Exception:
        return False


def _page_numbers_from_xml_text(xml: str) -> set[int]:
    """
    Extract N from 'Page N of M' by joining split w:t runs inside paragraphs.
    """
    found: set[int] = set()
    if not xml:
        return found
    paragraphs = re.findall(r"<w:p[\s>][\s\S]*?</w:p>", xml)
    for blob in paragraphs or [xml]:
        parts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", blob)
        joined = " ".join("".join(parts).split())
        if not joined:
            continue
        # No leading \b — chrome is often glued: "productPage 1 of 60"
        for match in re.finditer(r"page\s*(\d{1,4})\s*of\s*(\d{1,4})\b", joined, re.I):
            try:
                found.add(int(match.group(1)))
            except ValueError:
                continue
        # Short text-box fragments: "3 of 67" / "14 of"
        if len(joined) <= 40:
            match = re.fullmatch(r"(\d{1,4})\s+of(?:\s+(\d{1,4}))?", joined, re.I)
            if match:
                try:
                    found.add(int(match.group(1)))
                except ValueError:
                    pass
    return found


def _xml_has_page_field(xml: str) -> bool:
    """Detect Word PAGE / NUMPAGES field instructions anywhere in the part XML."""
    if not xml:
        return False
    for instr in re.findall(r"<w:instrText[^>]*>([^<]*)</w:instrText>", xml, re.I):
        token = " ".join(str(instr or "").upper().split())
        if re.search(r"\bPAGE\b", token) or re.search(r"\bNUMPAGES\b", token):
            return True
    if re.search(r'fldSimple[^>]*instr="[^"]*\b(?:PAGE|NUMPAGES)\b', xml, re.I):
        return True
    return False


def _clean_chrome_lines(values: list[str]) -> list[str]:
    """Keep readable header/footer lines once; drop page refs and OCR junk."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text or len(text) > 180:
            continue
        if re.fullmatch(r"page(\s+of)?", text, re.I):
            continue
        # Drop per-page footer/header page counters ("Page 12 of 67").
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", text, re.I):
            continue
        if re.fullmatch(r"page\s+\d+", text, re.I):
            continue
        # Strip glued page counters from otherwise useful lines.
        text = re.sub(r"\s*page\s+\d+\s+of\s+\d+\s*", " ", text, flags=re.I).strip()
        text = " ".join(text.split())
        if _looks_like_chrome_garbage(text):
            continue
        letters = re.sub(r"[^A-Za-z0-9]", "", text)
        if len(letters) < 3:
            continue
        # Drop heavily duplicated dual-column glue: "Title:Title:"
        if re.search(r"(.{6,}?)\1", text):
            text = re.sub(r"(.+?)\1+", r"\1", text).strip()
        if re.search(r"(.)\1{4,}", text):
            continue
        # Drop obvious OCR glyph shreds (almost no vowels / too short tokens).
        if len(text) <= 4 and not any(ch in text.casefold() for ch in "aeiou"):
            continue
        if '"' in text:
            continue
        # Drop dual-column glued junk (very long pasted lines).
        if len(text) > 90:
            continue
        # Drop broken OCR fragments ("- Microbiology", "Quality Contro-l ...").
        if text.startswith("-"):
            continue
        if "contro-l" in text.casefold():
            continue
        if _looks_like_chrome_garbage(text):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _looks_like_chrome_garbage(text: str) -> bool:
    """
    True for field/OCR debris that is not useful header/footer chrome.

    Morphology only — no fixed token denylist. Targets shreds like a digit
    glued mid-word without document-id separators (hyphen/slash/dot).
    """
    t = " ".join(str(text or "").split()).strip()
    if not t:
        return True
    # Real chrome lines are almost always multi-word or Key: Value / IDs.
    tokens = t.split()
    if len(tokens) == 1:
        tok = tokens[0]
        # Document IDs / versions usually contain separators or trailing digits.
        if "-" in tok or "/" in tok or tok.count(".") >= 1:
            return False
        # Digits sandwiched between letters → classic OCR/field fragment.
        if re.search(r"[A-Za-z]\d+[A-Za-z]", tok):
            return True
        letters = [c for c in tok if c.isalpha()]
        digits = [c for c in tok if c.isdigit()]
        if letters and digits and not any(ch in "-/._" for ch in tok):
            # Alphanumeric mash without separators (not SOP-001 style).
            if len(tok) <= 10 and len(digits) <= 2:
                vowels = sum(1 for c in tok.casefold() if c in "aeiou")
                if vowels <= 2:
                    return True
        # Short all-alpha tokens with almost no vowels (noise glyphs).
        if tok.isalpha() and 3 <= len(tok) <= 7:
            vowels = sum(1 for c in tok.casefold() if c in "aeiou")
            if vowels == 0:
                return True
    return False


def _chrome_block_quality(block: dict[str, Any]) -> tuple[int, int, int]:
    """Rank header/footer parts: prefer clean field-rich / short readable lines."""
    fields = len(block.get("fields") or [])
    clean = 0
    junk = 0
    for text in block.get("texts") or []:
        line = " ".join(str(text or "").split()).strip()
        if not line:
            continue
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", line, re.I):
            continue
        if _looks_like_chrome_garbage(line) or '"' in line or len(line) > 90:
            junk += 1
            continue
        clean += 1
    return (fields, clean, -junk)


def _chrome_once(
    blocks: list[dict[str, Any]],
    *,
    images: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Collapse all header/footer section variants into one content list.

    Prefer readable texts from the cleanest part (paragraphs + table cells).
    Also include discovered Key: Value fields. Skip page-number-only parts.
    """
    ranked = sorted(blocks or [], key=_chrome_block_quality, reverse=True)
    content: list[str] = []
    seen: set[str] = set()

    def _add_lines(lines: list[str]) -> int:
        added = 0
        for line in _clean_chrome_lines(lines):
            key = line.casefold()
            if key in seen:
                continue
            seen.add(key)
            content.append(line)
            added += 1
        return added

    for block in ranked:
        candidates: list[str] = []
        candidates.extend(str(t) for t in (block.get("texts") or []) if str(t or "").strip())
        for field in block.get("fields") or []:
            name = str(field.get("label") or field.get("name") or "").strip()
            value = str(field.get("value") or "").strip()
            if name and value:
                candidates.append(f"{name}: {value}")
        # Pair adjacent label / value lines that table cells often emit separately.
        candidates = _pair_adjacent_label_value_lines(candidates)
        added = _add_lines(candidates)
        # One substantial chrome part is enough (avoid merging every section variant).
        if added and _chrome_block_quality(block)[1] > 0:
            break

    return {
        "content": content,
        "images": _image_refs(images),
    }


def _pair_adjacent_label_value_lines(lines: list[str]) -> list[str]:
    """
    Turn consecutive chrome lines ``SOP Title`` + ``Handling of …`` into
    ``SOP Title: Handling of …`` when the left side looks like a short label.
    """
    items = [" ".join(str(x or "").split()).strip() for x in lines if str(x or "").strip()]
    if not items:
        return []
    out: list[str] = []
    i = 0
    while i < len(items):
        left = items[i]
        if i + 1 < len(items):
            right = items[i + 1]
            label = left.rstrip(":").strip()
            # Already a Key: Value line — keep as-is.
            if ":" in left and left.split(":", 1)[1].strip():
                out.append(left)
                i += 1
                continue
            if (
                is_valid_meta_key(label)
                and right
                and ":" not in right
                and not (is_valid_meta_key(right) and not any(ch.isdigit() for ch in right))
                and not re.fullmatch(r"page(\s+of)?", right, flags=re.I)
                and len(right) <= 160
            ):
                out.append(f"{label}: {right}")
                i += 2
                continue
        out.append(left)
        i += 1
    return out


def _extract_package_headers_footers(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read unique word/header*.xml and word/footer*.xml parts."""
    headers: list[dict[str, Any]] = []
    footers: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            headers, hdr_images = _parse_package_hf_parts(archive, role="header")
            footers, ftr_images = _parse_package_hf_parts(archive, role="footer")
            images = hdr_images + ftr_images
    except Exception:
        return [], [], []
    return headers, footers, images


def _parse_package_hf_parts(
    archive: zipfile.ZipFile,
    *,
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pattern = re.compile(rf"word/{role}(\d*)\.xml$", re.IGNORECASE)
    names = sorted((n for n in archive.namelist() if pattern.fullmatch(n)), key=lambda n: n.lower())
    blocks: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    seen_index: dict[str, int] = {}

    for index, name in enumerate(names):
        xml = archive.read(name).decode("utf-8", errors="ignore")
        texts, fields = _hf_texts_and_fields_from_xml(xml)
        stem = Path(name).stem
        rels = _rels_map(archive, f"word/_rels/{stem}.xml.rels")
        part_images: list[dict[str, Any]] = []
        for rid in set(re.findall(r'(?:r:embed|r:link)="([^"]+)"', xml)) | set(
            re.findall(r'r:id="([^"]+)"', xml)
        ):
            target = rels.get(rid, "")
            if not _target_looks_like_image(target):
                continue
            part = _resolve_media_part(target)
            part_images.append(
                {
                    "id": rid,
                    "part_name": part,
                    "content_type": Path(part).suffix.lstrip(".").lower(),
                    "width_emu": None,
                    "height_emu": None,
                    "width_inches": None,
                    "height_inches": None,
                    "location": role,
                    "section_index": index,
                    "in_table": False,
                    "table_index": None,
                    "row": None,
                    "column": None,
                }
            )

        text_key = hashlib_sha1("\n".join(texts))
        richness = (len(part_images), len(fields), len([t for t in texts if t.strip()]))
        if text_key in seen_index:
            prev_i = seen_index[text_key]
            prev = blocks[prev_i]
            prev_rich = (
                len(prev.get("images") or []),
                len(prev.get("fields") or []),
                len([t for t in (prev.get("texts") or []) if str(t).strip()]),
            )
            if richness > prev_rich:
                for img in list(prev.get("images") or []):
                    if img in images:
                        try:
                            images.remove(img)
                        except ValueError:
                            pass
                blocks[prev_i] = {
                    "section_index": index,
                    "part_name": name,
                    "texts": texts,
                    "tables": [],
                    "images": part_images,
                    "fields": fields,
                }
                images.extend(part_images)
            continue

        if (
            not any(t.strip() for t in texts)
            and not part_images
            and "a:blip" not in xml
            and "v:imagedata" not in xml
        ):
            continue

        seen_index[text_key] = len(blocks)
        images.extend(part_images)
        blocks.append(
            {
                "section_index": index,
                "part_name": name,
                "texts": texts,
                "tables": [],
                "images": part_images,
                "fields": fields,
            }
        )
    return blocks, images


def _package_role_has_image(path: Path, *, role: str) -> bool:
    """True when any word/header*.xml or word/footer*.xml references an image."""
    role_lc = str(role or "header").casefold()
    if role_lc not in {"header", "footer"}:
        return False
    pattern = re.compile(rf"word/{re.escape(role_lc)}\d*\.xml$", re.IGNORECASE)
    try:
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if pattern.fullmatch(n)]
            for name in names:
                xml = archive.read(name).decode("utf-8", errors="ignore")
                if "a:blip" not in xml and "v:imagedata" not in xml:
                    continue
                stem = Path(name).stem
                rels = _rels_map(archive, f"word/_rels/{stem}.xml.rels")
                for rid in set(re.findall(r'(?:r:embed|r:link)="([^"]+)"', xml)) | set(
                    re.findall(r'r:id="([^"]+)"', xml)
                ):
                    target = rels.get(rid, "")
                    if _target_looks_like_image(target):
                        return True
    except Exception:
        return False
    return False


def _hf_texts_and_fields_from_xml(xml: str) -> tuple[list[str], list[dict[str, str]]]:
    """
    Extract header/footer texts from paragraphs AND tables (cell order).

    Table rows with 2/4/... cells become Label|Value pairs when morphology fits.
    """
    texts: list[str] = []
    fields: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    except Exception:
        raw = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml)
        joined = " ".join(t.strip() for t in raw if t.strip())
        return ([joined] if joined else [], [])

    def _append_line(line: str) -> None:
        cleaned = " ".join(str(line or "").split()).strip()
        if not cleaned:
            return
        if texts and texts[-1] == cleaned:
            return
        texts.append(cleaned)

    def _walk(node: ET.Element) -> None:
        for child in list(node):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "tbl":
                _consume_table(child)
            elif tag == "p":
                line = _xml_paragraph_text(child)
                if line:
                    _append_line(line)
                    if ":" in line:
                        fields.extend(_colon_fields_from_lines([line]))
            elif tag in {"sdt", "sdtContent", "hdr", "ftr"}:
                _walk(child)

    def _consume_table(tbl: ET.Element) -> None:
        for row in tbl.findall(f"./{{{W_NS}}}tr"):
            cells: list[str] = []
            for tc in row.findall(f"./{{{W_NS}}}tc"):
                parts: list[str] = []
                for p in tc.findall(f".//{{{W_NS}}}p"):
                    line = _xml_paragraph_text(p)
                    if line:
                        parts.append(line)
                cell = " ".join(parts).strip()
                if cell and (not cells or cells[-1] != cell):
                    cells.append(cell)
            if not cells:
                continue
            if len(cells) == 1:
                _append_line(cells[0])
                if ":" in cells[0]:
                    fields.extend(_colon_fields_from_lines([cells[0]]))
                continue
            # Pair cells as Label | Value (2-col or 4-col chrome tables).
            pairs_ok = len(cells) % 2 == 0 and len(cells) <= 8
            if pairs_ok:
                for i in range(0, len(cells), 2):
                    left = cells[i]
                    right = cells[i + 1] if i + 1 < len(cells) else ""
                    label = left.split(":", 1)[0].strip().rstrip(".") if ":" in left else left.rstrip(":").strip()
                    inline = left.split(":", 1)[1].strip() if ":" in left else ""
                    value = inline or right.strip()
                    if (
                        is_valid_meta_key(label)
                        and value
                        and value.casefold() != label.casefold()
                        # Reject label|label rows; allow IDs/dates (digit-bearing values).
                        and not (
                            is_valid_meta_key(value) and not any(ch.isdigit() for ch in value)
                        )
                        and not re.fullmatch(r"page(\s+of)?", value, flags=re.I)
                    ):
                        fields.append({"label": label, "name": label, "value": value})
                        _append_line(f"{label}: {value}")
                    else:
                        _append_line(left)
                        if right:
                            _append_line(right)
            else:
                for cell in cells:
                    _append_line(cell)
                    if ":" in cell:
                        fields.extend(_colon_fields_from_lines([cell]))

    _walk(root)
    # Also pick up any top-level paragraphs the walk may miss in unusual roots.
    if not texts:
        for paragraph in root.iter(f"{{{W_NS}}}p"):
            line = _xml_paragraph_text(paragraph)
            if line:
                _append_line(line)
    return texts, _unique_fields(fields)


def _extract_headers_footers(doc: Document, *, role: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract unique header/footer parts via python-docx (skip linked duplicates)."""
    blocks: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for index, section in enumerate(doc.sections):
        container = section.header if role == "header" else section.footer
        if container is None:
            continue
        if index > 0:
            try:
                if container.is_linked_to_previous:
                    continue
            except Exception:
                pass

        texts = [
            " ".join((p.text or "").split()).strip()
            for p in container.paragraphs
            if " ".join((p.text or "").split()).strip()
        ]
        tables, table_images = _extract_tables_from_container(
            container.tables,
            location=f"{role}_table",
            section_index=index,
        )
        table_text: list[str] = []
        table_fields: list[dict[str, str]] = []
        for table in tables:
            cell_rows = table.get("cells") or []
            for row in cell_rows:
                cells = [" ".join(str(c.get("text") or "").split()).strip() for c in row]
                cells = [c for c in cells if c]
                for cell in cells:
                    table_text.append(cell)
                if len(cells) == 1:
                    texts.append(cells[0])
            table_fields.extend(_fields_from_table_cells(cell_rows))
        for field in table_fields:
            name = str(field.get("label") or field.get("name") or "").strip()
            value = str(field.get("value") or "").strip()
            if name and value:
                texts.append(f"{name}: {value}")
        signature = hashlib_sha1("\n".join(texts + table_text))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        images.extend(table_images)
        para_images = _images_from_paragraphs(
            container.paragraphs,
            location=role,
            section_index=index,
        )
        images.extend(para_images)
        fields = _unique_fields(_colon_fields_from_lines(texts) + table_fields)
        texts = _unique_strings(texts)

        if not texts and not tables and not para_images and not table_images and not fields:
            continue

        blocks.append(
            {
                "section_index": index,
                "texts": texts,
                "tables": tables,
                "images": para_images + table_images,
                "fields": fields,
            }
        )
    return blocks, images


def _extract_repeated_fields(
    doc: Document,
    result: dict[str, Any],
    *,
    body_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Return repeating fields AND repeating content once each:
      [{ "name", "value", "occurrences" }, ...]

    - Key/Value chrome (Facility / Department / …): name=label, value=value
    - Repeated plain content (Confidential…): name=text, value=\"\"
    """
    from .page_meta import is_valid_meta_key, metadata_slot_for_key, normalize_meta_key

    value_counts: dict[str, Counter[str]] = {}
    label_display: dict[str, str] = {}
    label_hits: Counter[str] = Counter()
    content_hits: Counter[str] = Counter()
    content_display: dict[str, str] = {}
    content_from_hf: set[str] = set()
    marker_keys: set[str] = set()

    def _note_field(label: str, value: str) -> None:
        name = " ".join(str(label or "").split()).strip().rstrip(":")
        val = " ".join(str(value or "").split()).strip()
        if not name:
            return
        tokens = name.split()
        letters = "".join(ch for ch in name if ch.isalpha())
        # Short ALL-CAPS annotation labels belong in repeated content, not Key/Value.
        if (
            len(tokens) == 1
            and tokens[0] == tokens[0].upper()
            and 3 <= len(letters) <= 12
            and not any(ch.isdigit() for ch in tokens[0])
        ):
            _note_marker(name)
            return
        if not val:
            return
        if not is_valid_meta_key(name):
            return
        if any(len(tok) == 1 for tok in tokens):
            return
        if not all(tok[:1].isupper() for tok in tokens if tok[:1].isalpha()):
            return
        if metadata_slot_for_key(name):
            return
        if len(val) > 60 or len(val.split()) > 8:
            return
        if sum(1 for ch in val if ch.isalnum()) < 2:
            return
        if name.count("-") >= 2 and any(ch.isdigit() for ch in name):
            return
        # Skip glued label rows like "Facility BMS1 Department".
        if len(tokens) >= 3 and tokens[0].casefold() == "facility":
            return
        key = normalize_meta_key(name)
        if not key:
            return
        label_display[key] = name
        label_hits[key] += 1
        value_counts.setdefault(key, Counter())[val] += 1

    def _note_marker(text: str) -> None:
        """Count short ALL-CAPS annotation markers (from 'LABEL:' lines) as repeated content."""
        cleaned = " ".join(str(text or "").split()).strip().rstrip(":")
        if not cleaned:
            return
        words = cleaned.split()
        if len(words) != 1:
            return
        token = words[0]
        letters = "".join(ch for ch in token if ch.isalpha())
        if len(letters) < 3 or len(letters) > 12:
            return
        if token != token.upper():
            return
        if any(ch.isdigit() for ch in token):
            return
        key = cleaned.casefold()
        content_hits[key] += 1
        content_display.setdefault(key, cleaned)
        marker_keys.add(key)

    def _note_content(text: str, *, from_hf: bool = False) -> None:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return
        # Annotation markers are written as LABEL: or LABEL: body (not bare PASS/FAIL tokens).
        if ":" in cleaned:
            left, _right = cleaned.split(":", 1)
            left = left.strip()
            words = left.split()
            if len(words) == 1:
                token = words[0]
                letters = "".join(ch for ch in token if ch.isalpha())
                if (
                    3 <= len(letters) <= 12
                    and token == token.upper()
                    and not any(ch.isdigit() for ch in token)
                ):
                    # Count marker itself; body text after ':' varies by occurrence.
                    _note_marker(left)
                    return
            return
        words = cleaned.split()
        if not (2 <= len(words) <= 10):
            return
        if len(cleaned) > 80:
            return
        if cleaned.startswith("-") or cleaned.endswith("."):
            return
        if "contro-l" in cleaned.casefold():
            return
        if "/" in cleaned or '"' in cleaned or "<" in cleaned or "$" in cleaned:
            return
        if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", cleaned, re.I):
            return
        if _SECTION_INLINE_RE.match(cleaned) or _SECTION_NUM_ONLY_RE.match(cleaned):
            return
        number, title = _parse_outline(cleaned)
        if number and title:
            return
        if cleaned[:1].islower():
            return
        letters = re.sub(r"[^A-Za-z0-9]", "", cleaned)
        if len(letters) < 8:
            return
        # Prefer banner/chrome phrasing: mostly capitals or Title Case words.
        upper_ratio = sum(1 for ch in cleaned if ch.isupper()) / max(1, sum(1 for ch in cleaned if ch.isalpha()))
        titleish = all(w[:1].isupper() for w in words if w[:1].isalpha())
        if not from_hf and upper_ratio < 0.35 and not titleish:
            return
        key = cleaned.casefold()
        content_hits[key] += 1
        content_display.setdefault(key, cleaned)
        if from_hf:
            content_from_hf.add(key)

    def _is_clean_value(value: str) -> bool:
        text = str(value or "").strip()
        if not text or text.endswith("-"):
            return False
        return all(ch.isalnum() or ch.isspace() for ch in text)

    # 1) Bold-label chrome rows in body + candidate body content.
    if body_items is None:
        for paragraph in _iter_paragraphs(doc):
            text = " ".join((paragraph.text or "").split()).strip()
            if text:
                _note_content(text, from_hf=False)
            for name, value in _bold_label_value_pairs_from_paragraph(paragraph):
                _note_field(name, value)
    else:
        for item in body_items:
            text = str(item.get("text") or "").strip()
            if text:
                _note_content(text, from_hf=False)
            for name, value in item.get("bold_pairs") or []:
                _note_field(str(name or ""), str(value or ""))

    # 2) Header/footer texts + colon pairs (chrome content is trustworthy).
    hf_texts: list[str] = []
    for block in (result.get("headers") or []) + (result.get("footers") or []):
        for line in block.get("texts") or []:
            hf_texts.append(str(line))
            _note_content(str(line), from_hf=True)
        for field in block.get("fields") or []:
            _note_field(str(field.get("label") or field.get("name") or ""), str(field.get("value") or ""))
    for field in _colon_fields_from_lines(hf_texts):
        _note_field(str(field.get("label") or field.get("name") or ""), str(field.get("value") or ""))

    field_name_keys = set(label_hits.keys())
    field_value_keys = {
        normalize_meta_key(val) for counter in value_counts.values() for val in counter.keys()
    }
    field_labels_cf = {label_display[k].casefold() for k in label_display}

    out: list[dict[str, Any]] = []
    for key, hits in label_hits.most_common():
        if hits < 2:
            continue
        common = value_counts.get(key)
        if not common:
            continue
        clean_pool = [(val, count) for val, count in common.items() if _is_clean_value(val) and count >= 2]
        pool = clean_pool or list(common.items())
        value = max(pool, key=lambda item: (item[1], sum(1 for ch in item[0] if ch.isalnum())))[0]
        out.append(
            {
                "name": label_display[key],
                "value": value,
                "occurrences": int(hits),
            }
        )

    for key, hits in content_hits.most_common():
        # HF/chrome / annotation markers: >=2. Body-only banners need more repetition.
        min_hits = 2 if key in content_from_hf or key in marker_keys else 5
        if hits < min_hits:
            continue
        display = content_display[key]
        norm = normalize_meta_key(display)
        if norm in field_name_keys or norm in field_value_keys:
            continue
        # Skip glued field rows already represented as Key:Value pairs.
        if any(label in display.casefold() for label in field_labels_cf):
            # Allow only pure confidentiality / SOP banners, not Facility/Department glue.
            banner = display.casefold()
            if not (
                "confidential" in banner
                or "uncontrolled" in banner
                or "standard operating procedure" in banner
            ):
                continue
        out.append(
            {
                "name": display,
                "value": "",
                "occurrences": int(hits),
            }
        )
    return out
