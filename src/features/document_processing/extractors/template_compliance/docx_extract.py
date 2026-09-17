"""Full DOCX template extractor → common JSON schema.

Implementation helpers live in sibling modules (docx_*), kept for
maintainability. Public entry: extract_docx_template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document

from .schema import empty_template_document
from .page_meta import metadata_from_colon_fields

from .docx_common import (
    _CORRUPTED_NUM_ONLY_RE,
    _SECTION_NUM_ONLY_RE,
    _document_type_from_chrome,
    _full_para_text,
    _is_valid_outline_title,
    _iter_paragraphs_with_meta,
    _load_numbering_resolver,
    _normalize_number,
    _number_sort_key,
    _outline_level,
    _parse_outline,
)
from .docx_fonts import (
    _package_hf_font,
)
from .docx_hf import (
    _chrome_once,
    _detect_page_field_present,
    _extract_headers_footers,
    _extract_package_headers_footers,
    _extract_repeated_fields,
    _package_role_has_image,
)
from .docx_images import (
    _count_body_images_and_tables,
    _dedupe_images,
    _extract_body_images,
    _images_from_paragraphs,
)
from .docx_page1 import (
    _extract_first_page_chrome,
)
from .docx_sections import (
    _extract_sections,
    scan_document_body,
)
from .docx_tables import (
    _extract_tables,
)

from .docx_hf import _clean_chrome_lines  # noqa: F401  (tests)

def extract_docx_template(path: str | Path) -> dict[str, Any]:
    """Extract a DOCX template into the shared compliance JSON schema."""
    file_path = Path(path)
    doc = Document(str(file_path))
    payload = empty_template_document(
        source=str(file_path.resolve()),
        source_format="docx",
        document_name=file_path.name,
    )
    result = payload["document"]
    result["file_name"] = file_path.name
    result["file_type"] = "docx"

    package_headers, package_footers, package_images = _extract_package_headers_footers(file_path)
    if package_headers:
        headers = package_headers
        header_images = [img for img in package_images if img.get("location") == "header"]
    else:
        headers, header_images = _extract_headers_footers(doc, role="header")
    if package_footers:
        footers = package_footers
        footer_images = [img for img in package_images if img.get("location") == "footer"]
    else:
        footers, footer_images = _extract_headers_footers(doc, role="footer")

    first_page = _extract_first_page_chrome(doc, path=file_path)
    # Metadata: page-1 control fields only (not body definitions / random colons).
    control_fields = list(first_page.get("fields") or [])
    metadata = metadata_from_colon_fields(control_fields)
    document_type = str(first_page.get("document_type") or "").strip()
    if not document_type:
        document_type = _document_type_from_chrome(
            list((first_page.get("lines") or [])),
            # header filled below — resolve later if empty
            [],
        )
    if document_type:
        metadata["document_type"] = document_type
    else:
        metadata["document_type"] = ""
    result["metadata"] = metadata
    result["signature_table"] = dict(
        first_page.get("signature_table")
        or {
            "title": "",
            "columns": [],
            "rows": [],
        }
    )
    result.pop("first_page_tables", None)
    header_imgs_raw = header_images or [img for block in headers for img in (block.get("images") or [])]
    footer_imgs_raw = footer_images or [img for block in footers for img in (block.get("images") or [])]
    # One header / one footer: unique field lines only (arrays, not per-page copies).
    header_chrome = _chrome_once(headers, images=header_imgs_raw)
    footer_chrome = _chrome_once(footers, images=footer_imgs_raw)
    result["header"] = list(header_chrome.get("content") or [])
    result["footer"] = list(footer_chrome.get("content") or [])
    if not str(result["metadata"].get("document_type") or "").strip():
        doc_type = _document_type_from_chrome([], list(result["header"] or []))
        if doc_type:
            result["metadata"]["document_type"] = doc_type
    header_imgs = list(header_chrome.get("images") or [])
    footer_imgs = list(footer_chrome.get("images") or [])
    # Logo is separate: present + location (header preferred over footer).
    in_header = _package_role_has_image(file_path, role="header") or bool(header_imgs)
    in_footer = _package_role_has_image(file_path, role="footer") or bool(footer_imgs)
    if in_header:
        result["logo"] = {"present": True, "location": "header"}
    elif in_footer:
        result["logo"] = {"present": True, "location": "footer"}
    else:
        result["logo"] = {"present": False, "location": ""}

    # One body walk: outline/sections + fonts + TOC + spacing + media rows + repeated notes.
    body_scan = scan_document_body(doc)

    fonts_raw = body_scan.get("fonts") or {}
    spacing_raw = body_scan.get("line_spacing") or {}
    header_font = _package_hf_font(file_path, role="header")
    footer_font = _package_hf_font(file_path, role="footer")
    body_name = str(fonts_raw.get("body_font") or "")
    body_size = fonts_raw.get("body_size_pt")
    heading_name = str(fonts_raw.get("heading_font") or body_name or "")
    heading_size = fonts_raw.get("heading_size_pt") or body_size
    header_name = str(header_font.get("name") or "")
    header_size = header_font.get("size")
    footer_name = str(footer_font.get("name") or "")
    footer_size = footer_font.get("size")
    # Fall back to body fonts only when the package part has no usable rFonts.
    if not header_name:
        header_name = heading_name or body_name or ""
    if header_size is None:
        header_size = heading_size or body_size
    if not footer_name:
        footer_name = body_name or heading_name or ""
    if footer_size is None:
        footer_size = body_size or heading_size
    result["fonts"] = {
        "header": {"name": header_name, "size": str(header_size or "")},
        "footer": {"name": footer_name, "size": str(footer_size or "")},
        "section_heading": {"name": heading_name, "size": str(heading_size or "")},
        "section_content": {"name": body_name, "size": str(body_size or "")},
        "minimum_line_spacing": str(spacing_raw.get("minimum") or ""),
    }

    toc_raw = body_scan.get("toc") or {}
    result["toc"] = {"present": bool(toc_raw.get("present"))}
    result["page"] = {
        "present": _detect_page_field_present(
            file_path,
            package_headers or headers,
            package_footers or footers,
        )
    }

    sections_raw, _indents = _extract_sections(doc, scan=body_scan)
    sections_out: list[dict[str, Any]] = []
    for section in sections_raw:
        if int(section.get("level") or 0) != 1:
            continue
        number = _display_section_number(str(section.get("number") or ""), level=1)
        title = str(section.get("title") or "").strip()
        children = [
            {
                "number": _display_section_number(str(child.get("number") or ""), level=2),
                "title": str(child.get("title") or "").strip(),
                "indentation": _format_indent(
                    child.get("indent_inches"),
                    child.get("indent_twips"),
                    child.get("alignment"),
                ),
                "image_count": 0,
                "table_count": 0,
            }
            for child in sorted(
                [
                    c
                    for c in (section.get("children") or [])
                    if int(c.get("level") or 0) == 2
                ],
                key=lambda c: _number_sort_key(str(c.get("number") or "")),
            )
        ]
        sections_out.append(
            {
                "number": number,
                "title": title,
                "indentation": _format_indent(
                    section.get("indent_inches"),
                    section.get("indent_twips"),
                    section.get("alignment"),
                ),
                "image_count": 0,
                "table_count": 0,
                "subsections": children,
            }
        )

    # Keep level-2 only through PROCEDURE (inclusive). Later majors stay flat.
    procedure_major = _procedure_section_major(sections_out)
    if procedure_major is not None:
        for section in sections_out:
            try:
                major = int(str(section.get("number") or "0").split(".")[0])
            except ValueError:
                major = 0
            if major > procedure_major:
                section["subsections"] = []

    body_tables, table_images = _extract_tables(doc)
    body_images = _extract_body_images(doc)
    all_body_images = _dedupe_images(
        body_images + table_images + list(first_page.get("images") or [])
    )
    body_only_images = [
        img for img in all_body_images if img.get("location") not in {"header", "footer"}
    ]

    _assign_section_media_counts(
        doc,
        sections_out,
        body_tables,
        body_only_images,
        media_rows=list(body_scan.get("media_rows") or []),
    )
    result["sections"] = sections_out

    # Accurate document-body totals from OOXML (python-docx misses nested/floating media).
    package_counts = _count_body_images_and_tables(file_path)
    result["total_images"] = int(package_counts.get("images") or 0)
    result["total_tables"] = int(package_counts.get("tables") or 0)
    result["repeated_fields"] = _extract_repeated_fields(
        doc,
        {"headers": headers, "footers": footers},
        body_items=list(body_scan.get("repeated_body") or []),
    )
    return payload


def _procedure_section_major(sections: list[dict[str, Any]]) -> int | None:
    """Return the major number of the PROCEDURE section, if present."""
    for section in sections:
        words = " ".join(str(section.get("title") or "").casefold().split()).split()
        # Exact title only — not "STANDARD OPERATING PROCEDURE".
        if words == ["procedure"]:
            try:
                return int(str(section.get("number") or "").split(".")[0])
            except ValueError:
                return None
    return None


def _assign_section_media_counts(
    doc: Document,
    sections: list[dict[str, Any]],
    body_tables: list[dict[str, Any]],
    body_images: list[dict[str, Any]],
    *,
    media_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Fill image_count / table_count on each section and subsection (document order)."""
    del body_tables, body_images  # attribution uses body order / media_rows
    by_number: dict[str, dict[str, Any]] = {}
    known_majors: set[str] = set()
    known_minors: set[str] = set()
    for section in sections:
        num = str(section.get("number") or "")
        by_number[num] = section
        known_majors.add(num)
        for child in section.get("subsections") or []:
            cnum = str(child.get("number") or "")
            by_number[cnum] = child
            known_minors.add(cnum)

    current_major = ""
    current_minor = ""
    counted_tables: set[int] = set()
    seen_image_ids: dict[str, set[str]] = {}

    def _target_key() -> str:
        if current_minor and current_minor in by_number:
            return current_minor
        if current_major and current_major in by_number:
            return current_major
        return ""

    def _update_outline(text: str) -> None:
        nonlocal current_major, current_minor
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return
        number, title = _parse_outline(cleaned)
        if not number:
            only = _SECTION_NUM_ONLY_RE.match(cleaned) or _CORRUPTED_NUM_ONLY_RE.match(cleaned)
            if only:
                number = _normalize_number(only.group(1))
                title = ""
        if not number:
            return
        level = _outline_level(number)
        if level == 1:
            display = _display_section_number(number, level=1)
            if display in known_majors and (not title or _is_valid_outline_title(title, 1)):
                current_major = display
                current_minor = ""
            return
        if level == 2:
            display = _display_section_number(number, level=2)
            major = display.split(".")[0]
            if major in known_majors:
                current_major = major
            if display in known_minors and (not title or _is_valid_outline_title(title, 2)):
                current_minor = display
            elif major in known_majors:
                current_minor = ""

    if media_rows is None:
        numbering = _load_numbering_resolver(doc)
        rows: list[dict[str, Any]] = []
        for paragraph, meta in _iter_paragraphs_with_meta(doc):
            text = _full_para_text(paragraph, numbering)
            in_table = bool((meta or {}).get("in_table"))
            table_index = int((meta or {}).get("table_index") or -1)
            images = _images_from_paragraphs(
                [paragraph],
                location="body",
                section_index=None,
                table_index=table_index if in_table and table_index >= 0 else None,
            )
            rows.append(
                {
                    "text": text,
                    "in_table": in_table,
                    "table_index": table_index,
                    "image_ids": [str(img.get("id") or "").strip() for img in images if img.get("id")],
                }
            )
        media_rows = rows

    for row in media_rows:
        text = str(row.get("text") or "")
        if text:
            _update_outline(text)
        in_table = bool(row.get("in_table"))
        table_index = int(row.get("table_index") or -1)
        target_key = _target_key()
        target = by_number.get(target_key) if target_key else None

        if in_table and table_index >= 0 and table_index not in counted_tables:
            counted_tables.add(table_index)
            if target is not None:
                target["table_count"] = int(target.get("table_count") or 0) + 1

        if target is None or not target_key:
            continue
        seen = seen_image_ids.setdefault(target_key, set())
        for rid in row.get("image_ids") or []:
            rid_s = str(rid or "").strip()
            if not rid_s or rid_s in seen:
                continue
            seen.add(rid_s)
            target["image_count"] = int(target.get("image_count") or 0) + 1


def _display_section_number(number: str, *, level: int) -> str:
    """Normalize outline ids: level-1 as '2', level-2 as '2.1'."""
    parts = [p for p in str(number or "").split(".") if p != ""]
    if not parts:
        return str(number or "")
    if level == 1:
        return str(int(parts[0])) if parts[0].isdigit() else parts[0]
    if len(parts) >= 2:
        major = str(int(parts[0])) if parts[0].isdigit() else parts[0]
        minor = str(int(parts[1])) if parts[1].isdigit() else parts[1]
        return f"{major}.{minor}"
    return str(number)


def _format_indent(inches: Any, twips: Any, alignment: Any = None) -> dict[str, str]:
    """Return indentation object: left indent inches + alignment label."""
    indent = ""
    if inches is not None:
        try:
            indent = f"{float(inches):.3f} in"
        except Exception:
            indent = ""
    if not indent and twips is not None:
        try:
            indent = f"{float(twips) / 1440.0:.3f} in"
        except Exception:
            indent = ""
    align = str(alignment or "").strip()
    if align not in {"Left", "Right", "Center", "Justified"}:
        # Normalize common variants; default empty when unknown.
        key = align.casefold()
        if key in {"left", "start"}:
            align = "Left"
        elif key in {"right", "end"}:
            align = "Right"
        elif key == "center":
            align = "Center"
        elif key in {"justified", "justify", "both"}:
            align = "Justified"
        else:
            align = ""
    return {"indent": indent, "alignment": align}
