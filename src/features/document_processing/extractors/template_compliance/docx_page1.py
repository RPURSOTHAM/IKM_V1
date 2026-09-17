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
    _colon_fields_from_lines,
    _document_type_from_chrome,
    _iter_paragraphs_with_meta,
    _unique_fields,
)
from .docx_images import (
    _images_from_paragraphs,
)
from .docx_tables import (
    _control_pairs_from_table,
    _dedupe_first_page_tables,
    _extract_page1_tables_from_docx,
    _is_body_section_table,
    _is_first_body_section_heading,
    _materialize_first_page_tables,
    _pick_signature_table,
)

def _extract_first_page_chrome(doc: Document, *, path: Path | None = None) -> dict[str, Any]:
    """
    Capture page-1 body chrome only:
      - colon Key: Value control fields
      - early tables from paragraph stream AND python-docx Table API
      - logo images

    Stream table indexes and ``doc.tables`` indexes can differ, so both
    sources are collected (early only) then deduped.
    """
    lines: list[str] = []
    images: list[dict[str, Any]] = []
    row_cells: dict[tuple[int, int], dict[int, str]] = {}
    table_titles: dict[int, str] = {}
    last_line = ""
    # Stream table indexes can diverge from doc.tables. Cap early; chrome filter
    # drops procedure noise even if a few mid-front tables sneak in.
    # Hard stop is the first body H1 (e.g. "1.0 PURPOSE").
    max_stream_tables = 4

    for paragraph, meta in _iter_paragraphs_with_meta(doc):
        text_line = " ".join((paragraph.text or "").split()).strip()
        in_table = bool((meta or {}).get("in_table"))

        if text_line and not in_table:
            if _is_first_body_section_heading(text_line):
                break

        para_images = _images_from_paragraphs(
            [paragraph],
            location="first_page_logo",
            section_index=0,
        )
        if para_images:
            images.extend(para_images)

        if in_table:
            try:
                table_index = int((meta or {}).get("table_index"))
            except (TypeError, ValueError):
                table_index = -1
            try:
                row_index = int((meta or {}).get("row_index"))
            except (TypeError, ValueError):
                row_index = -1
            try:
                cell_index = int((meta or {}).get("cell_index"))
            except (TypeError, ValueError):
                cell_index = -1
            if table_index < 0 or table_index >= max_stream_tables:
                continue
            if row_index >= 0 and cell_index >= 0 and text_line:
                if table_index not in table_titles and last_line and ":" not in last_line:
                    table_titles[table_index] = last_line
                bucket = row_cells.setdefault((table_index, row_index), {})
                prev = bucket.get(cell_index) or ""
                if len(text_line) > len(prev):
                    bucket[cell_index] = text_line
            continue

        if text_line:
            lines.append(text_line)
            last_line = text_line

    stream_tables = _materialize_first_page_tables(row_cells, table_titles)
    docx_tables = _extract_page1_tables_from_docx(doc, path=path)
    tables = _dedupe_first_page_tables(stream_tables + docx_tables)
    tables = [t for t in tables if not _is_body_section_table(t)]
    # Used only for signature_table + control metadata slots (not emitted as tables).
    signature_table = _pick_signature_table(tables)

    fields = _colon_fields_from_lines(lines)
    for table in tables:
        for pair in _control_pairs_from_table(table):
            fields.append({"label": pair["key"], "name": pair["key"], "value": pair["value"]})
    fields = _unique_fields(fields)
    fields = [
        {
            "label": str(f.get("label") or f.get("name") or "").strip(),
            "name": str(f.get("name") or f.get("label") or "").strip(),
            "value": str(f.get("value") or "").strip(),
        }
        for f in fields
        if str(f.get("value") or "").strip()
    ]

    document_type = _document_type_from_chrome(lines, [])
    if not document_type:
        for table in tables:
            title = str(table.get("title") or "").strip()
            if title and ":" not in title and 2 <= len(title.split()) <= 6:
                document_type = title
                break

    return {
        "lines": lines,
        "fields": fields,
        "images": images,
        "signature_table": signature_table,
        "document_type": document_type,
    }
