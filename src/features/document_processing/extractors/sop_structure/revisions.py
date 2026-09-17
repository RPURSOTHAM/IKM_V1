"""Revision history detection from tables and content."""

from __future__ import annotations

import re
from typing import Any

from src.features.document_processing.extractors.sop_structure.helpers import looks_like_heading_title

_VERSION_HEADERS = {"version", "rev", "revision", "rev no", "rev. no", "revision no", "ver", "ver."}
_DATE_HEADERS = {"date", "rev date", "revision date", "effective date", "approved date"}
_DESC_HEADERS = {
    "description",
    "change",
    "changes",
    "change description",
    "reason",
    "remarks",
    "details",
    "summary",
}
_AUTHOR_HEADERS = {"author", "prepared by", "revised by", "changed by", "by", "name", "owner"}


def _norm_header(cell: str) -> str:
    return re.sub(r"\s+", " ", str(cell or "").strip().lower()).rstrip(".")


def _map_columns(headers: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, raw in enumerate(headers):
        h = _norm_header(raw)
        if h in _VERSION_HEADERS and "version" not in mapping:
            mapping["version"] = idx
        elif h in _DATE_HEADERS and "date" not in mapping:
            mapping["date"] = idx
        elif h in _DESC_HEADERS and "description" not in mapping:
            mapping["description"] = idx
        elif h in _AUTHOR_HEADERS and "author" not in mapping:
            mapping["author"] = idx
    return mapping


def _is_revision_table(headers: list[str], title: str = "") -> bool:
    mapping = _map_columns(headers)
    if "version" in mapping and ("date" in mapping or "description" in mapping):
        return True
    if looks_like_heading_title(title, ("revision", "revision history", "document history", "change history")):
        return "version" in mapping or "date" in mapping
    return False


def extract_revisions(template_data: dict[str, Any]) -> list[dict[str, Any]]:
    revisions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for table in template_data.get("tables") or []:
        if not isinstance(table, dict):
            continue
        headers = [str(c or "") for c in (table.get("header_row") or [])]
        title = str(
            table.get("title")
            or table.get("caption")
            or table.get("table_caption")
            or ""
        )
        rows_probe = list(table.get("row_data") or [])
        if not _is_revision_table(headers, title):
            # Try first row of row_data as header if header_row empty / insufficient
            if rows_probe:
                probe_headers = [str(c or "") for c in (rows_probe[0] or [])]
                if _is_revision_table(probe_headers, title):
                    headers = probe_headers
                    data_rows = rows_probe[1:]
                else:
                    continue
            else:
                continue
        else:
            data_rows = rows_probe
            # If row_data still includes the header as first row, skip it
            if data_rows and _map_columns([str(c or "") for c in (data_rows[0] or [])]) == _map_columns(headers):
                # Compare normalized — drop duplicate header row
                first = [_norm_header(c) for c in (data_rows[0] or [])]
                hdr = [_norm_header(c) for c in headers]
                if first == hdr:
                    data_rows = data_rows[1:]

        colmap = _map_columns(headers)
        if not colmap:
            continue

        for row in data_rows:
            cells = [str(c or "").strip() for c in (row or [])]
            if not any(cells):
                continue
            # Skip if this row looks like a repeated header
            if _map_columns(cells) and len(_map_columns(cells)) >= 2:
                continue

            def cell(key: str) -> str:
                idx = colmap.get(key)
                if idx is None or idx >= len(cells):
                    return ""
                return cells[idx].strip()

            version = cell("version")
            date = cell("date")
            description = cell("description")
            author = cell("author")
            if not version and not description and not date:
                continue
            key = (version.lower(), date.lower(), description.lower()[:80])
            if key in seen:
                continue
            seen.add(key)
            revisions.append(
                {
                    "version": version,
                    "date": date,
                    "description": description,
                    "author": author,
                    "page": table.get("page_start") or table.get("page"),
                    "table_id": table.get("table_id") or table.get("id"),
                }
            )

    return revisions
