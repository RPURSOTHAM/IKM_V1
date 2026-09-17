"""Approval block detection (Prepared/Reviewed/Approved By)."""

from __future__ import annotations

import re
from typing import Any

_FIELD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("prepared_by", re.compile(r"^\s*prepared\s*by\s*[:\-–—]?\s*(?P<val>.*)$", re.I)),
    ("reviewed_by", re.compile(r"^\s*reviewed\s*by\s*[:\-–—]?\s*(?P<val>.*)$", re.I)),
    ("approved_by", re.compile(r"^\s*approved\s*by\s*[:\-–—]?\s*(?P<val>.*)$", re.I)),
    ("authorized_by", re.compile(r"^\s*authorized\s*by\s*[:\-–—]?\s*(?P<val>.*)$", re.I)),
    ("approved_date", re.compile(r"^\s*approved\s*date\s*[:\-–—]?\s*(?P<val>.*)$", re.I)),
    ("review_date", re.compile(r"^\s*review(?:ed)?\s*date\s*[:\-–—]?\s*(?P<val>.*)$", re.I)),
    ("prepared_date", re.compile(r"^\s*prepared\s*date\s*[:\-–—]?\s*(?P<val>.*)$", re.I)),
]

_INLINE = re.compile(
    r"(?P<label>Prepared\s*By|Reviewed\s*By|Approved\s*By|Authorized\s*By|"
    r"Approved\s*Date|Review(?:ed)?\s*Date|Prepared\s*Date)\s*[:\-–—]?\s*(?P<val>[^\n|]+)",
    re.I,
)

_LABEL_TO_FIELD = {
    "prepared by": "prepared_by",
    "reviewed by": "reviewed_by",
    "approved by": "approved_by",
    "authorized by": "authorized_by",
    "approved date": "approved_date",
    "review date": "review_date",
    "reviewed date": "review_date",
    "prepared date": "prepared_date",
}


def _clean_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip())
    cleaned = cleaned.strip(" ._-")
    # Drop signature placeholders
    if cleaned.lower() in {"", "n/a", "na", "sign", "signature", "name", "date"}:
        return ""
    return cleaned


def extract_approvals(
    template_data: dict[str, Any],
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return zero or one Approval dict when approval fields are detected."""
    fields: dict[str, str] = {}
    page = None

    # Scan content blocks
    for block in content_blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        for field, pattern in _FIELD_PATTERNS:
            m = pattern.match(text)
            if m:
                val = _clean_value(m.group("val"))
                if val or field not in fields:
                    if val:
                        fields[field] = val
                    page = page or block.get("page")
                break
        else:
            for m in _INLINE.finditer(text):
                label = re.sub(r"\s+", " ", m.group("label").strip().lower())
                field = _LABEL_TO_FIELD.get(label)
                if not field:
                    continue
                val = _clean_value(m.group("val"))
                if val:
                    fields[field] = val
                    page = page or block.get("page")

    # Also scan table cells (revision/approval tables)
    for table in template_data.get("tables") or []:
        if not isinstance(table, dict):
            continue
        rows = table.get("row_data") or []
        header = [str(c or "").strip() for c in (table.get("header_row") or [])]
        # Pair header+first data rows as label/value
        for row in rows:
            cells = [str(c or "").strip() for c in (row or [])]
            if len(cells) >= 2:
                label = cells[0]
                val = cells[1]
                for field, pattern in _FIELD_PATTERNS:
                    if pattern.match(label) or pattern.match(f"{label}: {val}"):
                        cleaned = _clean_value(val)
                        if cleaned:
                            fields[field] = cleaned
                            page = page or table.get("page_start") or table.get("page")
            # Also check "Label: value" in any cell
            for cell in cells:
                for field, pattern in _FIELD_PATTERNS:
                    m = pattern.match(cell)
                    if m:
                        cleaned = _clean_value(m.group("val"))
                        if cleaned:
                            fields[field] = cleaned
        # Header columns named Prepared By etc. with values in rows
        header_fields: list[str | None] = []
        for h in header:
            mapped = None
            for field, pattern in _FIELD_PATTERNS:
                if pattern.match(h) or pattern.match(h + ":"):
                    mapped = field
                    break
            if mapped is None:
                key = re.sub(r"\s+", " ", h.strip().lower())
                mapped = _LABEL_TO_FIELD.get(key)
            header_fields.append(mapped)
        if any(header_fields):
            for row in rows[:3]:
                cells = [str(c or "").strip() for c in (row or [])]
                for idx, field in enumerate(header_fields):
                    if field and idx < len(cells):
                        cleaned = _clean_value(cells[idx])
                        if cleaned:
                            fields[field] = cleaned
                            page = page or table.get("page_start") or table.get("page")

    if not fields:
        return []

    # Map authorized_by into approved_by if approved_by empty
    if fields.get("authorized_by") and not fields.get("approved_by"):
        fields["approved_by"] = fields["authorized_by"]

    approval = {
        "prepared_by": fields.get("prepared_by", ""),
        "reviewed_by": fields.get("reviewed_by", ""),
        "approved_by": fields.get("approved_by", ""),
        "approved_date": fields.get("approved_date", ""),
        "review_date": fields.get("review_date", ""),
        "prepared_date": fields.get("prepared_date", ""),
        "page": page,
    }
    # Require at least one meaningful person/date field
    if not any(approval[k] for k in ("prepared_by", "reviewed_by", "approved_by", "approved_date")):
        return []
    return [approval]
