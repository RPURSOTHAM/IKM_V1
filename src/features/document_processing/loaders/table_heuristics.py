"""Shared table classification heuristics for PDF and DOCX loaders."""

from __future__ import annotations

import re


_FORM_TABLE_MARKERS = (
    "date",
    "time",
    "from",
    "to",
    "sl. no",
    "s. no",
    "employee",
    "code",
    "sign",
    "signature",
    "checked by",
    "verified by",
    "approved by",
    "initiated by",
    "trainer",
    "comments",
    "remarks",
    "yes",
    "no",
    "na",
    "n/a",
)

_CONTENT_TABLE_MARKERS = (
    "purpose",
    "scope",
    "responsibility",
    "responsibilities",
    "responsible",
    "procedure",
    "definitions",
    "definition",
    "abbreviations",
    "abbreviation",
    "reference",
    "references",
    "activity",
    "description",
    "details",
    "investigation",
    "deviation",
    "out of specification",
    "out of trend",
    "oos",
    "oot",
    "sample",
    "material",
    "product",
    "batch",
    "rejection",
    "root cause",
    "corrective",
    "preventive",
    "acceptance criteria",
    "assessment",
    "impact assessment",
    "reportability",
    "reportable event",
    "justification",
    "event",
    "training",
    "attachment",
    "personnel",
    "issue date",
    "issuance",
    "issued to",
    "issued by",
    "qms",
    "outage",
    "archived",
    "remarks",
    "lab investigation",
    "department",
    "date range",
    "revision history",
    "summary of changes",
    "change control",
    "change control number",
    "version number",
    "effective date",
    "new annexure",
    "annexure introduction",
    "deviation observer",
    "deviation owner",
    "section qa",
    "investigation owner",
    "quality approver",
    "qualified person",
    "recurring deviations",
    "non-conformance",
    "nonconformance",
)


def form_marker_hits(normalized: str) -> int:
    hits = 0
    for marker in _FORM_TABLE_MARKERS:
        if marker == "to" and re.search(r"\d+\s+to\s+\d+", normalized):
            continue
        if marker == "from" and re.search(r"\d+\s+from\s+\d+", normalized):
            continue
        if not re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", normalized):
            continue
        hits += 1
    return hits


def looks_like_specification_table(text: str, column_count: int) -> bool:
    normalized = " ".join((text or "").lower().split())
    if not normalized or column_count < 2:
        return False
    has_serial = any(token in normalized for token in ("sl. no", "sl no", "s. no", "sr. no"))
    has_parameters = "parameter" in normalized
    has_requirements = any(token in normalized for token in ("requirement", "specification"))
    if (has_serial and has_parameters) or (has_parameters and has_requirements):
        return True
    filled_cells = [cell.strip() for row in text.splitlines() for cell in row.split("|") if cell.strip()]
    section_rows = sum(1 for cell in filled_cells if re.match(r"^\d+(?:\.\d+)+\s+\S", cell))
    return section_rows >= 1 and has_parameters


def looks_like_content_table(text: str, *, filled_cells: list[str], normalized: str) -> bool:
    """Return True for tables that carry business/procedure content, even if they look form-like."""
    if not normalized:
        return False
    marker_hits = sum(1 for marker in _CONTENT_TABLE_MARKERS if marker in normalized)
    numbered_sections = len(re.findall(r"(?<!\d)\d+\.\d+(?:\.\d+)*\.?\s+\S", text or ""))
    numbered_headings = len(re.findall(r"(?<!\d)[1-9]\.0\s+[A-Z][A-Z &/]+", text or ""))
    serial_rows = len(re.findall(r"\b(?:sl\.?\s*no|s\.?\s*no|sr\.?\s*no)\b", normalized))
    narrative_cells = [cell for cell in filled_cells if len(cell.split()) >= 8]
    narrative_word_count = sum(len(cell.split()) for cell in narrative_cells)

    if numbered_headings or numbered_sections >= 2:
        return True
    if "summary of changes" in normalized and len(filled_cells) >= 4:
        return True
    if "revision history" in normalized and len(filled_cells) >= 4:
        return True
    if "change control" in normalized and "summary" in normalized:
        return True
    if serial_rows and len(filled_cells) >= 5:
        return True
    if marker_hits >= 2 and len(filled_cells) >= 3:
        return True
    if marker_hits >= 1 and narrative_word_count >= 20:
        return True
    if narrative_word_count >= 60:
        return True
    return False


def looks_like_form_table(
    text: str,
    *,
    row_count: int,
    column_count: int,
    empty_ratio: float,
) -> bool:
    """Detect blank forms, sign-off grids, and checkbox tables."""
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return True

    if looks_like_specification_table(text, column_count):
        return False

    filled_cells = [cell.strip() for row in text.splitlines() for cell in row.split("|") if cell.strip()]
    if not filled_cells:
        return True
    if looks_like_content_table(text, filled_cells=filled_cells, normalized=normalized):
        return False

    short_filled = [cell for cell in filled_cells if len(cell.split()) <= 4]
    short_ratio = len(short_filled) / max(len(filled_cells), 1)
    narrative_cells = [cell for cell in filled_cells if len(cell.split()) >= 12]
    narrative_word_count = sum(len(cell.split()) for cell in narrative_cells)
    marker_hits = form_marker_hits(normalized)
    checkbox_hits = len(re.findall(r"(?:☐|□|✓|yes\s*/\s*no|\bna\b|\bn/a\b)", normalized, flags=re.IGNORECASE))

    if len(narrative_cells) >= 3 or narrative_word_count >= 80:
        return False
    if row_count >= 10 and column_count == 2 and empty_ratio < 0.15:
        return False

    mostly_blank_grid = row_count >= 5 and column_count >= 2 and empty_ratio >= 0.45
    label_heavy_form = (
        row_count >= 3
        and column_count >= 2
        and marker_hits >= 3
        and short_ratio >= 0.6
        and narrative_word_count < 40
    )
    checkbox_form = checkbox_hits >= 2 and marker_hits >= 2
    signature_form = marker_hits >= 2 and any(
        marker in normalized
        for marker in ("sign", "signature", "checked by", "verified by", "approved by", "prepared by")
    )

    return mostly_blank_grid or label_heavy_form or checkbox_form or signature_form


def table_matrix_stats(table: list[list[object]]) -> tuple[int, int, float]:
    rows = table or []
    cells = [str(cell or "").strip() for row in rows for cell in (row or [])]
    total_cells = len(cells)
    if total_cells == 0:
        return len(rows), 0, 1.0
    filled_cells = [cell for cell in cells if cell]
    empty_ratio = 1 - (len(filled_cells) / total_cells)
    column_count = max((len(row or []) for row in rows), default=0)
    return len(rows), column_count, empty_ratio
