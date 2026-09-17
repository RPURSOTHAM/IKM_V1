"""
ooxml_table_parser.py
=====================
OOXML-based structural DOCX table extraction, merged-cell resolution,
deterministic cleaning, table classification, and quality scoring.

Uses Python standard library (xml.etree.ElementTree) to parse w:tbl XML
elements directly from python-docx objects or zipfile document.xml.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

_VALID_SHORT_TOKENS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "it", "hr", "qa", "qc", "id", "no", "na", "n/a",
    "a1", "a2", "b1", "b2", "v1.0", "v1.1", "v1.2", "v2.0",
    "yes", "no", "true", "false", "pass", "fail",
    "active", "pending", "draft", "approved", "rejected",
}

_PAGE_MARKER_RE = re.compile(
    r"^(?:page\s+\d+(\s+of\s+\d+)?|---|===|\*\*\*|\[page break\]|lastrenderedpagebreak)$",
    re.IGNORECASE,
)

_SYMBOL_NOISE_RE = re.compile(r"^[\s\-_=\*\.·•–—\|\\/]+$")


def _get_node_text_and_features(tc_node: ET.Element) -> tuple[str, bool, bool]:
    """
    Extract text, line breaks, page-break markers, and image references from a w:tc element.
    Returns: (text, has_page_break, has_image)
    """
    paragraph_texts: list[str] = []
    has_page_break = False
    has_image = False

    for p in tc_node.findall(f".//{W_NS}p"):
        runs_parts: list[str] = []
        for child in p.iter():
            tag = child.tag
            if tag == f"{W_NS}t":
                if child.text:
                    runs_parts.append(child.text)
            elif tag == f"{W_NS}br":
                br_type = str(child.get(f"{W_NS}type") or "").lower()
                if br_type == "page":
                    has_page_break = True
                runs_parts.append("\n")
            elif tag == f"{W_NS}lastRenderedPageBreak":
                has_page_break = True
            elif tag in (f"{W_NS}drawing", f"{A_NS}blip"):
                has_image = True

        p_text = "".join(runs_parts).strip()
        if p_text:
            paragraph_texts.append(p_text)

    cell_text = "\n".join(paragraph_texts).strip()
    return cell_text, has_page_break, has_image


def parse_raw_ooxml_matrix(tbl_element: ET.Element) -> tuple[list[list[str]], list[dict[str, Any]], list[dict[str, Any]], bool, bool]:
    """
    Parse a w:tbl XML element into a resolved 2D cell matrix handling w:gridSpan and w:vMerge.
    Returns: (matrix, grid_spans, v_merges, has_page_breaks, has_images)
    """
    tr_elements = tbl_element.findall(f"{W_NS}tr")
    if not tr_elements:
        tr_elements = tbl_element.findall(f".//{W_NS}tr")

    if not tr_elements:
        return [], [], [], False, False

    grid_spans: list[dict[str, Any]] = []
    v_merges: list[dict[str, Any]] = []
    has_page_breaks = False
    has_images = False

    # First pass: determine max grid width across all rows
    row_cell_nodes: list[list[ET.Element]] = []
    for tr in tr_elements:
        tcs = tr.findall(f"{W_NS}tc")
        row_cell_nodes.append(tcs)

    # Resolve horizontal gridSpans per row to find total grid width
    row_spans: list[list[int]] = []
    max_grid_width = 0
    for row_idx, tcs in enumerate(row_cell_nodes):
        spans: list[int] = []
        for col_idx, tc in enumerate(tcs):
            tc_pr = tc.find(f"{W_NS}tcPr")
            span = 1
            if tc_pr is not None:
                grid_span_node = tc_pr.find(f"{W_NS}gridSpan")
                if grid_span_node is not None:
                    try:
                        span = int(grid_span_node.get(f"{W_NS}val") or 1)
                    except ValueError:
                        span = 1
                    grid_spans.append({
                        "row": row_idx,
                        "col": col_idx,
                        "span": span,
                    })
            spans.append(span)
        row_spans.append(spans)
        max_grid_width = max(max_grid_width, sum(spans))

    if max_grid_width == 0:
        return [], grid_spans, v_merges, False, False

    # Initialize matrix with empty strings
    matrix: list[list[str]] = [["" for _ in range(max_grid_width)] for _ in range(len(tr_elements))]

    # Keep track of active vertical merge origins per grid column: col_idx -> text
    active_vmerges: dict[int, str] = {}

    for row_idx, tcs in enumerate(row_cell_nodes):
        grid_col = 0
        for tc_idx, tc in enumerate(tcs):
            span = row_spans[row_idx][tc_idx]
            text, pb, img = _get_node_text_and_features(tc)
            if pb:
                has_page_breaks = True
            if img:
                has_images = True

            tc_pr = tc.find(f"{W_NS}tcPr")
            vmerge_val: str | None = None
            is_vmerge = False
            if tc_pr is not None:
                vmerge_node = tc_pr.find(f"{W_NS}vMerge")
                if vmerge_node is not None:
                    is_vmerge = True
                    vmerge_val = str(vmerge_node.get(f"{W_NS}val") or "continue").lower()

            if is_vmerge:
                if vmerge_val in ("restart", "true", "1"):
                    # Start of vertical merge
                    active_vmerges[grid_col] = text
                    cell_value = text
                    v_merges.append({"row": row_idx, "col": grid_col, "type": "restart"})
                else:
                    # Continuation of vertical merge
                    cell_value = active_vmerges.get(grid_col, text)
                    v_merges.append({"row": row_idx, "col": grid_col, "type": "continue"})
            else:
                cell_value = text
                if grid_col in active_vmerges:
                    del active_vmerges[grid_col]

            # Fill spanned grid cells
            for s in range(span):
                if grid_col + s < max_grid_width:
                    # Place text in primary cell, empty string in spanned sub-cells
                    matrix[row_idx][grid_col + s] = cell_value if s == 0 else ""

            grid_col += span

    return matrix, grid_spans, v_merges, has_page_breaks, has_images


def normalize_table_cell(text: str) -> str:
    """Normalize cell text whitespace while preserving short valid tokens and internal newlines."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        norm = " ".join(line.split()).strip()
        if norm:
            cleaned_lines.append(norm)
    return "\n".join(cleaned_lines).strip()


def is_noise_cell(cell_text: str) -> bool:
    """Check if a cell contains purely decorative/symbol noise."""
    if not cell_text:
        return False
    clean = cell_text.strip().lower()
    if clean in _VALID_SHORT_TOKENS:
        return False
    if _SYMBOL_NOISE_RE.match(clean):
        return True
    return False


def is_page_marker(text: str) -> bool:
    """Check if cell/row text represents a page break marker."""
    if not text:
        return False
    clean = " ".join(text.split()).strip()
    return bool(_PAGE_MARKER_RE.match(clean))


def remove_empty_rows(matrix: list[list[str]]) -> list[list[str]]:
    """Remove rows where all cells are empty or whitespace-only."""
    out: list[list[str]] = []
    for row in matrix:
        if any(cell.strip() for cell in row):
            out.append(row)
    return out


def remove_empty_columns(matrix: list[list[str]]) -> list[list[str]]:
    """Remove columns where all rows are empty."""
    if not matrix or not matrix[0]:
        return matrix
    cols_to_keep: list[int] = []
    num_cols = len(matrix[0])
    for col_idx in range(num_cols):
        col_has_content = any(row[col_idx].strip() for row in matrix if col_idx < len(row))
        if col_has_content:
            cols_to_keep.append(col_idx)

    if len(cols_to_keep) == num_cols:
        return matrix

    out: list[list[str]] = []
    for row in matrix:
        new_row = [row[idx] for idx in cols_to_keep if idx < len(row)]
        out.append(new_row)
    return out


def remove_duplicate_rows(matrix: list[list[str]]) -> list[list[str]]:
    """Remove duplicate data rows while preserving single header row."""
    if len(matrix) <= 2:
        return matrix
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = [matrix[0]]  # Keep header
    seen.add(tuple(c.strip().lower() for c in matrix[0]))

    for row in matrix[1:]:
        key = tuple(c.strip().lower() for c in row)
        if key in seen and any(k for k in key):
            continue
        seen.add(key)
        out.append(row)
    return out


def remove_repeated_headers(matrix: list[list[str]]) -> list[list[str]]:
    """Remove repeated header rows that re-appear mid-table due to page breaks."""
    if len(matrix) <= 3:
        return matrix
    header_key = tuple(c.strip().lower() for c in matrix[0])
    if not any(header_key):
        return matrix

    out: list[list[str]] = [matrix[0]]
    for idx, row in enumerate(matrix[1:], start=1):
        row_key = tuple(c.strip().lower() for c in row)
        if row_key == header_key and idx > 1:
            # Skip duplicate header inside body
            continue
        out.append(row)
    return out


def classify_table(matrix: list[list[str]]) -> str:
    """Classify table matrix into KEY_VALUE, GRID, SINGLE_COLUMN, or INVALID."""
    if not matrix or not any(any(c.strip() for c in r) for r in matrix):
        return "INVALID"

    num_rows = len(matrix)
    cols_per_row = [len([c for c in row if c.strip()]) for row in matrix]
    max_cols = max(cols_per_row, default=0)

    if max_cols <= 0:
        return "INVALID"
    if max_cols == 1:
        return "SINGLE_COLUMN"

    # Check key-value morphology (2 effective columns or key: value text)
    if max_cols == 2:
        return "KEY_VALUE"

    label_colon_count = 0
    total_checked = 0
    for row in matrix[:6]:
        non_empty = [c.strip() for c in row if c.strip()]
        if len(non_empty) >= 1:
            total_checked += 1
            if ":" in non_empty[0] or (len(non_empty) == 2 and not non_empty[0].isdigit()):
                label_colon_count += 1

    if total_checked > 0 and (label_colon_count / total_checked) >= 0.6:
        return "KEY_VALUE"

    return "GRID"


def calculate_table_quality(matrix: list[list[str]], table_type: str) -> dict[str, Any]:
    """
    Calculate deterministic quality score (0.0 to 1.0) and decision (ACCEPT/REVIEW/REJECT).
    Does NOT discard tables automatically — returns diagnostic metadata.
    """
    if table_type == "INVALID" or not matrix:
        return {"quality_score": 0.0, "decision": "REJECT", "table_type": "INVALID"}

    total_cells = sum(len(row) for row in matrix)
    if total_cells == 0:
        return {"quality_score": 0.0, "decision": "REJECT", "table_type": table_type}

    filled_cells = sum(1 for row in matrix for cell in row if cell.strip())
    fill_ratio = filled_cells / total_cells

    noise_cells = sum(1 for row in matrix for cell in row if is_noise_cell(cell))
    noise_ratio = noise_cells / max(total_cells, 1)

    # Base score driven by fill ratio and text quality
    score = (fill_ratio * 0.7) + ((1.0 - noise_ratio) * 0.3)

    # Boost score for well-formed GRID / KEY_VALUE tables
    if table_type in ("GRID", "KEY_VALUE") and len(matrix) >= 2:
        score = min(1.0, score + 0.1)

    score = round(max(0.0, min(1.0, score)), 2)

    if score >= 0.60:
        decision = "ACCEPT"
    elif score >= 0.40:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    return {
        "quality_score": score,
        "decision": decision,
        "table_type": table_type,
    }


def clean_docx_table(matrix: list[list[str]]) -> tuple[list[list[str]], str, float, str]:
    """
    Clean table matrix through normalization, empty row/col removal, header deduplication,
    noise filtering, classification, and quality scoring.
    Returns: (cleaned_matrix, table_type, quality_score, decision)
    """
    if not matrix:
        return [], "INVALID", 0.0, "REJECT"

    # Step 1: Cell normalization
    norm_matrix = [[normalize_table_cell(c) for c in row] for row in matrix]

    # Step 2: Empty row & column removal
    no_empty_rows = remove_empty_rows(norm_matrix)
    no_empty_cols = remove_empty_columns(no_empty_rows)

    if not no_empty_cols:
        return [], "INVALID", 0.0, "REJECT"

    # Step 3: Header deduplication and row deduplication
    no_rep_headers = remove_repeated_headers(no_empty_cols)
    deduped = remove_duplicate_rows(no_rep_headers)

    # Step 4: Classification and Quality Scoring
    table_type = classify_table(deduped)
    quality_info = calculate_table_quality(deduped, table_type)

    return (
        deduped,
        table_type,
        quality_info["quality_score"],
        quality_info["decision"],
    )


def parse_and_clean_ooxml_table(tbl_element: ET.Element) -> dict[str, Any]:
    """
    Primary entry point: Parse a w:tbl XML element, resolve merges, clean table, and compute score.
    Returns a dict with matrix, metadata, and quality metrics.
    """
    try:
        raw_matrix, grid_spans, v_merges, has_page_breaks, has_images = parse_raw_ooxml_matrix(tbl_element)
        if not raw_matrix:
            return {
                "matrix": [],
                "raw_matrix": [],
                "row_count": 0,
                "column_count": 0,
                "empty_ratio": 1.0,
                "grid_spans": [],
                "v_merges": [],
                "has_page_breaks": False,
                "has_images": False,
                "table_type": "INVALID",
                "quality_score": 0.0,
                "decision": "REJECT",
                "ooxml_extracted": False,
                "ooxml_fallback": True,
            }

        cleaned_matrix, table_type, quality_score, decision = clean_docx_table(raw_matrix)
        row_count = len(cleaned_matrix)
        column_count = max((len(r) for r in cleaned_matrix), default=0)
        total_cells = row_count * column_count if column_count else 1
        filled_cells = sum(1 for r in cleaned_matrix for c in r if c.strip())
        empty_ratio = round(1.0 - (filled_cells / total_cells), 2) if total_cells else 1.0

        return {
            "matrix": cleaned_matrix,
            "raw_matrix": raw_matrix,
            "row_count": row_count,
            "column_count": column_count,
            "empty_ratio": empty_ratio,
            "grid_spans": grid_spans,
            "v_merges": v_merges,
            "has_page_breaks": has_page_breaks,
            "has_images": has_images,
            "table_type": table_type,
            "quality_score": quality_score,
            "decision": decision,
            "ooxml_extracted": True,
            "ooxml_fallback": False,
        }
    except Exception as exc:
        logger.warning("OOXML table extraction failed on w:tbl element: %s", exc, exc_info=True)
        return {
            "matrix": [],
            "raw_matrix": [],
            "row_count": 0,
            "column_count": 0,
            "empty_ratio": 1.0,
            "grid_spans": [],
            "v_merges": [],
            "has_page_breaks": False,
            "has_images": False,
            "table_type": "INVALID",
            "quality_score": 0.0,
            "decision": "REJECT",
            "ooxml_extracted": False,
            "ooxml_fallback": True,
        }
