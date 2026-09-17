"""Extract tables, including columns, merges, alignment, shading, and cell content."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pymupdf

from ..document import BBox, CellPart, ImageRef, Table, TableCell, TextSpan
from ..errors import RECOVERABLE
from ..failures import log_failure
from .extract_image import images_overlapping
from .pdf_symbols import canonical_text

IMAGE_OVERLAP_THRESHOLD = 0.50
SHADING_COVERAGE = 0.45
NEAR_WHITE = 0.95
ALIGN_TOLERANCE = 8.0
BOUNDARY_TOLERANCE = 2.0
SAME_LINE_TOLERANCE = 3.0
ALIGNED_COL_MIN_GAP = 12.0
ALIGNED_COL_MIN_COLS = 2
ALIGNED_COL_MIN_ROWS = 2
ALIGNED_COL_MARGIN_X = 40.0
ALIGNED_COL_MAX_TERM_LEN = 10
ALIGNED_COL_MAX_ROW_GAP = 40.0
ALIGNED_TERM_RE = re.compile(r"^[A-Za-z°]+$")
ALIGNED_SECTION_RE = re.compile(r"^\d+(?:\.\d+)*\.?$")
ROW_SEP_MIN_WIDTH = 250.0
ROW_SEP_MIN_LINES = 3
ROW_SEP_Y_CLUSTER = 2.0
ROW_SEP_STAMP_X0 = 560.0
LINE_THICKNESS = 2.5
MIN_H_LINE_WIDTH = 80.0
MIN_V_LINE_HEIGHT = 15.0
MIN_CELL_WIDTH = 8.0
MIN_CELL_HEIGHT = 4.0
RULE_JOIN_TOLERANCE = 5.0
UNDERLINE_RULE_GAP = 4.0
TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_COLLECT_STYLES
CHAR_BOLD = pymupdf.mupdf.FZ_STEXT_BOLD
CHAR_UNDERLINE = pymupdf.mupdf.FZ_STEXT_UNDERLINE

CELL_BULLET_CHARS = (
    "\u2022\u2023\u25e6\u2043\u2219\u00b7\u25cf\u25cb\u25a0\u25a1"
    "\u25aa\u25ab\u25c6\u25c7\u25b6\u25b8*"
)
CELL_BULLET_RE = re.compile(
    rf"^\s*(?P<bullet>[{re.escape(CELL_BULLET_CHARS)}•])\s*"
)


def extract_page_tables(
    page: pymupdf.Page,
    page_images: Sequence[ImageRef] | None = None,
) -> list[Table]:
    """Detect tables on a page and capture structure plus cell content.

    Ruled grids come from ``find_tables()`` only when dark strokes exist.
    Term/description lists that have row rules but no verticals are picked
    up by ``extract_row_separated_tables``. Tab-aligned lists
    (Abbreviations) are picked up by ``extract_aligned_column_tables``.
    Both leftover paths run only on text outside ruled tables.
    """

    drawings = page.get_drawings()
    page_dict = page.get_text("dict", flags=TEXT_FLAGS, sort=True)
    page_images = list(page_images or [])
    extracted: list[Table] = []

    finder = _find_tables(page)
    if finder is not None:
        for table in finder.tables:
            table_bbox = BBox.from_rect(pymupdf.Rect(table.bbox))
            if _ruled_bbox(drawings, table_bbox) is None:
                continue
            extracted.append(
                _convert_table(table, drawings, page_images, page_dict)
            )

    extracted.extend(
        extract_row_separated_tables(
            page_dict,
            drawings,
            exclude_rects=[table.bbox for table in extracted if table.bbox],
            page_images=page_images,
        )
    )
    extracted.extend(
        extract_aligned_column_tables(
            page_dict,
            exclude_rects=[table.bbox for table in extracted if table.bbox],
            page_images=page_images,
        )
    )

    extracted.sort(
        key=lambda item: (
            item.bbox.y0 if item.bbox else 0,
            item.bbox.x0 if item.bbox else 0,
        )
    )
    return validate_tables(extracted)


def validate_tables(tables: list[Table]) -> list[Table]:
    """Drop detections that are probably not real tables.

    Rejected regions are left for the text extractor. Current rules
    (add more here later):
    - a ruled table must have dark stroked grid lines (checked before convert)
    - a row-separated table has horizontal rules only; it is built from
      leftover text by ``extract_row_separated_tables``
    - a tab-aligned glossary table has no strokes; it is built from leftover
      text by ``extract_aligned_column_tables``
    - a table must have cells and a usable grid
    - empty cells are allowed when a ruled grid is present (continuation
      pages of a revision table often fill only the last column)
    """

    return [table for table in tables if _is_probably_a_table(table)]


def _is_probably_a_table(table: Table) -> bool:
    """True when the table has at least one cell and a usable row/column count."""
    return bool(table.cells and table.row_count >= 1 and table.column_count >= 1)


def extract_aligned_column_tables(
    page_dict: dict,
    exclude_rects: Sequence[BBox] | None = None,
    page_images: Sequence[ImageRef] | None = None,
) -> list[Table]:
    """Build tables from tab-aligned glossary text that has no ruled grid.

    Abbreviations looks like a table because each row uses the same
    x-stops (term | definition | term | definition) with a clear gap
    between columns, but the PDF has no black borders.

    Do not call ``find_tables()`` here. That path treats word-gap fills
    as column edges and would revive false 40-column tables (VACUUM).
    This helper only inspects leftover text after ruled tables.

    Rules:
    - only leftover text (lines whose box is not inside a ruled table)
    - skip the left export stamp (``x1`` left of ``ALIGNED_COL_MARGIN_X``)
      and the right export stamp (``x0`` at or past ``ROW_SEP_STAMP_X0``)
    - group fragments on the same baseline into a row
    - merge fragments with a small x-gap (word spacing); keep a cell
      break only when the gap is at least ``ALIGNED_COL_MIN_GAP``
    - a lone ``:`` after a term is attached to that term, not a column
    - a row **starts** a table only when it has two or more cells *and*
      at least one short uppercase glossary term (``SOP:``, ``HMI``,
      ``°C:``). Numbered lists (``6.1.1``), form labels (``Title:``,
      ``Document No.:``), and site codes (``BLS2DP``) stay text
    - once a run is open, a 2-cell row that uses the same x-stops and a
      short left-hand token (``MS-Excel``, ``GxP``) stays in the table
      even if that token is not all-caps
    - emit a table only when two or more *consecutive* tabular rows
      share the same x-stop pattern and sit within
      ``ALIGNED_COL_MAX_ROW_GAP`` (gap is from the previous row bottom
      to the next row top); a non-glossary line between them ends the
      run. Wrapped definition fragments on the next baseline are folded
      into the previous row, not counted as a new row
    - a single mixed line (page header fields) stays as text
    """

    rows = _aligned_text_rows(page_dict, exclude_rects)
    tables: list[Table] = []
    run: list[list[dict]] = []
    for row in rows:
        if run and _aligned_is_wrap(run[-1], row):
            _merge_aligned_wrap(run[-1], row)
            continue
        if _is_glossary_row(row) or _is_aligned_run_term_row(run, row):
            if (
                run
                and _aligned_rows_compatible(run[0], row)
                and _aligned_row_gap(run[-1], row) <= ALIGNED_COL_MAX_ROW_GAP
            ):
                run.append(row)
            elif _is_glossary_row(row):
                table = _table_from_aligned_run(run, page_images)
                if table is not None:
                    tables.append(table)
                run = [row]
            else:
                table = _table_from_aligned_run(run, page_images)
                if table is not None:
                    tables.append(table)
                run = []
            continue
        table = _table_from_aligned_run(run, page_images)
        if table is not None:
            tables.append(table)
        run = []
    table = _table_from_aligned_run(run, page_images)
    if table is not None:
        tables.append(table)
    return tables


def _aligned_text_rows(
    page_dict: dict,
    exclude_rects: Sequence[BBox] | None,
) -> list[list[dict]]:
    """Group leftover text fragments into baseline rows for glossary detection."""
    pieces: list[dict] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text") or "" for span in line.get("spans") or [])
            text = " ".join(text.split())
            if not text:
                continue
            box = BBox(*line["bbox"])
            if box.x1 < ALIGNED_COL_MARGIN_X or box.x0 >= ROW_SEP_STAMP_X0:
                continue
            if _aligned_piece_excluded(box, exclude_rects):
                continue
            pieces.append({"text": text, "bbox": box, "x0": box.x0, "y0": box.y0})

    pieces.sort(key=lambda item: (round(item["y0"], 1), item["x0"]))
    rows: list[list[dict]] = []
    current: list[dict] = []
    current_y: float | None = None
    for piece in pieces:
        if current_y is None or abs(piece["y0"] - current_y) <= SAME_LINE_TOLERANCE:
            current.append(piece)
            current_y = piece["y0"] if current_y is None else current_y
        else:
            merged = _merge_close_row_pieces(current)
            if merged:
                rows.append(merged)
            current = [piece]
            current_y = piece["y0"]
    merged = _merge_close_row_pieces(current)
    if merged:
        rows.append(merged)
    return rows


def _aligned_piece_excluded(box: BBox, rects: Sequence[BBox] | None) -> bool:
    """True when a text fragment sits inside a ruled table or other exclude rect."""
    if not rects:
        return False
    return any(box.overlap_ratio(rect) >= IMAGE_OVERLAP_THRESHOLD for rect in rects)


def _merge_close_row_pieces(pieces: Sequence[dict]) -> list[dict]:
    """Join word-spaced fragments; keep a cell break only at a large x-gap."""
    ordered = sorted(pieces, key=lambda item: item["x0"])
    merged: list[dict] = []
    for piece in ordered:
        piece = dict(piece)
        text = (piece["text"] or "").strip()
        piece["text"] = text
        if not merged:
            merged.append(piece)
            continue
        previous = merged[-1]
        gap = piece["bbox"].x0 - previous["bbox"].x1
        if text == ":":
            previous["text"] = previous["text"].rstrip() + ":"
            continue
        if text.startswith(":") and (
            _is_glossary_term(previous["text"]) or _is_short_abbrev_token(previous["text"])
        ):
            piece["text"] = text.lstrip(":").strip()
            if piece["text"]:
                merged.append(piece)
            continue
        if gap < ALIGNED_COL_MIN_GAP:
            previous["text"] = f"{previous['text']} {piece['text']}".strip()
            previous["bbox"] = BBox.combine(previous["bbox"], piece["bbox"])
        else:
            merged.append(piece)
    return merged


def _aligned_term_body(text: str) -> str:
    """Strip trailing colons from a glossary cell for term matching."""
    return (text or "").strip().rstrip(":").strip()


def _is_short_abbrev_token(text: str) -> bool:
    """True for a short single-token label (``SOP``, ``MS-Excel``, ``GxP``)."""

    body = _aligned_term_body(text)
    if not body or " " in body:
        return False
    if len(body) > ALIGNED_COL_MAX_TERM_LEN:
        return False
    if ALIGNED_SECTION_RE.match(body):
        return False
    return bool(re.search(r"[A-Za-z]", body))


def _is_glossary_term(text: str) -> bool:
    """Short uppercase token used as an abbreviation / glossary label."""

    if not _is_short_abbrev_token(text):
        return False
    body = _aligned_term_body(text)
    if body != body.upper():
        return False
    return bool(ALIGNED_TERM_RE.match(body))


def _is_glossary_row(row: Sequence[dict]) -> bool:
    """True when a row has two+ cells and at least one short uppercase term."""
    if len(row) < ALIGNED_COL_MIN_COLS:
        return False
    return any(_is_glossary_term(piece["text"]) for piece in row)


def _is_aligned_run_term_row(run: Sequence[list[dict]], row: Sequence[dict]) -> bool:
    """True when an open glossary run can accept this mixed-case abbreviation row.

    ``_is_glossary_term`` stays strict so ``Title:`` cannot start a table.
    ``MS-Excel`` / ``GxP`` only join after a real glossary run is already open
    and they share that run's x-stops.
    """

    if not run or len(row) < ALIGNED_COL_MIN_COLS:
        return False
    if not _aligned_rows_compatible(run[0], row):
        return False
    left = min(row, key=lambda piece: piece["x0"])
    return _is_short_abbrev_token(left["text"])


def _aligned_is_wrap(previous: Sequence[dict], row: Sequence[dict]) -> bool:
    """True when ``row`` is a wrapped definition of ``previous``, not a new term."""
    if not previous or not row:
        return False
    if any(_is_glossary_term(piece["text"]) for piece in row):
        return False
    left = min(row, key=lambda piece: piece["x0"])
    if len(row) >= ALIGNED_COL_MIN_COLS and _is_short_abbrev_token(left["text"]):
        return False
    def_stops = [piece["x0"] for piece in previous if not _is_glossary_term(piece["text"])]
    if not def_stops:
        return False
    return all(_nearest_stop(piece["x0"], def_stops) is not None for piece in row)


def _aligned_row_gap(previous: Sequence[dict], row: Sequence[dict]) -> float:
    """Vertical gap from the previous row bottom to the next row top."""

    def _piece_y1(piece: dict) -> float:
        box = piece.get("bbox")
        return box.y1 if box is not None else piece["y0"]

    prev_bottom = max(_piece_y1(piece) for piece in previous)
    next_top = min(piece["y0"] for piece in row)
    return next_top - prev_bottom


def _merge_aligned_wrap(previous: list[dict], row: Sequence[dict]) -> None:
    """Fold wrapped definition fragments into the matching column of ``previous``."""
    definitions = [piece for piece in previous if not _is_glossary_term(piece["text"])]
    stops = [piece["x0"] for piece in definitions]
    for piece in row:
        index = _nearest_stop(piece["x0"], stops)
        if index is None:
            continue
        target = definitions[index]
        target["text"] = f"{target['text']} {piece['text']}".strip()
        target["bbox"] = BBox.combine(target["bbox"], piece["bbox"])


def _aligned_rows_compatible(first: Sequence[dict], following: Sequence[dict]) -> bool:
    """True when ``following`` uses the same column x-stops as ``first``."""
    stops = _cluster_x_stops([piece["x0"] for piece in first])
    if len(stops) < ALIGNED_COL_MIN_COLS:
        return False
    hits = 0
    for piece in following:
        if _nearest_stop(piece["x0"], stops) is not None:
            hits += 1
    return hits >= ALIGNED_COL_MIN_COLS


def _cluster_x_stops(values: Sequence[float]) -> list[float]:
    """Merge nearby x0 values into column stops."""
    unique: list[float] = []
    for value in sorted(values):
        if not unique or abs(value - unique[-1]) > ALIGN_TOLERANCE:
            unique.append(value)
    return unique


def _nearest_stop(x0: float, stops: Sequence[float]) -> int | None:
    """Index of the column stop nearest ``x0``, or ``None`` if none are close."""
    match: int | None = None
    best_delta: float | None = None
    for index, stop in enumerate(stops):
        delta = abs(x0 - stop)
        if delta > ALIGN_TOLERANCE:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            match = index
    return match


def _table_from_aligned_run(
    run: Sequence[list[dict]],
    page_images: Sequence[ImageRef] | None,
) -> Table | None:
    """Build a ``Table`` from consecutive aligned glossary (or row-separated) rows."""
    if len(run) < ALIGNED_COL_MIN_ROWS:
        return None
    stops: list[float] = []
    for row in run:
        for piece in row:
            stops.append(piece["x0"])
    columns_x = _cluster_x_stops(stops)
    if len(columns_x) < ALIGNED_COL_MIN_COLS:
        return None

    cells: list[TableCell] = []
    page_images = list(page_images or [])
    for row_idx, row in enumerate(run):
        by_col: dict[int, dict] = {}
        for piece in row:
            col = _nearest_stop(piece["x0"], columns_x)
            if col is None:
                continue
            if col in by_col:
                existing = by_col[col]
                existing["text"] = f"{existing['text']} {piece['text']}".strip()
                existing["bbox"] = BBox.combine(existing["bbox"], piece["bbox"])
            else:
                by_col[col] = dict(piece)
        row_y0 = min(piece["bbox"].y0 for piece in row)
        row_y1 = max(piece["bbox"].y1 for piece in row)
        for col_idx, x0 in enumerate(columns_x):
            piece = by_col.get(col_idx)
            if col_idx + 1 < len(columns_x):
                x1 = columns_x[col_idx + 1]
            elif piece is not None:
                x1 = piece["bbox"].x1
            else:
                x1 = x0 + 40.0
            text = piece["text"] if piece else ""
            bbox = (
                piece["bbox"]
                if piece
                else BBox(x0, row_y0, max(x0 + MIN_CELL_WIDTH, x1), row_y1)
            )
            images = images_overlapping(page_images, bbox, IMAGE_OVERLAP_THRESHOLD)
            parts = []
            if text:
                parts.append(CellPart(kind="text", text=text, spans=[TextSpan(text=text)]))
            cells.append(
                TableCell(
                    row=row_idx,
                    column=col_idx,
                    rowspan=1,
                    colspan=1,
                    align="left",
                    valign="top",
                    shading=None,
                    text=text,
                    parts=parts,
                    images=images,
                    bbox=bbox,
                    is_header=False,
                )
            )

    bbox = _bbox_from_cells(cells)
    return Table(
        columns=[f"Col{index + 1}" for index in range(len(columns_x))],
        column_count=len(columns_x),
        row_count=len(run),
        cells=cells,
        bbox=bbox,
        header_row_count=0,
    )


def extract_row_separated_tables(
    page_dict: dict,
    drawings: Sequence[dict],
    exclude_rects: Sequence[BBox] | None = None,
    page_images: Sequence[ImageRef] | None = None,
) -> list[Table]:
    """Build 2-column tables from leftover text sitting between row rules.

    GxP-style glossaries look like a table because each term sits on the
    left and its description on the right, with a full-width horizontal
    rule between rows, but **no verticals**. ``find_tables()`` returns
    nothing, and the Abbreviations detector ignores these rows because
    the terms are phrases (``GxP Spreadsheet:``), not short tokens.

    Do not call ``find_tables()`` here and do not loosen
    ``_is_glossary_term``. This path is gated by the stroke ladder, so
    VACUUM paragraphs and numbered lists (no matching row rules) stay
    text. Ruled grids are excluded first so existing tables are untouched.

    Rules:
    - only leftover text (outside ruled tables and the side export stamp)
    - detect a ladder of at least ``ROW_SEP_MIN_LINES`` long, parallel,
      same-span horizontal strokes (width ≥ ``ROW_SEP_MIN_WIDTH``)
    - ignore strokes that already sit inside a ruled table
    - each band between consecutive rules is one row
    - assign fragments in that band to columns by x-stop (term | description);
      wrapped lines in the same band join in that cell
    - emit a table only when the ladder yields at least two rows and two
      columns
    """

    ladders = _row_separator_ladders(drawings, exclude_rects)
    if not ladders:
        return []

    pieces = _row_separator_pieces(page_dict, exclude_rects)
    tables: list[Table] = []
    for ys, _x0, _x1 in ladders:
        rows = _row_separator_rows(pieces, ys)
        table = _table_from_aligned_run(rows, page_images)
        if table is not None:
            tables.append(table)
    return tables


def _row_separator_ladders(
    drawings: Sequence[dict],
    exclude_rects: Sequence[BBox] | None,
) -> list[tuple[list[float], float, float]]:
    """Find groups of long parallel h-strokes that form row bands."""
    raw: list[tuple[float, float, float]] = []
    for x0, y0, x1, y1 in _stroke_segments(drawings):
        left, right = (x0, x1) if x0 <= x1 else (x1, x0)
        top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)
        dx = right - left
        dy = bottom - top
        if dy > LINE_THICKNESS or dx < ROW_SEP_MIN_WIDTH:
            continue
        y = (top + bottom) / 2
        if _row_sep_line_excluded(y, left, right, exclude_rects):
            continue
        raw.append((y, left, right))

    raw.sort(key=lambda item: item[0])
    unique: list[tuple[float, float, float]] = []
    for y, left, right in raw:
        if unique and abs(y - unique[-1][0]) <= ROW_SEP_Y_CLUSTER:
            continue
        unique.append((y, left, right))

    groups: list[list[tuple[float, float, float]]] = []
    for line in unique:
        placed = False
        for group in groups:
            _, gx0, gx1 = group[0]
            if abs(line[1] - gx0) <= ALIGN_TOLERANCE and abs(line[2] - gx1) <= ALIGN_TOLERANCE:
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])

    ladders: list[tuple[list[float], float, float]] = []
    for group in groups:
        if len(group) < ROW_SEP_MIN_LINES:
            continue
        ys = [line[0] for line in group]
        x0 = min(line[1] for line in group)
        x1 = max(line[2] for line in group)
        ladders.append((ys, x0, x1))
    return ladders


def _row_sep_line_excluded(
    y: float,
    left: float,
    right: float,
    rects: Sequence[BBox] | None,
) -> bool:
    """True when a candidate row rule already sits inside a ruled table."""
    if not rects:
        return False
    probe = BBox(left, y - 1.0, right, y + 1.0)
    return any(
        rect.y0 - RULE_JOIN_TOLERANCE <= y <= rect.y1 + RULE_JOIN_TOLERANCE
        and probe.intersection_area(rect) > 0
        for rect in rects
        if rect is not None
    )


def _row_separator_pieces(
    page_dict: dict,
    exclude_rects: Sequence[BBox] | None,
) -> list[dict]:
    """Leftover text fragments for row-separated tables, skipping the side stamp."""
    pieces: list[dict] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text") or "" for span in line.get("spans") or [])
            text = " ".join(text.split())
            if not text:
                continue
            box = BBox(*line["bbox"])
            if box.x1 < ALIGNED_COL_MARGIN_X or box.x0 >= ROW_SEP_STAMP_X0:
                continue
            if _aligned_piece_excluded(box, exclude_rects):
                continue
            mid_y = (box.y0 + box.y1) / 2
            pieces.append({"text": text, "bbox": box, "x0": box.x0, "y0": box.y0, "mid_y": mid_y})
    return pieces


def _row_separator_rows(pieces: Sequence[dict], ys: Sequence[float]) -> list[list[dict]]:
    """Assign leftover fragments to bands between consecutive row-rule y values."""
    bands: list[list[dict]] = [[] for _ in range(len(ys) - 1)]
    for piece in pieces:
        mid_y = piece["mid_y"]
        for index in range(len(ys) - 1):
            if ys[index] < mid_y < ys[index + 1]:
                bands[index].append(piece)
                break
    rows: list[list[dict]] = []
    for band in bands:
        merged = _merge_close_row_pieces(band)
        if len(merged) >= ALIGNED_COL_MIN_COLS:
            rows.append(merged)
    return rows


def _find_tables(page: pymupdf.Page):
    """Run PyMuPDF ``find_tables()``; return the finder or ``None``."""
    try:
        finder = page.find_tables()
        if finder is not None and finder.tables:
            return finder
    except RECOVERABLE as exc:
        log_failure("extract", "PyMuPDF find_tables", exc)
    return None


def _convert_table(
    table,
    drawings: list[dict],
    page_images: Sequence[ImageRef],
    page_dict: dict,
) -> Table:
    """Turn a PyMuPDF table hit into a ``Table`` with cells, headers, and bbox."""
    cells = _extract_cells(table, drawings, page_images, page_dict)
    cells = _merge_continuation_rows(cells)

    if cells:
        row_count = max(cell.row + cell.rowspan for cell in cells)
        col_count = max(cell.column + cell.colspan for cell in cells)
    else:
        row_count = getattr(table, "row_count", 0) or 0
        col_count = getattr(table, "col_count", 0) or 0

    columns, header_row_count = _resolve_columns(table, cells, col_count)

    bbox = _bbox_from_cells(cells) or BBox.from_rect(pymupdf.Rect(table.bbox))
    for cell in cells:
        cell.is_header = cell.row < header_row_count

    return Table(
        columns=columns,
        column_count=col_count,
        row_count=row_count,
        cells=cells,
        bbox=bbox,
        header_row_count=header_row_count,
    )


def _header_names(table) -> list[str]:
    """Column titles from PyMuPDF's table header object, if present."""
    header = getattr(table, "header", None)
    if header is None:
        return []
    names = []
    for name in header.names or []:
        cleaned = " ".join((name or "").split())
        names.append(cleaned)
    return names


def _resolve_columns(table, cells: list[TableCell], col_count: int) -> tuple[list[str], int]:
    """Prefer a full row of unmerged cells as column titles.

    Skip rows that look like body data (bullets, multi-line cells) so a
    responsibility table's first data row is not treated as a header.
    """

    by_row: dict[int, list[TableCell]] = {}
    for cell in cells:
        by_row.setdefault(cell.row, []).append(cell)

    for row_idx in sorted(by_row):
        row = sorted(by_row[row_idx], key=lambda cell: cell.column)
        if not row:
            continue
        if not all(cell.colspan == 1 for cell in row):
            continue
        if sum(cell.colspan for cell in row) != col_count:
            continue
        if _row_looks_like_body(row):
            continue
        names = [
            " ".join(cell.text.split()) or f"Col{cell.column + 1}"
            for cell in row
        ]
        if any(name and not name.startswith("Col") for name in names):
            header_rows = max(int(getattr(table, "header_rows", 0) or 0), row_idx + 1)
            return names, header_rows

    names = _header_names(table)
    if len(names) < col_count:
        names = names + [f"Col{index + 1}" for index in range(len(names), col_count)]
    if not names:
        names = [f"Col{index + 1}" for index in range(col_count)]
    return names[:col_count], 0


def _row_looks_like_body(row: Sequence[TableCell]) -> bool:
    """True when any cell in the row looks like data, not a column title."""
    return any(_cell_looks_like_body(cell) for cell in row)


def _cell_looks_like_body(cell: TableCell) -> bool:
    """True for bullets or multi-line text that should not become a header."""
    text = cell.text or ""
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" in text or "•" in text:
        return True
    if CELL_BULLET_RE.match(stripped):
        return True
    return _is_cell_bullet_glyph(stripped[0])


def _extract_cells(
    table,
    drawings: list[dict],
    page_images: Sequence[ImageRef],
    page_dict: dict,
) -> list[TableCell]:
    """Build cells from placements or the ruled grid, with text, images, and shading."""
    table_bbox = BBox.from_rect(pymupdf.Rect(table.bbox))
    grid = _ruled_bbox(drawings, table_bbox)
    placements = getattr(table, "placements", None)
    if placements:
        specs = _clip_specs_to_grid(_cells_from_placements(placements), grid)
    else:
        specs = _cells_from_grid(table, grid)

    cells: list[TableCell] = []
    header_rows = int(getattr(table, "header_rows", 0) or 0)
    grid_rules = _table_stroke_lines(drawings, table_bbox)[0]

    for spec in specs:
        rect = spec["bbox"]
        if rect is None or rect.is_empty:
            continue

        parts, text, images, content_bbox = _cell_content(
            rect, page_images, page_dict, grid_rules
        )
        align, valign = _detect_alignment(rect, content_bbox)
        is_header = spec["row"] < header_rows or spec.get("is_header", False)

        cells.append(
            TableCell(
                row=spec["row"],
                column=spec["column"],
                rowspan=spec["rowspan"],
                colspan=spec["colspan"],
                align=align,
                valign=valign,
                shading=_cell_shading(drawings, rect),
                text=text,
                parts=parts,
                images=images,
                bbox=rect,
                is_header=is_header,
            )
        )

    return cells


def _join_cell_text(left: str, right: str) -> str:
    """Join wrapped cell text; keep hyphen/slash joins without an extra space."""
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("/") or left.endswith("-"):
        return f"{left}{right}"
    return f"{left} {right}"


def _merge_cell_text_parts(target: TableCell, extra: TableCell) -> None:
    """Append ``extra`` text spans and images onto ``target`` for a continuation row."""
    extra_text = [part for part in extra.parts if part.kind == "text"]
    extra_images = [part for part in extra.parts if part.kind == "image"]
    if extra_text:
        dest = next((part for part in reversed(target.parts) if part.kind == "text"), None)
        joined_spans: list[TextSpan] = []
        if dest is not None:
            joined_spans.extend(dest.spans or ([TextSpan(text=dest.text)] if dest.text else []))
        for part in extra_text:
            more = part.spans or ([TextSpan(text=part.text)] if part.text else [])
            joined_spans = _append_wrapped_spans(joined_spans, more)
        joined_spans = TextSpan.merge_adjacent(
            joined_spans,
            join_across_newlines=False,
        )
        if dest is None:
            target.parts.insert(
                0,
                CellPart(kind="text", text=target.text, spans=joined_spans),
            )
        else:
            dest.text = target.text
            dest.spans = joined_spans
    target.parts.extend(extra_images)


def _merge_continuation_rows(cells: list[TableCell]) -> list[TableCell]:
    """Fold wrapped-line split rows back into the row above.

    A following row with an empty first cell is treated as wrapped text of
    the row above (for example a Description that spilled onto the next
    grid line). Do **not** fold section banners into that row: shaded
    family/group headers such as ``Family 01: Motor Not Stopped`` also
    have an empty first cell, but they are their own rows.
    """

    by_row: dict[int, list[TableCell]] = {}
    for cell in cells:
        by_row.setdefault(cell.row, []).append(cell)

    kept: dict[int, list[TableCell]] = {}
    previous_row: int | None = None

    for row_idx in sorted(by_row):
        row_cells = by_row[row_idx]
        first = next((cell for cell in row_cells if cell.column == 0), None)
        is_continuation = (
            previous_row is not None
            and first is not None
            and not first.text.strip()
            and not first.images
            and any(cell.text.strip() or cell.images for cell in row_cells)
            and not _row_is_section_banner(row_cells)
        )

        if is_continuation:
            previous_cells = {cell.column: cell for cell in kept[previous_row]}
            for cell in row_cells:
                if not cell.text.strip() and not cell.images:
                    continue
                target = previous_cells.get(cell.column)
                if target is None:
                    cell.row = previous_row
                    kept[previous_row].append(cell)
                    previous_cells[cell.column] = cell
                    continue
                target.text = _join_cell_text(target.text, cell.text)
                _merge_cell_text_parts(target, cell)
                target.images.extend(cell.images)
                target.bbox = BBox.combine(target.bbox, cell.bbox)
            continue

        kept[row_idx] = row_cells
        previous_row = row_idx

    remapped: list[TableCell] = []
    for new_row, row_idx in enumerate(sorted(kept)):
        for cell in kept[row_idx]:
            cell.row = new_row
            remapped.append(cell)
    return remapped


def _row_is_section_banner(row_cells: Sequence[TableCell]) -> bool:
    """True for a full-width group header (shaded and/or a single filled cell)."""

    filled = [
        cell
        for cell in row_cells
        if (cell.text or "").strip() or cell.images
    ]
    if len(filled) == 1:
        return True
    return any(cell.shading for cell in filled)


def _cells_from_placements(placements) -> list[dict]:
    """Convert PyMuPDF cell placements into row/col/span/bbox specs."""
    specs: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    for row_idx, row in enumerate(placements):
        col_idx = 0
        for placement in row:
            if placement is None:
                col_idx += 1
                continue

            while (row_idx, col_idx) in occupied:
                col_idx += 1

            rowspan = max(1, int(getattr(placement, "rowspan", 1) or 1))
            colspan = max(1, int(getattr(placement, "colspan", 1) or 1))
            tag = getattr(placement, "tag", "td")

            for drow in range(rowspan):
                for dcol in range(colspan):
                    occupied.add((row_idx + drow, col_idx + dcol))

            bbox = None
            if getattr(placement, "bbox", None):
                bbox = BBox(*placement.bbox)

            specs.append(
                {
                    "row": row_idx,
                    "column": col_idx,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "bbox": bbox,
                    "is_header": tag == "th",
                }
            )
            col_idx += colspan

    return specs


def _cells_from_grid(table, grid: BBox | None) -> list[dict]:
    """Infer cell specs from unique cell rectangles clipped to the ruled grid."""
    rects: list[BBox] = []
    seen: set[tuple[float, float, float, float]] = set()

    for cell in table.cells:
        if cell is None:
            continue
        clipped = _clip_rect_to_grid(BBox.from_rect(pymupdf.Rect(cell)), grid)
        if clipped is None:
            continue
        key = (
            round(clipped.x0, 2),
            round(clipped.y0, 2),
            round(clipped.x1, 2),
            round(clipped.y1, 2),
        )
        if key in seen:
            continue
        seen.add(key)
        rects.append(clipped)

    x_bounds, y_bounds = _boundaries_from_rects(rects)
    specs: list[dict] = []
    for rect in rects:
        span = _span_from_boundaries(rect, x_bounds, y_bounds)
        specs.append({**span, "bbox": rect, "is_header": False})
    return specs


def _clip_specs_to_grid(specs: list[dict], grid: BBox | None) -> list[dict]:
    """Drop or shrink cell specs so they stay inside the stroked grid."""
    clipped: list[dict] = []
    for spec in specs:
        bbox = spec.get("bbox")
        if bbox is None:
            continue
        rect = _clip_rect_to_grid(bbox, grid)
        if rect is None:
            continue
        clipped.append({**spec, "bbox": rect})
    return clipped


def _clip_rect_to_grid(rect: BBox, grid: BBox | None) -> BBox | None:
    """Intersect ``rect`` with ``grid``; drop boxes that are too small."""
    if grid is None:
        return rect
    clipped = rect.intersection(grid)
    if clipped.is_empty or clipped.width < MIN_CELL_WIDTH or clipped.height < MIN_CELL_HEIGHT:
        return None
    return clipped


def _bbox_from_cells(cells: Sequence[TableCell]) -> BBox | None:
    """Union of every non-empty cell box."""
    boxes = [cell.bbox for cell in cells if cell.bbox is not None and not cell.bbox.is_empty]
    return BBox.union_all(boxes)


def _table_stroke_lines(
    drawings: Sequence[dict], table_bbox: BBox
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Horizontal and vertical dark strokes that overlap a table bbox."""

    h_lines: list[tuple[float, float, float]] = []
    v_lines: list[tuple[float, float, float]] = []
    if table_bbox.is_empty:
        return h_lines, v_lines

    for x0, y0, x1, y1 in _stroke_segments(drawings):
        left, right = (x0, x1) if x0 <= x1 else (x1, x0)
        top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)
        dx = right - left
        dy = bottom - top
        if not _segment_hits_bbox(left, top, right, bottom, table_bbox):
            continue
        if dy <= LINE_THICKNESS and dx >= MIN_H_LINE_WIDTH:
            overlap = min(right, table_bbox.x1 + RULE_JOIN_TOLERANCE) - max(
                left, table_bbox.x0 - RULE_JOIN_TOLERANCE
            )
            if overlap >= MIN_H_LINE_WIDTH:
                h_lines.append(((top + bottom) / 2, left, right))
        elif dx <= LINE_THICKNESS and dy >= MIN_V_LINE_HEIGHT:
            overlap = min(bottom, table_bbox.y1 + RULE_JOIN_TOLERANCE) - max(
                top, table_bbox.y0 - RULE_JOIN_TOLERANCE
            )
            if overlap >= MIN_V_LINE_HEIGHT:
                v_lines.append(((left + right) / 2, top, bottom))
    return h_lines, v_lines


def _ruled_bbox(drawings: Sequence[dict], table_bbox: BBox) -> BBox | None:
    """Clip a detected table to its stroked grid, ignoring margin/footer boxes."""

    h_lines, v_lines = _table_stroke_lines(drawings, table_bbox)
    if len(h_lines) < 2 and len(v_lines) < 2:
        return None

    frame_h = _frame_horizontal_lines(h_lines)
    if v_lines and frame_h:
        frame_h = [
            line for line in frame_h if _vertical_meets_y(line[0], v_lines)
        ] or frame_h

    if len(frame_h) >= 2:
        y0 = min(line[0] for line in frame_h)
        y1 = max(line[0] for line in frame_h)
    else:
        y0, y1 = table_bbox.y0, table_bbox.y1

    v_in_band = [
        line
        for line in v_lines
        if min(line[2], y1 + RULE_JOIN_TOLERANCE) - max(line[1], y0 - RULE_JOIN_TOLERANCE)
        >= MIN_V_LINE_HEIGHT
    ]
    if len(v_in_band) >= 2:
        x0 = min(line[0] for line in v_in_band)
        x1 = max(line[0] for line in v_in_band)
    elif frame_h:
        x0 = min(line[1] for line in frame_h)
        x1 = max(line[2] for line in frame_h)
    else:
        x0, x1 = table_bbox.x0, table_bbox.x1

    ruled = BBox(x0, y0, x1, y1)
    if ruled.is_empty:
        return None
    return ruled


def _frame_horizontal_lines(
    h_lines: Sequence[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Keep the longest horizontal strokes that form the table frame."""
    if not h_lines:
        return []
    longest = max(line[2] - line[1] for line in h_lines)
    min_width = max(MIN_H_LINE_WIDTH, 0.6 * longest)
    return [line for line in h_lines if line[2] - line[1] >= min_width]


def _vertical_meets_y(
    y: float,
    v_lines: Sequence[tuple[float, float, float]],
    tolerance: float = RULE_JOIN_TOLERANCE,
) -> bool:
    """True when a vertical stroke meets or crosses the horizontal at ``y``."""
    for _x, top, bottom in v_lines:
        if abs(top - y) <= tolerance or abs(bottom - y) <= tolerance:
            return True
        if top - tolerance <= y <= bottom + tolerance:
            return True
    return False


def _segment_hits_bbox(x0: float, y0: float, x1: float, y1: float, bbox: BBox) -> bool:
    """True when a stroke segment overlaps ``bbox`` within join tolerance."""
    pad = RULE_JOIN_TOLERANCE
    return not (
        x1 < bbox.x0 - pad
        or x0 > bbox.x1 + pad
        or y1 < bbox.y0 - pad
        or y0 > bbox.y1 + pad
    )


def _stroke_segments(drawings: Sequence[dict]) -> list[tuple[float, float, float, float]]:
    """Collect dark line segments from drawings, skipping near-white fills."""
    segments: list[tuple[float, float, float, float]] = []
    for drawing in drawings:
        color = drawing.get("color")
        if color is None or _is_near_white(color):
            continue
        items = drawing.get("items") or []
        found_line = False
        for item in items:
            if not item or item[0] != "l" or len(item) < 3:
                continue
            start, end = item[1], item[2]
            segments.append((float(start.x), float(start.y), float(end.x), float(end.y)))
            found_line = True
        if found_line:
            continue
        rect = drawing.get("rect")
        if rect is None:
            continue
        box = pymupdf.Rect(rect)
        if box.height <= LINE_THICKNESS or box.width <= LINE_THICKNESS:
            segments.append((float(box.x0), float(box.y0), float(box.x1), float(box.y1)))
    return segments


def _boundaries_from_rects(rects: Sequence[BBox]) -> tuple[list[float], list[float]]:
    """Sorted unique x and y edges used as the table grid."""
    x_values: list[float] = []
    y_values: list[float] = []
    for rect in rects:
        x_values.extend([rect.x0, rect.x1])
        y_values.extend([rect.y0, rect.y1])
    return _unique_sorted(x_values), _unique_sorted(y_values)


def _unique_sorted(values: Sequence[float], tolerance: float = BOUNDARY_TOLERANCE) -> list[float]:
    """Sort numbers and collapse values that differ by less than ``tolerance``."""
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _span_from_boundaries(rect: BBox, x_bounds: list[float], y_bounds: list[float]) -> dict:
    """Map a cell rectangle onto grid indices (row, column, rowspan, colspan)."""
    x0 = _boundary_index(rect.x0, x_bounds)
    x1 = _boundary_index(rect.x1, x_bounds)
    y0 = _boundary_index(rect.y0, y_bounds)
    y1 = _boundary_index(rect.y1, y_bounds)
    if None in (x0, x1, y0, y1):
        return {"row": 0, "column": 0, "rowspan": 1, "colspan": 1}
    return {
        "row": y0,
        "column": x0,
        "rowspan": max(1, y1 - y0),
        "colspan": max(1, x1 - x0),
    }


def _boundary_index(value: float, bounds: list[float]) -> int | None:
    """Index of the grid line nearest ``value``, or ``None`` if none match."""
    for index, bound in enumerate(bounds):
        if abs(value - bound) <= BOUNDARY_TOLERANCE:
            return index
    return None


def _spans_in_cell(
    page_dict: dict,
    cell: BBox,
    grid_rules: Sequence[tuple[float, float, float]] | None = None,
) -> list[tuple[TextSpan, BBox, float]]:
    """Return styled spans that belong in the cell.

    A span is used when its baseline sits in the cell's y-range. If the
    span box crosses a column edge, only the characters whose x falls
    inside this cell are kept (for example ``maintenance Superviso``
    splits so ``maintenance`` stays in one column and ``Superviso`` in
    the next).
    """

    spans: list[tuple[TextSpan, BBox, float]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text") or ""
                if not text.strip():
                    continue
                box = BBox(*span["bbox"])
                clipped = _clip_span_to_cell(text, box, cell)
                if clipped is None:
                    continue
                sliced, slice_box = clipped
                spans.append(
                    (_cell_span_style(span, sliced, box, grid_rules), slice_box, slice_box.y0)
                )
    spans.sort(
        key=lambda item: (
            round(item[2] / SAME_LINE_TOLERANCE) * SAME_LINE_TOLERANCE,
            item[1].x0,
        )
    )
    return spans


def _clip_span_to_cell(text: str, box: BBox, cell: BBox) -> tuple[str, BBox] | None:
    """Keep the part of a span that sits inside ``cell``.

    Characters are placed evenly across the span box. Spans that only
    stick a couple of points past the border (glyph ink) are kept whole.
    """

    cy = (box.y0 + box.y1) / 2
    if cy < cell.y0 or cy > cell.y1:
        return None
    if box.x1 <= cell.x0 or box.x0 >= cell.x1:
        return None
    if (
        box.x0 >= cell.x0 - BOUNDARY_TOLERANCE
        and box.x1 <= cell.x1 + BOUNDARY_TOLERANCE
    ):
        return text, box

    n = len(text)
    width = box.width if box.width > 0 else 1.0
    kept: list[tuple[int, str]] = []
    for index, char in enumerate(text):
        x = box.x0 + (index + 0.5) * width / n
        if cell.x0 <= x <= cell.x1:
            kept.append((index, char))
    if not kept:
        return None
    sliced = "".join(char for _index, char in kept)
    if not sliced.strip():
        return None
    first, last = kept[0][0], kept[-1][0]
    return sliced, BBox(
        box.x0 + first * width / n,
        box.y0,
        box.x0 + (last + 1) * width / n,
        box.y1,
    )


def _cell_span_style(
    span: dict,
    text: str,
    box: BBox | None = None,
    grid_rules: Sequence[tuple[float, float, float]] | None = None,
) -> TextSpan:
    """Build a cell span with emphasis; drop underline when it is a grid rule."""
    flags = int(span.get("flags") or 0)
    char_flags = int(span.get("char_flags") or 0)
    font = (span.get("font") or "").lower()
    bold = bool(flags & pymupdf.TEXT_FONT_BOLD) or bool(char_flags & CHAR_BOLD) or "bold" in font
    italic = (
        bool(flags & pymupdf.TEXT_FONT_ITALIC)
        or "italic" in font
        or "oblique" in font
    )
    underline = bool(char_flags & CHAR_UNDERLINE)
    if underline and box is not None and _span_rests_on_grid_rule(box, grid_rules):
        underline = False
    return TextSpan(text=canonical_text(text), bold=bold, italic=italic, underline=underline)


def _span_rests_on_grid_rule(
    box: BBox,
    grid_rules: Sequence[tuple[float, float, float]] | None,
) -> bool:
    """True when a table rule sits under the span (false MuPDF underline)."""

    if not grid_rules:
        return False
    for y, left, right in grid_rules:
        if box.x1 < left - RULE_JOIN_TOLERANCE or box.x0 > right + RULE_JOIN_TOLERANCE:
            continue
        gap = y - box.y1
        if -LINE_THICKNESS <= gap <= UNDERLINE_RULE_GAP:
            return True
    return False


def _cell_content(
    rect: BBox,
    page_images: Sequence[ImageRef],
    page_dict: dict,
    grid_rules: Sequence[tuple[float, float, float]] | None = None,
) -> tuple[list[CellPart], str, list[ImageRef], BBox | None]:
    """Collect text parts and overlapping images that belong in one cell."""
    images = images_overlapping(list(page_images), rect, IMAGE_OVERLAP_THRESHOLD)
    spans = _spans_in_cell(page_dict, rect, grid_rules)
    boxes: list[BBox] = [box for _, box, _ in spans]
    boxes.extend(image.bbox for image in images)

    line_groups: list[tuple[float, list[TextSpan]]] = []
    current_y = None
    current_line: list[TextSpan] = []
    for style, box, y0 in spans:
        if current_y is None or abs(y0 - current_y) <= SAME_LINE_TOLERANCE:
            current_line.append(style)
            current_y = y0 if current_y is None else current_y
        else:
            if current_line:
                line_groups.append((current_y, current_line))
            current_line = [style]
            current_y = y0
    if current_line and current_y is not None:
        line_groups.append((current_y, current_line))

    events: list[tuple[float, int, CellPart]] = []
    joined_text, joined_spans = _compose_cell_styled(line_groups)
    if joined_text:
        first_y = line_groups[0][0] if line_groups else 0.0
        events.append(
            (
                first_y,
                0,
                CellPart(kind="text", text=joined_text, spans=joined_spans),
            )
        )
    for image in images:
        events.append((image.bbox.y0, 1000, CellPart(kind="image", image=image)))
    events.sort(key=lambda item: (item[0], item[1]))
    parts = [event[2] for event in events]

    content_bbox = None
    if boxes:
        content_bbox = BBox(
            min(box.x0 for box in boxes),
            min(box.y0 for box in boxes),
            max(box.x1 for box in boxes),
            max(box.y1 for box in boxes),
        )

    return parts, joined_text, images, content_bbox


def _is_cell_bullet_glyph(char: str) -> bool:
    """True when ``char`` is a list bullet used inside table cells."""
    if len(char) != 1:
        return False
    return char in CELL_BULLET_CHARS or char == "•"


def _normalize_cell_line(text: str) -> str:
    """Keep original marks; only rewrite Symbol/Wingdings PUA to Unicode."""

    return canonical_text(text)


def _split_leading_bullet(text: str) -> tuple[str | None, str]:
    """Split a cell line into ``(bullet, rest)`` when it starts with a list mark."""
    match = CELL_BULLET_RE.match(text)
    if match:
        return "•", text[match.end():].strip()
    if text.startswith("•"):
        return "•", text[1:].strip()
    return None, text


def _compose_cell_text(lines: Sequence[str]) -> str:
    """Keep each bullet item on its own line; join wrapped continuations with spaces."""

    items: list[str] = []
    current: list[str] = []

    def flush() -> None:
        """Join the current wrapped line pieces into one bullet/paragraph."""
        if current:
            items.append(_join_line_pieces(current))
            current.clear()

    for raw in lines:
        line = _normalize_cell_line(raw).strip()
        if not line:
            continue
        bullet, rest = _split_leading_bullet(line)
        if bullet:
            flush()
            current.append(f"{bullet} {rest}".strip() if rest else bullet)
        else:
            current.append(line)
    flush()
    return "\n".join(items)


def _compose_cell_styled(
    line_groups: Sequence[tuple[float, list[TextSpan]]],
) -> tuple[str, list[TextSpan]]:
    """Join cell lines into plain text plus styled spans, keeping bullet breaks."""
    prepared: list[list[TextSpan]] = []
    for _y0, spans in line_groups:
        normalized = _normalize_span_line(spans)
        if normalized:
            prepared.append(normalized)

    items: list[list[TextSpan]] = []
    current: list[TextSpan] = []

    def flush() -> None:
        """Move the current bullet/paragraph spans onto ``items``."""
        if current:
            items.append(TextSpan.merge_adjacent(current, join_across_newlines=False))
            current.clear()

    for spans in prepared:
        line_text = "".join(span.text for span in spans).strip()
        bullet, _rest = _split_leading_bullet(line_text)
        if bullet:
            flush()
            current.extend(spans)
        else:
            current = _append_wrapped_spans(current, spans)
    flush()

    result: list[TextSpan] = []
    texts: list[str] = []
    for spans in items:
        piece = "".join(span.text for span in spans).strip()
        if not piece:
            continue
        texts.append(piece)
        if result:
            result.append(TextSpan(text="\n"))
        result.extend(spans)
    return "\n".join(texts), TextSpan.merge_adjacent(result, join_across_newlines=False)


def _normalize_span_line(spans: Sequence[TextSpan]) -> list[TextSpan]:
    """Canonicalize glyphs in a line of spans and space a leading ``•``."""
    result: list[TextSpan] = []
    for span in spans:
        text = _normalize_cell_line(span.text)
        if text:
            result.append(TextSpan(text, span.bold, span.italic, span.underline))
    result = TextSpan.merge_adjacent(result, join_across_newlines=False)
    if result and result[0].text.startswith("•") and not result[0].text.startswith("• "):
        first = result[0]
        rest = first.text[1:].lstrip()
        first.text = f"• {rest}" if rest else "• "
    return result


def _append_wrapped_spans(
    current: list[TextSpan], extra: Sequence[TextSpan]
) -> list[TextSpan]:
    """Append wrapped spans with a space, unless the last token ends in ``/`` or ``-``."""
    extra_list = [span for span in extra if span.text]
    if not extra_list:
        return current
    if not current:
        return list(extra_list)
    last = current[-1]
    if last.text.endswith("/") or last.text.endswith("-") or last.text.endswith(" "):
        return TextSpan.merge_adjacent(current + extra_list, join_across_newlines=False)
    space = TextSpan(
        text=" ",
        bold=last.bold,
        italic=last.italic,
        underline=last.underline,
    )
    return TextSpan.merge_adjacent(current + [space] + extra_list, join_across_newlines=False)


def _join_line_pieces(parts: Sequence[str]) -> str:
    """Join wrapped cell fragments with spaces, keeping hyphen/slash joins."""
    result = ""
    for piece in parts:
        piece = piece.strip()
        if not piece:
            continue
        if not result:
            result = piece
        elif result.endswith("/") or result.endswith("-"):
            result += piece
        else:
            result += " " + piece
    return result


def _detect_alignment(cell: BBox, content: BBox | None) -> tuple[str, str]:
    """Infer alignment from leftover space. Default is left/top (typical SOP cells)."""

    if content is None or content.is_empty or cell.width <= 0 or cell.height <= 0:
        return "left", "top"

    left_gap = content.x0 - cell.x0
    right_gap = cell.x1 - content.x1
    top_gap = content.y0 - cell.y0
    bottom_gap = cell.y1 - content.y1
    width_ratio = content.width / cell.width
    height_ratio = content.height / cell.height

    # Wrapped or near-full-width text is left-aligned, not center/right.
    if width_ratio >= 0.65:
        horizontal = "left"
    elif (
        min(left_gap, right_gap) > ALIGN_TOLERANCE
        and abs(left_gap - right_gap) <= max(ALIGN_TOLERANCE, 0.08 * cell.width)
    ):
        horizontal = "center"
    elif right_gap <= ALIGN_TOLERANCE and left_gap > right_gap + ALIGN_TOLERANCE:
        horizontal = "right"
    else:
        horizontal = "left"

    if height_ratio >= 0.65:
        vertical = "top"
    elif (
        min(top_gap, bottom_gap) > ALIGN_TOLERANCE
        and abs(top_gap - bottom_gap) <= max(ALIGN_TOLERANCE, 0.12 * cell.height)
    ):
        vertical = "middle"
    elif bottom_gap <= ALIGN_TOLERANCE and top_gap > bottom_gap + ALIGN_TOLERANCE:
        vertical = "bottom"
    else:
        vertical = "top"

    return horizontal, vertical


def _cell_shading(drawings: Sequence[dict], cell: BBox) -> str | None:
    """Return a hex fill color when a non-white fill covers most of the cell."""
    best_color: str | None = None
    best_ratio = 0.0

    for drawing in drawings:
        fill = drawing.get("fill")
        if fill is None:
            continue
        rect = drawing.get("rect")
        if rect is None:
            continue

        fill_box = BBox.from_rect(rect)
        fill_cx = (fill_box.x0 + fill_box.x1) / 2
        fill_cy = (fill_box.y0 + fill_box.y1) / 2
        if not cell.contains_point(fill_cx, fill_cy):
            continue

        cover = cell.intersection_area(fill_box)
        if cell.area <= 0:
            continue
        ratio = cover / cell.area
        if ratio < SHADING_COVERAGE:
            continue

        color = _rgb_to_hex(fill)
        if color is None or _is_near_white(fill):
            continue
        if ratio > best_ratio:
            best_ratio = ratio
            best_color = color

    return best_color


def _rgb_to_hex(color) -> str | None:
    """Convert a 0–1 RGB triple to ``#RRGGBB``, or ``None`` if it is invalid."""
    try:
        red, green, blue = color[:3]
        return (
            f"#{int(max(0, min(255, red * 255))):02X}"
            f"{int(max(0, min(255, green * 255))):02X}"
            f"{int(max(0, min(255, blue * 255))):02X}"
        )
    except RECOVERABLE:
        return None


def _is_near_white(color) -> bool:
    """True when an RGB triple is close enough to white to ignore as a stroke/fill."""
    try:
        red, green, blue = color[:3]
        return red >= NEAR_WHITE and green >= NEAR_WHITE and blue >= NEAR_WHITE
    except RECOVERABLE:
        return False
