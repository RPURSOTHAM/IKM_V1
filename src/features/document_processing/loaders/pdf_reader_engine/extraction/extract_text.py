"""Extract text blocks in reading order, including indentation and bullets."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pymupdf

from ..document import BBox, TextBlock, TextLine, TextSpan
from .pdf_symbols import STANDARD_BULLET, canonical_symbol, canonical_text, is_list_bullet_marker

INDENT_STEP = 18.0
INSIDE_RECT_THRESHOLD = 0.55
PAGE_OVERLAP_THRESHOLD = 0.80
TEXT_ORDER_Y_BAND = 6.0
MISSED_BLOCK_OVERLAP = 0.45
LABEL_COLON_MIN = 2
LABEL_COLON_MAX = 20
SAME_BASELINE_TOLERANCE = 2.0
POINT_TO_MM = 0.3528
SAME_LINE_MERGE_MM = 1.0
SAME_LINE_MERGE_Y_PT = SAME_LINE_MERGE_MM / POINT_TO_MM
# Two phrases on one baseline with at least this x-gap stay separate blocks
# unless either side looks like a labeled field (``Document No.:``).
SAME_LINE_LARGE_GAP = 40.0
# Tab encoded as spaces inside one span (not ordinary word spacing).
SPACE_RUN_RE = re.compile(r" {4,}")
SENTENCE_END_PUNCT = ".?!"
WRAP_BLOCK_MAX_GAP = 8.0
WRAP_BLOCK_MAX_OVERLAP = 2.0
WRAP_BLOCK_MAX_LEFT_SHIFT = 20.0
# Do not wrap-merge an uppercase "continuation" when the previous line
# ends this far short of the content column (figure titles / captions).
WRAP_BLOCK_MIN_RIGHT_SLACK = 80.0
# Lone ``Title:`` label plus the nearby title value (no colon).
TITLE_LABEL_RE = re.compile(r"^title\s*:\s*$", re.IGNORECASE)
TITLE_MERGE_MAX_GAP = 90.0
# Standalone sub-header: one capitalized word, colon suffix, whole line.
SUB_HEADER_RE = re.compile(r"^[A-Z][A-Za-z]*\s*:\s*$")

SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?$")
# Two+ dotted parts. Allow a glued capital (``6.3.1Following``) so the
# number is still a new line, not a wrap of the heading above.
SECTION_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?(?:\s+|$|(?=[A-Z]))")

BULLET_CHARS = (
    "\u2022\u2023\u25e6\u2043\u2219\u00b7\u25cf\u25cb\u25a0\u25a1"
    "\u25aa\u25ab\u25c6\u25c7\u25b6\u25b8*"
)

BULLET_RE = re.compile(
    rf"^\s*(?P<bullet>"
    rf"[{re.escape(BULLET_CHARS)}]"
    rf"|[0-9]{{1,3}}[.)]"
    rf"|[A-Za-z][.)]"
    rf"|[ivxlcdmIVXLCDM]{{1,6}}[.)]"
    rf")(?:\s+|$)"
)

TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_COLLECT_STYLES
CHAR_BOLD = pymupdf.mupdf.FZ_STEXT_BOLD
CHAR_UNDERLINE = pymupdf.mupdf.FZ_STEXT_UNDERLINE


def extract_text_blocks(
    page: pymupdf.Page,
    exclude_rects: Sequence[BBox] | None = None,
) -> list[TextBlock]:
    """Return text blocks in original page order, skipping excluded regions.

    Table regions are excluded per line, not per block, so a heading that
    PyMuPDF grouped with a table header (e.g. ``6.8. Pictorial Autoclave``)
    is kept while ``Sr. No.`` / ``IMAGE`` lines inside the table are dropped.
    """

    page_box = BBox.from_rect(page.rect)
    data = page.get_text("dict", flags=TEXT_FLAGS, sort=True)
    raw_blocks: list[tuple[int, BBox, list[dict]]] = []

    for block in data.get("blocks", []):
        if block.get("type", 0) != 0 or "lines" not in block:
            continue

        bbox = BBox(*block["bbox"])
        if bbox.is_empty:
            continue
        if bbox.area and bbox.intersection_area(page_box) / bbox.area < PAGE_OVERLAP_THRESHOLD:
            continue

        kept_lines = _lines_outside_rects(block["lines"], exclude_rects)
        if not kept_lines:
            continue

        raw_blocks.append((len(raw_blocks), bbox, kept_lines))

    raw_blocks = _expand_gapped_same_row_lines(raw_blocks)

    blocks: list[TextBlock] = []
    for order, bbox, lines in raw_blocks:
        block = _build_text_block(order, bbox, lines)
        if block is None:
            continue
        blocks.append(block)

    expected = [
        _copy_text_block(block)
        for block in blocks
        if not is_ignored_text_block(block)
    ]

    blocks = _normalize_list_layout(blocks)
    blocks = validate_text_blocks(blocks)
    blocks = merge_same_line_text_blocks(blocks)
    blocks = merge_orphan_section_number_blocks(blocks)
    blocks = merge_title_label_text_blocks(blocks)
    blocks = merge_wrapped_text_blocks(blocks)
    for block in blocks:
        block.lines = apply_line_break_rules(block.lines)
    blocks = split_standalone_sub_header_blocks(blocks)
    kept = [block for block in blocks if not is_ignored_text_block(block)]
    kept = validate_no_missed_text_blocks(kept, expected)
    return validate_text_blocks(kept)


def merge_same_line_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Merge text blocks that belong on one visual line.

    Coordinates are read as ``(x1, y1)`` top-left and ``(x2, y2)`` bottom-right
    (``BBox.x0/y0`` and ``BBox.x1/y1``). PDF extractors often emit each
    header field as its own block when the tops are a fraction of a millimetre
    apart, for example::

        Document No.: | BLS2B5-QCM-SOP-0041 | Version No: | 5.0
        Effective Date: 04/09/2026 | Review Date: 03/09/2028

    Merge ``following`` into ``previous`` when both are true:

    - ``x1`` of the following block is greater than ``x1`` of the previous
      block (the following field starts further right, even if the boxes
      overlap horizontally)
    - ``y1`` of the two blocks differs by less than 1 mm
      (1 point ≈ 0.3528 mm, so 1 mm ≈ 2.83 points)

    After a merge the combined box becomes the new ``previous``, so three
    or more fields on the same row are collapsed in one left-to-right pass.
    A large empty x-gap (``SAME_LINE_LARGE_GAP``) is not merged unless at
    least one side contains ``:`` (letterhead ``Effective Date:`` /
    ``Review Date:``). Paired captions such as ``Before wrapping`` /
    ``After wrapping`` stay separate. Blocks that fail the test start a
    new group. Ignored boilerplate is left untouched.
    """

    if len(blocks) < 2:
        return blocks

    ordered = sorted(blocks, key=_block_reading_key)
    merged: list[TextBlock] = []
    current = ordered[0]
    for following in ordered[1:]:
        if _should_merge_same_line_blocks(current, following):
            current = _merge_text_blocks(current, following)
        else:
            merged.append(current)
            current = following
    merged.append(current)
    return merged


def _expand_gapped_same_row_lines(
    raw_blocks: list[tuple[int, BBox, list[dict]]],
) -> list[tuple[int, BBox, list[dict]]]:
    """Split a PDF block when two same-row pieces sit far apart in x.

    PyMuPDF often emits paired captions as two lines (or span groups) in
    one block. A tab can also be many spaces inside a single span
    (``Before wrapping          After covering…``). Letterhead fields
    with ``:`` stay grouped so same-line merge can still collapse
    ``Document No.:`` with its value.
    """

    expanded: list[tuple[int, BBox, list[dict]]] = []
    for _order, bbox, lines in raw_blocks:
        pieces: list[dict] = []
        for line in lines:
            pieces.extend(_split_line_span_clusters(line))
        if not pieces:
            expanded.append((len(expanded), bbox, lines))
            continue
        groups: list[list[dict]] = [[pieces[0]]]
        for piece in pieces[1:]:
            previous = groups[-1][-1]
            if _same_row_large_gap_raw(previous, piece) and not _letterhead_pair_texts(
                _raw_line_text(previous),
                _raw_line_text(piece),
            ):
                groups.append([piece])
            else:
                groups[-1].append(piece)
        for group in groups:
            expanded.append((len(expanded), _bbox_from_raw_lines(group) or bbox, group))
    return expanded


def _split_line_span_clusters(line: dict) -> list[dict]:
    """Return one or more line dicts when spans are separated by a large gap.

    Also splits a single span whose text contains a wide run of spaces
    (a visual tab). Ordinary two-space word gaps are left intact.
    """

    spans: list[dict] = []
    for span in line.get("spans") or []:
        if not (span.get("text") or "").strip():
            continue
        spans.extend(_split_span_on_space_runs(span))
    if len(spans) < 2:
        return [line]
    spans = sorted(spans, key=lambda span: (span.get("bbox") or [0.0])[0])
    clusters: list[list[dict]] = [[spans[0]]]
    for span in spans[1:]:
        previous = clusters[-1][-1]
        previous_box = previous.get("bbox")
        box = span.get("bbox")
        if previous_box and box and (box[0] - previous_box[2]) >= SAME_LINE_LARGE_GAP:
            clusters.append([span])
        else:
            clusters[-1].append(span)
    if len(clusters) == 1:
        return [line]
    parts: list[dict] = []
    for group in clusters:
        part = dict(line)
        part["spans"] = group
        xs0 = min(span["bbox"][0] for span in group)
        ys0 = min(span["bbox"][1] for span in group)
        xs1 = max(span["bbox"][2] for span in group)
        ys1 = max(span["bbox"][3] for span in group)
        part["bbox"] = (xs0, ys0, xs1, ys1)
        parts.append(part)
    return parts


def _split_span_on_space_runs(span: dict) -> list[dict]:
    """Split one span when a space run is wide enough to be a column tab."""

    text = span.get("text") or ""
    bbox = span.get("bbox")
    if not bbox or len(bbox) < 4 or len(text) < 5:
        return [span]
    matches = list(SPACE_RUN_RE.finditer(text))
    if not matches:
        return [span]
    width = bbox[2] - bbox[0]
    if width <= 0:
        return [span]
    unit = width / len(text)
    cuts = [
        match
        for match in matches
        if (match.end() - match.start()) * unit >= SAME_LINE_LARGE_GAP
    ]
    if not cuts:
        return [span]

    pieces: list[dict] = []
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    cursor = 0
    for match in cuts:
        chunk = text[cursor : match.start()]
        if chunk.strip():
            pieces.append(
                _copy_span_slice(
                    span,
                    chunk,
                    (x0 + cursor * unit, y0, x0 + match.start() * unit, y1),
                )
            )
        cursor = match.end()
    chunk = text[cursor:]
    if chunk.strip():
        pieces.append(_copy_span_slice(span, chunk, (x0 + cursor * unit, y0, x1, y1)))
    return pieces if len(pieces) >= 2 else [span]


def _copy_span_slice(span: dict, text: str, bbox: tuple[float, float, float, float]) -> dict:
    """Copy a PyMuPDF span dict with new wording and box."""

    piece = dict(span)
    piece["text"] = text
    piece["bbox"] = bbox
    return piece


def _same_row_large_gap_raw(previous: dict, following: dict) -> bool:
    """True when two raw lines share a baseline but sit far apart in x."""

    previous_box = previous.get("bbox")
    following_box = following.get("bbox")
    if not previous_box or not following_box:
        return False
    if abs(following_box[1] - previous_box[1]) >= SAME_LINE_MERGE_Y_PT:
        return False
    return (following_box[0] - previous_box[2]) >= SAME_LINE_LARGE_GAP


def _raw_line_text(line: dict) -> str:
    """Plain text of a raw PyMuPDF line or span cluster."""

    return "".join(span.get("text") or "" for span in (line.get("spans") or []))


def _bbox_from_raw_lines(lines: Sequence[dict]) -> BBox | None:
    """Smallest box covering the raw PyMuPDF lines (and their spans)."""

    boxes: list[BBox] = []
    for line in lines:
        bbox = line.get("bbox")
        if bbox and len(bbox) >= 4:
            boxes.append(BBox(*bbox[:4]))
        for span in line.get("spans") or []:
            span_box = span.get("bbox")
            if span_box and len(span_box) >= 4:
                boxes.append(BBox(*span_box[:4]))
    if not boxes:
        return None
    return BBox(
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _letterhead_pair_texts(left: str, right: str) -> bool:
    """True when a same-row pair should stay together as labeled fields."""

    return _has_field_colon(left) or _has_field_colon(right)


def _has_field_colon(text: str) -> bool:
    """True when ``text`` contains a labeled-field colon."""

    return ":" in (text or "")


def merge_orphan_section_number_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Prepend a lone section number onto the related text to its right.

    PyMuPDF sometimes emits ``6.1.10`` as its own block because the body
    line starts a few points higher. Same-line merge misses that (tops
    differ by more than 1 mm, and reading order sees the body first).

    Join when all of these hold:

    - the number block is only a dotted section number (``6.1``, ``6.1.10``)
    - a partner block starts to the right of the number
    - the two boxes overlap vertically
    - the partner does not already start with a section number
    """

    if len(blocks) < 2:
        return blocks

    ordered = sorted(blocks, key=_block_reading_key)
    skip: set[int] = set()
    merged: list[TextBlock] = []
    for index, block in enumerate(ordered):
        if index in skip:
            continue
        if not _is_lone_section_number_block(block):
            merged.append(block)
            continue
        partner_index = _find_orphan_number_partner(ordered, index)
        if partner_index is None:
            merged.append(block)
            continue
        skip.add(partner_index)
        partner = ordered[partner_index]
        if partner_index < index:
            merged = [item for item in merged if item is not partner]
        merged.append(_prepend_text_block(block, partner))
    return merged


def _is_lone_section_number_block(block: TextBlock) -> bool:
    """True when ``block`` is only a dotted section number, with no other text."""

    if is_ignored_text_block(block):
        return False
    stripped = (block.text or "").strip()
    if "." not in stripped:
        return False
    return bool(SECTION_NUMBER_RE.match(stripped))


def _find_orphan_number_partner(
    blocks: list[TextBlock],
    number_index: int,
) -> int | None:
    """Index of the right-hand text that vertically overlaps the lone number."""

    number = blocks[number_index]
    if number.bbox is None:
        return None
    best_index: int | None = None
    best_overlap = 0.0
    for index, partner in enumerate(blocks):
        if index == number_index or partner.bbox is None:
            continue
        if is_ignored_text_block(partner):
            continue
        if _is_lone_section_number_block(partner):
            continue
        if partner.bbox.x0 <= number.bbox.x0:
            continue
        if partner.lines and _starts_with_section_number(partner.lines[0]):
            continue
        overlap = _vertical_overlap(number.bbox, partner.bbox)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    return best_index


def _vertical_overlap(left: BBox, right: BBox) -> float:
    """Height of the shared vertical span; 0 when the boxes do not overlap in y."""

    return max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))


def _prepend_text_block(prefix: TextBlock, target: TextBlock) -> TextBlock:
    """Put ``prefix`` at the start of ``target``'s first line and union the boxes."""

    merged = _copy_text_block(target)
    extra = _copy_text_block(prefix)
    if not extra.lines:
        return merged
    if not merged.lines:
        merged.lines = extra.lines
    else:
        merged.lines[0] = _join_prefix_line(extra.lines[0], merged.lines[0])
    for line_number, line in enumerate(merged.lines, start=1):
        line.line_number = line_number
    merged.bbox = BBox.combine(extra.bbox, merged.bbox)
    merged.indent = min(merged.indent, extra.indent)
    return merged


def _join_prefix_line(prefix: TextLine, body: TextLine) -> TextLine:
    """Put ``prefix`` text in front of ``body`` with a space."""

    prefix_text = (prefix.text or "").rstrip()
    body_text = (body.text or "").lstrip()
    if not prefix_text:
        return body
    if not body_text:
        return prefix
    prefix_spans = list(prefix.spans) if prefix.spans else [TextSpan(text=prefix_text)]
    body_spans = list(body.spans) if body.spans else [TextSpan(text=body_text)]
    if prefix_spans and not prefix_spans[-1].text.endswith(" "):
        prefix_spans[-1].text += " "
    body.text = f"{prefix_text} {body_text}"
    body.spans = TextSpan.merge_adjacent(prefix_spans + body_spans)
    body.bbox = BBox.combine(prefix.bbox, body.bbox) or body.bbox
    body.indent = min(prefix.indent, body.indent)
    body.indent_level = min(prefix.indent_level, body.indent_level)
    return body


def merge_title_label_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Merge a lone ``Title:`` label with the related title value next to it.

    SOP letterheads often emit ``Title:`` as its own block and the document
    name as another, slightly above or below (or to the right). Same-line
    merge misses them when the tops differ by more than 1 mm.

    Join the label with a previous or following block when all of these hold:

    - one block is exactly ``Title:`` (optional spaces)
    - the other block's text contains no ``:``
    - the two boxes do not overlap
    - the gap between the boxes (horizontal if on one row, vertical if
      stacked) is at most ``TITLE_MERGE_MAX_GAP``
    """

    if len(blocks) < 2:
        return blocks

    ordered = sorted(blocks, key=_block_reading_key)
    skip: set[int] = set()
    merged: list[TextBlock] = []
    for index, block in enumerate(ordered):
        if index in skip:
            continue
        if not _is_title_label_block(block):
            merged.append(block)
            continue
        partner_index = _find_title_value_neighbor(ordered, index)
        if partner_index is None:
            merged.append(block)
            continue
        skip.add(partner_index)
        value = ordered[partner_index]
        if partner_index < index:
            merged = [item for item in merged if item is not value]
        merged.append(_merge_text_blocks(block, value))
    return merged


def _is_title_label_block(block: TextBlock) -> bool:
    """True when the block is only the letterhead label ``Title:``."""

    if is_ignored_text_block(block):
        return False
    return bool(TITLE_LABEL_RE.match((block.text or "").strip()))


def _is_title_value_block(block: TextBlock) -> bool:
    """True when the block looks like a title value, not another labeled field."""

    if is_ignored_text_block(block):
        return False
    text = block.text or ""
    if not text.strip():
        return False
    return ":" not in text


def _find_title_value_neighbor(ordered: Sequence[TextBlock], index: int) -> int | None:
    """Return the closer previous or following value block, if related."""

    label = ordered[index]
    best_index: int | None = None
    best_gap: float | None = None
    for partner_index in (index - 1, index + 1):
        if partner_index < 0 or partner_index >= len(ordered):
            continue
        other = ordered[partner_index]
        if not _is_title_value_block(other):
            continue
        if not _title_blocks_are_related(label, other):
            continue
        gap = _title_pair_gap(label, other)
        if best_gap is None or gap < best_gap:
            best_index = partner_index
            best_gap = gap
    return best_index


def _title_blocks_are_related(label: TextBlock, value: TextBlock) -> bool:
    """True when ``label`` and ``value`` sit nearby and do not overlap."""

    if label.bbox is None or value.bbox is None:
        return False
    if not label.bbox.intersection(value.bbox).is_empty:
        return False
    return _title_pair_gap(label, value) <= TITLE_MERGE_MAX_GAP


def _title_pair_gap(label: TextBlock, value: TextBlock) -> float:
    """Horizontal gap if on one row, otherwise the vertical stacked gap."""

    assert label.bbox is not None and value.bbox is not None
    left = label.bbox
    right = value.bbox
    y_overlap = min(left.y1, right.y1) - max(left.y0, right.y0)
    if y_overlap > -SAME_LINE_MERGE_Y_PT:
        if left.x1 <= right.x0:
            return right.x0 - left.x1
        if right.x1 <= left.x0:
            return left.x0 - right.x1
        return float("inf")
    if left.y1 <= right.y0:
        return right.y0 - left.y1
    if right.y1 <= left.y0:
        return left.y0 - right.y1
    return float("inf")


def merge_wrapped_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Merge a wrapped continuation that PyMuPDF split into the next block.

    ``merge_same_line_text_blocks`` only joins fields on one visual row
    (tops within 1 mm). ``apply_line_break_rules`` only joins lines that
    already sit inside the same block. A wrapped sentence such as::

        …the duration of the entire process from
        the start until the objects can be taken out…

    is often two blocks a full line apart, so neither of those helpers
    fires.

    Append ``following`` onto ``previous`` when all of these hold:

    - ``previous`` does not end with sentence punctuation (``.`` ``?``
      ``!``). A comma or any other unfinished line qualifies
    - ``following`` is not a bullet and not a dotted section heading
    - ``following`` is a wrap: lowercase, or a hyphenated id after ``-``,
      or an uppercase continuation when ``previous`` is not a heading
      **and** ``previous`` reaches the content column (right slack
      ≤ ``WRAP_BLOCK_MIN_RIGHT_SLACK``). A short figure title above a
      caption is not treated as a wrap.
    - ``following`` sits just under ``previous`` (small gap, not a
      side-by-side column whose boxes overlap in y)
    - ``following`` does not start far to the left of ``previous``

    Same-line header merges and intra-block line-break rules are
    unchanged. Ignored boilerplate (export stamp, headers) is not used
    as a barrier, so a wrap split around the stamp still joins.
    After this pass, ``apply_line_break_rules`` still runs on the
    combined lines and turns the wrap into a space.
    """

    if len(blocks) < 2:
        return blocks

    ordered = sorted(blocks, key=_block_reading_key)
    ignored = [block for block in ordered if is_ignored_text_block(block)]
    candidates = [block for block in ordered if not is_ignored_text_block(block)]
    if len(candidates) < 2:
        return blocks

    merged: list[TextBlock] = []
    current = candidates[0]
    column_right = _content_column_right(candidates)
    for following in candidates[1:]:
        if _should_merge_wrapped_blocks(current, following, column_right):
            current = _append_wrapped_block(current, following)
        else:
            merged.append(current)
            current = following
    merged.append(current)
    merged.extend(ignored)
    return merged


def _should_merge_wrapped_blocks(
    previous: TextBlock,
    following: TextBlock,
    column_right: float | None = None,
) -> bool:
    """True when ``following`` continues an unfinished ``previous`` block."""

    if previous.bbox is None or following.bbox is None:
        return False
    if is_ignored_text_block(previous) or is_ignored_text_block(following):
        return False
    if not previous.lines or not following.lines:
        return False

    previous_text = (previous.text or "").rstrip()
    following_text = (following.text or "").lstrip()
    if not previous_text or not following_text:
        return False
    if _is_standalone_sub_header_line(previous.lines[-1]) or _is_standalone_sub_header_line(
        following.lines[0]
    ):
        return False
    if previous_text[-1] in SENTENCE_END_PUNCT:
        return False
    if _starts_with_bullet_point(following.lines[0]) or _starts_with_section_number(
        following.lines[0]
    ):
        return False
    if not _is_wrapped_continuation(previous.lines[0], previous_text, following_text):
        return False
    if _uppercase_wrap_has_large_right_slack(previous, following_text, column_right):
        return False

    gap = following.bbox.y0 - previous.bbox.y1
    if gap < -WRAP_BLOCK_MAX_OVERLAP:
        return False
    if gap > WRAP_BLOCK_MAX_GAP:
        return False
    if following.bbox.x0 < previous.bbox.x0 - WRAP_BLOCK_MAX_LEFT_SHIFT:
        return False
    return True


def _content_column_right(blocks: Sequence[TextBlock]) -> float | None:
    """Right edge of leftover body text, used as the content-column end.

    Ignored stamps are not in ``blocks``. A wide figure caption on the
    same page does not move this edge; the longest remaining line does.
    """

    rights: list[float] = []
    for block in blocks:
        for line in block.lines:
            if line.bbox is None or line.bbox.is_empty:
                continue
            rights.append(line.bbox.x1)
        if block.bbox is not None and not block.bbox.is_empty:
            rights.append(block.bbox.x1)
    if not rights:
        return None
    return max(rights)


def _uppercase_wrap_has_large_right_slack(
    previous: TextBlock,
    following_text: str,
    column_right: float | None,
) -> bool:
    """True when an uppercase follower is a new line, not a wrap.

    Lowercase wraps and hyphenated ids are excluded so existing wrap
    rules stay in force. Slack is measured from the last line of
    ``previous`` to the content column, not from the page edge.
    """

    if column_right is None or not following_text or not following_text[0].isupper():
        return False
    if _ends_with_hyphen((previous.text or "").rstrip()):
        return False
    last = previous.lines[-1]
    last_x1 = last.bbox.x1 if last.bbox is not None else previous.bbox.x1
    return (column_right - last_x1) > WRAP_BLOCK_MIN_RIGHT_SLACK


def _append_wrapped_block(previous: TextBlock, following: TextBlock) -> TextBlock:
    """Copy ``previous`` and append ``following`` lines; do not flatten to one line."""

    merged = _copy_text_block(previous)
    extra = _copy_text_block(following)
    merged.lines.extend(extra.lines)
    for line_number, line in enumerate(merged.lines, start=1):
        line.line_number = line_number
    merged.bbox = BBox.combine(merged.bbox, extra.bbox)
    merged.indent = min(merged.indent, extra.indent)
    return merged


def _should_merge_same_line_blocks(previous: TextBlock, following: TextBlock) -> bool:
    """True when ``following`` starts further right and on the same baseline."""
    if previous.bbox is None or following.bbox is None:
        return False
    if is_ignored_text_block(previous) or is_ignored_text_block(following):
        return False
    previous_x0, previous_y0 = previous.bbox.x0, previous.bbox.y0
    following_x0, following_y0 = following.bbox.x0, following.bbox.y0
    if following_x0 <= previous_x0:
        return False
    if abs(following_y0 - previous_y0) >= SAME_LINE_MERGE_Y_PT:
        return False
    gap = following.bbox.x0 - previous.bbox.x1
    if gap >= SAME_LINE_LARGE_GAP and not _letterhead_pair_texts(
        previous.text or "",
        following.text or "",
    ):
        return False
    return True


def _merge_text_blocks(previous: TextBlock, following: TextBlock) -> TextBlock:
    """Join ``following`` onto the last line of ``previous`` and union the boxes."""
    merged = _copy_text_block(previous)
    extra = _copy_text_block(following)
    if not merged.lines:
        merged.lines = extra.lines
    else:
        for line in extra.lines:
            merged.lines[-1] = _join_wrapped_lines(merged.lines[-1], line)
    for line_number, line in enumerate(merged.lines, start=1):
        line.line_number = line_number
    merged.bbox = BBox.combine(merged.bbox, extra.bbox)
    merged.indent = min(merged.indent, extra.indent)
    return merged


def validate_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Check extracted text and repair it when reading order is not intact.

    Current rules (add more here later):
    - lines inside a block must run top-to-bottom, then left-to-right
    - blocks on the page must follow the same reading order
    """

    ordered: list[TextBlock] = []
    for block in blocks:
        lines = sorted(block.lines, key=_text_reading_key)
        if not _lines_are_in_order(block.lines):
            for line_number, line in enumerate(lines, start=1):
                line.line_number = line_number
            block.lines = lines
        ordered.append(block)

    ordered.sort(key=_block_reading_key)
    for order, block in enumerate(ordered):
        block.order = order
    return ordered


def validate_no_missed_text_blocks(
    extracted: list[TextBlock],
    expected: Sequence[TextBlock],
) -> list[TextBlock]:
    """Restore source text blocks that never made it into the extraction result.

    Current rules (add more here later):
    - every expected block must appear in the result by text or bbox overlap
    """

    result = list(extracted)
    for source in expected:
        if is_ignored_text_block(source):
            continue
        if _text_block_is_represented(source, result):
            continue
        recovered = _copy_text_block(source)
        recovered.lines = apply_line_break_rules(recovered.lines)
        result.append(recovered)
    return result


def _text_block_is_represented(
    source: TextBlock,
    extracted: Sequence[TextBlock],
) -> bool:
    """True when ``source`` already appears in ``extracted`` by text or overlap."""
    source_text = _coverage_text(source)
    if not source_text:
        return True

    extracted_text = " ".join(_coverage_text(block) for block in extracted)
    if source_text and source_text in extracted_text:
        return True

    if source.bbox is None or source.bbox.is_empty:
        return False
    for block in extracted:
        if block.bbox is None or block.bbox.is_empty:
            continue
        if source.bbox.overlap_ratio(block.bbox) >= MISSED_BLOCK_OVERLAP:
            return True
        if block.bbox.overlap_ratio(source.bbox) >= MISSED_BLOCK_OVERLAP:
            return True
    return False


def _coverage_text(block: TextBlock) -> str:
    """Collapsed whitespace used to test whether a block's wording survived."""
    return " ".join((block.text or "").split())


def _copy_text_block(block: TextBlock) -> TextBlock:
    """Deep-copy a text block so later merges do not mutate the original."""
    lines = [
        TextLine(
            line_number=line.line_number,
            text=line.text,
            indent=line.indent,
            indent_level=line.indent_level,
            is_bullet=line.is_bullet,
            bullet=line.bullet,
            bbox=line.bbox,
            spans=[
                TextSpan(
                    text=span.text,
                    bold=span.bold,
                    italic=span.italic,
                    underline=span.underline,
                )
                for span in line.spans
            ],
        )
        for line in block.lines
    ]
    return TextBlock(
        order=block.order,
        lines=lines,
        indent=block.indent,
        bbox=block.bbox,
    )


def _build_text_block(order: int, bbox: BBox, lines: Sequence[dict]) -> TextBlock | None:
    """Turn PyMuPDF line dicts into a ``TextBlock`` with bullets and indents."""
    built: list[tuple[dict, str | None, str, BBox, list[TextSpan]]] = []
    for line in lines:
        parsed = _parse_line(line)
        if parsed is not None:
            built.append(parsed)

    if not built:
        return _fallback_text_block(order, bbox, lines)

    block_left = min(item[3].x0 for item in built)
    text_lines: list[TextLine] = []
    for line_number, (_raw, bullet, text, line_bbox, spans) in enumerate(built, start=1):
        indent = round(line_bbox.x0, 2)
        relative = max(0.0, line_bbox.x0 - block_left)
        text_lines.append(
            TextLine(
                line_number=line_number,
                text=text,
                indent=indent,
                indent_level=int(round(relative / INDENT_STEP)),
                is_bullet=bullet is not None,
                bullet=bullet,
                bbox=line_bbox,
                spans=spans,
            )
        )
    return TextBlock(
        order=order,
        lines=text_lines,
        indent=round(block_left, 2),
        bbox=_bbox_from_text_lines(text_lines) or bbox,
    )


def _fallback_text_block(order: int, bbox: BBox, lines: Sequence[dict]) -> TextBlock | None:
    """Keep a block even when line parsing drops every span."""

    text_lines: list[TextLine] = []
    for line in lines:
        spans = line.get("spans") or []
        styled: list[TextSpan] = []
        pieces: list[str] = []
        for span in spans:
            raw = span.get("text") or ""
            if not raw.strip():
                continue
            pieces.append(raw)
            styled.append(_span_style(span, raw))
        text = "".join(pieces).strip()
        if not text:
            continue
        line_bbox = BBox(*(line.get("bbox") or bbox.to_tuple()))
        text_lines.append(
            TextLine(
                line_number=len(text_lines) + 1,
                text=text,
                indent=round(line_bbox.x0, 2),
                indent_level=0,
                is_bullet=False,
                bullet=None,
                bbox=line_bbox,
                spans=TextSpan.merge_adjacent(styled),
            )
        )
    if not text_lines:
        return None
    return TextBlock(
        order=order,
        lines=text_lines,
        indent=min(line.indent for line in text_lines),
        bbox=_bbox_from_text_lines(text_lines) or bbox,
    )


def _lines_are_in_order(lines: Sequence[TextLine]) -> bool:
    """True when lines already run top-to-bottom then left-to-right."""
    previous: tuple[float, float] | None = None
    for line in lines:
        key = _text_reading_key(line)
        if previous is not None and key < previous:
            return False
        previous = key
    return True


def _text_reading_key(line: TextLine) -> tuple[float, float]:
    """Sort key for a line: y-band then left edge."""
    band = round(line.bbox.y0 / TEXT_ORDER_Y_BAND) * TEXT_ORDER_Y_BAND
    return (band, line.bbox.x0)


def _block_reading_key(block: TextBlock) -> tuple[float, float]:
    """Sort key for a block: y-band then left edge."""
    bbox = block.bbox
    if bbox is None:
        return (0.0, 0.0)
    band = round(bbox.y0 / TEXT_ORDER_Y_BAND) * TEXT_ORDER_Y_BAND
    return (band, bbox.x0)


def apply_line_break_rules(lines: list[TextLine]) -> list[TextLine]:
    """Turn wrapped PDF line feeds into spaces, except where a break must stay.

    Edit `_should_retain_line_feed` to change when a newline is kept.
    """

    if len(lines) <= 1:
        return lines

    result: list[TextLine] = []
    current = lines[0]
    for following in lines[1:]:
        if _should_retain_line_feed(current, following):
            result.append(current)
            current = following
        else:
            current = _join_wrapped_lines(current, following)
    result.append(current)

    for line_number, line in enumerate(result, start=1):
        line.line_number = line_number
    return result


def split_standalone_sub_header_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Keep ``Definitions:`` / ``Abbreviations:`` as their own text blocks.

    A sub-header is a whole line that is one capitalized word plus ``:``.
    Those lines are not merged with the paragraph above or below.
    ``Title:`` letterhead labels are joined earlier and are left intact
    once they already include the title value.
    """

    split: list[TextBlock] = []
    for block in blocks:
        split.extend(_split_block_on_sub_headers(block))
    return split


def _split_block_on_sub_headers(block: TextBlock) -> list[TextBlock]:
    """Slice ``block`` so each standalone sub-header line is its own block."""

    if len(block.lines) <= 1:
        return [block]
    groups: list[list[TextLine]] = []
    current: list[TextLine] = []
    for line in block.lines:
        if _is_standalone_sub_header_line(line):
            if current:
                groups.append(current)
                current = []
            groups.append([line])
            continue
        current.append(line)
    if current:
        groups.append(current)
    if len(groups) <= 1:
        return [block]
    return [_text_block_from_lines(block, group) for group in groups]


def _text_block_from_lines(original: TextBlock, lines: list[TextLine]) -> TextBlock:
    """Build a block that keeps ``original.order`` and the union of ``lines``."""

    numbered: list[TextLine] = []
    for line_number, line in enumerate(lines, start=1):
        numbered.append(
            TextLine(
                line_number=line_number,
                text=line.text,
                indent=line.indent,
                indent_level=line.indent_level,
                is_bullet=line.is_bullet,
                bullet=line.bullet,
                bbox=line.bbox,
                spans=list(line.spans),
            )
        )
    return TextBlock(
        order=original.order,
        lines=numbered,
        indent=min((line.indent for line in numbered), default=original.indent),
        bbox=_bbox_from_text_lines(numbered) or original.bbox,
    )


def is_standalone_sub_header(text: str) -> bool:
    """True for ``Definitions:`` / ``Abbreviations:``, not ``Document No.:``."""

    return bool(SUB_HEADER_RE.match((text or "").strip()))


def _is_standalone_sub_header_line(line: TextLine) -> bool:
    """True when ``line`` is only a capitalized one-word label plus ``:``."""

    return is_standalone_sub_header(line.text or "")


def _is_standalone_sub_header_text(text: str) -> bool:
    """True for a standalone sub-header line."""

    return is_standalone_sub_header(text)


def _should_retain_line_feed(current: TextLine, following: TextLine) -> bool:
    """Return True when the newline between two lines should be kept.

    Lines on the same baseline are joined (``6.1.`` + title) unless they
    sit far apart in x and neither side is a labeled field.

    Otherwise keep the break when any of these is true:

    - the current line starts with a section number (``6.12 Service
      Programs``) **and** the next line starts with an uppercase letter,
      so a new sentence under a heading is not folded into the title.
      A lowercase wrap (``and password.`` after ``6.9.3 … Comment``) is
      still joined. A hyphenated document id (``GL-`` / ``QC-SOP-0284``)
      is joined even when the next piece starts with a capital.
    - the current statement ends with a sentence ``.`` (not a lone section
      number such as ``6.5.``)
    - the following line is a bullet: a symbol (``•``, dingbats) **or** a
      numeric/letter marker (``1.``, ``2)``, ``a)``)
    - the following line is a section number (``6.1.``, ``6.1.1``,
      ``6.2.1 Do not...``, or a glued ``6.3.1Following``) — dotted
      numbers with at least two parts, so they are not treated as a
      wrapped continuation of the heading above
    - the current or following line is a standalone sub-header
      (``Definitions:``, ``Abbreviations:``)
    - the following line has a ``:`` between characters 2 and 20, with no
      ``.`` or ``,`` before that colon

    Every other line feed is replaced with a single space.
    """

    if _on_same_baseline(current, following):
        gap = following.bbox.x0 - current.bbox.x1
        if gap >= SAME_LINE_LARGE_GAP and not _letterhead_pair_texts(
            current.text or "",
            following.text or "",
        ):
            return True
        return False
    if _ends_with_hyphen(current.text or "") and not _starts_with_section_number(following):
        return False
    if _starts_with_section_number(current) and _starts_with_uppercase(following):
        return True
    if _ends_with_sentence_period(current.text or ""):
        return True
    if _starts_with_bullet_point(following):
        return True
    if _starts_with_section_number(following):
        return True
    if _is_standalone_sub_header_line(current) or _is_standalone_sub_header_line(following):
        return True
    return _following_has_short_label_colon(following)


def _on_same_baseline(current: TextLine, following: TextLine) -> bool:
    """True when two lines share a baseline within ``SAME_BASELINE_TOLERANCE``."""
    return abs(current.bbox.y0 - following.bbox.y0) <= SAME_BASELINE_TOLERANCE


def _ends_with_sentence_period(text: str) -> bool:
    """True when ``text`` ends with ``.`` that is not a lone section number."""
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return False
    return SECTION_NUMBER_RE.match(stripped) is None


def _following_has_short_label_colon(line: TextLine) -> bool:
    """Keep a newline when the next line looks like a short label ending in ``:``.

    The colon must sit on characters 2-20 of the following line, and the text
    before it must not contain ``.`` or ``,``.
    """

    text = (line.text or "").lstrip()
    colon = text.find(":")
    if colon + 1 < LABEL_COLON_MIN or colon + 1 > LABEL_COLON_MAX:
        return False
    prefix = text[:colon]
    return "." not in prefix and "," not in prefix


def _starts_with_section_number(line: TextLine) -> bool:
    """True when the line starts with a dotted section number (``6.1.1``)."""

    return bool(SECTION_HEADING_RE.match(line.text or ""))


def _starts_with_uppercase(line: TextLine) -> bool:
    """True when the first non-space character of ``line`` is uppercase."""

    stripped = (line.text or "").lstrip()
    return bool(stripped) and stripped[0].isupper()


def _ends_with_hyphen(text: str) -> bool:
    """True when ``text`` ends with ``-`` after trailing space is stripped."""

    stripped = (text or "").rstrip()
    return bool(stripped) and stripped[-1] == "-"


def _is_wrapped_continuation(
    previous_line: TextLine,
    previous_text: str,
    following_text: str,
) -> bool:
    """True when ``following_text`` is a wrap of ``previous_text``, not a new heading sentence."""

    if following_text[0].islower():
        return True
    if _ends_with_hyphen(previous_text):
        return True
    return not _starts_with_section_number(previous_line)


def _starts_with_bullet_point(line: TextLine) -> bool:
    """True when the line is a symbol, numeric, or letter list item."""
    if line.is_bullet or line.bullet:
        return True

    text = line.text or ""
    if BULLET_RE.match(text):
        return True

    stripped = text.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    if first in BULLET_CHARS or first == "•":
        return True
    return 0xE000 <= ord(first) <= 0xF8FF


def _join_wrapped_lines(current: TextLine, following: TextLine) -> TextLine:
    """Append ``following`` onto ``current``, unioning boxes and spans.

    Insert a space unless ``current`` already ends with ``-`` (document-id wrap).
    """
    extra_text = (following.text or "").strip()
    extra_spans = list(following.spans) if following.spans else []
    if extra_text and not extra_spans:
        extra_spans = [TextSpan(text=extra_text)]

    current_text = (current.text or "").rstrip()
    if not extra_text:
        return current
    if not current_text:
        current.text = extra_text
        current.spans = extra_spans
    else:
        separator = "" if _ends_with_hyphen(current_text) else " "
        current.text = f"{current_text}{separator}{extra_text}"
        spans = list(current.spans) if current.spans else [TextSpan(text=current_text)]
        if separator and spans and not spans[-1].text.endswith(" "):
            spans[-1].text += " "
        spans.extend(extra_spans)
        current.spans = TextSpan.merge_adjacent(spans)

    current.bbox = BBox(
        min(current.bbox.x0, following.bbox.x0),
        min(current.bbox.y0, following.bbox.y0),
        max(current.bbox.x1, following.bbox.x1),
        max(current.bbox.y1, following.bbox.y1),
    )
    return current


def is_ignored_text_block(block: TextBlock) -> bool:
    """Return True when a boilerplate text block should be dropped.

    Edit the exact phrases, contains phrases, markers, and patterns in this
    function when templates change. Matching is case-insensitive and ignores
    extra whitespace and ``*``.
    """

    text = _ignore_normalized(block.text)
    if not text:
        return True

    exact_phrases = (
        "standard operating procedure",
    )
    if text in exact_phrases:
        return True

    contains_phrases = (
        "confidential work product",
        "reference copy",
        "this document has been electronically signed",
    )
    if any(phrase in text for phrase in contains_phrases):
        return True

    markers = (
        "exported by",
        "exported date",
        "export by",
        "export date",
    )
    if any(
        text == marker
        or text.startswith(marker + " ")
        or text.endswith(" " + marker)
        for marker in markers
    ):
        return True

    if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", text):
        return True

    return False


def _ignore_normalized(text: str) -> str:
    """Lowercase, collapse whitespace, and drop ``*`` for ignore matching."""
    cleaned = text.replace("*", " ")
    return " ".join(cleaned.lower().split())


def _parse_line(line: dict) -> tuple[dict, str | None, str, BBox, list[TextSpan]] | None:
    """Parse a PyMuPDF line into bullet, body text, bbox, and styled spans."""
    spans = line.get("spans") or []
    if not spans:
        return None

    styled: list[TextSpan] = []
    bullet: str | None = None

    for index, span in enumerate(spans):
        raw = span.get("text") or ""
        if not raw:
            continue

        if index == 0 and bullet is None:
            detected = _span_bullet(raw)
            if detected is not None:
                bullet, remainder = detected
                if remainder:
                    styled.append(_span_style(span, remainder))
                continue

        styled.append(_span_style(span, raw))

    styled = TextSpan.merge_adjacent(styled)
    text = "".join(span.text for span in styled).strip()
    if bullet:
        text = text.lstrip()
    else:
        match = BULLET_RE.match(text)
        if match:
            bullet = _canonical_bullet_marker(match.group("bullet"))
            text = text[match.end():].lstrip()
            styled = _strip_prefix_from_spans(styled, match.group(0))

    if not text and not bullet:
        return None

    return line, bullet, text, BBox(*line["bbox"]), styled


def _span_style(span: dict, text: str) -> TextSpan:
    """Build a ``TextSpan`` with canonical glyphs and bold/italic/underline flags."""
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
    return TextSpan(text=canonical_text(text), bold=bold, italic=italic, underline=underline)


def _strip_prefix_from_spans(spans: list[TextSpan], prefix: str) -> list[TextSpan]:
    """Remove a leading bullet/prefix string from the start of ``spans``."""
    remaining = prefix
    result: list[TextSpan] = []
    for span in spans:
        if remaining and span.text.startswith(remaining):
            leftover = span.text[len(remaining):]
            remaining = ""
            if leftover:
                result.append(TextSpan(leftover, span.bold, span.italic, span.underline))
            continue
        if remaining and remaining.startswith(span.text):
            remaining = remaining[len(span.text):]
            continue
        result.append(span)
    return TextSpan.merge_adjacent(result)


def _normalize_list_layout(blocks: list[TextBlock]) -> list[TextBlock]:
    """Join hanging bullets with the text to their right and set list indent levels."""

    consumed: set[tuple[int, int]] = set()
    for block_index, block in enumerate(blocks):
        for line in block.lines:
            if not line.is_bullet or line.text:
                continue
            match = _nearest_right_text(blocks, block_index, line, consumed)
            if match is None:
                continue
            other_block, other_line_index, other = match
            line.text = other.text
            line.spans = list(other.spans)
            line.bbox = BBox(
                line.bbox.x0,
                min(line.bbox.y0, other.bbox.y0),
                other.bbox.x1,
                max(line.bbox.y1, other.bbox.y1),
            )
            consumed.add((other_block, other_line_index))

    merged: list[TextBlock] = []
    for block_index, block in enumerate(blocks):
        lines = [
            line
            for line_index, line in enumerate(block.lines)
            if (block_index, line_index) not in consumed
        ]
        if not lines:
            continue
        for line_number, line in enumerate(lines, start=1):
            line.line_number = line_number
        block.lines = lines
        block.indent = min(line.indent for line in lines)
        block.bbox = BBox(
            min(line.bbox.x0 for line in lines),
            min(line.bbox.y0 for line in lines),
            max(line.bbox.x1 for line in lines),
            max(line.bbox.y1 for line in lines),
        )
        merged.append(block)

    _apply_indent_levels(merged)
    return merged


def _nearest_right_text(
    blocks: list[TextBlock],
    block_index: int,
    bullet: TextLine,
    consumed: set[tuple[int, int]],
) -> tuple[int, int, TextLine] | None:
    """Find the closest text line sitting to the right of a hanging bullet."""
    bullet_mid = (bullet.bbox.y0 + bullet.bbox.y1) / 2
    best: tuple[int, int, TextLine] | None = None
    best_dx = None

    for other_index, block in enumerate(blocks):
        for line_index, line in enumerate(block.lines):
            if (other_index, line_index) in consumed:
                continue
            if other_index == block_index and line is bullet:
                continue
            if not line.text or line.bbox.x0 <= bullet.bbox.x1 - 1:
                continue
            other_mid = (line.bbox.y0 + line.bbox.y1) / 2
            if abs(other_mid - bullet_mid) > 6 and abs(line.bbox.y0 - bullet.bbox.y0) > 6:
                continue
            dx = line.bbox.x0 - bullet.bbox.x1
            if dx < -1 or dx > 90:
                continue
            if best_dx is None or dx < best_dx:
                best = (other_index, line_index, line)
                best_dx = dx

    return best


def _apply_indent_levels(blocks: list[TextBlock]) -> None:
    """Set ``indent_level`` from bullet stops or relative left edge."""
    bullet_indents = sorted(
        {
            round(line.indent / INDENT_STEP) * INDENT_STEP
            for block in blocks
            for line in block.lines
            if line.is_bullet
        }
    )
    for block in blocks:
        block_left = min((line.indent for line in block.lines), default=block.indent)
        for line in block.lines:
            if line.is_bullet and bullet_indents:
                nearest = min(
                    bullet_indents,
                    key=lambda stop, current=line.indent: abs(stop - current),
                )
                line.indent_level = bullet_indents.index(nearest)
            else:
                line.indent_level = int(round(max(0.0, line.indent - block_left) / INDENT_STEP))


def _span_bullet(raw: str) -> tuple[str, str] | None:
    """Split ``raw`` into ``(marker, remainder)`` when it starts with a bullet."""
    stripped = raw.strip()

    if stripped and _is_bullet_token(stripped):
        remainder = raw[raw.find(stripped) + len(stripped):]
        return _canonical_bullet_marker(stripped), remainder

    match = BULLET_RE.match(raw)
    if match:
        return _canonical_bullet_marker(match.group("bullet")), raw[match.end():]

    return None


def _canonical_bullet_marker(token: str) -> str:
    """Map Symbol/PUA dingbat bullets (e.g. U+F0B7) to a real ``•``.

    Ticks and other non-bullet dingbats keep their Unicode stand-in
    (``U+F0FC`` -> ``✓``). Numeric/letter markers such as ``1.`` or
    ``a)`` are left unchanged.
    """

    mapped = canonical_text(token)
    if is_list_bullet_marker(token) or is_list_bullet_marker(mapped):
        return STANDARD_BULLET
    if token in BULLET_CHARS:
        return STANDARD_BULLET
    return mapped


def _is_bullet_token(text: str) -> bool:
    """True when ``text`` is a list bullet glyph (including mapped PUA)."""
    if is_list_bullet_marker(text):
        return True
    if len(text) != 1:
        return False
    if text in BULLET_CHARS:
        return True
    mapped = canonical_symbol(text)
    if mapped != text and is_list_bullet_marker(mapped):
        return True
    if 0xE000 <= ord(text) <= 0xF8FF:
        return is_list_bullet_marker(mapped)
    return False


def _bbox_from_text_lines(lines: Sequence[TextLine]) -> BBox | None:
    """Union of every line box in ``lines``."""
    bbox: BBox | None = None
    for line in lines:
        bbox = BBox.combine(bbox, line.bbox)
    return bbox


def _lines_outside_rects(
    lines: Sequence[dict],
    rects: Sequence[BBox] | None,
) -> list[dict]:
    """Keep PyMuPDF lines whose boxes are not mostly inside ``rects``."""
    if not rects:
        return list(lines)
    kept: list[dict] = []
    for line in lines:
        raw_bbox = line.get("bbox")
        if not raw_bbox:
            kept.append(line)
            continue
        bbox = BBox(*raw_bbox)
        if bbox.is_empty or not _is_inside_any(bbox, rects):
            kept.append(line)
    return kept


def _is_inside_any(bbox: BBox, rects: Sequence[BBox] | None) -> bool:
    """True when ``bbox`` is covered by any exclude rect at the inside threshold."""
    if not rects:
        return False
    return any(bbox.overlap_ratio(rect) >= INSIDE_RECT_THRESHOLD for rect in rects)
