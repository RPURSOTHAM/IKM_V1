"""Split a text block when an icon sits in the middle of a sentence.

Help
----
SOP steps sometimes put a button/icon in the middle of a line
(``Press the OPEN button [icon] to open…``). Extraction already keeps
the wording as one TEXT block (the hole is smaller than the same-line
gap). Reconstruction needs that sentence **broken** at the icon:

    TEXT (left) → IMAGE → TEXT (right)

``split_text_around_inline_images`` does that for **icon-sized**
pictures only. Large figures stay standalone IMAGE items and the
surrounding text is not split. The icon itself stays in the image
list; page ordering by bbox then yields left-text, icon, right-text.

Match when all of these hold:

- the image has no callout ``labels`` (those stay on the photo)
- image height ≤ ``INLINE_IMAGE_MAX_HEIGHT`` and width ≤
  ``INLINE_IMAGE_MAX_WIDTH``
- at least ``INLINE_INSIDE_RATIO`` of the image box is covered by a
  ``TextLine`` box
- the image is not much taller than that line
- the split point is inside the line (text on both sides)

What this does not change
-------------------------
Same-line / wrap / hyphen text merges, side-callout absorb, table
cells, infographic clips, filtering, or how headings open sections.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..document import BBox, ImageRef, TextBlock, TextLine, TextSpan

INLINE_IMAGE_MAX_HEIGHT = 36.0
INLINE_IMAGE_MAX_WIDTH = 72.0
INLINE_INSIDE_RATIO = 0.80
INLINE_HEIGHT_FACTOR = 1.5
INLINE_HEIGHT_PAD = 4.0
INLINE_CENTER_Y_SLACK = 2.0


@dataclass(frozen=True)
class _Split:
    """One icon cut on a text line, in original-line character coordinates."""

    image: ImageRef
    line_index: int
    char_offset: int


def split_text_around_inline_images(
    images: list[ImageRef],
    text_blocks: list[TextBlock],
) -> list[TextBlock]:
    """Break text blocks that have an icon in the middle of a line.

    ``images`` is not modified. Returned blocks replace ``text_blocks``;
    fragments keep bboxes that sort around the icon in reading order.
    """

    if not images or not text_blocks:
        return text_blocks

    splits_by_block: dict[int, list[_Split]] = {}
    for image in images:
        host = _best_host_line(image, text_blocks)
        if host is None:
            continue
        block, line_index, line = host
        offset = _char_offset(line, image)
        text = line.text or ""
        if offset <= 0 or offset >= len(text):
            continue
        splits_by_block.setdefault(id(block), []).append(
            _Split(image=image, line_index=line_index, char_offset=offset)
        )

    if not splits_by_block:
        return text_blocks

    result: list[TextBlock] = []
    for block in text_blocks:
        splits = splits_by_block.get(id(block))
        if not splits:
            result.append(block)
            continue
        result.extend(_split_block(block, splits))
    return result


def _best_host_line(
    image: ImageRef,
    text_blocks: list[TextBlock],
) -> tuple[TextBlock, int, TextLine] | None:
    """Tightest text line that contains an icon-sized ``image``, or ``None``."""

    if image.labels:
        return None
    if image.bbox.is_empty:
        return None
    if image.bbox.height > INLINE_IMAGE_MAX_HEIGHT:
        return None
    if image.bbox.width > INLINE_IMAGE_MAX_WIDTH:
        return None

    best_block: TextBlock | None = None
    best_line: TextLine | None = None
    best_index = 0
    best_area = 0.0
    best_overlap = -1.0
    for block in text_blocks:
        for line_index, line in enumerate(block.lines):
            if not _image_sits_in_line(image, line):
                continue
            area = line.bbox.area
            overlap = image.bbox.overlap_ratio(line.bbox)
            tighter = best_line is None or area < best_area or (
                area == best_area and overlap > best_overlap
            )
            if not tighter:
                continue
            best_block = block
            best_line = line
            best_index = line_index
            best_area = area
            best_overlap = overlap
    if best_block is None or best_line is None:
        return None
    return best_block, best_index, best_line


def _image_sits_in_line(image: ImageRef, line: TextLine) -> bool:
    """True when ``image`` is a line-sized icon inside ``line``."""

    box = line.bbox
    if box.is_empty:
        return False
    if image.bbox.overlap_ratio(box) < INLINE_INSIDE_RATIO:
        return False
    center_y = (image.bbox.y0 + image.bbox.y1) / 2.0
    if center_y < box.y0 - INLINE_CENTER_Y_SLACK:
        return False
    if center_y > box.y1 + INLINE_CENTER_Y_SLACK:
        return False
    max_height = box.height * INLINE_HEIGHT_FACTOR + INLINE_HEIGHT_PAD
    return image.bbox.height <= max_height


def _char_offset(line: TextLine, image: ImageRef) -> int:
    """Character index on ``line`` at the icon’s left edge."""

    text = line.text or ""
    width = line.bbox.width
    if not text or width <= 0:
        return 0
    ratio = (image.bbox.x0 - line.bbox.x0) / width
    ratio = max(0.0, min(1.0, ratio))
    return round(ratio * len(text))


def _split_block(block: TextBlock, splits: list[_Split]) -> list[TextBlock]:
    """Yield left/right text fragments around each icon on ``block``."""

    ordered = sorted(
        splits,
        key=lambda item: (item.line_index, item.char_offset, item.image.bbox.x0),
    )
    by_line: dict[int, list[_Split]] = {}
    for split in ordered:
        by_line.setdefault(split.line_index, []).append(split)

    fragments: list[TextBlock] = []
    current: list[TextLine] = []

    def flush() -> None:
        """Emit ``current`` as a text block and start a new fragment."""

        nonlocal current
        if not current:
            return
        fragments.append(_block_from_lines(block, current))
        current = []

    for line_index, line in enumerate(block.lines):
        line_splits = by_line.get(line_index)
        if not line_splits:
            current.append(line)
            continue
        cursor = 0
        left_x = line.bbox.x0
        for split in line_splits:
            left = _slice_line(
                line,
                start=cursor,
                end=split.char_offset,
                x0=left_x,
                x1=split.image.bbox.x0,
            )
            if left is not None:
                current.append(left)
            flush()
            cursor = split.char_offset
            left_x = split.image.bbox.x1
        right = _slice_line(
            line,
            start=cursor,
            end=len(line.text or ""),
            x0=left_x,
            x1=line.bbox.x1,
        )
        if right is not None:
            current.append(right)
    flush()
    return fragments or [block]


def _slice_line(
    line: TextLine,
    start: int,
    end: int,
    x0: float,
    x1: float,
) -> TextLine | None:
    """Copy ``line[start:end]`` with a bbox from ``x0`` to ``x1``."""

    text = (line.text or "")[start:end]
    if not text.strip():
        return None
    box = BBox(
        min(x0, x1),
        line.bbox.y0,
        max(x0, x1),
        line.bbox.y1,
    )
    if box.width <= 0:
        box = BBox(line.bbox.x0, line.bbox.y0, line.bbox.x1, line.bbox.y1)
    indent = round(box.x0, 2)
    relative = max(0.0, indent - line.indent) if line.indent else 0.0
    indent_level = (
        line.indent_level + int(round(relative / 18.0)) if relative else line.indent_level
    )
    return TextLine(
        line_number=line.line_number,
        text=text,
        indent=indent,
        indent_level=indent_level,
        is_bullet=line.is_bullet and start == 0,
        bullet=line.bullet if start == 0 else None,
        bbox=box,
        spans=_slice_spans(line.spans, start, end),
    )


def _slice_spans(spans: list[TextSpan], start: int, end: int) -> list[TextSpan]:
    """Keep the portion of ``spans`` covering characters ``[start, end)``."""

    if not spans or end <= start:
        return []
    sliced: list[TextSpan] = []
    pos = 0
    for span in spans:
        chunk = span.text or ""
        span_end = pos + len(chunk)
        if span_end <= start or pos >= end:
            pos = span_end
            continue
        cut_start = max(0, start - pos)
        cut_end = min(len(chunk), end - pos)
        piece = chunk[cut_start:cut_end]
        if piece:
            sliced.append(
                TextSpan(
                    text=piece,
                    bold=span.bold,
                    italic=span.italic,
                    underline=span.underline,
                )
            )
        pos = span_end
    return sliced


def _block_from_lines(original: TextBlock, lines: list[TextLine]) -> TextBlock:
    """Build a fragment that keeps ``original.order`` and the union bbox."""

    numbered = []
    for index, line in enumerate(lines, start=1):
        numbered.append(
            TextLine(
                line_number=index,
                text=line.text,
                indent=line.indent,
                indent_level=line.indent_level,
                is_bullet=line.is_bullet,
                bullet=line.bullet,
                bbox=line.bbox,
                spans=list(line.spans),
            )
        )
    bbox = BBox(
        min(line.bbox.x0 for line in numbered),
        min(line.bbox.y0 for line in numbered),
        max(line.bbox.x1 for line in numbered),
        max(line.bbox.y1 for line in numbered),
    )
    return TextBlock(
        order=original.order,
        lines=numbered,
        indent=min((line.indent for line in numbered), default=original.indent),
        bbox=bbox,
    )
