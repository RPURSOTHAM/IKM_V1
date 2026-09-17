"""Attach callout labels that sit beside a picture so they are part of the image.

Help
----
SOP photos often have short labels to the left or right, joined by arrows
(``Emergency Switch``, ``Control Switch``). PyMuPDF saves only the raster
placement, so those words become a separate TEXT block.

``attach_nearby_text_to_images`` folds a text block into the nearest image
when all of these hold:

- the text box overlaps the image vertically (beside the photo, not a
  caption above or below)
- the horizontal gap is at most ``IMAGE_TEXT_MAX_GAP`` (or the boxes
  already overlap in x)
- the text block is short (height ≤ ``IMAGE_TEXT_MAX_HEIGHT``) so a
  procedure paragraph in a second column is not absorbed
- the text does not start with a dotted section number (``6.2.2 …``)

Matching labels are removed from the text list. The image clip is
re-rasterized to the union box so the words appear in the PNG, and
``ImageRef.labels`` records the wording.

What this does not change
-------------------------
Same-line / wrap / orphan-number text merges, table extraction (tables
already ran), infographic detection, grouping, or letterhead filters.
Captions under a figure stay text (no vertical overlap). Headings such
as ``6.2.2 The screen as shown…`` stay text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from ..document import BBox, ImageRef, TextBlock
from .extract_image import MIN_DISPLAY_SIZE, _save_displayed_image

IMAGE_TEXT_MAX_GAP = 36.0
IMAGE_TEXT_MAX_HEIGHT = 56.0
IMAGE_TEXT_CLIP_PAD = 4.0
SECTION_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?(?:\s+|$|(?=[A-Z]))")


def attach_nearby_text_to_images(
    page: pymupdf.Page,
    images: list[ImageRef],
    text_blocks: list[TextBlock],
    page_number: int,
    output_dir: str | Path,
) -> list[TextBlock]:
    """Fold close callout labels into ``images`` and return the leftover text.

    ``images`` is updated in place (bbox, pixels, ``labels``). Text blocks
    that were attached are omitted from the returned list.
    """

    if not images or not text_blocks:
        return text_blocks

    output_dir = Path(output_dir)
    attached: set[int] = set()
    for image in images:
        partners = _adjacent_label_blocks(image, text_blocks, attached)
        if not partners:
            continue
        _absorb_labels(page, image, partners, page_number, output_dir)
        attached.update(id(block) for block in partners)

    if not attached:
        return text_blocks
    return [block for block in text_blocks if id(block) not in attached]


def _adjacent_label_blocks(
    image: ImageRef,
    text_blocks: list[TextBlock],
    already_attached: set[int],
) -> list[TextBlock]:
    """Text blocks that look like callouts for ``image``, nearest first."""

    matches: list[tuple[float, TextBlock]] = []
    for block in text_blocks:
        if id(block) in already_attached:
            continue
        if not _is_image_callout(image.bbox, block):
            continue
        gap = _horizontal_gap(image.bbox, block.bbox)
        matches.append((gap, block))
    matches.sort(key=lambda item: item[0])
    return [block for _gap, block in matches]


def _is_image_callout(image_box: BBox, block: TextBlock) -> bool:
    """True when ``block`` is a short label sitting beside ``image_box``."""

    if block.bbox is None or image_box.is_empty:
        return False
    text = (block.text or "").strip()
    if not text:
        return False
    if SECTION_HEADING_RE.match(text):
        return False
    if block.bbox.height > IMAGE_TEXT_MAX_HEIGHT:
        return False
    if _vertical_overlap(image_box, block.bbox) <= 0:
        return False
    return _horizontal_gap(image_box, block.bbox) <= IMAGE_TEXT_MAX_GAP


def _absorb_labels(
    page: pymupdf.Page,
    image: ImageRef,
    partners: list[TextBlock],
    page_number: int,
    output_dir: Path,
) -> None:
    """Union callout boxes into ``image``, re-clip, and record label wording."""

    boxes = [image.bbox, *[block.bbox for block in partners if block.bbox is not None]]
    union = _padded_union(boxes, page)
    if union is None:
        return
    saved = _save_displayed_image(
        page=page,
        bbox=union,
        page_number=page_number,
        xref=image.xref,
        index=_next_clip_index(image),
        output_dir=output_dir,
    )
    if saved is None:
        return
    path, width, height, ext = saved
    image.path = path
    image.width = width
    image.height = height
    image.bbox = union
    image.ext = ext
    image.labels = [text for text in ((block.text or "").strip() for block in partners) if text]


def _padded_union(boxes: list[BBox], page: pymupdf.Page) -> BBox | None:
    """Union of ``boxes``, padded and clipped to the page."""

    if not boxes:
        return None
    union = BBox(
        min(box.x0 for box in boxes) - IMAGE_TEXT_CLIP_PAD,
        min(box.y0 for box in boxes) - IMAGE_TEXT_CLIP_PAD,
        max(box.x1 for box in boxes) + IMAGE_TEXT_CLIP_PAD,
        max(box.y1 for box in boxes) + IMAGE_TEXT_CLIP_PAD,
    )
    page_box = BBox.from_rect(page.rect)
    clipped = union.intersection(page_box)
    if clipped.is_empty or clipped.width < MIN_DISPLAY_SIZE or clipped.height < MIN_DISPLAY_SIZE:
        return None
    return clipped


def _next_clip_index(image: ImageRef) -> int:
    """Index suffix for a re-clipped file so the original PNG is kept."""

    stem = Path(image.path).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1]) + 1000
    return 1000


def _horizontal_gap(left: BBox, right: BBox) -> float:
    """Gap between the boxes on the x-axis; 0 when they overlap horizontally."""

    if right.x0 >= left.x1:
        return right.x0 - left.x1
    if left.x0 >= right.x1:
        return left.x0 - right.x1
    return 0.0


def _vertical_overlap(left: BBox, right: BBox) -> float:
    """Height of the shared vertical span; 0 when the boxes miss in y."""

    return max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
