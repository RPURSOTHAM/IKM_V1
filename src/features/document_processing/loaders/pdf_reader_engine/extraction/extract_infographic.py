"""Detect vector infographics (flowcharts, box-and-arrow drawings) and clip them.

A cluster of stacked filled/stroked rectangles joined by short connectors is
rasterized as one image, including the text that sits inside the boxes.
Existing text, table, and embedded-image extractors are left unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pymupdf

from ..document import BBox, ImageRef
from .extract_image import MIN_DISPLAY_SIZE, rasterize_page_clip

# Synthetic xref: this PNG is a page clip, not an embedded image object.
INFOGRAPHIC_XREF = 0

MIN_STEP_BOXES = 3
MIN_CONNECTORS = 2
STEP_MIN_WIDTH = 80.0
STEP_MIN_HEIGHT = 18.0
STEP_MAX_HEIGHT = 120.0
CONNECTOR_MAX_SPAN = 80.0
CONNECTOR_MIN_SIDE = 40.0
CLUSTER_GAP = 80.0
BOX_OVERLAP_FRACTION = 0.40
CLIP_PAD = 4.0
STAMP_X0 = 560.0
TABLE_OVERLAP = 0.50
EXISTING_IMAGE_COVER = 0.70


def extract_page_infographics(
    page: pymupdf.Page,
    page_number: int,
    output_dir: str | Path,
    exclude_rects: Sequence[BBox] | None = None,
    page_images: Sequence[ImageRef] | None = None,
) -> list[ImageRef]:
    """Find leftover drawing clusters and save each as one PNG clip.

    Ignores drawings inside table regions and the right-side export stamp.
    A cluster becomes an infographic when it has several filled/stroked
    step boxes joined by short arrows. Raster charts already extracted as
    images are skipped. Heading text above or below the boxes stays text.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude_rects = list(exclude_rects or [])
    page_images = list(page_images or [])
    page_box = BBox.from_rect(page.rect)

    step_boxes: list[BBox] = []
    connectors: list[BBox] = []
    for drawing in page.get_drawings():
        box = _drawing_bbox(drawing)
        if box is None or not _on_page(box, page_box):
            continue
        if box.x0 >= STAMP_X0:
            continue
        if _inside_excluded(box, exclude_rects):
            continue
        if _is_step_box(drawing, box):
            step_boxes.append(box)
        elif _is_connector(drawing, box):
            connectors.append(box)

    references: list[ImageRef] = []
    for cluster in _cluster_boxes(step_boxes):
        if len(cluster) < MIN_STEP_BOXES:
            continue
        union = _union(cluster)
        if union is None:
            continue
        nearby = [box for box in connectors if _near_union(box, union, CLUSTER_GAP)]
        if len(nearby) < MIN_CONNECTORS:
            continue
        clip = _padded_union([union, *nearby], page_box)
        if clip is None:
            continue
        if _covered_by_existing_image(clip, page_images):
            continue
        saved = _save_infographic_clip(
            page=page,
            bbox=clip,
            page_number=page_number,
            index=len(references),
            output_dir=output_dir,
        )
        if saved is None:
            continue
        path, width, height, ext = saved
        references.append(
            ImageRef(
                path=path,
                xref=INFOGRAPHIC_XREF,
                width=width,
                height=height,
                bbox=clip,
                ext=ext,
            )
        )

    references.sort(key=lambda item: (item.bbox.y0, item.bbox.x0))
    return references


def _drawing_bbox(drawing: dict) -> BBox | None:
    """Return the drawing's rectangle, or ``None`` if missing or empty."""
    rect = drawing.get("rect")
    if rect is None:
        return None
    box = BBox.from_rect(rect)
    if box.is_empty:
        return None
    return box


def _on_page(box: BBox, page_box: BBox) -> bool:
    """True when ``box`` intersects the visible page (not an off-page frame)."""
    if box.y0 >= page_box.y1 - 1.0 or box.y1 <= page_box.y0 + 1.0:
        return False
    return box.intersection_area(page_box) > 0.0


def _inside_excluded(box: BBox, rects: Sequence[BBox]) -> bool:
    """True when ``box`` sits mostly inside a table (or other exclude) rect."""
    return any(box.overlap_ratio(rect) >= TABLE_OVERLAP for rect in rects)


def _is_step_box(drawing: dict, box: BBox) -> bool:
    """Filled and stroked rectangle that can be a flowchart step."""

    kind = str(drawing.get("type") or "")
    if "s" not in kind or "f" not in kind:
        return False
    items = drawing.get("items") or []
    if not any(item and item[0] == "re" for item in items):
        return False
    if box.width < STEP_MIN_WIDTH:
        return False
    if box.height < STEP_MIN_HEIGHT or box.height > STEP_MAX_HEIGHT:
        return False
    return True


def _is_connector(drawing: dict, box: BBox) -> bool:
    """Short arrow or line between step boxes, not a table rule or stamp."""

    items = drawing.get("items") or []
    if not any(item and item[0] == "l" for item in items):
        return False
    if max(box.width, box.height) >= CONNECTOR_MAX_SPAN:
        return False
    if min(box.width, box.height) >= CONNECTOR_MIN_SIDE:
        return False
    return True


def _boxes_linked(left: BBox, right: BBox) -> bool:
    """True when two step boxes are stacked or side-by-side within ``CLUSTER_GAP``."""
    x_overlap = min(left.x1, right.x1) - max(left.x0, right.x0)
    y_overlap = min(left.y1, right.y1) - max(left.y0, right.y0)
    x_gap = max(0.0, max(left.x0, right.x0) - min(left.x1, right.x1))
    y_gap = max(0.0, max(left.y0, right.y0) - min(left.y1, right.y1))
    min_width = min(left.width, right.width)
    min_height = min(left.height, right.height)

    stacked = (
        x_overlap >= min_width * BOX_OVERLAP_FRACTION and y_gap <= CLUSTER_GAP
    )
    side_by_side = (
        y_overlap >= min_height * BOX_OVERLAP_FRACTION and x_gap <= CLUSTER_GAP
    )
    return stacked or side_by_side


def _cluster_boxes(boxes: Sequence[BBox]) -> list[list[BBox]]:
    """Group nearby step boxes with union-find; one cluster per infographic."""
    if not boxes:
        return []

    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        """Return the union-find root for ``index``."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        """Merge the clusters that contain ``left`` and ``right``."""
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if _boxes_linked(left, boxes[j]):
                union(i, j)

    groups: dict[int, list[BBox]] = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)
    return list(groups.values())


def _union(boxes: Sequence[BBox]) -> BBox | None:
    """Smallest box that contains every box in ``boxes``."""
    return BBox.union_all(boxes)


def _padded_union(boxes: Sequence[BBox], page_box: BBox) -> BBox | None:
    """Union of ``boxes``, padded for strokes, clipped to the page."""
    union = _union(boxes)
    if union is None:
        return None
    # Extra bottom pad would swallow a heading that sits just under the last box.
    padded = BBox(
        union.x0 - CLIP_PAD,
        union.y0 - CLIP_PAD,
        union.x1 + CLIP_PAD,
        union.y1 + 1.0,
    )
    clipped = padded.intersection(page_box)
    if clipped.is_empty:
        return None
    if clipped.width < MIN_DISPLAY_SIZE or clipped.height < MIN_DISPLAY_SIZE:
        return None
    return clipped


def _near_union(box: BBox, region: BBox, gap: float) -> bool:
    """True when ``box`` intersects ``region`` expanded by ``gap``."""
    expanded = BBox(
        region.x0 - gap,
        region.y0 - gap,
        region.x1 + gap,
        region.y1 + gap,
    )
    return box.intersection_area(expanded) > 0.0


def _covered_by_existing_image(clip: BBox, images: Sequence[ImageRef]) -> bool:
    """True when an embedded image already covers most of this clip."""
    return any(clip.overlap_ratio(image.bbox) >= EXISTING_IMAGE_COVER for image in images)


def _save_infographic_clip(
    page: pymupdf.Page,
    bbox: BBox,
    page_number: int,
    index: int,
    output_dir: Path,
) -> tuple[str, int, int, str] | None:
    """Rasterize ``bbox`` at ``DISPLAY_SCALE`` and save ``page_N_infographic_I.png``."""
    filename = f"page_{page_number}_infographic_{index}.png"
    path = output_dir / filename
    size = rasterize_page_clip(page, bbox, path)
    if size is None:
        return None
    width, height = size
    return str(path), width, height, "png"
