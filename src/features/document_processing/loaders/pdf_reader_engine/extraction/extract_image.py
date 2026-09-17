"""Extract images as they appear on the PDF page.

Each placement is saved at the document's displayed size and orientation.
The document model keeps only a file reference.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ..document import BBox, ImageRef
from ..errors import RECOVERABLE
from ..failures import log_failure

MIN_DISPLAY_SIZE = 4.0
LOGO_Y0_MAX = 20.0
# 2 pixels per PDF point keeps page orientation/aspect and readable detail.
DISPLAY_SCALE = 2.0
IMAGE_ASPECT_TOLERANCE = 0.12
IMAGE_SIZE_TOLERANCE = 0.25
IMAGE_DUPLICATE_OVERLAP = 0.90


def extract_page_images(
    page: pymupdf.Page,
    page_number: int,
    output_dir: str | Path,
    saved_xrefs: dict | None = None,
) -> list[ImageRef]:
    """Save each image placement as shown on the page, in reading order."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if saved_xrefs is None:
        saved_xrefs = {}

    references: list[ImageRef] = []
    seen_locations: set[tuple[int, float, float, float, float]] = set()
    placement_index = 0

    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref <= 0:
            continue

        try:
            placements = page.get_image_rects(xref, transform=True)
        except RECOVERABLE as exc:
            log_failure("extract", f"page {page_number} image xref {xref} rects", exc)
            continue

        for placement in placements:
            rect, _matrix = _split_placement(placement)
            bbox = BBox.from_rect(rect)
            if bbox.y0 < LOGO_Y0_MAX:
                continue
            if bbox.width < MIN_DISPLAY_SIZE or bbox.height < MIN_DISPLAY_SIZE:
                continue

            location = (
                xref,
                round(bbox.x0, 2),
                round(bbox.y0, 2),
                round(bbox.x1, 2),
                round(bbox.y1, 2),
            )
            if location in seen_locations:
                continue
            seen_locations.add(location)

            saved = saved_xrefs.get(location)
            if saved is None:
                saved = _save_displayed_image(
                    page=page,
                    bbox=bbox,
                    page_number=page_number,
                    xref=xref,
                    index=placement_index,
                    output_dir=output_dir,
                )
                placement_index += 1
                if saved is None:
                    continue
                saved_xrefs[location] = saved

            path, width, height, ext = saved
            references.append(
                ImageRef(
                    path=path,
                    xref=xref,
                    width=width,
                    height=height,
                    bbox=bbox,
                    ext=ext,
                )
            )

    references.sort(key=lambda item: (item.bbox.y0, item.bbox.x0))
    return validate_images(references)


def validate_images(images: list[ImageRef]) -> list[ImageRef]:
    """Keep images that match the page placement and are not the same picture twice.

    Current rules (add more here later):
    - saved width/height must be usable
    - landscape vs portrait must match the page box
    - aspect ratio must match the page box
    - pixel size must match the displayed box at DISPLAY_SCALE
    - two xrefs on almost the same box (image + soft-mask companion) are
      one picture: if each box covers the other by at least
      ``IMAGE_DUPLICATE_OVERLAP`` (90%), keep the larger box only
    """

    sized = [image for image in images if _image_size_and_orientation_match(image)]
    return _drop_overlapping_duplicates(sized)


def _drop_overlapping_duplicates(images: list[ImageRef]) -> list[ImageRef]:
    """Keep the larger box when two placements cover each other by 90%."""
    ordered = sorted(images, key=lambda image: image.bbox.area, reverse=True)
    kept: list[ImageRef] = []
    for image in ordered:
        if any(_same_displayed_placement(image.bbox, other.bbox) for other in kept):
            continue
        kept.append(image)
    kept.sort(key=lambda item: (item.bbox.y0, item.bbox.x0))
    return kept


def _same_displayed_placement(left: BBox, right: BBox) -> bool:
    """True when each box covers the other by ``IMAGE_DUPLICATE_OVERLAP``."""
    if left.is_empty or right.is_empty:
        return False
    return (
        left.overlap_ratio(right) >= IMAGE_DUPLICATE_OVERLAP
        and right.overlap_ratio(left) >= IMAGE_DUPLICATE_OVERLAP
    )


def _image_size_and_orientation_match(image: ImageRef) -> bool:
    """True when saved pixels match the page box size, aspect, and orientation."""
    if image.width < 1 or image.height < 1 or image.bbox.is_empty:
        return False
    if image.bbox.width < MIN_DISPLAY_SIZE or image.bbox.height < MIN_DISPLAY_SIZE:
        return False

    saved_landscape = image.width >= image.height
    page_landscape = image.bbox.width >= image.bbox.height
    if saved_landscape != page_landscape:
        return False

    page_aspect = image.bbox.width / image.bbox.height
    saved_aspect = image.width / image.height
    if abs(saved_aspect - page_aspect) / page_aspect > IMAGE_ASPECT_TOLERANCE:
        return False

    expected_width = image.bbox.width * DISPLAY_SCALE
    expected_height = image.bbox.height * DISPLAY_SCALE
    if expected_width <= 0 or expected_height <= 0:
        return False
    if abs(image.width - expected_width) / expected_width > IMAGE_SIZE_TOLERANCE:
        return False
    if abs(image.height - expected_height) / expected_height > IMAGE_SIZE_TOLERANCE:
        return False
    return True


def images_overlapping(
    images: list[ImageRef],
    rect: BBox,
    threshold: float = 0.50,
) -> list[ImageRef]:
    """Return images whose displayed box is mostly inside ``rect``."""

    matched: list[ImageRef] = []
    for image in images:
        if image.bbox.overlap_ratio(rect) >= threshold:
            matched.append(image)
    return matched


def _split_placement(placement):
    """Unpack ``(rect, matrix)`` from ``get_image_rects(..., transform=True)``."""
    if isinstance(placement, (tuple, list)) and len(placement) == 2:
        return placement[0], placement[1]
    return placement, None


def rasterize_page_clip(
    page: pymupdf.Page,
    bbox: BBox,
    output_path: Path,
) -> tuple[int, int] | None:
    """Save a page clip at ``DISPLAY_SCALE``. Return ``(width, height)`` or ``None``."""

    clip = pymupdf.Rect(bbox.to_tuple()) & page.rect
    if clip.is_empty or clip.width < MIN_DISPLAY_SIZE or clip.height < MIN_DISPLAY_SIZE:
        return None
    try:
        pixmap = page.get_pixmap(
            clip=clip,
            matrix=pymupdf.Matrix(DISPLAY_SCALE, DISPLAY_SCALE),
            alpha=False,
        )
    except RECOVERABLE as exc:
        log_failure("extract", f"rasterize clip {output_path.name}", exc)
        return None
    if pixmap.width < 1 or pixmap.height < 1:
        return None
    pixmap.save(str(output_path))
    return int(pixmap.width), int(pixmap.height)


def _save_displayed_image(
    page: pymupdf.Page,
    bbox: BBox,
    *,
    page_number: int,
    xref: int,
    index: int,
    output_dir: Path,
) -> tuple[str, int, int, str] | None:
    """Rasterize the image as painted on the page (size and orientation)."""

    filename = f"page_{page_number}_xref_{xref}_{index}.png"
    path = output_dir / filename
    size = rasterize_page_clip(page, bbox, path)
    if size is None:
        return None
    width, height = size
    return str(path), width, height, "png"
