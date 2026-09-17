"""Run the core extractors on a PDF and return ordered page items.

This layer does not filter or group. Callers that need a consumer-ready
result should use ``src.main.process``, which sends this output through
filtering and grouping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pymupdf

from ..document import (
    BBox,
    ContentKind,
    DocumentContent,
    DocumentItem,
    ImageRef,
    Table,
    TextBlock,
)
from ..errors import RECOVERABLE
from ..failures import log_failure, run_continuing
from .attach_image_text import attach_nearby_text_to_images
from .attach_inline_images import split_text_around_inline_images
from .extract_image import extract_page_images
from .extract_infographic import extract_page_infographics
from .extract_table import extract_page_tables
from .extract_text import extract_text_blocks

Y_BAND = 6.0
INSIDE_THRESHOLD = 0.55


def extract_document(
    document_name: str,
    image_dir: str | Path | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> DocumentContent:
    """Open ``document_name`` and extract text, tables, images, and infographics.

    Per page the order is: embedded images, tables, infographic clips, then
    text outside table and infographic regions. Items are merged in reading
    order. Saved pictures go under ``image_dir``. ``on_progress`` receives
    a stage message and a 0–1 fraction of extraction work.
    """

    pdf_path = Path(document_name)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if image_dir is None:
        image_dir = Path("extracted_images") / pdf_path.stem
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    _emit_progress(on_progress, "Opening PDF…", 0.0)
    document = pymupdf.open(pdf_path)
    page_count = document.page_count
    items: list[DocumentItem] = []
    saved_placements: dict = {}

    try:
        _emit_progress(
            on_progress,
            f"Opened PDF ({page_count} page{'s' if page_count != 1 else ''})",
            0.0,
        )
        for page_index, page in enumerate(document, start=1):
            try:
                items.extend(
                    _extract_one_page(
                        page,
                        page_index,
                        page_count,
                        image_dir,
                        saved_placements,
                        on_progress,
                    )
                )
            except RECOVERABLE as exc:
                log_failure(
                    "extract",
                    f"page {page_index} of {page_count}",
                    exc,
                )
        _emit_progress(on_progress, "Extraction complete", 1.0)
    finally:
        document.close()

    return DocumentContent(
        name=str(pdf_path),
        page_count=page_count,
        items=items,
    )


def _extract_one_page(
    page,
    page_index: int,
    page_count: int,
    image_dir: Path,
    saved_placements: dict,
    on_progress: Callable[[str, float], None] | None,
) -> list[DocumentItem]:
    """Extract one page; a failed extractor returns empty and the rest continue."""

    page_base = (page_index - 1) / max(page_count, 1)
    page_span = 1 / max(page_count, 1)
    page_label = f"Page {page_index} of {page_count}"
    _emit_progress(
        on_progress,
        f"{page_label}: extracting images",
        page_base + 0.05 * page_span,
    )
    images = run_continuing(
        "extract",
        lambda: extract_page_images(
            page,
            page_number=page_index,
            output_dir=image_dir,
            saved_xrefs=saved_placements,
        ),
        fallback=[],
        message=f"{page_label} images",
    )
    _emit_progress(
        on_progress,
        f"{page_label}: extracting tables",
        page_base + 0.25 * page_span,
    )
    tables = run_continuing(
        "extract",
        lambda: extract_page_tables(page, page_images=images),
        fallback=[],
        message=f"{page_label} tables",
    )
    table_rects = [table.bbox for table in tables if table.bbox]
    _emit_progress(
        on_progress,
        f"{page_label}: extracting infographics",
        page_base + 0.65 * page_span,
    )
    infographics = run_continuing(
        "extract",
        lambda: extract_page_infographics(
            page,
            page_number=page_index,
            output_dir=image_dir,
            exclude_rects=table_rects,
            page_images=images,
        ),
        fallback=[],
        message=f"{page_label} infographics",
    )
    exclude_rects = table_rects + [image.bbox for image in infographics]
    standalone_images = [
        image for image in images if not _inside_any(image.bbox, exclude_rects)
    ]
    standalone_images.extend(infographics)
    _emit_progress(
        on_progress,
        f"{page_label}: extracting text",
        page_base + 0.82 * page_span,
    )
    text_blocks = run_continuing(
        "extract",
        lambda: extract_text_blocks(page, exclude_rects=exclude_rects),
        fallback=[],
        message=f"{page_label} text",
    )
    text_blocks = run_continuing(
        "extract",
        lambda: attach_nearby_text_to_images(
            page,
            standalone_images,
            text_blocks,
            page_number=page_index,
            output_dir=image_dir,
        ),
        fallback=text_blocks,
        message=f"{page_label} image captions",
    )
    text_blocks = run_continuing(
        "extract",
        lambda: split_text_around_inline_images(standalone_images, text_blocks),
        fallback=text_blocks,
        message=f"{page_label} inline images",
    )
    _emit_progress(
        on_progress,
        f"{page_label}: ordering blocks",
        page_base + 0.95 * page_span,
    )
    return run_continuing(
        "extract",
        lambda: _ordered_page_items(
            page_index, text_blocks, tables, standalone_images
        ),
        fallback=[],
        message=f"{page_label} ordering",
    )


def _ordered_page_items(
    page_number: int,
    text_blocks: list[TextBlock],
    tables: list[Table],
    images: list[ImageRef],
) -> list[DocumentItem]:
    """Merge text, tables, and images on one page into reading-order items."""

    pending: list[tuple[BBox, ContentKind, TextBlock | Table | ImageRef]] = []

    for block in text_blocks:
        if block.bbox is None:
            continue
        pending.append((block.bbox, ContentKind.TEXT, block))

    for table in tables:
        if table.bbox is None:
            continue
        pending.append((table.bbox, ContentKind.TABLE, table))

    for image in images:
        pending.append((image.bbox, ContentKind.IMAGE, image))

    pending.sort(key=lambda item: _reading_key(item[0]))

    ordered: list[DocumentItem] = []
    for line_number, (bbox, kind, content) in enumerate(pending, start=1):
        ordered.append(
            DocumentItem(
                page=page_number,
                line_number=line_number,
                kind=kind,
                content=content,
                bbox=bbox,
            )
        )
    return ordered


def _reading_key(bbox: BBox) -> tuple[float, float]:
    """Sort key: top-to-bottom in ``Y_BAND`` rows, then left-to-right."""

    band = round(bbox.y0 / Y_BAND) * Y_BAND
    return (band, bbox.x0)


def _inside_any(bbox: BBox, rects: list[BBox]) -> bool:
    """True when ``bbox`` is mostly covered by any rectangle in ``rects``."""

    return any(bbox.overlap_ratio(rect) >= INSIDE_THRESHOLD for rect in rects)


def _emit_progress(
    on_progress: Callable[[str, float], None] | None,
    message: str,
    fraction: float,
) -> None:
    """Report a stage if the caller supplied a progress callback."""

    if on_progress is not None:
        on_progress(message, fraction)
