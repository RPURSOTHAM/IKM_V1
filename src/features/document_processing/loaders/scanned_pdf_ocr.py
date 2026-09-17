"""OCR fallback for scanned / image-only PDFs that have no usable text layer."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import os

from src.features.document_processing.loaders.component_classification import is_image_block
from src.features.document_processing.loaders.image_ocr_loader import (
    blocks_from_ocr_text,
    ocr_image_best,
    ocr_text_score,
)
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdf_visual_content import (
    combine_searchable_visual_text,
    describe_image_bytes_with_vision,
    should_request_vision_description,
    vision_text_quality,
)

logger = logging.getLogger(__name__)

_RENDER_DPI = 300
_CID_NOISE = re.compile(r"\(cid:\d+\)", re.IGNORECASE)


def _ocr_processor_enabled() -> bool:
    try:
        from src.features.document_processing.shared_processor.deployment import (
            ProcessorConfigurationProvider,
        )

        return ProcessorConfigurationProvider.is_enabled("ocr")
    except Exception:
        return True


def native_text_for_page(blocks: list[Block], page: int) -> str:
    parts = [
        str(block.text or "")
        for block in blocks
        if int(block.page or 0) == page and not is_image_block(block)
    ]
    return " ".join(part.strip() for part in parts if part.strip())


def looks_like_usable_native_text(text: str) -> bool:
    """True when the PDF already has a real selectable text layer."""
    cleaned = _CID_NOISE.sub(" ", text or "")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) < 40:
        return False
    letters = sum(ch.isalpha() for ch in cleaned)
    words = [part for part in cleaned.split() if any(ch.isalpha() for ch in part)]
    if letters < 12 or len(words) < 5:
        return False
    return (letters / max(len(cleaned), 1)) >= 0.28


def pdf_pages_needing_ocr(blocks: list[Block], page_count: int) -> list[int]:
    needed: list[int] = []
    for page in range(1, max(page_count, 1) + 1):
        if not looks_like_usable_native_text(native_text_for_page(blocks, page)):
            needed.append(page)
    return needed


def render_pdf_page_png(document_path: Path, page_number: int, destination: Path, dpi: int = _RENDER_DPI) -> None:
    import fitz

    with fitz.open(str(document_path)) as doc:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"PDF page {page_number} is out of range for {document_path.name}")
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        destination.write_bytes(pixmap.tobytes("png"))


def _vision_for_scanned_pages_enabled() -> bool:
    raw = (os.getenv("PDF_VISION_FOR_SCANNED_PAGES") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _extract_scanned_page_text(png_path: Path) -> str:
    ocr_text = ocr_image_best(png_path)
    vision_text = ""
    if _vision_for_scanned_pages_enabled() or should_request_vision_description(ocr_text):
        try:
            vision_text = describe_image_bytes_with_vision(
                png_path.read_bytes(),
                filename=png_path.name,
            )
            if vision_text and ocr_text_score(ocr_text) > vision_text_quality(vision_text):
                vision_text = ""
        except Exception as exc:
            logger.warning("Vision description failed for scanned page %s: %s", png_path.name, exc)
    return combine_searchable_visual_text(ocr_text, vision_text)


def ocr_scanned_pdf_pages(document_path: Path, pages: list[int]) -> list[Block]:
    ocr_blocks: list[Block] = []
    with tempfile.TemporaryDirectory(prefix="pdf-ocr-") as tmp:
        tmp_dir = Path(tmp)
        for page_number in pages:
            png_path = tmp_dir / f"page-{page_number}.png"
            render_pdf_page_png(document_path, page_number, png_path)
            text = _extract_scanned_page_text(png_path)
            page_blocks = blocks_from_ocr_text(text, page=page_number)
            if not page_blocks:
                logger.warning(
                    "OCR produced no text for %s page %s",
                    document_path.name,
                    page_number,
                )
            for block in page_blocks:
                metadata = dict(getattr(block, "metadata", {}) or {})
                metadata["scanned_page_ocr"] = True
                block.metadata = metadata
            ocr_blocks.extend(page_blocks)
    return ocr_blocks


def apply_ocr_to_scanned_pdf(document_path: Path, blocks: list[Block]) -> list[Block]:
    """Replace unusable native text with Tesseract OCR for scanned PDF pages."""
    if not _ocr_processor_enabled():
        return blocks

    try:
        import fitz

        with fitz.open(str(document_path)) as doc:
            page_count = len(doc)
    except Exception:
        page_count = max((int(block.page or 1) for block in blocks), default=1)

    pages = pdf_pages_needing_ocr(blocks, page_count)
    if not pages:
        return blocks

    logger.info(
        "Running Tesseract OCR on scanned PDF %s pages=%s",
        document_path.name,
        pages,
    )
    ocr_blocks = ocr_scanned_pdf_pages(document_path, pages)
    if not ocr_blocks:
        return blocks

    kept = [
        block
        for block in blocks
        if int(block.page or 0) not in pages or is_image_block(block)
    ]
    merged = kept + ocr_blocks
    merged.sort(key=lambda block: (int(block.page or 0), int(block.line_number or 0)))
    return merged
