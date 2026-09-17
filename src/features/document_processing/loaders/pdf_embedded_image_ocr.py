"""OCR and optional vision for embedded images detected inside native PDF pages."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from src.features.document_processing.loaders.component_classification import (
    COMPONENT_IMAGE,
    is_image_block,
)
from src.features.document_processing.loaders.image_ocr_loader import ocr_image_best, ocr_text_score
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdf_visual_content import (
    attach_visual_extraction,
    describe_image_bytes_with_vision,
    image_block_bbox,
    merge_bboxes,
    should_request_vision_description,
    vision_for_embedded_images_enabled,
    vision_text_quality,
)
from src.features.document_processing.loaders.scanned_pdf_ocr import _ocr_processor_enabled

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 200
_DEFAULT_RETRY_DPI = 300
_DEFAULT_MAX_IMAGES = 100
_DEFAULT_MAX_IMAGES_PER_PAGE = 8
_MIN_IMAGE_SIDE = 48
_BBOX_OVERLAP_THRESHOLD = 0.55


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or str(default)).strip())
    except ValueError:
        return default


def _image_processor_enabled() -> bool:
    try:
        from src.features.document_processing.shared_processor.deployment import (
            ProcessorConfigurationProvider,
        )

        return ProcessorConfigurationProvider.is_enabled("image")
    except Exception:
        return True


def embedded_image_ocr_enabled() -> bool:
    if not _env_bool("PDF_EMBEDDED_IMAGE_OCR_ENABLED", False):
        return False
    if not _ocr_processor_enabled():
        return False
    return _image_processor_enabled()


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])


def _bbox_overlap_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    top = max(left[1], right[1])
    x1 = min(left[2], right[2])
    bottom = min(left[3], right[3])
    if x1 <= x0 or bottom <= top:
        return 0.0
    overlap = _bbox_area((x0, top, x1, bottom))
    smaller = min(_bbox_area(left), _bbox_area(right))
    if smaller <= 0:
        return 0.0
    return overlap / smaller


def _image_area(block: Block) -> float:
    metadata = dict(getattr(block, "metadata", {}) or {})
    width = metadata.get("width")
    height = metadata.get("height")
    try:
        if width is not None and height is not None:
            return abs(float(width)) * abs(float(height))
    except (TypeError, ValueError):
        pass
    bbox = image_block_bbox(block)
    if bbox is None:
        return 0.0
    return _bbox_area(bbox)


def _image_large_enough(block: Block, min_side: int = _MIN_IMAGE_SIDE) -> bool:
    metadata = dict(getattr(block, "metadata", {}) or {})
    width = metadata.get("width")
    height = metadata.get("height")
    try:
        if width is not None and height is not None:
            return min(abs(float(width)), abs(float(height))) >= min_side
    except (TypeError, ValueError):
        pass
    bbox = image_block_bbox(block)
    if bbox is None:
        return False
    return min(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])) >= min_side


def _image_has_extracted_text(block: Block) -> bool:
    return bool(str(getattr(block, "text", "") or "").strip())


def _select_ocr_candidates(
    candidates: list[Block],
    *,
    max_total: int,
    max_per_page: int,
) -> list[Block]:
    """Pick largest images per page, then apply a document-wide cap."""
    if max_per_page <= 0:
        per_page = max_total if max_total > 0 else len(candidates)
    else:
        per_page = max(1, max_per_page)
    by_page: dict[int, list[Block]] = {}
    for block in candidates:
        page = int(block.page or 1)
        by_page.setdefault(page, []).append(block)
    selected: list[Block] = []
    for page in sorted(by_page):
        page_blocks = sorted(by_page[page], key=_image_area, reverse=True)
        selected.extend(page_blocks[:per_page])
    selected.sort(key=_image_area, reverse=True)
    if max_total > 0:
        selected = selected[:max_total]
    return selected


def _block_from_bbox(
    page_num: int,
    bbox: tuple[float, float, float, float],
    *,
    image_index: int,
    source: str,
) -> Block:
    x0, top, x1, bottom = bbox
    width = abs(x1 - x0)
    height = abs(bottom - top)
    return Block(
        text="",
        page=page_num,
        line_number=0,
        block_type="image",
        component_type=COMPONENT_IMAGE,
        metadata={
            "image_index": image_index,
            "x0": x0,
            "top": top,
            "x1": x1,
            "bottom": bottom,
            "width": width,
            "height": height,
            "component_type": COMPONENT_IMAGE,
            "image_discovery_source": source,
            "source_top": top,
        },
    )


def _discover_fitz_image_blocks(document_path: Path, blocks: list[Block]) -> list[Block]:
    """Add image blocks reported by PyMuPDF when pdfplumber missed raster images."""
    if not _env_bool("PDF_EMBEDDED_IMAGE_FITZ_DISCOVERY", True):
        return []

    overlap_threshold = min(
        1.0,
        max(0.0, _env_float("PDF_EMBEDDED_IMAGE_BBOX_OVERLAP_THRESHOLD", _BBOX_OVERLAP_THRESHOLD)),
    )
    min_side = max(1, _env_int("PDF_EMBEDDED_IMAGE_MIN_SIDE", _MIN_IMAGE_SIDE))

    existing = [block for block in blocks if is_image_block(block)]
    existing_bboxes = [bbox for block in existing if (bbox := image_block_bbox(block)) is not None]
    discovered: list[Block] = []

    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF unavailable; skipping supplemental embedded-image discovery")
        return []

    try:
        with fitz.open(str(document_path)) as doc:
            for page_num in range(1, len(doc) + 1):
                page = doc[page_num - 1]
                seen_xrefs: set[int] = set()
                image_index = 0
                for image in page.get_images(full=True):
                    xref = int(image[0])
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        rects = page.get_image_rects(xref)
                    except Exception:
                        rects = []
                    for rect in rects or []:
                        bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                        if min(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])) < min_side:
                            continue
                        if any(_bbox_overlap_ratio(bbox, known) >= overlap_threshold for known in existing_bboxes):
                            continue
                        image_index += 1
                        block = _block_from_bbox(
                            page_num,
                            bbox,
                            image_index=image_index,
                            source="fitz_image_rect",
                        )
                        discovered.append(block)
                        existing_bboxes.append(bbox)
    except Exception as exc:
        logger.warning("Supplemental embedded-image discovery failed for %s: %s", document_path.name, exc)
        return []

    if discovered:
        logger.info(
            "Discovered %s supplemental embedded image block(s) in %s via PyMuPDF",
            len(discovered),
            document_path.name,
        )
    return discovered


def render_pdf_image_crop(
    document_path: Path,
    page_number: int,
    bbox: tuple[float, float, float, float],
    destination: Path,
    *,
    dpi: int = _DEFAULT_DPI,
) -> None:
    import fitz

    x0, top, x1, bottom = bbox
    with fitz.open(str(document_path)) as doc:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"PDF page {page_number} is out of range for {document_path.name}")
        page = doc[page_number - 1]
        clip = fitz.Rect(x0, top, x1, bottom)
        if clip.is_empty or clip.width < 1 or clip.height < 1:
            raise ValueError(f"Invalid image crop on page {page_number}")
        pixmap = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
        destination.write_bytes(pixmap.tobytes("png"))


def _vision_description_for_crop(png_path: Path) -> str:
    try:
        return describe_image_bytes_with_vision(
            png_path.read_bytes(),
            filename=png_path.name,
        )
    except Exception as exc:
        logger.warning(
            "Vision description failed for %s: %s",
            png_path.name,
            exc,
        )
        return ""


def _process_image_crop(
    document_path: Path,
    block: Block,
    *,
    dpi: int,
    png_path: Path,
    pass_label: str,
) -> None:
    bbox = image_block_bbox(block)
    if bbox is None:
        return
    render_pdf_image_crop(
        document_path,
        int(block.page or 1),
        bbox,
        png_path,
        dpi=dpi,
    )
    ocr_text = ocr_image_best(png_path)
    vision_text = ""
    if should_request_vision_description(ocr_text):
        vision_text = _vision_description_for_crop(png_path)
        if vision_text and ocr_text_score(ocr_text) > vision_text_quality(vision_text):
            vision_text = ""

    if not ocr_text.strip() and not vision_text.strip():
        return

    source = "tesseract_ocr"
    if vision_text and ocr_text:
        source = "tesseract_and_vision"
    elif vision_text:
        source = "vision"
    if pass_label != "primary":
        source = f"{source}_{pass_label}"

    attach_visual_extraction(
        block,
        ocr_text=ocr_text,
        vision_text=vision_text,
        source=source,
    )


def _ocr_selected_blocks(
    document_path: Path,
    selected: list[Block],
    *,
    dpi: int,
    tmp_dir: Path,
    pass_label: str,
) -> int:
    success_count = 0
    for index, block in enumerate(selected, start=1):
        if _image_has_extracted_text(block):
            continue
        png_path = tmp_dir / f"{pass_label}-page-{block.page}-img-{index}.png"
        try:
            _process_image_crop(
                document_path,
                block,
                dpi=dpi,
                png_path=png_path,
                pass_label=pass_label,
            )
        except Exception as exc:
            logger.warning(
                "Embedded image processing failed for page %s image %s in %s (%s): %s",
                block.page,
                (getattr(block, "metadata", {}) or {}).get("image_index"),
                document_path.name,
                pass_label,
                exc,
            )
            continue
        if _image_has_extracted_text(block):
            success_count += 1
        else:
            logger.warning(
                "Embedded image processing produced no text for page %s image %s in %s (%s)",
                block.page,
                (getattr(block, "metadata", {}) or {}).get("image_index"),
                document_path.name,
                pass_label,
            )
    return success_count


def _retry_merged_regions(
    document_path: Path,
    blocks: list[Block],
    selected: list[Block],
    *,
    dpi: int,
    tmp_dir: Path,
) -> None:
    if not _env_bool("PDF_EMBEDDED_IMAGE_MERGED_REGION_RETRY", True):
        return

    by_page: dict[int, list[Block]] = {}
    for block in selected:
        if _image_has_extracted_text(block):
            continue
        page = int(block.page or 1)
        by_page.setdefault(page, []).append(block)

    for page, page_blocks in sorted(by_page.items()):
        bboxes = [bbox for block in page_blocks if (bbox := image_block_bbox(block)) is not None]
        merged = merge_bboxes(bboxes)
        if merged is None:
            continue
        target = max(page_blocks, key=_image_area)
        png_path = tmp_dir / f"merged-page-{page}.png"
        merged_block = _block_from_bbox(
            page,
            merged,
            image_index=int((getattr(target, "metadata", {}) or {}).get("image_index") or 0),
            source="merged_image_region",
        )
        try:
            _process_image_crop(
                document_path,
                merged_block,
                dpi=dpi,
                png_path=png_path,
                pass_label="merged",
            )
            if _image_has_extracted_text(merged_block):
                metadata = dict(getattr(target, "metadata", {}) or {})
                merged_metadata = dict(getattr(merged_block, "metadata", {}) or {})
                metadata.update(merged_metadata)
                metadata["merged_region_retry"] = True
                target.text = merged_block.text
                target.metadata = metadata
        except Exception as exc:
            logger.warning(
                "Merged embedded-image retry failed for page %s in %s: %s",
                page,
                document_path.name,
                exc,
            )


def apply_ocr_to_embedded_pdf_images(document_path: Path, blocks: list[Block]) -> list[Block]:
    """Run OCR/vision on embedded PDF image blocks and attach text for chunking."""
    if not embedded_image_ocr_enabled():
        return blocks

    supplemental = _discover_fitz_image_blocks(document_path, blocks)
    if supplemental:
        blocks = list(blocks) + supplemental

    candidates = [block for block in blocks if is_image_block(block) and _image_large_enough(block)]
    if not candidates:
        logger.info("No embedded image blocks eligible for OCR in %s", document_path.name)
        return blocks

    max_total = _env_int("PDF_MAX_EMBEDDED_IMAGES_OCR", _DEFAULT_MAX_IMAGES)
    max_per_page = _env_int("PDF_MAX_EMBEDDED_IMAGES_PER_PAGE", _DEFAULT_MAX_IMAGES_PER_PAGE)
    dpi = max(72, _env_int("PDF_OCR_DPI", _DEFAULT_DPI))
    retry_dpi = max(dpi, _env_int("PDF_EMBEDDED_IMAGE_OCR_RETRY_DPI", _DEFAULT_RETRY_DPI))
    selected = _select_ocr_candidates(
        candidates,
        max_total=max_total,
        max_per_page=max_per_page,
    )

    logger.info(
        "Running embedded-image processing on %s image(s) (total_limit=%s, per_page_limit=%s, dpi=%s, vision=%s) in %s",
        len(selected),
        max_total,
        max_per_page,
        dpi,
        vision_for_embedded_images_enabled(),
        document_path.name,
    )

    with tempfile.TemporaryDirectory(prefix="pdf-embed-ocr-") as tmp:
        tmp_dir = Path(tmp)
        _ocr_selected_blocks(
            document_path,
            selected,
            dpi=dpi,
            tmp_dir=tmp_dir,
            pass_label="primary",
        )

        if retry_dpi > dpi and _env_bool("PDF_EMBEDDED_IMAGE_OCR_RETRY_ENABLED", True):
            retry_targets = [block for block in selected if not _image_has_extracted_text(block)]
            if retry_targets:
                logger.info(
                    "Retrying embedded-image processing at %s dpi for %s image(s) in %s",
                    retry_dpi,
                    len(retry_targets),
                    document_path.name,
                )
                _ocr_selected_blocks(
                    document_path,
                    retry_targets,
                    dpi=retry_dpi,
                    tmp_dir=tmp_dir,
                    pass_label="retry",
                )

        _retry_merged_regions(
            document_path,
            blocks,
            selected,
            dpi=retry_dpi,
            tmp_dir=tmp_dir,
        )

    return blocks
