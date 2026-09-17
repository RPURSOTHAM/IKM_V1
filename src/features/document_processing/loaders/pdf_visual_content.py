"""Shared helpers for multimodal PDF visual content extraction."""

from __future__ import annotations

import os
import re
from typing import Any

from src.features.document_processing.loaders.image_ocr_loader import ocr_text_score
from src.features.document_processing.loaders.loader import Block

_STRUCTURED_VISION_PROMPT = (
    "Describe this visual for document search indexing. Include only information that is "
    "clearly visible. Do not invent values.\n"
    "Format:\n"
    "PURPOSE: brief purpose of the figure\n"
    "VISIBLE_TEXT: all readable text, labels, titles, and annotations\n"
    "AXES: axis names and scales if present\n"
    "LEGENDS: legend entries if present\n"
    "STRUCTURE: tables, panels, diagrams, or chart layout\n"
    "OBSERVATIONS: important visible relationships or trends\n"
    "Return plain text only."
)

_TABLE_LINE_PATTERN = re.compile(r"\|.+\|")
_MULTI_COLUMN_LINE = re.compile(r"\S+\s{2,}\S+")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except ValueError:
        return default


def vision_for_embedded_images_enabled() -> bool:
    if _env_bool("PDF_VISION_FOR_EMBEDDED_IMAGES", False):
        return True
    return _env_bool("OPEN_WEIGHT_VLM_FOR_OCR", False)


def vision_when_ocr_chars_below() -> int:
    return max(0, _env_int("PDF_VISION_WHEN_OCR_CHARS_BELOW", 280))


def looks_like_tabular_ocr_text(text: str) -> bool:
    """Generic heuristic for OCR that resembles a table grid."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    pipe_rows = sum(1 for line in lines if _TABLE_LINE_PATTERN.search(line) or " | " in line)
    if pipe_rows >= 2:
        return True
    multi_column_rows = sum(1 for line in lines if _MULTI_COLUMN_LINE.search(line))
    token_rows = sum(1 for line in lines if len(line.split()) >= 4)
    return multi_column_rows >= 2 or token_rows >= max(3, len(lines) // 2)


def image_block_bbox(block: Any) -> tuple[float, float, float, float] | None:
    metadata = dict(getattr(block, "metadata", {}) or {})
    x0 = metadata.get("x0")
    top = metadata.get("top")
    x1 = metadata.get("x1")
    bottom = metadata.get("bottom")
    if x0 is None or top is None or x1 is None or bottom is None:
        return None
    try:
        rect = (float(x0), float(top), float(x1), float(bottom))
    except (TypeError, ValueError):
        return None
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return None
    return rect


def merge_bboxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    x0 = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    if x1 <= x0 or bottom <= top:
        return None
    return (x0, top, x1, bottom)


def combine_searchable_visual_text(*parts: str) -> str:
    cleaned_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = " ".join((part or "").split()).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned_parts.append(cleaned)
    return "\n\n".join(cleaned_parts).strip()


def infer_visual_content_type(ocr_text: str, vision_text: str) -> str:
    if looks_like_tabular_ocr_text(ocr_text):
        return "image_table"
    if vision_text.strip() and not ocr_text.strip():
        return "image_vision"
    if vision_text.strip() and ocr_text.strip():
        return "image_multimodal"
    return "image_ocr"


def image_block_extraction_metadata(block: Any) -> dict[str, Any]:
    metadata = dict(getattr(block, "metadata", {}) or {})
    bbox = image_block_bbox(block)
    ocr_text = str(metadata.get("ocr_text") or "").strip()
    vision_text = str(metadata.get("visual_description") or "").strip()
    content_type = str(metadata.get("content_type") or infer_visual_content_type(ocr_text, vision_text))
    image_ref = {
        "image_index": metadata.get("image_index"),
        "page_number": int(getattr(block, "page", 0) or metadata.get("page_number") or 0) or None,
        "bbox": list(bbox) if bbox else None,
        "discovery_source": metadata.get("image_discovery_source"),
    }
    return {
        **metadata,
        "content_type": content_type,
        "content_source": metadata.get("image_text_source") or metadata.get("content_source") or "",
        "ocr_text": ocr_text,
        "visual_description": vision_text,
        "embedded_image_ocr": bool(metadata.get("embedded_image_ocr")),
        "image_reference": image_ref,
        "images": [image_ref],
    }


def attach_visual_extraction(
    block: Block,
    *,
    ocr_text: str = "",
    vision_text: str = "",
    source: str,
) -> None:
    ocr_clean = " ".join((ocr_text or "").split()).strip()
    vision_clean = " ".join((vision_text or "").split()).strip()
    combined = combine_searchable_visual_text(ocr_clean, vision_clean)
    if not combined:
        return

    metadata = dict(getattr(block, "metadata", {}) or {})
    metadata["ocr_text"] = ocr_clean
    if vision_clean:
        metadata["visual_description"] = vision_clean
    metadata["image_caption"] = combined
    metadata["ocr_used"] = bool(ocr_clean)
    metadata["embedded_image_ocr"] = True
    metadata["image_text_source"] = source
    metadata["content_source"] = source
    metadata["content_type"] = infer_visual_content_type(ocr_clean, vision_clean)
    block.text = combined
    block.metadata = metadata


def should_request_vision_description(ocr_text: str) -> bool:
    if not vision_for_embedded_images_enabled():
        return False
    threshold = vision_when_ocr_chars_below()
    if threshold <= 0:
        return True
    return len((ocr_text or "").strip()) < threshold


def describe_image_bytes_with_vision(png_bytes: bytes, *, filename: str) -> str:
    from src.features.content_intelligence_hub.application.ollama_vlm import (
        summarize_image_bytes,
        vlm_enabled,
    )

    if not vlm_enabled():
        return ""
    payload = summarize_image_bytes(
        png_bytes,
        filename=filename,
        prompt=_STRUCTURED_VISION_PROMPT,
        mime_type="image/png",
    )
    return str(payload.get("summary") or payload.get("text") or "").strip()


def vision_text_quality(text: str) -> int:
    return ocr_text_score(text)
