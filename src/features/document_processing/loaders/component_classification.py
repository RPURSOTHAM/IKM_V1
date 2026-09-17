"""Classify extracted document blocks into structural component types."""

from __future__ import annotations

import re
from typing import Any

COMPONENT_HEADER = "header"
COMPONENT_FOOTER = "footer"
COMPONENT_TITLE = "title"
COMPONENT_SUBTITLE = "subtitle"
COMPONENT_PARAGRAPH = "paragraph"
COMPONENT_TABLE = "table"
COMPONENT_IMAGE = "image"

_STRUCTURED_COMPONENTS = frozenset(
    {
        COMPONENT_HEADER,
        COMPONENT_FOOTER,
        COMPONENT_TITLE,
        COMPONENT_SUBTITLE,
        COMPONENT_PARAGRAPH,
        COMPONENT_TABLE,
        COMPONENT_IMAGE,
    }
)


def normalize_component_type(value: str | None) -> str:
    key = str(value or "").strip().lower()
    return key if key in _STRUCTURED_COMPONENTS else COMPONENT_PARAGRAPH


def heading_level_from_style(style: str) -> int | None:
    """Map Word paragraph styles to heading levels (1 = title, 2+ = subtitle)."""
    normalized = str(style or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"title", "document title", "cover page title"}:
        return 1
    match = re.search(r"heading\s*(\d+)", normalized)
    if match:
        return int(match.group(1))
    if "subtitle" in normalized:
        return 2
    return None


def component_type_from_heading_level(level: int | None) -> str:
    if level is None:
        return COMPONENT_PARAGRAPH
    if level <= 1:
        return COMPONENT_TITLE
    return COMPONENT_SUBTITLE


def classify_text_block(
    block: Any,
    *,
    region: str | None = None,
    median_font_size: float | None = None,
) -> tuple[str, int | None]:
    """Return (component_type, heading_level) for a text block."""
    block_type = str(getattr(block, "block_type", "text") or "text").lower()
    if block_type == "table":
        return COMPONENT_TABLE, None
    if block_type == "image":
        return COMPONENT_IMAGE, None

    # A PDF line in the page margin is a running header/footer even when it is
    # bold, large, centred, or happens to look like a numbered heading.  Region
    # provenance is more reliable than typography for this decision.  Checking
    # it before an existing component value also prevents a prior pass from
    # preserving a false title classification.
    if region == "header":
        return COMPONENT_HEADER, None
    if region == "footer":
        return COMPONENT_FOOTER, None

    existing = str(getattr(block, "component_type", "") or "").strip().lower()
    if existing and existing in _STRUCTURED_COMPONENTS and existing != COMPONENT_PARAGRAPH:
        return existing, getattr(block, "heading_level", None)

    style = str(getattr(block, "style", "") or "")
    level = heading_level_from_style(style)
    if level is not None:
        return component_type_from_heading_level(level), level

    text = str(getattr(block, "text", "") or "").strip()
    if text:
        numbered = re.match(r"^\s*(?P<num>\d+(?:\.\d+)*\.?)\s+(?P<title>\S.+)$", text)
        if numbered:
            parts = [part for part in numbered.group("num").rstrip(".").split(".") if part]
            if len(parts) == 1 or parts[-1] == "0":
                title_words = numbered.group("title").split()
                if len(title_words) <= 8 and (numbered.group("title").isupper() or numbered.group("title").istitle()):
                    return COMPONENT_TITLE, 1
            elif len(parts) >= 2:
                title = numbered.group("title").strip().rstrip(":")
                title_words = title.split()
                if (
                    len(title_words) <= 10
                    and (title.isupper() or title.istitle() or text.rstrip().endswith(":"))
                    and not re.search(r"\b(?:select|click|press|enter|verify|ensure|record|update)\b", title, re.IGNORECASE)
                ):
                    return COMPONENT_SUBTITLE, min(len(parts), 6)

    font_size = float(getattr(block, "font_size", 0) or 0)
    bold = bool(getattr(block, "bold", False))
    word_count = len(text.split())
    alpha_chars = len(re.findall(r"[A-Za-z]", text))

    if median_font_size and font_size >= median_font_size * 1.25 and bold and word_count <= 12 and alpha_chars >= 3:
        return COMPONENT_TITLE, 1
    if median_font_size and font_size >= median_font_size * 1.1 and bold and word_count <= 16 and alpha_chars >= 3:
        return COMPONENT_SUBTITLE, 2
    if bold and word_count <= 8 and text.isupper() and alpha_chars >= 3:
        return COMPONENT_TITLE, 1
    if bold and word_count <= 12 and text.istitle() and alpha_chars >= 3:
        return COMPONENT_SUBTITLE, 2

    return COMPONENT_PARAGRAPH, None


def apply_component_metadata(
    block: Any,
    *,
    region: str | None = None,
    median_font_size: float | None = None,
) -> Any:
    """Set component_type, heading_level, and block_type on a block in-place."""
    component_type, heading_level = classify_text_block(
        block,
        region=region,
        median_font_size=median_font_size,
    )
    try:
        block.component_type = component_type
        block.heading_level = heading_level
        if component_type == COMPONENT_TABLE:
            block.block_type = "table"
        elif component_type == COMPONENT_IMAGE:
            block.block_type = "image"
        else:
            block.block_type = "text"
    except Exception:
        pass
    metadata = dict(getattr(block, "metadata", {}) or {})
    metadata["component_type"] = component_type
    if heading_level is not None:
        metadata["heading_level"] = heading_level
    if region:
        metadata["region"] = region
    try:
        block.metadata = metadata
    except Exception:
        pass
    return block


def block_component_type(block: Any) -> str:
    explicit = str(getattr(block, "component_type", "") or "").strip().lower()
    if explicit in _STRUCTURED_COMPONENTS:
        return explicit
    metadata = getattr(block, "metadata", {}) or {}
    if isinstance(metadata, dict):
        meta_type = str(metadata.get("component_type") or "").strip().lower()
        if meta_type in _STRUCTURED_COMPONENTS:
            return meta_type
    block_type = str(getattr(block, "block_type", "text") or "text").lower()
    if block_type == "table":
        return COMPONENT_TABLE
    if block_type == "image":
        return COMPONENT_IMAGE
    return COMPONENT_PARAGRAPH


def should_chunk_component(component_type: str) -> bool:
    """All structural components participate in chunking."""
    return normalize_component_type(component_type) in _STRUCTURED_COMPONENTS


def is_image_block(block: Any) -> bool:
    """Return True when a block represents an embedded image (not chunkable text)."""
    block_type = str(getattr(block, "block_type", "text") or "text").lower()
    if block_type == "image":
        return True
    return block_component_type(block) == COMPONENT_IMAGE


def image_block_chunk_text(block: Any) -> str:
    """Return searchable text extracted for an image block, if any."""
    from src.features.document_processing.loaders.pdf_visual_content import combine_searchable_visual_text

    metadata = getattr(block, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("image_caption", "caption", "description"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    combined = combine_searchable_visual_text(
        str(metadata.get("ocr_text") or "").strip(),
        str(metadata.get("visual_description") or "").strip(),
    )
    if combined:
        return combined
    return str(getattr(block, "text", "") or "").strip()


def image_block_content_types(block: Any) -> list[str]:
    metadata = getattr(block, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    content_type = str(metadata.get("content_type") or "").strip().lower()
    if content_type == "image_table":
        return ["image_table", "table"]
    if content_type in {"image_vision", "image_multimodal"}:
        return ["image_caption", "vision"]
    if content_type:
        return [content_type]
    return ["image_caption"]


def blocks_for_chunking(blocks: list[Any]) -> list[Any]:
    """Return structural blocks that can be represented as separate chunks."""
    return list(blocks)
