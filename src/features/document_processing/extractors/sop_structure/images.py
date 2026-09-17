"""Image / figure caption enrichment for template graphs."""

from __future__ import annotations

import re
from typing import Any

from src.features.document_processing.extractors.sop_structure.helpers import (
    normalize_heading_text,
    walk_tree,
)

_FIGURE_CAPTION = re.compile(
    r"^\s*(?:figure|fig\.?)\s*(?P<num>\d+[A-Za-z]?)\s*[:.\-–—]?\s*(?P<caption>.*)$",
    re.I,
)


def _images_from_tree(document_tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for node in walk_tree(document_tree):
        if node.get("type") != "image":
            continue
        images.append(dict(node))
    return images


def extract_images(
    template_data: dict[str, Any],
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich existing image nodes with figure captions and section titles."""
    existing = list(template_data.get("images") or [])
    tree_images = _images_from_tree(list(template_data.get("document_tree") or []))

    # Prefer explicit images list; fall back to tree
    base = existing if existing else tree_images
    if not base:
        # Synthesize from content blocks marked as image
        for idx, block in enumerate(content_blocks):
            if str(block.get("block_type") or "").lower() in {"image", "figure"} or str(
                block.get("component_type") or ""
            ).lower() in {"image", "figure"}:
                base.append(
                    {
                        "image_index": idx,
                        "page": block.get("page"),
                        "caption": str(block.get("text") or ""),
                        "width": block.get("width"),
                        "height": block.get("height"),
                        "image_type": block.get("image_type") or "embedded",
                    }
                )

    captions_by_page: dict[int, list[dict[str, Any]]] = {}
    section_by_page: dict[int, str] = {}
    current_heading = ""
    for block in content_blocks:
        text = str(block.get("text") or "").strip()
        page = int(block.get("page") or 0) or 0
        style = str(block.get("style") or "").lower()
        if style.startswith("heading") and text:
            current_heading = text
            if page:
                section_by_page[page] = text
        m = _FIGURE_CAPTION.match(text)
        if m:
            captions_by_page.setdefault(page, []).append(
                {
                    "figure_number": m.group("num"),
                    "caption": (m.group("caption") or "").strip() or text,
                    "text": text,
                }
            )

    enriched: list[dict[str, Any]] = []
    used_captions: set[tuple[int, str]] = set()

    for idx, img in enumerate(base):
        if not isinstance(img, dict):
            continue
        page = int(img.get("page") or 0) or None
        caption = str(img.get("caption") or "").strip()
        figure_number = str(img.get("figure_number") or "").strip()
        section_title = str(img.get("section_title") or "").strip()

        # Match nearby figure caption on same page
        page_key = int(page or 0)
        for cap in captions_by_page.get(page_key, []):
            key = (page_key, cap["figure_number"])
            if key in used_captions:
                continue
            used_captions.add(key)
            figure_number = figure_number or cap["figure_number"]
            caption = caption or cap["caption"]
            break

        # Caption may already be on the image node
        if not figure_number and caption:
            m = _FIGURE_CAPTION.match(caption)
            if m:
                figure_number = m.group("num")
                caption = (m.group("caption") or caption).strip()

        if not section_title and page_key in section_by_page:
            section_title = section_by_page[page_key]
        elif not section_title:
            section_title = current_heading

        enriched.append(
            {
                "image_index": int(img.get("image_index") if img.get("image_index") is not None else idx),
                "page": page,
                "caption": caption,
                "width": img.get("width"),
                "height": img.get("height"),
                "image_type": img.get("image_type") or img.get("type") or "embedded",
                "figure_number": figure_number,
                "section_title": section_title,
            }
        )

    # Orphan figure captions without image nodes still create Image entries
    for page, caps in captions_by_page.items():
        for cap in caps:
            key = (page, cap["figure_number"])
            if key in used_captions:
                continue
            used_captions.add(key)
            enriched.append(
                {
                    "image_index": len(enriched),
                    "page": page or None,
                    "caption": cap["caption"],
                    "width": None,
                    "height": None,
                    "image_type": "figure_caption",
                    "figure_number": cap["figure_number"],
                    "section_title": section_by_page.get(page, ""),
                }
            )

    return enriched
