"""Shared helpers for SOP structure detectors."""

from __future__ import annotations

import re
from typing import Any, Iterable


def normalize_heading_text(text: str | None) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    # Drop leading section numbers: "5 Procedure" / "5. Procedure"
    cleaned = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", cleaned)
    return cleaned.strip().lower()


def looks_like_heading_title(text: str, keywords: Iterable[str]) -> bool:
    norm = normalize_heading_text(text)
    if not norm:
        return False
    for keyword in keywords:
        key = keyword.lower()
        if norm == key or norm.startswith(key + " ") or key in norm.split():
            return True
        if key in norm and len(norm) <= max(40, len(key) + 20):
            return True
    return False


def walk_tree(nodes: list[Any] | None) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        found.append(node)
        found.extend(walk_tree(node.get("children") or []))
    return found


def content_block(
    *,
    text: str,
    page: int | None = None,
    style: str = "",
    block_type: str = "text",
    component_type: str = "paragraph",
    bold: bool = False,
) -> dict[str, Any]:
    return {
        "text": str(text or "").strip(),
        "page": int(page or 0) or None,
        "style": str(style or ""),
        "block_type": str(block_type or "text"),
        "component_type": str(component_type or "paragraph"),
        "bold": bool(bold),
    }
