"""Select document pages used for key field extraction."""

from __future__ import annotations

from typing import Any


def select_extraction_page_numbers(page_count: int) -> list[int]:
    """Return 1-based page numbers for extraction (first 2 + last, or all when <= 3 pages)."""
    total = max(1, int(page_count))
    if total <= 3:
        return list(range(1, total + 1))
    return [1, 2, total]


def filter_blocks_for_extraction(
    blocks: list[Any],
    page_count: int,
    *,
    strict_page_scope: bool = False,
) -> list[Any]:
    """Keep blocks on selected pages.

    When strict_page_scope is enabled and page metadata is missing on all blocks,
    return no blocks so caller can skip extraction instead of scanning full content.
    """
    pages = set(select_extraction_page_numbers(page_count))
    tagged = [block for block in blocks if getattr(block, "page", None) in pages]
    if tagged:
        return tagged
    has_any_page_tag = any(getattr(block, "page", None) is not None for block in blocks)
    if strict_page_scope and not has_any_page_tag:
        return []
    return list(blocks)
