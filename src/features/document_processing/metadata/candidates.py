"""Build searchable line and table candidates from document blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.features.document_processing.loaders.component_classification import block_component_type, is_image_block
from src.features.document_processing.utilities.extraction_filters import clean_extracted_line, looks_like_toc_entry

_PAGE_MARKER_RE = re.compile(r"^(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?$", re.IGNORECASE)
_FIELD_PATTERN = re.compile(r"(?P<field>[A-Za-z][A-Za-z0-9_ /-]{1,80})\s*(?:\:|=|\|\s*)\s*(?P<value>.+)")


@dataclass(frozen=True)
class LineCandidate:
    text: str
    page: int | None
    line_number: int | None
    block_type: str
    component_type: str


@dataclass(frozen=True)
class LabelValuePair:
    label: str
    value: str
    page: int | None
    line_number: int | None
    source: str


def _is_candidate_line(text: str) -> bool:
    value = clean_extracted_line(text)
    if not value:
        return False
    if _PAGE_MARKER_RE.fullmatch(value):
        return False
    if _FIELD_PATTERN.search(value) or re.search(r"\|\s*\S", value):
        return True
    if looks_like_toc_entry(value):
        return False
    return True


def build_line_candidates(blocks: list[Any]) -> list[LineCandidate]:
    candidates: list[LineCandidate] = []
    for block in blocks:
        if is_image_block(block):
            continue
        block_type = str(getattr(block, "block_type", "text") or "text")
        component_type = block_component_type(block)
        page = getattr(block, "page", None)
        base_line = getattr(block, "line_number", None)
        for offset, line in enumerate(str(getattr(block, "text", "") or "").splitlines()):
            cleaned = clean_extracted_line(line)
            if not _is_candidate_line(cleaned):
                continue
            line_number = int(base_line + offset) if isinstance(base_line, int) else base_line
            candidates.append(
                LineCandidate(
                    text=cleaned,
                    page=int(page) if page is not None else None,
                    line_number=line_number,
                    block_type=block_type,
                    component_type=component_type,
                )
            )
    return candidates


def build_label_value_pairs(blocks: list[Any]) -> list[LabelValuePair]:
    pairs: list[LabelValuePair] = []
    lines = build_line_candidates(blocks)
    for idx, candidate in enumerate(lines):
        match = _FIELD_PATTERN.match(candidate.text)
        if match:
            pairs.append(
                LabelValuePair(
                    label=match.group("field").strip(),
                    value=match.group("value").strip(),
                    page=candidate.page,
                    line_number=candidate.line_number,
                    source="inline_pattern",
                )
            )
            continue

        label_only = re.match(r"^([A-Za-z][A-Za-z0-9_ /-]{1,80})\s*[:=\-]\s*$", candidate.text)
        if label_only and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if next_line.page == candidate.page:
                pairs.append(
                    LabelValuePair(
                        label=label_only.group(1).strip(),
                        value=next_line.text.strip(),
                        page=candidate.page,
                        line_number=candidate.line_number,
                        source="next_line",
                    )
                )

    for block in blocks:
        if is_image_block(block):
            continue
        block_type = str(getattr(block, "block_type", "text") or "text")
        if block_type != "table" and block_component_type(block) != "table":
            continue
        page = getattr(block, "page", None)
        base_line = getattr(block, "line_number", None)
        for offset, line in enumerate(str(getattr(block, "text", "") or "").splitlines()):
            cleaned = clean_extracted_line(line)
            if not cleaned or cleaned.lower().startswith("table:"):
                continue
            cells = [cell.strip() for cell in re.split(r"\||\t", cleaned) if cell.strip()]
            if len(cells) == 2:
                pairs.append(
                    LabelValuePair(
                        label=cells[0],
                        value=cells[1],
                        page=int(page) if page is not None else None,
                        line_number=int(base_line + offset) if isinstance(base_line, int) else base_line,
                        source="table_row",
                    )
                )
    return pairs
