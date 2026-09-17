"""Procedure step detection from content blocks under Procedure sections."""

from __future__ import annotations

import re
from typing import Any

from src.features.document_processing.extractors.sop_structure.helpers import (
    looks_like_heading_title,
    normalize_heading_text,
    walk_tree,
)

_PROCEDURE_KEYWORDS = (
    "procedure",
    "procedures",
    "method",
    "methods",
    "operating procedure",
    "steps",
    "instructions",
)

_EXIT_SECTION_KEYWORDS = (
    "purpose",
    "scope",
    "responsibility",
    "responsibilities",
    "roles",
    "approval",
    "approvals",
    "revision",
    "revision history",
    "reference",
    "references",
    "definition",
    "definitions",
    "annexure",
    "appendix",
)

# 5.1 text / 5.1.2 text / 1. text / 1) text  (dot/paren after number optional for decimals)
_DECIMAL_STEP = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)+|\d+)\s*(?:[.)]\s+|\s+)(?P<text>\S.+?)\s*$"
)
_LETTER_STEP = re.compile(
    r"^\s*(?P<num>[A-Za-z])\s*[.)]\s+(?P<text>\S.+?)\s*$"
)
_ROMAN_STEP = re.compile(
    r"^\s*(?P<num>[IVXLCDMivxlcdm]+)\s*[.)]\s+(?P<text>\S.+?)\s*$"
)
_ROMAN_VALUES = set("ivxlcdmIVXLCDM")


def _is_roman(token: str) -> bool:
    token = (token or "").strip()
    if not token or len(token) > 8:
        return False
    return all(ch in _ROMAN_VALUES for ch in token)


def _match_step(text: str) -> tuple[str, str, str] | None:
    """Return (step_number, step_text, detection_method) or None."""
    raw = str(text or "").strip()
    if not raw or len(raw) > 500:
        return None

    m = _DECIMAL_STEP.match(raw)
    if m:
        return m.group("num"), m.group("text").strip(), "numbered_decimal"

    m = _LETTER_STEP.match(raw)
    if m and len(m.group("num")) == 1:
        return m.group("num").upper(), m.group("text").strip(), "lettered"

    m = _ROMAN_STEP.match(raw)
    if m and _is_roman(m.group("num")):
        # Avoid treating single "I." as roman when it could be letter — still valid per spec
        return m.group("num").upper(), m.group("text").strip(), "roman"

    return None


def _collect_procedure_section_titles(template_data: dict[str, Any]) -> set[str]:
    titles: set[str] = set()
    for node in walk_tree(list(template_data.get("document_tree") or [])):
        if node.get("type") != "heading":
            continue
        text = str(node.get("text") or "").strip()
        if looks_like_heading_title(text, _PROCEDURE_KEYWORDS):
            titles.add(normalize_heading_text(text))
    for item in template_data.get("outline") or []:
        text = str(item.get("text") or "").strip()
        if looks_like_heading_title(text, _PROCEDURE_KEYWORDS):
            titles.add(normalize_heading_text(text))
    return titles


def _is_heading_block(block: dict[str, Any], known_headings: set[str]) -> bool:
    text = str(block.get("text") or "").strip()
    if not text:
        return False
    style = str(block.get("style") or "").lower()
    if style.startswith("heading") or style in {"title", "subtitle"}:
        return True
    component = str(block.get("component_type") or "").lower()
    if component in {"title", "subtitle", "heading"}:
        return True
    return normalize_heading_text(text) in known_headings


def extract_procedure_steps(
    template_data: dict[str, Any],
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect numbered / lettered / roman steps under procedure-like sections."""
    procedure_titles = _collect_procedure_section_titles(template_data)
    known_headings = {
        normalize_heading_text(h.get("text"))
        for h in (template_data.get("outline") or [])
        if isinstance(h, dict) and h.get("text")
    }
    for node in walk_tree(list(template_data.get("document_tree") or [])):
        if node.get("type") == "heading" and node.get("text"):
            known_headings.add(normalize_heading_text(node.get("text")))

    steps: list[dict[str, Any]] = []
    order = 0
    in_procedure = False
    current_section_title = ""
    current_section_level = 1
    # If no explicit procedure heading, still accept clear step patterns document-wide.
    allow_document_wide = not procedure_titles

    for block in content_blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue

        if _is_heading_block(block, known_headings):
            norm = normalize_heading_text(text)
            level = int(block.get("level") or 1)
            if looks_like_heading_title(text, _PROCEDURE_KEYWORDS) or norm in procedure_titles:
                in_procedure = True
                current_section_title = text
                current_section_level = level
                continue
            if looks_like_heading_title(text, _EXIT_SECTION_KEYWORDS):
                in_procedure = False
                current_section_title = text
                current_section_level = level
                continue
            # Numbered sub-headings under a procedure section are also steps
            matched_as_step = _match_step(text) if (in_procedure or allow_document_wide) else None
            if matched_as_step and (in_procedure or allow_document_wide):
                step_number, step_text, method = matched_as_step
                order += 1
                steps.append(
                    {
                        "step_number": step_number,
                        "text": step_text,
                        "order": order,
                        "page": block.get("page"),
                        "section_title": current_section_title if in_procedure else (current_section_title or text),
                        "section_level": current_section_level if in_procedure else level,
                        "detection_method": method,
                    }
                )
                if in_procedure:
                    # Keep parent procedure section title for linking
                    pass
                continue
            if in_procedure:
                current_section_title = text
                current_section_level = level
            else:
                current_section_title = text
                current_section_level = level
            continue

        if not in_procedure and not allow_document_wide:
            continue

        matched = _match_step(text)
        if not matched:
            continue

        step_number, step_text, method = matched
        # Prefer decimal nested steps (5.1) even outside explicit section if a procedure exists
        if not in_procedure and not allow_document_wide:
            continue
        if not in_procedure and allow_document_wide:
            # Skip lettered/roman noise outside procedure sections unless clearly numbered
            if method not in {"numbered_decimal", "lettered", "roman"}:
                continue

        order += 1
        steps.append(
            {
                "step_number": step_number,
                "text": step_text,
                "order": order,
                "page": block.get("page"),
                "section_title": current_section_title if in_procedure else (current_section_title or ""),
                "section_level": current_section_level if in_procedure else 1,
                "detection_method": method,
            }
        )

    return steps
