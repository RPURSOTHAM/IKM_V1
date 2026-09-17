"""Responsibility / roles section detection."""

from __future__ import annotations

import re
from typing import Any

from src.features.document_processing.extractors.sop_structure.helpers import (
    looks_like_heading_title,
    normalize_heading_text,
    walk_tree,
)

_ROLE_SECTION_KEYWORDS = (
    "responsibility",
    "responsibilities",
    "roles",
    "role",
    "roles and responsibilities",
)

_KNOWN_ROLES = (
    "qa",
    "qc",
    "quality assurance",
    "quality control",
    "production",
    "engineering",
    "warehouse",
    "manufacturing",
    "maintenance",
    "operator",
    "supervisor",
    "manager",
    "hse",
    "ehs",
    "store",
    "stores",
    "procurement",
    "validation",
)

_ROLE_LINE = re.compile(
    r"^\s*(?P<role>[A-Za-z][A-Za-z0-9/ &]{0,40}?)\s*[:\-–—]\s*(?P<desc>\S.+?)\s*$"
)
_ROLE_BULLET = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])\s*(?P<role>[A-Za-z][A-Za-z0-9/ &]{0,40}?)\s*[:\-–—]\s*(?P<desc>\S.+?)\s*$"
)


def _normalize_role(role: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(role or "").strip())
    # Preserve common acronyms
    upper = cleaned.upper()
    if upper in {"QA", "QC", "HSE", "EHS"}:
        return upper
    return cleaned


def _is_known_role(role: str) -> bool:
    norm = role.strip().lower()
    if not norm:
        return False
    if norm in _KNOWN_ROLES:
        return True
    if norm.upper() in {"QA", "QC", "HSE", "EHS"}:
        return True
    return any(k in norm for k in _KNOWN_ROLES if len(k) > 3)


def extract_responsibilities(
    template_data: dict[str, Any],
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outline = list(template_data.get("outline") or [])
    tree = list(template_data.get("document_tree") or [])
    section_titles = set()
    for node in walk_tree(tree):
        if node.get("type") == "heading" and looks_like_heading_title(
            str(node.get("text") or ""), _ROLE_SECTION_KEYWORDS
        ):
            section_titles.add(normalize_heading_text(node.get("text")))
    for item in outline:
        if looks_like_heading_title(str(item.get("text") or ""), _ROLE_SECTION_KEYWORDS):
            section_titles.add(normalize_heading_text(item.get("text")))

    results: list[dict[str, Any]] = []
    in_section = False
    current_heading = ""
    seen: set[tuple[str, str]] = set()

    for block in content_blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        norm = normalize_heading_text(text)
        style = str(block.get("style") or "").lower()
        is_heading = style.startswith("heading") or norm in section_titles

        if looks_like_heading_title(text, _ROLE_SECTION_KEYWORDS) or norm in section_titles:
            in_section = True
            current_heading = text
            continue

        if in_section and is_heading and not looks_like_heading_title(text, _ROLE_SECTION_KEYWORDS):
            # New major section ends responsibilities
            if looks_like_heading_title(
                text,
                ("purpose", "scope", "procedure", "approval", "revision", "reference", "definition"),
            ):
                in_section = False
                continue

        if not in_section and not section_titles:
            # Document-wide known role lines
            pass
        elif not in_section:
            continue

        m = _ROLE_BULLET.match(text) or _ROLE_LINE.match(text)
        if not m:
            # Bare role name on its own line followed by description is hard; accept known roles alone
            if _is_known_role(text) and len(text) <= 40:
                role = _normalize_role(text)
                key = (role.lower(), "")
                if key not in seen:
                    seen.add(key)
                    results.append(
                        {
                            "role": role,
                            "description": "",
                            "page": block.get("page"),
                            "section_title": current_heading,
                        }
                    )
            continue

        role = _normalize_role(m.group("role"))
        desc = m.group("desc").strip()
        if not in_section and not _is_known_role(role):
            continue
        key = (role.lower(), desc.lower())
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "role": role,
                "description": desc,
                "page": block.get("page"),
                "section_title": current_heading,
            }
        )

    return results
