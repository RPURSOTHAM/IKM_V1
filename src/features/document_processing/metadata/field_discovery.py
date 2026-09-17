"""Discover structured label/value pairs from document content without a field catalog.

Uses existing ``build_label_value_pairs``. Validation is generic (shape/quality),
not an allowlist of business labels.
"""

from __future__ import annotations

import re
from typing import Any

from src.features.document_processing.metadata.candidates import LabelValuePair, build_label_value_pairs
from src.features.logical_folders.application.hierarchy import normalize_folder_name

_SENTENCE_PUNCT = re.compile(r"[.?!;]")
_WORD_CHARS = re.compile(r"[A-Za-z0-9]")
_MULTI_SPACE = re.compile(r"\s+")
_MAX_VALUE_WORDS = 8


def normalize_discovered_field_name(label: Any) -> str | None:
    text = _MULTI_SPACE.sub(" ", str(label or "").strip())
    if not text:
        return None
    if len(text) > 64:
        return None
    if _SENTENCE_PUNCT.search(text):
        return None
    words = text.split()
    if not words or len(words) > 6:
        return None
    if not _WORD_CHARS.search(text):
        return None
    if sum(1 for ch in text if ch.isalpha()) < 2:
        return None
    return text[:64]


def _usable_value(raw: Any) -> str | None:
    name = normalize_folder_name(raw)
    if not name:
        return None
    if _SENTENCE_PUNCT.search(name) and len(name.split()) > 3:
        return None
    if len(name.split()) > _MAX_VALUE_WORDS:
        return None
    if len(name) > 256:
        name = name[:256].strip()
    if not name:
        return None
    return name


def discover_structured_fields(blocks: list[Any]) -> list[dict[str, Any]]:
    """Return extracted fields in document order. First occurrence of a label wins."""
    pairs = build_label_value_pairs(blocks)
    return fields_from_label_value_pairs(pairs)


def fields_from_label_value_pairs(pairs: list[LabelValuePair]) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        field_name = normalize_discovered_field_name(pair.label)
        value = _usable_value(pair.value)
        if not field_name or not value:
            continue
        if field_name.lower() == value.lower():
            continue
        key = field_name.lower()
        if key in seen:
            continue
        seen.add(key)
        discovered.append(
            {
                "field_name": field_name,
                "value": value,
                "confidence": 0.9,
                "extraction_method": "discovered_label_value",
                "source": pair.source,
            }
        )
    return discovered


def merge_extracted_with_discovered(
    extracted: list[dict[str, Any]] | None,
    discovered: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep configured extraction results; append discovered names not already present.

    Empty configured values stay in the result unless a discovered pair fills them.
    """
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    pending_empty: dict[str, dict[str, Any]] = {}

    def _name(item: dict[str, Any]) -> str:
        return str(item.get("field_name") or item.get("name") or "").strip()

    def _has_value(item: dict[str, Any]) -> bool:
        return item.get("value") is not None and str(item.get("value") or "").strip() != ""

    for item in extracted or []:
        if not isinstance(item, dict):
            continue
        name = _name(item)
        if not name:
            continue
        key = name.lower()
        if not _has_value(item):
            pending_empty.setdefault(key, item)
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    for item in discovered or []:
        if not isinstance(item, dict):
            continue
        name = _name(item)
        if not name or not _has_value(item):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        pending_empty.pop(key, None)
        merged.append(item)

    for key, item in pending_empty.items():
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged
