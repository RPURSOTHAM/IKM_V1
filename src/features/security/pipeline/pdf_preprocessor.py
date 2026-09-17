"""Layout-aware PDF/text preprocessor — strips noise from NER input, preserves confidentiality markers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.features.security.dlp.policy_loader import load_pdf_preprocessor_config


def _collect_markers(text: str, marker_list: list[str]) -> list[str]:
    found: list[str] = []
    for marker in marker_list:
        pattern = rf"(?i)\b{re.escape(marker)}\b"
        if re.search(pattern, text):
            found.append(marker.upper() if marker.isupper() else marker)
    # De-duplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _line_is_marker_only(line: str, markers: list[str]) -> bool:
    remaining = line
    for marker in markers:
        remaining = re.sub(rf"(?i)\b{re.escape(marker)}\b", "", remaining)
    remaining = re.sub(r"[\s\-|/.,]+", "", remaining)
    return not remaining.strip()


def _strip_marker_lines(text: str, markers: list[str]) -> str:
    if not markers:
        return text
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if _line_is_marker_only(stripped, markers):
            continue
        kept.append(line)
    return "\n".join(kept)


def _remove_page_numbers(text: str, patterns: list[str]) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        drop = False
        for pattern in patterns:
            try:
                if re.match(pattern, stripped):
                    drop = True
                    break
            except re.error:
                continue
        if not drop:
            kept.append(line)
    return "\n".join(kept)


def _remove_repeated_watermarks(text: str, min_count: int) -> tuple[str, list[str]]:
    """Drop lines repeated across many pages (typical watermark/header noise)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text, []
    counts = Counter(lines)
    repeated = {ln for ln, cnt in counts.items() if cnt >= min_count and len(ln.split()) <= 8}
    if not repeated:
        return text, []
    kept: list[str] = []
    removed: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in repeated:
            removed.append(stripped)
            continue
        kept.append(line)
    return "\n".join(kept), sorted(set(removed))


def _blocks_to_ner_text(blocks: list[Any], remove_types: set[str]) -> str:
    parts: list[str] = []
    for block in blocks:
        component = str(getattr(block, "component_type", "") or "").strip().lower()
        meta = getattr(block, "metadata", {}) or {}
        region = str(meta.get("region") or "").strip().lower()
        if component in remove_types or region in remove_types:
            continue
        text = str(getattr(block, "text", "") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def preprocess_for_ner(
    text: str,
    *,
    blocks: list[Any] | None = None,
    file_name: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare clean semantic text for NER while preserving confidentiality markers separately.

    Returns:
        ner_input_text: text with noise regions removed for NER
        confidentiality_markers: labels extracted from full document (not deleted from audit)
        document_metadata: augmented metadata including confidentiality_markers
        removed_regions: diagnostic list of stripped repeated/header/footer noise
    """
    cfg = load_pdf_preprocessor_config()
    marker_cfg = list(cfg.get("confidentiality_markers") or [])
    page_patterns = list(cfg.get("page_number_patterns") or [])
    remove_types = {str(t).lower() for t in (cfg.get("remove_component_types") or ["header", "footer"])}
    min_repeat = int(cfg.get("min_repeated_line_count") or 3)

    full_text = (text or "").strip()
    base_meta = dict(metadata or {})

    markers = _collect_markers(full_text, marker_cfg)

    if blocks:
        ner_source = _blocks_to_ner_text(blocks, remove_types)
        if not ner_source.strip():
            ner_source = full_text
    else:
        ner_source = full_text

    removed_regions: list[str] = []
    ner_source, repeated = _remove_repeated_watermarks(ner_source, min_repeat)
    removed_regions.extend(repeated)

    ner_source = _strip_marker_lines(ner_source, markers)
    ner_source = _remove_page_numbers(ner_source, page_patterns)

    # Collapse excessive blank lines
    ner_input = re.sub(r"\n{3,}", "\n\n", ner_source).strip()

    doc_metadata = {
        **base_meta,
        "confidentiality_markers": markers,
        "original_file_name": base_meta.get("original_file_name") or file_name,
    }

    return {
        "ner_input_text": ner_input,
        "confidentiality_markers": markers,
        "document_metadata": doc_metadata,
        "removed_regions": removed_regions,
        "strategy": "layout_aware_preprocessor",
    }
