"""Multi-technique metadata field extraction from document blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

import numpy as np

from src.features.document_processing.metadata.candidates import LabelValuePair, LineCandidate, build_label_value_pairs, build_line_candidates
from src.features.document_processing.metadata.field_definitions import (
    SKIP_CONTENT_EXTRACTION,
    MetadataFieldDefinition,
    normalize_field_definitions,
)
from src.features.document_processing.metadata.type_coercion import coerce_metadata_value
from src.features.document_processing.utilities.extraction_filters import clean_extracted_line

_VALUE_SEPARATOR = r"\s*(?:\:|=|\|\s*)\s*"
_FIELD_PATTERN = re.compile(rf"(?P<field>[A-Za-z][A-Za-z0-9_ /-]{{1,80}}){_VALUE_SEPARATOR}(?P<value>.+)")
_TITLE_FIELD_NAMES = frozenset({"title", "document_name", "document_title"})
_TITLE_SEPARATOR_ONLY_RE = re.compile(r"^[\s\-_:|/\\.…·•–—~=+*#@]+$")
# Accept Unicode letters/digits (and underscore) — not ASCII-only Latin alnum.
_TITLE_HAS_ALNUM_RE = re.compile(r"\w", re.UNICODE)


@dataclass(frozen=True)
class FieldExtractionHit:
    value: Any
    confidence: float
    method: str


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _truncate(value: str, max_length: int | None) -> str:
    text = str(value or "").strip()
    if max_length is not None and max_length > 0:
        return text[:max_length]
    return text[:512]


def _label_match_score(label: str, target: str) -> float:
    left = _normalize_key(label)
    right = _normalize_key(target)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.92
    return SequenceMatcher(None, left, right).ratio()


def _best_label_match(field: MetadataFieldDefinition, label: str) -> float:
    return max(_label_match_score(alias, label) for alias in field.searchable_labels)


def _extract_by_rules(full_text: str, field: MetadataFieldDefinition) -> FieldExtractionHit | None:
    for alias in field.searchable_labels:
        label = re.escape(alias.replace("_", " "))
        underscore = re.escape(alias)
        match = re.search(
            rf"(?:{label}|{underscore}){_VALUE_SEPARATOR}(.+)",
            full_text,
            flags=re.IGNORECASE,
        )
        if match:
            raw = match.group(1).strip().splitlines()[0]
            coerced = coerce_metadata_value(raw, field.data_type, enum_values=field.enum_values)
            if coerced is not None:
                return FieldExtractionHit(value=coerced, confidence=0.93, method="rule_pattern")
    for match in _FIELD_PATTERN.finditer(full_text):
        if _best_label_match(field, match.group("field")) >= 0.88:
            raw = match.group("value").strip().splitlines()[0]
            coerced = coerce_metadata_value(raw, field.data_type, enum_values=field.enum_values)
            if coerced is not None:
                return FieldExtractionHit(value=coerced, confidence=0.90, method="rule_scan")
    return None


def _extract_from_pairs(pairs: list[LabelValuePair], field: MetadataFieldDefinition) -> FieldExtractionHit | None:
    best: FieldExtractionHit | None = None
    for pair in pairs:
        score = _best_label_match(field, pair.label)
        if score < 0.82:
            continue
        coerced = coerce_metadata_value(pair.value, field.data_type, enum_values=field.enum_values)
        if coerced is None:
            continue
        confidence = min(0.95, 0.78 + (score * 0.15))
        method = {
            "inline_pattern": "pattern_pair",
            "next_line": "adjacent_line",
            "table_row": "table_cell",
        }.get(pair.source, "label_pair")
        hit = FieldExtractionHit(value=coerced, confidence=confidence, method=method)
        if best is None or hit.confidence > best.confidence:
            best = hit
    return best


def _extract_by_fuzzy(lines: list[LineCandidate], field: MetadataFieldDefinition) -> FieldExtractionHit | None:
    best: FieldExtractionHit | None = None
    for candidate in lines:
        match = _FIELD_PATTERN.match(candidate.text)
        if not match:
            continue
        score = _best_label_match(field, match.group("field"))
        if score < 0.78:
            continue
        coerced = coerce_metadata_value(match.group("value"), field.data_type, enum_values=field.enum_values)
        if coerced is None:
            continue
        hit = FieldExtractionHit(value=coerced, confidence=min(0.88, 0.65 + score * 0.2), method="fuzzy_label")
        if best is None or hit.confidence > best.confidence:
            best = hit
    return best


def _extract_by_embedding(
    lines: list[LineCandidate],
    field: MetadataFieldDefinition,
    embed_fn: Callable[[list[str]], np.ndarray],
) -> FieldExtractionHit | None:
    if not lines:
        return None
    labels = list(field.searchable_labels)
    if not labels:
        return None
    line_texts = [candidate.text for candidate in lines]
    vectors = embed_fn(labels + line_texts)
    label_vectors = vectors[: len(labels)]
    line_vectors = vectors[len(labels) :]
    best_idx = -1
    best_score = 0.0
    for idx, line_vector in enumerate(line_vectors):
        for label_vector in label_vectors:
            denom = float(np.linalg.norm(label_vector) * np.linalg.norm(line_vector))
            score = float(np.dot(label_vector, line_vector) / denom) if denom else 0.0
            if score > best_score:
                best_score = score
                best_idx = idx
    if best_idx < 0 or best_score < 0.55:
        return None

    candidate = lines[best_idx]
    inline = _FIELD_PATTERN.match(candidate.text)
    raw_value = inline.group("value") if inline else candidate.text
    coerced = coerce_metadata_value(raw_value, field.data_type, enum_values=field.enum_values)
    if coerced is None:
        return None
    return FieldExtractionHit(
        value=coerced,
        confidence=min(0.82, 0.45 + best_score * 0.35),
        method="embedding_similarity",
    )


def _is_usable_document_title_value(text: str | None) -> bool:
    """Reject punctuation-only title values while keeping titles that contain punctuation."""
    cleaned = clean_extracted_line(str(text or ""))
    if not cleaned:
        return False
    if _TITLE_SEPARATOR_ONLY_RE.fullmatch(cleaned):
        return False
    if not _TITLE_HAS_ALNUM_RE.search(cleaned):
        return False
    return True


def _accept_title_hit(field: MetadataFieldDefinition, hit: FieldExtractionHit | None) -> FieldExtractionHit | None:
    if hit is None:
        return None
    if field.field_name != "document_title":
        return hit
    if isinstance(hit.value, str) and not _is_usable_document_title_value(hit.value):
        return None
    return hit


def _extract_title_heuristic(lines: list[LineCandidate], field: MetadataFieldDefinition) -> FieldExtractionHit | None:
    if field.field_name not in _TITLE_FIELD_NAMES:
        return None

    # Prefer layout-marked title/subtitle blocks; merge adjacent same-page fragments.
    merged_candidates: list[str] = []
    pending: list[str] = []
    pending_page: int | None = None

    def flush_pending() -> None:
        nonlocal pending, pending_page
        if pending:
            merged = clean_extracted_line(" ".join(pending))
            if _is_usable_document_title_value(merged):
                merged_candidates.append(merged)
        pending = []
        pending_page = None

    for candidate in lines[:20]:
        text = clean_extracted_line(candidate.text)
        if candidate.component_type not in {"title", "subtitle"}:
            flush_pending()
            continue
        if not _is_usable_document_title_value(text):
            continue
        if pending and candidate.page == pending_page and len(pending[-1].split()) <= 8 and len(text.split()) <= 8:
            pending.append(text)
            continue
        flush_pending()
        pending = [text]
        pending_page = candidate.page
    flush_pending()

    for text in merged_candidates:
        if field.field_name == "document_title" or len(text.split()) >= 2 or len(text) >= 8:
            return FieldExtractionHit(
                value=_truncate(text, field.max_length),
                confidence=0.84,
                method="title_block",
            )

    # Fallback for document_title: first meaningful early line that is not a bare label.
    if field.field_name == "document_title":
        for candidate in lines[:12]:
            text = clean_extracted_line(candidate.text)
            if not _is_usable_document_title_value(text):
                continue
            match = _FIELD_PATTERN.match(text)
            if match:
                if _best_label_match(field, match.group("field")) >= 0.78:
                    value = clean_extracted_line(match.group("value"))
                    if _is_usable_document_title_value(value):
                        return FieldExtractionHit(
                            value=_truncate(value, field.max_length),
                            confidence=0.8,
                            method="title_inline",
                        )
                continue
            if re.match(r"^[A-Za-z][A-Za-z0-9_ /-]{1,80}\s*[:=\-]\s*$", text):
                continue
            if len(text.split()) >= 2 or (candidate.component_type in {"title", "subtitle"} and len(text) >= 4):
                return FieldExtractionHit(
                    value=_truncate(text, field.max_length),
                    confidence=0.72,
                    method="title_early_line",
                )
    return None


def extract_metadata_fields(
    blocks: list[Any],
    fields: list[dict[str, Any]],
    *,
    extraction_model: dict[str, Any] | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Extract metadata values using rule, structural, fuzzy, and optional embedding techniques."""
    definitions = normalize_field_definitions(fields)
    if not definitions:
        return []

    lines = build_line_candidates(blocks)
    pairs = build_label_value_pairs(blocks)
    full_text = "\n".join(candidate.text for candidate in lines)
    model_hint = (extraction_model or {}).get("model_id") or (extraction_model or {}).get("provider")

    results: list[dict[str, Any]] = []
    unresolved: list[MetadataFieldDefinition] = []

    for field_def in definitions:
        if field_def.field_name.lower() in SKIP_CONTENT_EXTRACTION:
            results.append(
                {
                    "field_name": field_def.field_name,
                    "metadata_field_id": field_def.metadata_field_id,
                    "value": None,
                    "confidence": 0.0,
                    "extraction_method": "system_managed",
                    "extraction_model": model_hint or "none",
                }
            )
            continue

        hit = (
            _accept_title_hit(field_def, _extract_by_rules(full_text, field_def))
            or _accept_title_hit(field_def, _extract_from_pairs(pairs, field_def))
            or _accept_title_hit(field_def, _extract_title_heuristic(lines, field_def))
            or _accept_title_hit(field_def, _extract_by_fuzzy(lines, field_def))
        )
        if hit is None:
            unresolved.append(field_def)
            continue

        value = hit.value
        if isinstance(value, str):
            value = _truncate(value, field_def.max_length)
        results.append(
            {
                "field_name": field_def.field_name,
                "metadata_field_id": field_def.metadata_field_id,
                "value": value,
                "confidence": round(hit.confidence, 3),
                "extraction_method": hit.method,
                "extraction_model": model_hint or hit.method,
            }
        )

    if unresolved and embed_fn is not None:
        for field_def in unresolved:
            hit = _extract_by_embedding(lines, field_def, embed_fn)
            if hit is None:
                results.append(
                    {
                        "field_name": field_def.field_name,
                        "metadata_field_id": field_def.metadata_field_id,
                        "value": None,
                        "confidence": 0.0,
                        "extraction_method": "none",
                        "extraction_model": model_hint or "embedding",
                    }
                )
                continue
            value = hit.value
            if isinstance(value, str):
                value = _truncate(value, field_def.max_length)
            results.append(
                {
                    "field_name": field_def.field_name,
                    "metadata_field_id": field_def.metadata_field_id,
                    "value": value,
                    "confidence": round(hit.confidence, 3),
                    "extraction_method": hit.method,
                    "extraction_model": model_hint or "embedding",
                }
            )
    else:
        for field_def in unresolved:
            results.append(
                {
                    "field_name": field_def.field_name,
                    "metadata_field_id": field_def.metadata_field_id,
                    "value": None,
                    "confidence": 0.0,
                    "extraction_method": "none",
                    "extraction_model": model_hint or "rule_based",
                }
            )

    return results
