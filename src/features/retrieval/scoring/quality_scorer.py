"""Chunk-level grounding, quality, and highlight enrichment for retrieval results.

Quality / grounding formulas (deterministic, bounded to [0, 1]):

grounding_score
    Independent of retrieval/rerank score.
    = 0.70 * query_term_coverage
    + 0.20 * lexical_hit_density (clamped)
    + 0.10 * citation_evidence_presence
    Fallback when query_terms empty: 0.0 (no lexical evidence to ground against).

quality_score
    Independent of raw retrieval ranking; starts from structural chunk quality.
    = 0.40 * text_completeness
    + 0.30 * metadata_completeness
    + 0.30 * evidence_support
    Penalties: duplicate (-0.40), OCR uncertainty (-0.15).
    retrieval_score is NOT copied; it is only used as a weak optional prior (max 0.10 weight)
    when evidence is present so ranking quality and chunk quality stay distinct.

quality_flags (stable vocabulary, evidence-based only)
    missing_text, incomplete_chunk, missing_metadata, weak_match,
    low_score, low_grounding, duplicate_result
"""

from __future__ import annotations

import re
from typing import Any

# Stable flag vocabulary required by retrieval/catalog audit contract.
QUALITY_FLAG_MISSING_TEXT = "missing_text"
QUALITY_FLAG_INCOMPLETE_CHUNK = "incomplete_chunk"
QUALITY_FLAG_MISSING_METADATA = "missing_metadata"
QUALITY_FLAG_WEAK_MATCH = "weak_match"
QUALITY_FLAG_LOW_SCORE = "low_score"
QUALITY_FLAG_LOW_GROUNDING = "low_grounding"
QUALITY_FLAG_DUPLICATE_RESULT = "duplicate_result"


def compute_highlight_spans(text: str, query_terms: list[str]) -> list[tuple[int, int]]:
    """Return character spans for query-term matches (case-insensitive, word boundaries)."""
    if not text or not query_terms:
        return []
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for term in query_terms:
        token = str(term or "").strip()
        if len(token) < 2:
            continue
        pattern = re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span not in seen:
                seen.add(span)
                spans.append(span)
    spans.sort(key=lambda item: item[0])
    return spans


def _coverage(evidence_terms: list[str], query_terms: list[str]) -> float:
    required = {term.lower() for term in query_terms if len(str(term)) >= 2}
    if not required:
        return 0.0
    matched = {str(term).lower() for term in evidence_terms}
    return len(matched & required) / len(required)


def compute_grounding_score(
    *,
    retrieval_score: float,
    evidence_terms: list[str],
    query_terms: list[str],
    text: str = "",
    metadata: dict[str, Any] | None = None,
) -> float:
    """Evidence-based grounding; deliberately not an alias of retrieval_score.

    ``retrieval_score`` is accepted for API compatibility but is not used in the
    primary calculation so grounding remains independently auditable.
    """
    _ = retrieval_score  # intentionally unused — independence from ranking score
    meta = metadata or {}
    if not query_terms:
        # No query terms → no lexical grounding evidence.
        return 0.0

    coverage = _coverage(evidence_terms, query_terms)
    highlights = compute_highlight_spans(text or "", query_terms)
    density = min(1.0, len(highlights) / max(len(query_terms), 1))
    citation_present = 1.0 if (
        meta.get("citation_anchor")
        or meta.get("citation")
        or meta.get("page") is not None
        or meta.get("section")
        or meta.get("section_name")
    ) else 0.0

    grounded = (0.70 * coverage) + (0.20 * density) + (0.10 * citation_present)
    return round(max(0.0, min(1.0, grounded)), 4)


def compute_quality_score(
    *,
    retrieval_score: float,
    text: str,
    metadata: dict[str, Any],
    evidence_terms: list[str],
    query_terms: list[str],
) -> float:
    """Structural/evidence quality; not a copy of retrieval or rerank score."""
    stripped = (text or "").strip()
    if not stripped:
        text_completeness = 0.0
    elif len(stripped) < 40:
        text_completeness = 0.45
    elif len(stripped) < 120:
        text_completeness = 0.75
    else:
        text_completeness = 1.0

    meta_bits = 0
    if metadata.get("page") is not None:
        meta_bits += 1
    if metadata.get("line_start") is not None or metadata.get("line_range"):
        meta_bits += 1
    if metadata.get("section") or metadata.get("section_name") or metadata.get("heading"):
        meta_bits += 1
    metadata_completeness = meta_bits / 3.0

    if not query_terms:
        evidence_support = 0.5
    else:
        evidence_support = _coverage(evidence_terms, query_terms)

    # Optional weak prior from retrieval so empty-evidence high-rank chunks are
    # not scored identically to strong-evidence ones, without aliasing score.
    prior = 0.0
    try:
        prior = max(0.0, min(1.0, float(retrieval_score))) * 0.10
    except (TypeError, ValueError):
        prior = 0.0

    score = (
        (0.40 * text_completeness)
        + (0.30 * metadata_completeness)
        + (0.30 * evidence_support)
        + prior
    )
    if metadata.get("is_duplicate"):
        score -= 0.40
    ocr_confidence = metadata.get("ocr_confidence")
    if ocr_confidence is not None:
        try:
            if float(ocr_confidence) < 0.5:
                score -= 0.15
        except (TypeError, ValueError):
            pass
    return round(max(0.0, min(1.0, score)), 4)


def compute_quality_flags(
    *,
    text: str,
    metadata: dict[str, Any],
    evidence_terms: list[str],
    query_terms: list[str],
    retrieval_score: float | None = None,
    grounding_score: float | None = None,
) -> list[str]:
    """Return evidence-based quality flags from the stable vocabulary only."""
    flags: list[str] = []
    stripped = (text or "").strip()
    if not stripped:
        flags.append(QUALITY_FLAG_MISSING_TEXT)
    elif len(stripped) < 40:
        flags.append(QUALITY_FLAG_INCOMPLETE_CHUNK)

    if not metadata.get("page") and not metadata.get("line_start") and not (
        metadata.get("section") or metadata.get("section_name")
    ):
        flags.append(QUALITY_FLAG_MISSING_METADATA)

    if query_terms and not evidence_terms:
        flags.append(QUALITY_FLAG_WEAK_MATCH)

    if metadata.get("is_duplicate"):
        flags.append(QUALITY_FLAG_DUPLICATE_RESULT)

    if retrieval_score is not None:
        try:
            if float(retrieval_score) < 0.25:
                flags.append(QUALITY_FLAG_LOW_SCORE)
        except (TypeError, ValueError):
            pass

    if grounding_score is not None:
        try:
            if float(grounding_score) < 0.5 and query_terms:
                flags.append(QUALITY_FLAG_LOW_GROUNDING)
        except (TypeError, ValueError):
            pass

    # Stable order for deterministic output.
    order = [
        QUALITY_FLAG_MISSING_TEXT,
        QUALITY_FLAG_INCOMPLETE_CHUNK,
        QUALITY_FLAG_MISSING_METADATA,
        QUALITY_FLAG_WEAK_MATCH,
        QUALITY_FLAG_LOW_SCORE,
        QUALITY_FLAG_LOW_GROUNDING,
        QUALITY_FLAG_DUPLICATE_RESULT,
    ]
    return [flag for flag in order if flag in flags]


def enrich_chunk_scores(
    *,
    text: str,
    retrieval_score: float,
    metadata: dict[str, Any],
    query_terms: list[str],
) -> tuple[list[tuple[int, int]], float, float, list[str]]:
    evidence_terms = metadata.get("query_evidence_terms") or []
    if not isinstance(evidence_terms, list):
        evidence_terms = []
    evidence_terms = [str(term) for term in evidence_terms]
    highlights = compute_highlight_spans(text, query_terms)
    grounding = compute_grounding_score(
        retrieval_score=retrieval_score,
        evidence_terms=evidence_terms,
        query_terms=query_terms,
        text=text,
        metadata=metadata,
    )
    quality = compute_quality_score(
        retrieval_score=retrieval_score,
        text=text,
        metadata=metadata,
        evidence_terms=evidence_terms,
        query_terms=query_terms,
    )
    flags = compute_quality_flags(
        text=text,
        metadata=metadata,
        evidence_terms=evidence_terms,
        query_terms=query_terms,
        retrieval_score=retrieval_score,
        grounding_score=grounding,
    )
    return highlights, grounding, quality, flags
