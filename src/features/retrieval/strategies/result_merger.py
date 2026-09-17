"""Candidate merge utilities for hybrid retrieval."""

from __future__ import annotations

from typing import Any

from src.features.retrieval.strategies.models import MergedCandidate
from src.features.retrieval.domain.models import RetrievalCandidate


def _document_id_from_candidate(candidate: RetrievalCandidate) -> str | None:
    metadata = candidate.metadata or {}
    doc_id = metadata.get("document_id")
    return str(doc_id) if doc_id else None


def _section_from_candidate(candidate: RetrievalCandidate) -> str | None:
    section = candidate.section_name or (candidate.metadata or {}).get("section")
    return str(section) if section else None


def merge_candidates_by_chunk_id(
    vector_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
) -> list[MergedCandidate]:
    """Merge vector and BM25 lists by ``chunk_id``, preserving both source scores."""
    merged: dict[str, MergedCandidate] = {}

    def _upsert(candidate: RetrievalCandidate, *, source: str) -> None:
        chunk_id = candidate.chunk_id or candidate.identity()
        entry = merged.get(chunk_id)
        if entry is None:
            metadata = dict(candidate.metadata or {})
            entry = MergedCandidate(
                chunk_id=chunk_id,
                document_id=_document_id_from_candidate(candidate),
                text=candidate.text,
                page=candidate.page,
                section=_section_from_candidate(candidate),
                metadata=metadata,
                doc_name=candidate.doc_name or str(metadata.get("doc_name") or ""),
            )
            merged[chunk_id] = entry
        else:
            if not entry.document_id:
                entry.document_id = _document_id_from_candidate(candidate)
            if not entry.section:
                entry.section = _section_from_candidate(candidate)
            if not entry.doc_name and candidate.doc_name:
                entry.doc_name = candidate.doc_name
            entry.metadata.update(candidate.metadata or {})

        score = float(candidate.score)
        if source == "vector":
            entry.vector_score = score
        else:
            entry.bm25_score = score

    for candidate in vector_candidates:
        _upsert(candidate, source="vector")
    for candidate in bm25_candidates:
        _upsert(candidate, source="bm25")

    return list(merged.values())


def apply_normalized_scores(candidates: list[MergedCandidate]) -> None:
    """Attach min-max normalized vector/BM25 scores in-place."""
    from src.features.retrieval.strategies.normalization import min_max_normalize

    norm_vector = min_max_normalize([item.vector_score for item in candidates])
    norm_bm25 = min_max_normalize([item.bm25_score for item in candidates])
    for item, n_vector, n_bm25 in zip(candidates, norm_vector, norm_bm25):
        item.normalized_vector_score = n_vector
        item.normalized_bm25_score = n_bm25


def alpha_weighted_hybrid_score(
    *,
    normalized_vector_score: float | None,
    normalized_bm25_score: float | None,
    alpha: float,
) -> float:
    """Combine normalized dense/sparse scores: alpha*vector + (1-alpha)*bm25."""
    bounded_alpha = max(0.0, min(1.0, float(alpha)))
    vector_component = float(normalized_vector_score or 0.0)
    bm25_component = float(normalized_bm25_score or 0.0)
    if normalized_vector_score is None and normalized_bm25_score is not None:
        return bm25_component
    if normalized_bm25_score is None and normalized_vector_score is not None:
        return vector_component
    return (bounded_alpha * vector_component) + ((1.0 - bounded_alpha) * bm25_component)


def rank_merged_by_alpha(
    merged: list[MergedCandidate],
    *,
    alpha: float,
) -> list[tuple[MergedCandidate, float]]:
    ranked: list[tuple[MergedCandidate, float]] = []
    for item in merged:
        score = alpha_weighted_hybrid_score(
            normalized_vector_score=item.normalized_vector_score,
            normalized_bm25_score=item.normalized_bm25_score,
            alpha=alpha,
        )
        ranked.append((item, score))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked
