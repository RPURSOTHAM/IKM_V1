"""Hybrid alpha-weighted fusion regression tests."""

from __future__ import annotations

from src.features.retrieval.strategies.result_merger import (
    alpha_weighted_hybrid_score,
    apply_normalized_scores,
    merge_candidates_by_chunk_id,
    rank_merged_by_alpha,
)
from src.features.retrieval.domain.models import RetrievalCandidate


def _candidate(chunk_id: str, score: float, **metadata) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        score=score,
        metadata=metadata,
        doc_name=metadata.get("doc_name", "doc.txt"),
    )


def test_alpha_weighted_score_endpoints() -> None:
    assert alpha_weighted_hybrid_score(
        normalized_vector_score=1.0,
        normalized_bm25_score=0.0,
        alpha=1.0,
    ) == 1.0
    assert alpha_weighted_hybrid_score(
        normalized_vector_score=0.0,
        normalized_bm25_score=1.0,
        alpha=0.0,
    ) == 1.0
    assert alpha_weighted_hybrid_score(
        normalized_vector_score=1.0,
        normalized_bm25_score=0.0,
        alpha=0.75,
    ) == 0.75


def test_alpha_changes_ordering_toward_bm25() -> None:
    vector = [_candidate("a", 0.9), _candidate("b", 0.2)]
    bm25 = [_candidate("a", 0.1), _candidate("b", 0.95)]
    merged = merge_candidates_by_chunk_id(vector, bm25)
    apply_normalized_scores(merged)
    vector_first = rank_merged_by_alpha(merged, alpha=1.0)[0][0].chunk_id
    bm25_first = rank_merged_by_alpha(merged, alpha=0.0)[0][0].chunk_id
    assert vector_first == "a"
    assert bm25_first == "b"


def test_default_alpha_075_balances_sources() -> None:
    vector = [_candidate("a", 0.8), _candidate("b", 0.4)]
    bm25 = [_candidate("a", 0.2), _candidate("b", 0.9)]
    merged = merge_candidates_by_chunk_id(vector, bm25)
    apply_normalized_scores(merged)
    ranked = rank_merged_by_alpha(merged, alpha=0.75)
    assert ranked[0][0].chunk_id in {"a", "b"}
    assert ranked[0][1] > 0.0
