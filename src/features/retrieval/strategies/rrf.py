"""Reciprocal Rank Fusion for dense + sparse retrieval lists."""

from __future__ import annotations

from typing import Iterable

from src.features.retrieval.domain.models import RetrievalCandidate


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[RetrievalCandidate]],
    *,
    k: int = 60,
    source_names: list[str] | None = None,
) -> list[RetrievalCandidate]:
    """Merge ranked candidate lists with classic RRF: score = Σ 1/(k + rank)."""
    lists = list(ranked_lists)
    names = source_names or [f"list_{idx}" for idx in range(len(lists))]
    fused: dict[str, RetrievalCandidate] = {}
    rrf_scores: dict[str, float] = {}

    for list_idx, candidates in enumerate(lists):
        source = names[list_idx] if list_idx < len(names) else f"list_{list_idx}"
        for rank, candidate in enumerate(candidates, start=1):
            identity = candidate.identity()
            contribution = 1.0 / (float(k) + float(rank))
            existing = fused.get(identity)
            if existing is None:
                existing = RetrievalCandidate(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    doc_name=candidate.doc_name,
                    section_name=candidate.section_name,
                    page=candidate.page,
                    score=0.0,
                    source_ranks=dict(candidate.source_ranks),
                    source_scores=dict(candidate.source_scores),
                    metadata=dict(candidate.metadata),
                    retrieval_signals=dict(candidate.retrieval_signals),
                )
                fused[identity] = existing
            existing.source_ranks[source] = rank
            existing.source_scores[source] = float(candidate.score)
            rrf_scores[identity] = rrf_scores.get(identity, 0.0) + contribution

    ordered = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    results: list[RetrievalCandidate] = []
    for identity, score in ordered:
        candidate = fused[identity]
        candidate.score = float(score)
        signals = dict(candidate.retrieval_signals)
        signals["rrf_score"] = round(float(score), 6)
        signals["fused_score"] = round(float(score), 6)
        for source, rank in candidate.source_ranks.items():
            signals[f"{source}_rank"] = float(rank)
        for source, source_score in candidate.source_scores.items():
            signals[f"{source}_score"] = round(float(source_score), 4)
        candidate.retrieval_signals = signals
        results.append(candidate)
    return results
