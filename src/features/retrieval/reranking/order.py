"""Rerank-score ordering helpers for final retrieval responses."""

from __future__ import annotations

import logging
from typing import Any

from src.features.retrieval.schemas.retrieval_schemas import ChunkOut

_logger = logging.getLogger(__name__)


def chunk_rerank_score(chunk: ChunkOut) -> float:
    """Return the raw cross-encoder rerank score used for final ordering."""
    signals = chunk.retrieval_signals if isinstance(chunk.retrieval_signals, dict) else {}
    raw = signals.get("rerank_score")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
    meta_score = metadata.get("rerank_score")
    if meta_score is not None:
        try:
            return float(meta_score)
        except (TypeError, ValueError):
            pass
    return float("-inf")


def sort_chunks_by_rerank_score(chunks: list[ChunkOut]) -> list[ChunkOut]:
    """Sort chunks by raw ``rerank_score`` descending (stable for ties)."""
    indexed = list(enumerate(chunks))
    indexed.sort(
        key=lambda pair: (chunk_rerank_score(pair[1]), -pair[0]),
        reverse=True,
    )
    return [chunk for _, chunk in indexed]


def rerank_order_is_monotonic(chunks: list[ChunkOut]) -> bool:
    """True when each result has rerank_score >= the next (when scores exist)."""
    scores = [chunk_rerank_score(chunk) for chunk in chunks]
    if not scores or all(score == float("-inf") for score in scores):
        return True
    for left, right in zip(scores, scores[1:]):
        if left < right:
            return False
    return True


def log_final_rerank_order(chunks: list[ChunkOut], *, query: str | None = None) -> None:
    """Log final response order for rerank verification."""
    for rank, chunk in enumerate(chunks, start=1):
        signals = chunk.retrieval_signals if isinstance(chunk.retrieval_signals, dict) else {}
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        _logger.info(
            "Final Rank=%d rerank_score=%s bm25_score=%s original_score=%s "
            "section=%s chunk_id=%s query=%r",
            rank,
            signals.get("rerank_score", metadata.get("rerank_score")),
            signals.get("bm25_score"),
            signals.get("original_score", signals.get("score")),
            chunk.section_name or metadata.get("section"),
            chunk.chunk_id,
            query,
        )


def finalize_reranked_chunks(
    chunks: list[ChunkOut],
    *,
    top_k: int | None,
    rerank_applied: bool,
    query: str | None = None,
) -> list[ChunkOut]:
    """Enforce rerank-score ordering and apply final top-K after reranking."""
    if not chunks:
        return []
    if not rerank_applied:
        return chunks[:top_k] if top_k is not None else list(chunks)

    ordered = sort_chunks_by_rerank_score(chunks)
    if top_k is not None:
        ordered = ordered[: max(0, int(top_k))]
    if not rerank_order_is_monotonic(ordered):
        _logger.warning(
            "Reranked chunk order was not monotonic by rerank_score; re-sorted query=%r",
            query,
        )
        ordered = sort_chunks_by_rerank_score(ordered)
    log_final_rerank_order(ordered, query=query)
    return ordered
