"""Score normalization helpers for hybrid retrieval fusion.

Normalization approach (min-max per source list):
- Vector similarity scores are min-max scaled to [0, 1] across the vector candidate set.
  When Weaviate returns distance instead of score, similarity is computed as max(0, 1 - distance)
  before normalization.
- BM25 scores are min-max scaled to [0, 1] across the BM25 candidate set.

Original raw scores are never modified; normalized values are stored separately in
``normalized_vector_score`` and ``normalized_bm25_score`` for downstream inspection.
RRF ranking uses rank positions from the original per-source orderings, not normalized scores.
"""

from __future__ import annotations


def min_max_normalize(values: list[float | None]) -> list[float | None]:
    """Scale non-null values to [0, 1]; return None for missing entries."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return [None for _ in values]
    lo = min(present)
    hi = max(present)
    if hi == lo:
        return [1.0 if v is not None else None for v in values]
    span = hi - lo
    return [((float(v) - lo) / span) if v is not None else None for v in values]


def distance_to_similarity(distance: float | None) -> float | None:
    if distance is None:
        return None
    return max(0.0, min(1.0, 1.0 - float(distance)))
