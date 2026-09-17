"""Data models for retrieval strategy execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.features.retrieval.schemas.retrieval_schemas import ChunkOut


@dataclass
class StrategyMetrics:
    """Timing and candidate-count metrics for a retrieval strategy run."""

    vector_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    merge_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0
    vector_candidates: int = 0
    bm25_candidates: int = 0
    merged_candidates: int = 0
    returned_candidates: int = 0

    def as_trace(self) -> dict[str, float]:
        return {
            "vector_search_ms": round(self.vector_search_ms, 2),
            "bm25_search_ms": round(self.bm25_search_ms, 2),
            "merge_ms": round(self.merge_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }

    def as_summary(self) -> dict[str, int]:
        return {
            "vector_candidates": self.vector_candidates,
            "bm25_candidates": self.bm25_candidates,
            "merged_candidates": self.merged_candidates,
            "returned_candidates": self.returned_candidates,
        }


@dataclass
class MergedCandidate:
    """Single chunk after vector/BM25 merge (pre- or post-fusion)."""

    chunk_id: str
    document_id: str | None
    text: str
    page: int | None
    section: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_name: str = ""
    vector_score: float | None = None
    bm25_score: float | None = None
    normalized_vector_score: float | None = None
    normalized_bm25_score: float | None = None
    rrf_score: float | None = None


@dataclass
class StrategyResult:
    """Output of a retrieval strategy search."""

    chunks: list[ChunkOut]
    metrics: StrategyMetrics
    pipeline_stages: list[str]
    total_candidates: int
    strategy: str
