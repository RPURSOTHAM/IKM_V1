"""Reranker interface and result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.features.retrieval.schemas.retrieval_schemas import ChunkOut


@dataclass
class RerankMetrics:
    rerank_ms: float = 0.0
    batch_count: int = 0
    candidate_count: int = 0
    average_score: float = 0.0
    model_load_time_ms: float = 0.0
    model: str = ""
    batch_size: int = 0
    top_score: float = 0.0

    def as_trace(self) -> dict[str, float | int | str]:
        return {
            "rerank_ms": round(self.rerank_ms, 2),
            "batch_count": self.batch_count,
            "candidate_count": self.candidate_count,
            "average_score": round(self.average_score, 4),
            "model_load_time_ms": round(self.model_load_time_ms, 2),
            "rerank_top_score": round(self.top_score, 4),
        }


@dataclass
class RerankResult:
    chunks: list[ChunkOut]
    applied: bool
    metrics: RerankMetrics = field(default_factory=RerankMetrics)


class Reranker(Protocol):
    """Rerank fused retrieval candidates without performing additional retrieval."""

    def rerank(
        self,
        query: str,
        candidates: list[ChunkOut],
        *,
        top_k: int | None = None,
    ) -> RerankResult: ...
