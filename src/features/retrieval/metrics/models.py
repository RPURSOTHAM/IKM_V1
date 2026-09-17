"""Retrieval metrics data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LatencyMetrics:
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    merge_ms: float = 0.0
    normalization_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    dlp_ms: float = 0.0
    serialization_ms: float = 0.0
    total_ms: float = 0.0

    def pipeline_trace(self, *, debug: bool = False) -> dict[str, float]:
        full = {
            "embedding_ms": round(self.embedding_ms, 2),
            "vector_search_ms": round(self.vector_search_ms, 2),
            "bm25_search_ms": round(self.bm25_search_ms, 2),
            "merge_ms": round(self.merge_ms, 2),
            "normalization_ms": round(self.normalization_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "dlp_ms": round(self.dlp_ms, 2),
            "serialization_ms": round(self.serialization_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }
        if debug:
            return full
        return {
            "total_ms": full["total_ms"],
            "vector_search_ms": full["vector_search_ms"],
            "bm25_search_ms": full["bm25_search_ms"],
            "merge_ms": full["merge_ms"],
            "fusion_ms": full["fusion_ms"],
            "rerank_ms": full["rerank_ms"],
        }


@dataclass
class QualityMetrics:
    average_vector_score: float | None = None
    average_bm25_score: float | None = None
    average_rrf_score: float | None = None
    average_rerank_score: float | None = None
    highest_rerank_score: float | None = None
    lowest_rerank_score: float | None = None
    median_rerank_score: float | None = None
    average_chunk_length: float = 0.0
    average_chunk_tokens: float = 0.0
    average_document_hits: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "average_vector_score": self.average_vector_score,
            "average_bm25_score": self.average_bm25_score,
            "average_rrf_score": self.average_rrf_score,
            "average_rerank_score": self.average_rerank_score,
            "highest_rerank_score": self.highest_rerank_score,
            "lowest_rerank_score": self.lowest_rerank_score,
            "median_rerank_score": self.median_rerank_score,
            "average_chunk_length": round(self.average_chunk_length, 1),
            "average_chunk_tokens": round(self.average_chunk_tokens, 1),
            "average_document_hits": round(self.average_document_hits, 2),
        }


@dataclass
class CandidateAnalysis:
    duplicate_chunks_removed: int = 0
    unique_documents: int = 0
    unique_sections: int = 0
    vector_only_hits: int = 0
    bm25_only_hits: int = 0
    hybrid_hits: int = 0
    shared_hits: int = 0
    candidate_overlap_percentage: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "duplicate_chunks_removed": self.duplicate_chunks_removed,
            "unique_documents": self.unique_documents,
            "unique_sections": self.unique_sections,
            "vector_only_hits": self.vector_only_hits,
            "bm25_only_hits": self.bm25_only_hits,
            "hybrid_hits": self.hybrid_hits,
            "shared_hits": self.shared_hits,
            "candidate_overlap_percentage": round(self.candidate_overlap_percentage, 1),
        }


@dataclass
class RetrievalMetrics:
    strategy: str
    repository_id: str | None
    query: str
    timestamp: str
    top_k_requested: int
    top_k_returned: int
    vector_candidates: int = 0
    bm25_candidates: int = 0
    merged_candidates: int = 0
    duplicate_chunks_removed: int = 0
    reranked_candidates: int = 0
    documents_retrieved: int = 0
    pages_retrieved: int = 0
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    candidate_analysis: CandidateAnalysis = field(default_factory=CandidateAnalysis)
    errors: list[str] = field(default_factory=list)

    def retrieval_summary(self, *, debug: bool = False) -> dict[str, Any]:
        base = {
            "strategy": self.strategy,
            "repository_id": self.repository_id,
            "documents_retrieved": self.documents_retrieved,
            "pages_retrieved": self.pages_retrieved,
            "returned_candidates": self.top_k_returned,
            "candidate_overlap": self.candidate_analysis.candidate_overlap_percentage,
        }
        if not debug:
            return base
        return {
            **base,
            "query": self.query,
            "timestamp": self.timestamp,
            "top_k_requested": self.top_k_requested,
            "vector_candidates": self.vector_candidates,
            "bm25_candidates": self.bm25_candidates,
            "merged_candidates": self.merged_candidates,
            "duplicates_removed": self.duplicate_chunks_removed,
            "reranked_candidates": self.reranked_candidates,
            "candidate_overlap": self.candidate_analysis.candidate_overlap_percentage,
            **self.candidate_analysis.as_dict(),
            **self.quality.as_dict(),
        }

    def export_record(self) -> dict[str, Any]:
        """Structured record for Prometheus / OpenTelemetry adapters."""
        return {
            "strategy": self.strategy,
            "repository_id": self.repository_id or "",
            "latency": self.latency.pipeline_trace(debug=True),
            "quality": self.quality.as_dict(),
            "candidate_analysis": self.candidate_analysis.as_dict(),
            "counts": {
                "vector_candidates": self.vector_candidates,
                "bm25_candidates": self.bm25_candidates,
                "merged_candidates": self.merged_candidates,
                "returned_candidates": self.top_k_returned,
                "documents_retrieved": self.documents_retrieved,
                "pages_retrieved": self.pages_retrieved,
            },
            "errors": list(self.errors),
        }
