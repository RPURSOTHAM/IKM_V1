"""Central retrieval metrics collector."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.features.retrieval.metrics.analysis import (
    compute_candidate_analysis,
    compute_quality_metrics,
    count_documents_and_pages,
)
from src.features.retrieval.metrics.models import LatencyMetrics, RetrievalMetrics
from src.features.retrieval.metrics.timing import StageTimer
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut

_logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class RetrievalMetricsCollector:
    """Build retrieval telemetry from strategy output without changing retrieval behavior."""

    @staticmethod
    def build(
        *,
        strategy: str,
        repository_id: str | None,
        query: str,
        top_k_requested: int,
        chunks: list[ChunkOut],
        strategy_metrics: Any | None = None,
        stage_timer: StageTimer | None = None,
        embedding_ms: float = 0.0,
        rerank_metrics: Any | None = None,
        rerank_applied: bool = False,
        total_ms: float | None = None,
        errors: list[str] | None = None,
    ) -> RetrievalMetrics:
        vector_candidates = int(getattr(strategy_metrics, "vector_candidates", 0) or 0)
        bm25_candidates = int(getattr(strategy_metrics, "bm25_candidates", 0) or 0)
        merged_candidates = int(getattr(strategy_metrics, "merged_candidates", 0) or 0)

        if strategy == "vector" and vector_candidates == 0:
            vector_candidates = len(chunks)
        if strategy == "keyword" and bm25_candidates == 0:
            bm25_candidates = len(chunks)
        if merged_candidates == 0:
            merged_candidates = max(len(chunks), vector_candidates, bm25_candidates)

        trace = getattr(strategy_metrics, "as_trace", lambda: {})() if strategy_metrics else {}
        if stage_timer is not None and not embedding_ms:
            embedding_ms = _safe_float(stage_timer.as_dict().get("embedding", 0.0))
        latency = LatencyMetrics(
            embedding_ms=_safe_float(embedding_ms),
            vector_search_ms=_safe_float(trace.get("vector_search_ms", 0.0)),
            bm25_search_ms=_safe_float(trace.get("bm25_search_ms", 0.0)),
            merge_ms=_safe_float(trace.get("merge_ms", 0.0)),
            normalization_ms=_safe_float(trace.get("normalization_ms", 0.0)),
            fusion_ms=_safe_float(trace.get("fusion_ms", 0.0)),
            rerank_ms=_safe_float(getattr(rerank_metrics, "rerank_ms", 0.0)),
        )
        if stage_timer is not None:
            timer = stage_timer.as_dict()
            for field_name in (
                "vector_search_ms",
                "bm25_search_ms",
                "merge_ms",
                "normalization_ms",
                "fusion_ms",
                "rerank_ms",
            ):
                timer_val = _safe_float(timer.get(field_name, 0.0))
                if timer_val > 0:
                    current = float(getattr(latency, field_name, 0.0))
                    setattr(latency, field_name, max(current, timer_val))
            # Production pipeline records stage timing under the stage class name.
            stage_rerank_ms = max(
                _safe_float(timer.get("cross_encoder_reranker", 0.0)),
                _safe_float(timer.get("rerank", 0.0)),
                _safe_float(trace.get("cross_encoder_reranker", 0.0)),
                _safe_float(trace.get("rerank", 0.0)),
            )
            if stage_rerank_ms > 0:
                latency.rerank_ms = max(float(latency.rerank_ms), stage_rerank_ms)
            latency.dlp_ms = stage_timer.get("dlp")
            latency.serialization_ms = stage_timer.get("serialization")
            latency.total_ms = total_ms if total_ms is not None else stage_timer.total_ms
        elif total_ms is not None:
            latency.total_ms = total_ms

        documents, pages = count_documents_and_pages(chunks)
        analysis = compute_candidate_analysis(
            vector_candidates=vector_candidates,
            bm25_candidates=bm25_candidates,
            merged_candidates=merged_candidates,
            chunks=chunks,
        )
        quality = compute_quality_metrics(chunks)

        reranked_candidates = int(getattr(rerank_metrics, "candidate_count", 0) or 0) if rerank_applied else 0

        return RetrievalMetrics(
            strategy=strategy,
            repository_id=repository_id,
            query=query,
            timestamp=datetime.now(timezone.utc).isoformat(),
            top_k_requested=top_k_requested,
            top_k_returned=len(chunks),
            vector_candidates=vector_candidates,
            bm25_candidates=bm25_candidates,
            merged_candidates=merged_candidates,
            duplicate_chunks_removed=analysis.duplicate_chunks_removed,
            reranked_candidates=reranked_candidates,
            documents_retrieved=documents,
            pages_retrieved=pages,
            latency=latency,
            quality=quality,
            candidate_analysis=analysis,
            errors=list(errors or []),
        )

    @staticmethod
    def log_metrics(metrics: RetrievalMetrics) -> None:
        _logger.info(
            "retrieval_metrics strategy=%s repository_id=%s query=%r "
            "total_ms=%.1f vector_candidates=%d bm25_candidates=%d merged_candidates=%d "
            "duplicates_removed=%d documents=%d pages=%d rerank_ms=%.1f "
            "avg_rerank=%s top_rerank=%s overlap_pct=%.1f errors=%d",
            metrics.strategy,
            metrics.repository_id,
            metrics.query,
            metrics.latency.total_ms,
            metrics.vector_candidates,
            metrics.bm25_candidates,
            metrics.merged_candidates,
            metrics.duplicate_chunks_removed,
            metrics.documents_retrieved,
            metrics.pages_retrieved,
            metrics.latency.rerank_ms,
            metrics.quality.average_rerank_score,
            metrics.quality.highest_rerank_score,
            metrics.candidate_analysis.candidate_overlap_percentage,
            len(metrics.errors),
        )

    @staticmethod
    def finalize_response_fields(
        metrics: RetrievalMetrics,
        *,
        debug: bool,
        existing_quality_summary: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
        pipeline_trace = metrics.latency.pipeline_trace(debug=debug)
        retrieval_summary = metrics.retrieval_summary(debug=debug)
        quality_summary = dict(existing_quality_summary or {})
        quality_summary["returned"] = metrics.top_k_returned
        if debug:
            quality_summary.update(metrics.quality.as_dict())
            quality_summary["candidate_analysis"] = metrics.candidate_analysis.as_dict()
        return pipeline_trace, retrieval_summary, quality_summary
