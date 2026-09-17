"""Hybrid retrieval: parallel vector + BM25, merge, normalize, alpha-weighted fusion (RRF optional).

Fusion contract:
- Default / explicit alpha → alpha-weighted fusion (platform default alpha=0.75).
- RRF only when use_rrf=True AND the request did not set alpha/hybrid_alpha.
- Explicit alpha is never silently ignored.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.features.retrieval.strategies.base import RetrievalStrategyContext, merged_to_chunk_out
from src.features.retrieval.strategies.result_merger import (
    apply_normalized_scores,
    merge_candidates_by_chunk_id,
    rank_merged_by_alpha,
)
from src.features.retrieval.strategies.rrf import reciprocal_rank_fusion

_logger = logging.getLogger(__name__)


class HybridRetrievalStrategy:
    """Vector + BM25 candidate fusion with configurable alpha weighting (RRF optional)."""

    name = "hybrid"

    def search(self, ctx: RetrievalStrategyContext) -> StrategyResult:
        from src.features.retrieval.strategies.models import StrategyMetrics, StrategyResult

        started = time.perf_counter()
        fetch = ctx.fetch_candidates
        if fetch is None:
            raise RuntimeError("RetrievalStrategyContext.fetch_candidates is required")

        vector_common = dict(ctx.common)
        vector_common["limit"] = max(ctx.vector_limit, ctx.top_k)
        bm25_common = dict(ctx.common)
        bm25_common["limit"] = max(ctx.bm25_limit, ctx.top_k)

        vector_started = time.perf_counter()
        bm25_started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as pool:
            if ctx.query_vector is None:
                bm25_future = pool.submit(
                    fetch,
                    mode="keyword",
                    query_text=ctx.query,
                    query_vector=None,
                    common=bm25_common,
                )
                vector_candidates = []
                bm25_candidates = bm25_future.result()
                vector_ms = 0.0
                bm25_ms = (time.perf_counter() - bm25_started) * 1000.0
            else:
                vector_future = pool.submit(
                    fetch,
                    mode="vector",
                    query_text=ctx.query,
                    query_vector=ctx.query_vector,
                    common=vector_common,
                )
                bm25_future = pool.submit(
                    fetch,
                    mode="keyword",
                    query_text=ctx.query,
                    query_vector=None,
                    common=bm25_common,
                )
                vector_candidates = vector_future.result()
                bm25_candidates = bm25_future.result()
                vector_ms = (time.perf_counter() - vector_started) * 1000.0
                bm25_ms = (time.perf_counter() - bm25_started) * 1000.0

        merge_started = time.perf_counter()
        merged = merge_candidates_by_chunk_id(vector_candidates, bm25_candidates)
        apply_normalized_scores(merged)
        merge_ms = (time.perf_counter() - merge_started) * 1000.0

        fusion_started = time.perf_counter()
        use_rrf = bool(getattr(ctx, "use_rrf", False))
        fusion_stage = "alpha_weighted_fusion"
        if use_rrf:
            fusion_stage = "reciprocal_rank_fusion"
            vector_ranked = sorted(vector_candidates, key=lambda item: float(item.score), reverse=True)
            bm25_ranked = sorted(bm25_candidates, key=lambda item: float(item.score), reverse=True)
            fused = reciprocal_rank_fusion(
                [vector_ranked, bm25_ranked],
                k=ctx.rrf_k,
                source_names=["vector", "bm25"],
            )
            merged_by_id = {item.chunk_id: item for item in merged}
            ranked_pairs: list[tuple[Any, float]] = []
            for candidate in fused:
                merged_item = merged_by_id.get(candidate.chunk_id)
                if merged_item is None:
                    continue
                merged_item.rrf_score = float(candidate.score)
                ranked_pairs.append((merged_item, float(candidate.score)))
        else:
            ranked_pairs = rank_merged_by_alpha(merged, alpha=ctx.hybrid_alpha)

        fusion_ms = (time.perf_counter() - fusion_started) * 1000.0

        chunks = []
        pool_k = ctx.candidate_k if ctx.candidate_k > 0 else ctx.top_k
        for merged_item, final_score in ranked_pairs[:pool_k]:
            signals: dict[str, Any] = {}
            if merged_item.vector_score is not None:
                signals["vector_score"] = round(float(merged_item.vector_score), 4)
            if merged_item.bm25_score is not None:
                signals["bm25_score"] = round(float(merged_item.bm25_score), 4)
            if merged_item.normalized_vector_score is not None:
                signals["normalized_vector_score"] = round(float(merged_item.normalized_vector_score), 4)
            if merged_item.normalized_bm25_score is not None:
                signals["normalized_bm25_score"] = round(float(merged_item.normalized_bm25_score), 4)
            if getattr(merged_item, "rrf_score", None) is not None:
                signals["rrf_score"] = round(float(merged_item.rrf_score), 6)
            signals["hybrid_alpha"] = round(float(ctx.hybrid_alpha), 4)
            signals["fusion_method"] = fusion_stage
            signals["score"] = round(float(final_score), 6)
            # Prefer chunk metadata repository_id; do not copy the request scope.
            chunks.append(
                merged_to_chunk_out(
                    merged_item,
                    final_score=float(final_score),
                    signals=signals,
                    query_terms=ctx.query_terms,
                    repository_id=None,
                )
            )

        total_ms = (time.perf_counter() - started) * 1000.0
        metrics = StrategyMetrics(
            vector_search_ms=vector_ms,
            bm25_search_ms=bm25_ms,
            merge_ms=merge_ms,
            fusion_ms=fusion_ms,
            total_ms=total_ms,
            vector_candidates=len(vector_candidates),
            bm25_candidates=len(bm25_candidates),
            merged_candidates=len(merged),
            returned_candidates=len(chunks),
        )

        _logger.info(
            "hybrid_retrieval repository_id=%s query=%r strategy=hybrid fusion=%s "
            "vector_candidates=%d bm25_candidates=%d merged_candidates=%d total_ms=%.1f",
            ctx.repository_id,
            ctx.query,
            fusion_stage,
            metrics.vector_candidates,
            metrics.bm25_candidates,
            metrics.merged_candidates,
            metrics.total_ms,
        )

        stages = ["embedding", "near_vector", "bm25", "candidate_merge", "score_normalization", fusion_stage]
        if ctx.query_vector is None:
            stages = ["bm25"]

        return StrategyResult(
            chunks=chunks,
            metrics=metrics,
            pipeline_stages=stages,
            total_candidates=len(vector_candidates) + len(bm25_candidates),
            strategy=self.name,
        )
