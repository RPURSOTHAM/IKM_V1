"""Vector-only retrieval strategy."""

from __future__ import annotations

import time

from src.features.retrieval.strategies.base import RetrievalStrategyContext, merged_to_chunk_out
from src.features.retrieval.strategies.models import StrategyMetrics, StrategyResult


class VectorRetrievalStrategy:
    name = "vector"

    def search(self, ctx: RetrievalStrategyContext) -> StrategyResult:
        started = time.perf_counter()
        common = dict(ctx.common)
        common["limit"] = max(int(common.get("limit") or ctx.top_k), ctx.vector_limit, ctx.top_k)

        fetch = ctx.fetch_candidates
        if fetch is None:
            raise RuntimeError("RetrievalStrategyContext.fetch_candidates is required")

        candidates = fetch(
            mode="vector",
            query_text=ctx.query,
            query_vector=ctx.query_vector,
            common=common,
        )
        search_ms = (time.perf_counter() - started) * 1000.0

        chunks = []
        for candidate in candidates[: ctx.top_k]:
            signals = dict(candidate.retrieval_signals or {})
            vector_score = float(candidate.score)
            signals.setdefault("vector_score", round(vector_score, 4))
            signals["score"] = round(vector_score, 4)
            chunk = merged_to_chunk_out(
                candidate,
                final_score=vector_score,
                signals=signals,
                query_terms=ctx.query_terms,
                repository_id=ctx.repository_id,
            )
            chunks.append(chunk)

        metrics = StrategyMetrics(
            vector_search_ms=search_ms,
            total_ms=search_ms,
            vector_candidates=len(candidates),
            returned_candidates=len(chunks),
        )
        return StrategyResult(
            chunks=chunks,
            metrics=metrics,
            pipeline_stages=["embedding", "near_vector"],
            total_candidates=len(candidates),
            strategy=self.name,
        )
