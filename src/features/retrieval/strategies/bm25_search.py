"""BM25 keyword retrieval strategy (delegates to existing BM25 service path)."""

from __future__ import annotations

import time

from src.features.retrieval.strategies.base import RetrievalStrategyContext, merged_to_chunk_out
from src.features.retrieval.strategies.models import StrategyMetrics, StrategyResult


class Bm25RetrievalStrategy:
    name = "keyword"

    def search(self, ctx: RetrievalStrategyContext) -> StrategyResult:
        started = time.perf_counter()
        common = dict(ctx.common)
        common["limit"] = max(int(common.get("limit") or ctx.top_k), ctx.bm25_limit, ctx.top_k)

        fetch = ctx.fetch_candidates
        if fetch is None:
            raise RuntimeError("RetrievalStrategyContext.fetch_candidates is required")

        candidates = fetch(
            mode="keyword",
            query_text=ctx.query,
            query_vector=None,
            common=common,
        )
        search_ms = (time.perf_counter() - started) * 1000.0

        chunks = []
        pool_k = ctx.candidate_k if ctx.candidate_k > 0 else ctx.top_k
        for candidate in candidates[:pool_k]:
            signals = dict(candidate.retrieval_signals or {})
            bm25_score = float(candidate.score)
            signals.setdefault("bm25_score", round(bm25_score, 4))
            signals["score"] = round(bm25_score, 4)
            chunk = merged_to_chunk_out(
                candidate,
                final_score=bm25_score,
                signals=signals,
                query_terms=ctx.query_terms,
                repository_id=ctx.repository_id,
            )
            chunks.append(chunk)

        metrics = StrategyMetrics(
            bm25_search_ms=search_ms,
            total_ms=search_ms,
            bm25_candidates=len(candidates),
            returned_candidates=len(chunks),
        )
        return StrategyResult(
            chunks=chunks,
            metrics=metrics,
            pipeline_stages=["bm25"],
            total_candidates=len(candidates),
            strategy=self.name,
        )
