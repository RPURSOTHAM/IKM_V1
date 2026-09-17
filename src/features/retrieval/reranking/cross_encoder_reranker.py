"""Cross-encoder reranker for fused retrieval candidates."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any

from src.features.retrieval.reranking.reranker import RerankMetrics, RerankResult
from src.features.retrieval.reranking.config import RerankerConfig
from src.features.retrieval.reranking.order import sort_chunks_by_rerank_score
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut

_logger = logging.getLogger(__name__)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


from src.features.retrieval.reranking.reranker_model_resolver import resolve_reranker_model_path


class CrossEncoderReranker:
    """Batch cross-encoder reranking over strategy-produced candidates only."""

    def __init__(
        self,
        model_loader: Callable[[str], Any | None],
        *,
        config: RerankerConfig | None = None,
    ) -> None:
        self._model_loader = model_loader
        self._config = config or RerankerConfig.from_env()
        self._last_model_load_time_ms = 0.0

    @property
    def config(self) -> RerankerConfig:
        return self._config

    def rerank(
        self,
        query: str,
        candidates: list[ChunkOut],
        *,
        top_k: int | None = None,
        top_k_before: int | None = None,
        repo_settings: dict[str, Any] | None = None,
    ) -> RerankResult:
        """Rerank candidates by cross-encoder score; no retrieval is performed."""
        metrics = RerankMetrics(
            batch_size=self._config.batch_size,
            model=self._config.model,
        )
        if not candidates:
            return RerankResult(chunks=[], applied=False, metrics=metrics)

        if not self._config.enabled:
            limit = top_k if top_k is not None else self._config.top_k_after
            return RerankResult(chunks=candidates[:limit], applied=False, metrics=metrics)

        before = top_k_before if top_k_before is not None else self._config.top_k_before
        after = top_k if top_k is not None else self._config.top_k_after
        pool = list(candidates[: max(1, before)])
        metrics.candidate_count = len(pool)

        model_path = resolve_reranker_model_path(repo_settings, config=self._config)
        metrics.model = model_path

        load_started = time.perf_counter()
        model = self._model_loader(model_path)
        self._last_model_load_time_ms = (time.perf_counter() - load_started) * 1000.0
        metrics.model_load_time_ms = self._last_model_load_time_ms

        if model is None:
            _logger.warning("Cross-encoder reranker unavailable at %s; preserving RRF order", model_path)
            return RerankResult(chunks=pool[:after], applied=False, metrics=metrics)

        pairs = [(query, chunk.text) for chunk in pool]
        started = time.perf_counter()
        raw_scores: list[float] = []
        batch_size = max(1, int(self._config.batch_size))
        batch_count = 0
        try:
            for offset in range(0, len(pairs), batch_size):
                batch = pairs[offset : offset + batch_size]
                batch_scores = model.predict(batch)
                raw_scores.extend(float(score) for score in batch_scores)
                batch_count += 1
        except Exception as exc:
            _logger.warning("Cross-encoder rerank failed; preserving candidate order: %s", exc)
            metrics.rerank_ms = (time.perf_counter() - started) * 1000.0
            metrics.batch_count = batch_count
            return RerankResult(chunks=pool[:after], applied=False, metrics=metrics)

        metrics.rerank_ms = (time.perf_counter() - started) * 1000.0
        metrics.batch_count = batch_count

        reranked: list[ChunkOut] = []
        for chunk, raw_score in zip(pool, raw_scores):
            score = float(raw_score)
            normalized = _sigmoid(score)
            signals = dict(chunk.retrieval_signals or {})
            prior_score = signals.get("rrf_score")
            if prior_score is None:
                prior_score = signals.get("score", chunk.score)
            signals["original_score"] = float(prior_score)
            signals["rerank_score"] = round(score, 4)
            signals["rerank_score_normalized"] = round(normalized, 4)
            signals["rerank_model"] = model_path
            metadata = dict(chunk.metadata or {})
            metadata["rerank_score"] = round(score, 4)
            metadata["rerank_model"] = model_path
            reranked.append(
                chunk.model_copy(
                    update={
                        "score": round(normalized, 4),
                        # Preserve independently computed grounding/quality; do not alias rerank score.
                        "retrieval_signals": signals,
                        "metadata": metadata,
                    }
                )
            )

        if self._config.score_threshold > 0:
            reranked = [
                item
                for item in reranked
                if float((item.retrieval_signals or {}).get("rerank_score") or 0.0)
                >= self._config.score_threshold
            ]

        reranked = sort_chunks_by_rerank_score(reranked)
        reranked = reranked[:after]

        if reranked:
            top_score = float(reranked[0].retrieval_signals.get("rerank_score") or 0.0)
            metrics.top_score = top_score
            metrics.average_score = sum(
                float((item.retrieval_signals or {}).get("rerank_score") or 0.0) for item in reranked
            ) / len(reranked)

        _logger.info(
            "cross_encoder_rerank model=%s batch_size=%d candidate_count=%d batch_count=%d "
            "rerank_ms=%.1f top_score=%.4f",
            model_path,
            batch_size,
            metrics.candidate_count,
            metrics.batch_count,
            metrics.rerank_ms,
            metrics.top_score,
        )

        return RerankResult(chunks=reranked, applied=True, metrics=metrics)
