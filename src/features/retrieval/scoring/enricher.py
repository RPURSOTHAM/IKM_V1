"""Orchestrate complexity and confidence enrichment for retrieved chunks."""

from __future__ import annotations

import logging
from typing import Any

from src.features.retrieval.schemas.retrieval_schemas import ChunkOut
from src.features.retrieval.scoring.complexity import ComplexityAnalyzer
from src.features.retrieval.scoring.confidence_scorer import (
    RetrievalConfidenceScorer,
    confidence_distribution_counts,
)
from src.features.retrieval.scoring.models import ScoringSummary

_logger = logging.getLogger(__name__)


def _complexity_distribution_key(level: str) -> str:
    return level.lower()


class ChunkScoringEnricher:
    """Attach complexity and retrieval-confidence metadata to returned chunks."""

    def __init__(
        self,
        *,
        complexity_analyzer: ComplexityAnalyzer | None = None,
        confidence_scorer: RetrievalConfidenceScorer | None = None,
    ) -> None:
        self._complexity = complexity_analyzer or ComplexityAnalyzer()
        self._confidence = confidence_scorer or RetrievalConfidenceScorer()

    def enrich(
        self,
        chunks: list[ChunkOut],
        *,
        candidate_overlap_pct: float = 0.0,
        strategy: str | None = None,
        repository_id: str | None = None,
    ) -> tuple[list[ChunkOut], ScoringSummary]:
        if not chunks:
            return [], ScoringSummary()

        confidence_results = self._confidence.score_chunks(
            chunks,
            candidate_overlap_pct=candidate_overlap_pct,
        )

        enriched: list[ChunkOut] = []
        complexity_scores: list[float] = []
        confidence_scores: list[float] = []
        complexity_distribution = {"low": 0, "medium": 0, "high": 0}

        for chunk, confidence in zip(chunks, confidence_results):
            metadata = dict(chunk.metadata or {})
            entities = metadata.get("entities")
            if not isinstance(entities, list):
                entities = []
            complexity = self._complexity.analyze(chunk.text, entities=entities)

            signals = dict(chunk.retrieval_signals or {})
            signals.update(confidence.as_signals_dict())
            signals["complexity"] = complexity.as_signals_dict()

            enriched.append(chunk.model_copy(update={"retrieval_signals": signals}))
            complexity_scores.append(complexity.complexity_score)
            confidence_scores.append(confidence.retrieval_confidence)
            level_key = _complexity_distribution_key(complexity.complexity_level)
            complexity_distribution[level_key] = complexity_distribution.get(level_key, 0) + 1

        summary = ScoringSummary(
            average_complexity=sum(complexity_scores) / len(complexity_scores),
            average_confidence=sum(confidence_scores) / len(confidence_scores),
            highest_confidence=max(confidence_scores),
            lowest_confidence=min(confidence_scores),
            complexity_distribution=complexity_distribution,
            confidence_distribution=confidence_distribution_counts(confidence_results),
        )

        _logger.info(
            "retrieval_scoring strategy=%s repository_id=%s chunks=%d "
            "avg_confidence=%.3f avg_complexity=%.3f highest_confidence=%.3f lowest_confidence=%.3f",
            strategy or "unknown",
            repository_id,
            len(enriched),
            summary.average_confidence,
            summary.average_complexity,
            summary.highest_confidence,
            summary.lowest_confidence,
        )
        return enriched, summary


def enrich_retrieval_chunks(
    chunks: list[ChunkOut],
    *,
    candidate_overlap_pct: float = 0.0,
    strategy: str | None = None,
    repository_id: str | None = None,
) -> tuple[list[ChunkOut], dict[str, Any]]:
    enricher = ChunkScoringEnricher()
    enriched, summary = enricher.enrich(
        chunks,
        candidate_overlap_pct=candidate_overlap_pct,
        strategy=strategy,
        repository_id=repository_id,
    )
    return enriched, summary.as_retrieval_summary()
