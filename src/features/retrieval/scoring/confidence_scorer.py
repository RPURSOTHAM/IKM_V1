"""Retrieval confidence scoring from existing retrieval signals."""

from __future__ import annotations

from typing import Any

from src.features.retrieval.scoring.config import (
    CONFIDENCE_HIGH_MIN,
    CONFIDENCE_MEDIUM_MIN,
    CONFIDENCE_VERY_HIGH_MIN,
    ConfidenceWeights,
)
from src.features.retrieval.scoring.models import ConfidenceResult
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _confidence_level(score: float) -> str:
    if score >= CONFIDENCE_VERY_HIGH_MIN:
        return "Very High"
    if score >= CONFIDENCE_HIGH_MIN:
        return "High"
    if score >= CONFIDENCE_MEDIUM_MIN:
        return "Medium"
    return "Low"


def _distribution_key(level: str) -> str:
    return level.lower().replace(" ", "_")


def _document_key(chunk: ChunkOut) -> str:
    metadata = chunk.metadata or {}
    doc_id = metadata.get("document_id")
    if doc_id:
        return str(doc_id)
    if chunk.doc_name:
        return str(chunk.doc_name)
    return str(chunk.chunk_id)


def _signal_float(signals: dict[str, Any], key: str) -> float | None:
    raw = signals.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _min_max_normalize(values: list[float | None]) -> list[float | None]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return [None for _ in values]
    lo = min(present)
    hi = max(present)
    if hi == lo:
        return [1.0 if v is not None else None for v in values]
    span = hi - lo
    return [((float(v) - lo) / span) if v is not None else None for v in values]


class RetrievalConfidenceScorer:
    """Compute retrieval confidence from normalized retrieval signals."""

    def __init__(self, *, weights: ConfidenceWeights | None = None) -> None:
        self._weights = weights or ConfidenceWeights.from_env()

    def score_chunks(
        self,
        chunks: list[ChunkOut],
        *,
        candidate_overlap_pct: float = 0.0,
    ) -> list[ConfidenceResult]:
        if not chunks:
            return []

        signals_list = [dict(chunk.retrieval_signals or {}) for chunk in chunks]
        vector_norm = _min_max_normalize([_signal_float(s, "normalized_vector_score") for s in signals_list])
        bm25_norm = _min_max_normalize([_signal_float(s, "normalized_bm25_score") for s in signals_list])
        rrf_norm = _min_max_normalize([_signal_float(s, "rrf_score") for s in signals_list])
        rerank_norm = [
            _signal_float(s, "rerank_score_normalized") or _signal_float(s, "rerank_score")
            for s in signals_list
        ]
        rerank_norm = _min_max_normalize(rerank_norm)

        overlap_factor = _clamp(candidate_overlap_pct / 100.0)
        doc_groups: dict[str, list[int]] = {}
        for index, chunk in enumerate(chunks):
            doc_groups.setdefault(_document_key(chunk), []).append(index)

        results: list[ConfidenceResult] = []
        weights = self._weights
        for index, signals in enumerate(signals_list):
            components: list[tuple[float, float]] = []

            rerank_value = rerank_norm[index]
            if rerank_value is not None:
                components.append((weights.rerank, rerank_value))

            vector_value = vector_norm[index]
            if vector_value is None:
                vector_value = _signal_float(signals, "vector_score")
                if vector_value is not None:
                    vector_value = _clamp(vector_value)
            if vector_value is not None:
                components.append((weights.vector, _clamp(vector_value)))

            bm25_value = bm25_norm[index]
            if bm25_value is None:
                raw_bm25 = _signal_float(signals, "bm25_score")
                if raw_bm25 is not None:
                    bm25_value = _clamp(raw_bm25 / max(raw_bm25, 1.0))
            if bm25_value is not None:
                components.append((weights.bm25, _clamp(bm25_value)))

            rrf_value = rrf_norm[index]
            if rrf_value is None:
                rrf_value = _signal_float(signals, "rrf_score")
            if rrf_value is not None:
                components.append((weights.rrf, _clamp(rrf_value)))

            active_weight = sum(weight for weight, _ in components)
            if active_weight <= 0:
                base_confidence = 0.0
            else:
                base_confidence = sum(weight * value for weight, value in components) / active_weight

            doc_key = _document_key(chunks[index])
            siblings = doc_groups.get(doc_key, [])
            sibling_scores = [
                rerank_norm[i] or vector_norm[i] or rrf_norm[i] or 0.0
                for i in siblings
                if (rerank_norm[i] is not None or vector_norm[i] is not None or rrf_norm[i] is not None)
            ]
            if sibling_scores:
                doc_agreement = _clamp((len(siblings) / max(len(chunks), 1)) * (sum(sibling_scores) / len(sibling_scores)))
            else:
                doc_agreement = 0.0

            hybrid_bonus = overlap_factor * 0.02 if vector_value is not None and bm25_value is not None else 0.0
            confidence = _clamp(
                base_confidence
                + (weights.document_agreement * doc_agreement)
                + hybrid_bonus
            )
            results.append(
                ConfidenceResult(
                    retrieval_confidence=confidence,
                    confidence_level=_confidence_level(confidence),
                )
            )
        return results


def confidence_distribution_counts(results: list[ConfidenceResult]) -> dict[str, int]:
    counts = {"very_high": 0, "high": 0, "medium": 0, "low": 0}
    for item in results:
        key = _distribution_key(item.confidence_level)
        counts[key] = counts.get(key, 0) + 1
    return counts
