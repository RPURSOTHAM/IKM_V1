"""Configuration for complexity and retrieval-confidence scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ComplexityWeights:
    readability: float = 0.30
    technical: float = 0.30
    entity: float = 0.20
    structural: float = 0.20

    @classmethod
    def from_env(cls) -> ComplexityWeights:
        return cls(
            readability=_env_float("COMPLEXITY_WEIGHT_READABILITY", 0.30),
            technical=_env_float("COMPLEXITY_WEIGHT_TECHNICAL", 0.30),
            entity=_env_float("COMPLEXITY_WEIGHT_ENTITY", 0.20),
            structural=_env_float("COMPLEXITY_WEIGHT_STRUCTURAL", 0.20),
        )


@dataclass(frozen=True)
class ConfidenceWeights:
    rerank: float = 0.40
    vector: float = 0.25
    bm25: float = 0.20
    rrf: float = 0.10
    document_agreement: float = 0.05

    @classmethod
    def from_env(cls) -> ConfidenceWeights:
        return cls(
            rerank=_env_float("CONFIDENCE_WEIGHT_RERANK", 0.40),
            vector=_env_float("CONFIDENCE_WEIGHT_VECTOR", 0.25),
            bm25=_env_float("CONFIDENCE_WEIGHT_BM25", 0.20),
            rrf=_env_float("CONFIDENCE_WEIGHT_RRF", 0.10),
            document_agreement=_env_float("CONFIDENCE_WEIGHT_DOCUMENT_AGREEMENT", 0.05),
        )


COMPLEXITY_LOW_MAX = _env_float("COMPLEXITY_THRESHOLD_LOW_MAX", 0.30)
COMPLEXITY_HIGH_MIN = _env_float("COMPLEXITY_THRESHOLD_HIGH_MIN", 0.71)

CONFIDENCE_VERY_HIGH_MIN = _env_float("CONFIDENCE_THRESHOLD_VERY_HIGH", 0.85)
CONFIDENCE_HIGH_MIN = _env_float("CONFIDENCE_THRESHOLD_HIGH", 0.70)
CONFIDENCE_MEDIUM_MIN = _env_float("CONFIDENCE_THRESHOLD_MEDIUM", 0.50)
