"""Data models for chunk complexity and retrieval-confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComplexityResult:
    complexity_score: float
    complexity_level: str
    readability_score: float
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    sentence_count: int
    word_count: int
    average_sentence_length: float
    average_word_length: float
    technical_term_density: float
    entity_density: float
    list_density: float
    table_density: float
    structural_density: float

    def as_signals_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.complexity_score, 4),
            "level": self.complexity_level,
            "readability": round(self.readability_score, 4),
            "technical_density": round(self.technical_term_density, 4),
            "entity_density": round(self.entity_density, 4),
            "structural_density": round(self.structural_density, 4),
            "list_density": round(self.list_density, 4),
            "table_density": round(self.table_density, 4),
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "average_sentence_length": round(self.average_sentence_length, 2),
            "average_word_length": round(self.average_word_length, 2),
            "flesch_reading_ease": round(self.flesch_reading_ease, 2),
            "flesch_kincaid_grade": round(self.flesch_kincaid_grade, 2),
        }


@dataclass
class ConfidenceResult:
    retrieval_confidence: float
    confidence_level: str

    def as_signals_dict(self) -> dict[str, Any]:
        return {
            "retrieval_confidence": round(self.retrieval_confidence, 4),
            "confidence_level": self.confidence_level,
        }


@dataclass
class ScoringSummary:
    average_complexity: float = 0.0
    average_confidence: float = 0.0
    highest_confidence: float = 0.0
    lowest_confidence: float = 0.0
    complexity_distribution: dict[str, int] = field(default_factory=lambda: {"low": 0, "medium": 0, "high": 0})
    confidence_distribution: dict[str, int] = field(
        default_factory=lambda: {"very_high": 0, "high": 0, "medium": 0, "low": 0}
    )

    def as_retrieval_summary(self) -> dict[str, Any]:
        return {
            "average_complexity": round(self.average_complexity, 4),
            "average_confidence": round(self.average_confidence, 4),
            "highest_confidence": round(self.highest_confidence, 4),
            "lowest_confidence": round(self.lowest_confidence, 4),
            "complexity_distribution": dict(self.complexity_distribution),
            "confidence_distribution": dict(self.confidence_distribution),
        }
