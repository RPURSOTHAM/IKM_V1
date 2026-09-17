"""Chunk text complexity analysis."""

from __future__ import annotations

import re
from typing import Any

from src.features.retrieval.scoring.config import (
    COMPLEXITY_HIGH_MIN,
    COMPLEXITY_LOW_MAX,
    ComplexityWeights,
)
from src.features.retrieval.scoring.models import ComplexityResult
from src.features.retrieval.scoring.readability import compute_readability

_ABBREVIATION = re.compile(r"\b[A-Z]{2,}(?:[-/][A-Z0-9]{1,})*\b")
_CAMEL_CASE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")
_IDENTIFIER = re.compile(r"\b[A-Z0-9]{1,3}[-_][A-Z0-9][A-Z0-9\-_]*\b", re.IGNORECASE)
_CHEMICAL = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
_PROCEDURE_CODE = re.compile(r"\b(?:SOP|WI|CAPA|OOS|OOT|GMP|GDP|GLP|QA|QC|IQ|OQ|PQ)[-_]?\d{2,}\b", re.IGNORECASE)
_REGULATORY = re.compile(
    r"\b(?:21\s*CFR|ICH|FDA|EMA|ISO\s*\d+|USP|EP|BP|Annex\s+\d+|Part\s+\d+)\b",
    re.IGNORECASE,
)
_STANDARD = re.compile(r"\b(?:ASTM|IEEE|ANSI|NIST|ASME|API|SAE)\s+[A-Z0-9\-.]+\b", re.IGNORECASE)
_EQUIPMENT = re.compile(
    r"\b(?:HPLC|GC(?:-MS)?|LC(?:-MS)?|FTIR|NMR|SEM|TEM|PCR|ELISA|Bioreactor|Autoclave|Centrifuge)\b",
    re.IGNORECASE,
)
_BULLET_LINE = re.compile(r"(?m)^\s*(?:[-*•●▪◦]|\d+[.)])\s+\S")
_TABLE_PIPE = re.compile(r"\|.+\|")
_TABLE_TSV = re.compile(r"(?m)^[^\n|\t]+\t[^\n|\t]+(?:\t[^\n|\t]+)+$")
_EQUATION = re.compile(r"(?:\$[^$]+\$|=\s*[\w\d+\-*/^().\s]{4,})")
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INDENTED_CODE = re.compile(r"(?m)^ {4,}\S")
_DATE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?(?:%|mg|ml|kg|mm|cm|°C|ppm|ppb)?\b", re.IGNORECASE)
_ORG = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b")
_CAPITAL_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}\b")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _density(count: int, word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    return _clamp(count / max(word_count, 1))


def _complexity_level(score: float) -> str:
    if score <= COMPLEXITY_LOW_MAX:
        return "Low"
    if score >= COMPLEXITY_HIGH_MIN:
        return "High"
    return "Medium"


def _extract_entities(metadata: dict[str, Any] | None) -> list[Any]:
    if not metadata:
        return []
    entities = metadata.get("entities")
    if isinstance(entities, list):
        return entities
    if isinstance(entities, dict):
        flattened: list[Any] = []
        for value in entities.values():
            if isinstance(value, list):
                flattened.extend(value)
            elif value:
                flattened.append(value)
        return flattened
    return []


class ComplexityAnalyzer:
    """Estimate chunk complexity from text structure and linguistic features."""

    def __init__(self, *, weights: ComplexityWeights | None = None) -> None:
        self._weights = weights or ComplexityWeights.from_env()

    def analyze(self, text: str, *, entities: list[Any] | None = None) -> ComplexityResult:
        readability = compute_readability(text)
        words = readability.word_count
        content = text or ""

        technical_matches = set()
        for pattern in (
            _ABBREVIATION,
            _CAMEL_CASE,
            _IDENTIFIER,
            _CHEMICAL,
            _PROCEDURE_CODE,
            _REGULATORY,
            _STANDARD,
            _EQUIPMENT,
        ):
            technical_matches.update(match.group(0) for match in pattern.finditer(content))
        technical_term_density = _density(len(technical_matches), words)

        entity_count = len(entities or [])
        if entity_count == 0:
            estimated_entities = set()
            estimated_entities.update(_DATE.findall(content))
            estimated_entities.update(_NUMBER.findall(content))
            estimated_entities.update(_ORG.findall(content))
            estimated_entities.update(_CAPITAL_TOKEN.findall(content))
            estimated_entities.update(_PROCEDURE_CODE.findall(content))
            entity_count = len(estimated_entities)
        entity_density = _density(entity_count, words)

        bullet_lines = len(_BULLET_LINE.findall(content))
        list_density = _density(bullet_lines, max(readability.sentence_count, 1))

        table_hits = len(_TABLE_PIPE.findall(content)) + len(_TABLE_TSV.findall(content))
        table_density = _density(table_hits, max(readability.sentence_count, 1))

        equation_hits = len(_EQUATION.findall(content))
        code_hits = len(_CODE_FENCE.findall(content)) + len(_INDENTED_CODE.findall(content))
        structural_hits = bullet_lines + table_hits + equation_hits + code_hits
        structural_density = _density(structural_hits, max(readability.sentence_count, 1))
        structural_density = _clamp(
            (list_density * 0.35)
            + (table_density * 0.30)
            + (_density(equation_hits, max(words, 1)) * 0.20)
            + (_density(code_hits, max(readability.sentence_count, 1)) * 0.15)
        )

        weights = self._weights
        complexity_score = _clamp(
            (readability.readability_complexity * weights.readability)
            + (technical_term_density * weights.technical)
            + (entity_density * weights.entity)
            + (structural_density * weights.structural)
        )

        return ComplexityResult(
            complexity_score=complexity_score,
            complexity_level=_complexity_level(complexity_score),
            readability_score=readability.readability_complexity,
            flesch_reading_ease=readability.flesch_reading_ease,
            flesch_kincaid_grade=readability.flesch_kincaid_grade,
            sentence_count=readability.sentence_count,
            word_count=readability.word_count,
            average_sentence_length=readability.average_sentence_length,
            average_word_length=readability.average_word_length,
            technical_term_density=technical_term_density,
            entity_density=entity_density,
            list_density=list_density,
            table_density=table_density,
            structural_density=structural_density,
        )
