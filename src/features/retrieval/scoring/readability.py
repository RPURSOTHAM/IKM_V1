"""Readability metrics for chunk complexity analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_VOWEL_GROUP = re.compile(r"[aeiouyAEIOUY]+")


@dataclass
class ReadabilityMetrics:
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    readability_complexity: float
    sentence_count: int
    word_count: int
    average_sentence_length: float
    average_word_length: float
    syllable_count: int


def _count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1
    groups = _VOWEL_GROUP.findall(cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _split_sentences(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(stripped) if part.strip()]
    return parts or [stripped]


def _split_words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text or "")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_readability(text: str) -> ReadabilityMetrics:
    sentences = _split_sentences(text)
    words = _split_words(text)
    sentence_count = max(len(sentences), 1 if words else 0)
    word_count = len(words)
    syllable_count = sum(_count_syllables(word) for word in words)

    if word_count == 0:
        return ReadabilityMetrics(
            flesch_reading_ease=100.0,
            flesch_kincaid_grade=0.0,
            readability_complexity=0.0,
            sentence_count=0,
            word_count=0,
            average_sentence_length=0.0,
            average_word_length=0.0,
            syllable_count=0,
        )

    average_sentence_length = word_count / sentence_count
    average_word_length = sum(len(word) for word in words) / word_count
    syllables_per_word = syllable_count / word_count
    words_per_sentence = word_count / sentence_count

    flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    flesch_reading_ease = max(0.0, min(100.0, flesch_reading_ease))
    flesch_kincaid_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59
    flesch_kincaid_grade = max(0.0, flesch_kincaid_grade)

    ease_complexity = 1.0 - _clamp(flesch_reading_ease / 100.0)
    grade_complexity = _clamp(flesch_kincaid_grade / 18.0)
    readability_complexity = _clamp((ease_complexity * 0.65) + (grade_complexity * 0.35))

    return ReadabilityMetrics(
        flesch_reading_ease=flesch_reading_ease,
        flesch_kincaid_grade=flesch_kincaid_grade,
        readability_complexity=readability_complexity,
        sentence_count=sentence_count if words else 0,
        word_count=word_count,
        average_sentence_length=average_sentence_length,
        average_word_length=average_word_length,
        syllable_count=syllable_count,
    )
