"""Chunk quality, grounding, and highlight span regression tests."""

from __future__ import annotations

from src.features.retrieval.scoring.quality_scorer import (
    compute_highlight_spans,
    enrich_chunk_scores,
)


def test_highlight_spans_match_query_terms() -> None:
    text = "The cleaning procedure requires validated acceptance criteria."
    spans = compute_highlight_spans(text, ["cleaning", "acceptance"])
    assert spans
    for start, end in spans:
        fragment = text[start:end].lower()
        assert fragment in {"cleaning", "acceptance"}


def test_grounding_score_reflects_evidence_not_only_retrieval_score() -> None:
    highlights, grounding, quality, flags = enrich_chunk_scores(
        text="Short",
        retrieval_score=0.9,
        metadata={"query_evidence_terms": []},
        query_terms=["missing-term"],
    )
    assert grounding < 0.9
    assert grounding != 0.9
    assert "weak_match" in flags
    assert "incomplete_chunk" in flags
    assert highlights == []


def test_quality_flags_for_low_text_quality() -> None:
    _, _, quality, flags = enrich_chunk_scores(
        text="tiny",
        retrieval_score=0.8,
        metadata={},
        query_terms=[],
    )
    assert quality < 0.8
    assert "incomplete_chunk" in flags
