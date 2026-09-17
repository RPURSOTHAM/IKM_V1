"""Tests for rerank-score final ordering and candidate pool resolution."""

from __future__ import annotations

import pytest

from src.features.retrieval.configuration.retrieval_config import (
    RetrievalPipelineConfig,
    resolve_candidate_pool_k,
)
from src.features.retrieval.reranking.order import (
    chunk_rerank_score,
    finalize_reranked_chunks,
    rerank_order_is_monotonic,
    sort_chunks_by_rerank_score,
)
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut


def _chunk(chunk_id: str, *, rerank: float | None, bm25: float = 1.0) -> ChunkOut:
    signals: dict[str, float] = {"bm25_score": bm25, "score": bm25}
    if rerank is not None:
        signals["rerank_score"] = rerank
    return ChunkOut(
        chunk_id=chunk_id,
        id=chunk_id,
        text=f"chunk {chunk_id}",
        doc_name="doc.pdf",
        section_name=f"section-{chunk_id}",
        page=1,
        score=bm25,
        source="doc.pdf",
        graph_context=[],
        highlight_spans=[],
        metadata={"rerank_score": rerank} if rerank is not None else {},
        retrieval_signals=signals,
    )


def test_sort_chunks_by_rerank_score_descending() -> None:
    chunks = [
        _chunk("c", rerank=0.20),
        _chunk("a", rerank=0.95),
        _chunk("b", rerank=0.50),
    ]
    ordered = sort_chunks_by_rerank_score(chunks)
    assert [item.chunk_id for item in ordered] == ["a", "b", "c"]
    assert rerank_order_is_monotonic(ordered)


def test_finalize_reranked_chunks_applies_top_k_after_sort() -> None:
    chunks = [
        _chunk("low", rerank=0.10),
        _chunk("high", rerank=0.90),
        _chunk("mid", rerank=0.55),
    ]
    finalized = finalize_reranked_chunks(chunks, top_k=2, rerank_applied=True, query="formula")
    assert [item.chunk_id for item in finalized] == ["high", "mid"]
    assert rerank_order_is_monotonic(finalized)


def test_bm25_scores_do_not_override_rerank_order() -> None:
    chunks = [
        _chunk("bm25-high", rerank=0.20, bm25=99.0),
        _chunk("rerank-high", rerank=0.95, bm25=1.0),
    ]
    ordered = sort_chunks_by_rerank_score(chunks)
    assert ordered[0].chunk_id == "rerank-high"
    assert chunk_rerank_score(ordered[0]) > chunk_rerank_score(ordered[1])


def test_znx_formula_chunk_promoted_when_highest_rerank_score() -> None:
    formula_chunk = ChunkOut(
        chunk_id="formula",
        id="formula",
        text="The empirical/molecular formula is C20H16FN3O2 for ZNX-101.",
        doc_name="report.docx",
        section_name="2",
        page=1,
        score=6.0,
        source="report.docx",
        graph_context=[],
        highlight_spans=[],
        metadata={"rerank_score": 0.9853},
        retrieval_signals={"bm25_score": 6.0, "rerank_score": 0.9853},
    )
    conclusion_chunk = ChunkOut(
        chunk_id="conclusion",
        id="conclusion",
        text="8. Conclusion The evidence supports molecular formula assignment of ZNX-101.",
        doc_name="report.docx",
        section_name="8",
        page=2,
        score=3.9,
        source="report.docx",
        graph_context=[],
        highlight_spans=[],
        metadata={"rerank_score": 0.5055},
        retrieval_signals={"bm25_score": 3.9, "rerank_score": 0.5055},
    )
    ordered = finalize_reranked_chunks(
        [conclusion_chunk, formula_chunk],
        top_k=2,
        rerank_applied=True,
        query="What is the molecular formula of ZNX-101?",
    )
    assert ordered[0].chunk_id == "formula"
    assert "C20H16FN3O2" in ordered[0].text


def test_resolve_candidate_pool_k_uses_existing_config_defaults() -> None:
    cfg = RetrievalPipelineConfig(candidate_k=40, rerank_top_k=20, final_top_k=10)
    assert resolve_candidate_pool_k(final_top_k=10, pipeline_cfg=cfg) == 40
    assert resolve_candidate_pool_k(final_top_k=10, rerank_top_k=25, pipeline_cfg=cfg) == 40
    assert resolve_candidate_pool_k(final_top_k=10, rerank_top_k=25, candidate_k=30, pipeline_cfg=cfg) == 30
