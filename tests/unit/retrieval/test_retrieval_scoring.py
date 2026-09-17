"""Unit tests for Phase 6 complexity analysis and retrieval-confidence scoring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from tests.unit.retrieval_auth_helpers import ADMIN_ACTOR, SECURITY_ALLOW

from src.features.retrieval.schemas.retrieval_schemas import ChunkOut, RetrieveRequest
from src.features.retrieval.scoring.complexity import ComplexityAnalyzer
from src.features.retrieval.scoring.confidence_scorer import RetrievalConfidenceScorer
from src.features.retrieval.scoring.enricher import ChunkScoringEnricher, enrich_retrieval_chunks
from src.features.retrieval.scoring.readability import compute_readability
from src.features.retrieval.application.retrieval_service import RetrievalService


def _chunk(
    chunk_id: str,
    text: str,
    *,
    signals: dict | None = None,
    entities: list | None = None,
) -> ChunkOut:
    return ChunkOut(
        chunk_id=chunk_id,
        id=chunk_id,
        text=text,
        doc_name="doc.pdf",
        section_name="Section",
        page=1,
        score=0.5,
        source="doc.pdf",
        graph_context=[],
        highlight_spans=[],
        metadata={"document_id": "doc-1", "entities": entities or []},
        retrieval_signals=signals or {},
    )


def test_simple_text_has_low_complexity() -> None:
    result = ComplexityAnalyzer().analyze("The cat sat on the mat. It was a sunny day.")
    assert result.complexity_level == "Low"
    assert 0.0 <= result.complexity_score <= 0.35
    assert result.word_count >= 8


def test_technical_text_has_higher_complexity() -> None:
    text = (
        "Per 21 CFR Part 211 and ICH Q7, the HPLC method SOP-1201 validates API assay. "
        "Execute CAPA-4421 when OOS results exceed USP limits for Batch B-9912."
    )
    result = ComplexityAnalyzer().analyze(text)
    assert result.technical_term_density > 0.05
    assert result.complexity_score > 0.25
    assert result.complexity_level in {"Medium", "High"}


def test_structural_features_detect_lists_tables_and_code() -> None:
    list_text = "- Step one\n- Step two\n- Step three\n- Step four"
    table_text = "Name | Value | Unit\nTemperature | 25 | C"
    code_text = "```python\nvalue = calculate()\nprint(value)\n```"

    list_result = ComplexityAnalyzer().analyze(list_text)
    table_result = ComplexityAnalyzer().analyze(table_text)
    code_result = ComplexityAnalyzer().analyze(code_text)

    assert list_result.list_density > 0
    assert table_result.table_density > 0
    assert code_result.structural_density > 0


def test_readability_metrics_normalized() -> None:
    metrics = compute_readability("This is simple. Very simple indeed.")
    assert 0.0 <= metrics.readability_complexity <= 1.0
    assert metrics.flesch_reading_ease > 0
    assert metrics.sentence_count >= 1


def test_entity_density_uses_extracted_entities() -> None:
    text = "Batch release completed."
    without = ComplexityAnalyzer().analyze(text)
    with_entities = ComplexityAnalyzer().analyze(
        text,
        entities=["Batch B-001", "Site A", "2024-01-15"],
    )
    assert with_entities.entity_density >= without.entity_density


def test_very_short_and_long_chunks() -> None:
    short = ComplexityAnalyzer().analyze("Go.")
    long_text = " ".join(["Technical validation step"] * 120) + " ISO 9001 SOP-2200."
    long = ComplexityAnalyzer().analyze(long_text)
    assert short.word_count <= 2
    assert long.word_count > 100
    assert long.complexity_score >= short.complexity_score


def test_confidence_scoring_normalization() -> None:
    chunks = [
        _chunk(
            "a",
            "alpha",
            signals={
                "normalized_vector_score": 0.9,
                "normalized_bm25_score": 0.7,
                "rrf_score": 0.04,
                "rerank_score_normalized": 0.95,
            },
        ),
        _chunk(
            "b",
            "beta",
            signals={
                "normalized_vector_score": 0.4,
                "normalized_bm25_score": 0.2,
                "rrf_score": 0.01,
                "rerank_score_normalized": 0.55,
            },
        ),
    ]
    results = RetrievalConfidenceScorer().score_chunks(chunks, candidate_overlap_pct=55.0)
    assert results[0].retrieval_confidence > results[1].retrieval_confidence
    assert results[0].confidence_level in {"Very High", "High"}
    assert 0.0 <= results[1].retrieval_confidence <= 1.0


def test_enricher_adds_signals_and_summary() -> None:
    chunks = [
        _chunk(
            "a",
            "Simple sentence for testing.",
            signals={"normalized_vector_score": 0.8, "rrf_score": 0.03},
        ),
        _chunk(
            "b",
            "HPLC SOP-100 validates 21 CFR Part 11 controls.",
            signals={"normalized_bm25_score": 0.6, "rrf_score": 0.02},
        ),
    ]
    enriched, summary = ChunkScoringEnricher().enrich(chunks, candidate_overlap_pct=40.0)
    assert len(enriched) == 2
    assert "retrieval_confidence" in enriched[0].retrieval_signals
    assert "complexity" in enriched[0].retrieval_signals
    assert summary.average_confidence > 0
    assert sum(summary.complexity_distribution.values()) == 2
    assert sum(summary.confidence_distribution.values()) == 2


def test_distribution_statistics() -> None:
    chunks = [
        _chunk("1", "Easy text.", signals={"rerank_score_normalized": 0.95}),
        _chunk("2", "Also easy.", signals={"rerank_score_normalized": 0.88}),
        _chunk("3", "ISO HPLC SOP validation.", signals={"rerank_score_normalized": 0.62}),
    ]
    _enriched, summary_dict = enrich_retrieval_chunks(chunks)
    assert "complexity_distribution" in summary_dict
    assert "confidence_distribution" in summary_dict
    assert summary_dict["highest_confidence"] >= summary_dict["lowest_confidence"]


class _FakeQuery:
    def __init__(self, objects):
        self.objects = objects

    def bm25(self, **_kwargs):
        return SimpleNamespace(objects=list(reversed(self.objects)))

    def near_vector(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)


class _FakeCollection:
    def __init__(self, objects):
        self.query = _FakeQuery(objects)


class _FakeCollections:
    def __init__(self, collection):
        self.collection = collection

    def get(self, _name):
        return self.collection


class _FakeClient:
    def __init__(self, collection):
        self.collections = _FakeCollections(collection)


def _fake_obj(uuid: str, text: str, score: float):
    return SimpleNamespace(
        uuid=uuid,
        properties={
            "chunk_id": uuid,
            "doc_name": "doc.pdf",
            "section_name": "Test",
            "page": 1,
            "text": text,
            "document_id": f"doc-{uuid}",
            "entities": ["Batch A"],
        },
        metadata=SimpleNamespace(score=score, distance=None),
    )


def test_retrieve_response_includes_complexity_and_confidence_signals() -> None:
    objects = [
        _fake_obj("first", "First more relevant shared result", 0.95),
        _fake_obj("second", "Second more relevant shared result", 0.20),
    ]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.1, 0.2]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    body = RetrieveRequest(
        query="more relevant shared",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="hybrid",
        use_production_pipeline=False,
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch("src.features.retrieval.metrics.exporter.export_retrieval_metrics"),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {}},
        ),
    ):
        response = asyncio.run(service.retrieve(body))

    assert response.retrieval_summary is not None
    assert "average_complexity" in response.retrieval_summary
    assert "average_confidence" in response.retrieval_summary
    assert response.results
    signals = response.results[0].retrieval_signals
    assert "retrieval_confidence" in signals
    assert "confidence_level" in signals
    assert "complexity" in signals
    assert "score" in signals["complexity"]
    assert "level" in signals["complexity"]
