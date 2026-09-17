"""Unit tests for Phase 5 retrieval metrics and observability."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from src.features.retrieval.metrics.analysis import (
    compute_candidate_analysis,
    compute_quality_metrics,
)
from src.features.retrieval.metrics.collector import RetrievalMetricsCollector
from src.features.retrieval.metrics.models import LatencyMetrics
from src.features.retrieval.metrics.timing import StageTimer
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut, RetrieveRequest
from src.features.retrieval.application.retrieval_service import RetrievalService
from src.features.retrieval.strategies.models import StrategyMetrics
from tests.unit.retrieval_auth_helpers import ADMIN_ACTOR, SECURITY_ALLOW


def _chunk(
    chunk_id: str,
    text: str,
    *,
    vector: float | None = 0.9,
    bm25: float | None = 12.0,
    rrf: float = 0.03,
    rerank: float | None = None,
    page: int = 1,
) -> ChunkOut:
    signals = {
        "vector_score": vector,
        "bm25_score": bm25,
        "rrf_score": rrf,
        "score": rrf,
    }
    if rerank is not None:
        signals["rerank_score"] = rerank
    if vector is None:
        signals.pop("vector_score", None)
    if bm25 is None:
        signals.pop("bm25_score", None)
    return ChunkOut(
        chunk_id=chunk_id,
        id=chunk_id,
        text=text,
        doc_name="doc.pdf",
        section_name="Section A",
        page=page,
        score=rrf,
        source="doc.pdf",
        graph_context=[],
        highlight_spans=[],
        metadata={"document_id": f"doc-{chunk_id}"},
        retrieval_signals=signals,
    )


def test_stage_timer_accumulates_latencies() -> None:
    timer = StageTimer()
    timer.record("merge", 0.4)
    timer.record("fusion", 0.1)
    assert timer.get("merge") == pytest.approx(0.4)
    assert "merge" in timer.as_dict()


def test_candidate_analysis_overlap() -> None:
    analysis = compute_candidate_analysis(
        vector_candidates=40,
        bm25_candidates=40,
        merged_candidates=58,
        chunks=[_chunk("a", "text a"), _chunk("b", "text b", page=2)],
    )
    assert analysis.duplicate_chunks_removed == 22
    assert analysis.shared_hits == 22
    assert analysis.candidate_overlap_percentage == pytest.approx(55.0)


def test_quality_metrics_include_rerank_stats() -> None:
    quality = compute_quality_metrics(
        [
            _chunk("a", "alpha", rerank=4.0),
            _chunk("b", "beta", rerank=1.0),
        ]
    )
    assert quality.average_rerank_score == pytest.approx(2.5)
    assert quality.highest_rerank_score == pytest.approx(4.0)
    assert quality.median_rerank_score == pytest.approx(2.5)


def test_retrieval_metrics_collector_builds_summary() -> None:
    timer = StageTimer()
    timer.record("vector_search_ms", 18.0)
    timer.record("bm25_search_ms", 12.0)
    metrics = RetrievalMetricsCollector.build(
        strategy="hybrid",
        repository_id="repo-1",
        query="batch release",
        top_k_requested=10,
        chunks=[_chunk("c1", "batch release criteria")],
        strategy_metrics=StrategyMetrics(
            vector_candidates=40,
            bm25_candidates=40,
            merged_candidates=58,
            returned_candidates=1,
        ),
        stage_timer=timer,
        embedding_ms=15.0,
        total_ms=89.0,
    )
    summary = metrics.retrieval_summary(debug=False)
    assert summary["strategy"] == "hybrid"
    assert summary["documents_retrieved"] == 1
    assert summary["returned_candidates"] == 1

    debug_summary = metrics.retrieval_summary(debug=True)
    assert debug_summary["vector_candidates"] == 40
    assert debug_summary["duplicates_removed"] == 22

    trace = metrics.latency.pipeline_trace(debug=True)
    assert trace["embedding_ms"] == 15.0
    assert trace["vector_search_ms"] == 18.0


def test_latency_metrics_lightweight_trace() -> None:
    latency = LatencyMetrics(total_ms=89.0, vector_search_ms=18.0, bm25_search_ms=12.0, fusion_ms=0.1, rerank_ms=42.0)
    lightweight = latency.pipeline_trace(debug=False)
    assert "total_ms" in lightweight
    assert "embedding_ms" not in lightweight
    assert lightweight["rerank_ms"] == 42.0


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
        },
        metadata=SimpleNamespace(score=score, distance=None),
    )


def test_hybrid_retrieve_exposes_retrieval_summary_and_trace() -> None:
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
        use_rerank=False,
        use_production_pipeline=False,
        debug_retrieval=True,
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
    assert response.retrieval_summary["strategy"] == "hybrid"
    assert response.pipeline_trace is not None
    assert "total_ms" in response.pipeline_trace
    assert "vector_search_ms" in response.pipeline_trace
    assert response.retrieval_summary["duplicates_removed"] >= 0


def test_debug_mode_includes_quality_and_candidate_stats() -> None:
    chunks = [_chunk("a", "alpha " * 10, rerank=3.0), _chunk("b", "beta " * 5, rerank=1.0, page=2)]
    metrics = RetrievalMetricsCollector.build(
        strategy="hybrid",
        repository_id="repo-1",
        query="alpha",
        top_k_requested=10,
        chunks=chunks,
        strategy_metrics=StrategyMetrics(vector_candidates=4, bm25_candidates=4, merged_candidates=6),
        total_ms=50.0,
    )
    _trace, _summary, quality = RetrievalMetricsCollector.finalize_response_fields(
        metrics,
        debug=True,
        existing_quality_summary={"min_score": 0.0},
    )
    assert quality["average_rerank_score"] == pytest.approx(2.0)
    assert "candidate_analysis" in quality


def test_non_debug_mode_returns_lightweight_summary() -> None:
    objects = [_fake_obj("only", "Vector only result", 0.91)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.2, 0.3]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    body = RetrieveRequest(
        query="vector only",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="vector",
        use_production_pipeline=False,
        debug_retrieval=False,
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
    assert "vector_candidates" not in response.retrieval_summary
    assert response.retrieval_summary["strategy"] == "vector"
    assert "embedding_ms" not in (response.pipeline_trace or {})


def test_bm25_search_exposes_candidate_counts_in_debug_mode() -> None:
    objects = [_fake_obj("b1", "Keyword match text", 2.5)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    body = RetrieveRequest(
        query="keyword match",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="keyword",
        use_production_pipeline=False,
        debug_retrieval=True,
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
    assert response.retrieval_summary["bm25_candidates"] >= 1
    assert response.pipeline_trace is not None
    assert response.pipeline_trace["bm25_search_ms"] >= 0


def test_large_candidate_pool_metrics() -> None:
    analysis = compute_candidate_analysis(
        vector_candidates=80,
        bm25_candidates=80,
        merged_candidates=120,
        chunks=[_chunk(f"c{i}", f"text {i}", page=(i % 5) + 1) for i in range(10)],
    )
    assert analysis.duplicate_chunks_removed == 40
    assert analysis.unique_documents >= 1
    assert analysis.candidate_overlap_percentage == pytest.approx(50.0)


def test_reranked_search_adds_rerank_latency_to_trace() -> None:
    objects = [
        _fake_obj("first", "First more relevant shared chunk", 0.95),
        _fake_obj("second", "Second more relevant shared chunk", 0.20),
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
    fake_reranker = SimpleNamespace(predict=lambda pairs: [4.0 if "second" in t.lower() else 0.1 for _q, t in pairs])
    body = RetrieveRequest(
        query="more relevant shared",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="hybrid",
        use_rerank=True,
        use_production_pipeline=False,
        debug_retrieval=True,
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch("src.features.retrieval.metrics.exporter.export_retrieval_metrics"),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(service, "_get_reranker", return_value=fake_reranker),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {"reranking": True}},
        ),
    ):
        response = asyncio.run(service.retrieve(body))

    assert response.pipeline_trace is not None
    assert response.pipeline_trace.get("rerank_ms", 0) > 0
    assert response.results[0].retrieval_signals.get("vector_score") is not None
    assert response.results[0].retrieval_signals.get("rrf_score") is not None
    assert response.results[0].retrieval_signals.get("rerank_score") is not None


def test_successful_retrieve_records_persisted_retrieval_duration() -> None:
    objects = [
        _fake_obj("first", "First more relevant shared chunk", 0.95),
        _fake_obj("second", "Second more relevant shared chunk", 0.20),
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
        use_rerank=False,
        use_production_pipeline=False,
        debug_retrieval=True,
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch("src.features.retrieval.metrics.exporter.export_retrieval_metrics"),
        patch(
            "src.features.observability.hooks.retrieval_hooks.record_search_completed"
        ) as record_completed,
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {}},
        ),
    ):
        response = asyncio.run(service.retrieve(body))

    assert response.latency_ms > 0
    assert record_completed.called
    kwargs = record_completed.call_args.kwargs
    assert kwargs.get("repository_id") == body.repository_id
    assert kwargs.get("duration_ms") is not None
    assert kwargs.get("result_count") == len(response.results)
    assert kwargs.get("search_mode") == "hybrid"


def test_production_pipeline_missing_reranker_does_not_claim_applied() -> None:
    from src.features.retrieval.application.pipeline.stages import ExistingCrossEncoderReranker
    from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
    from src.features.retrieval.domain.models import PipelineState, RetrievalCandidate

    def _no_score_rerank(query, candidates, config):
        return candidates

    stage = ExistingCrossEncoderReranker(_no_score_rerank)
    state = PipelineState(
        original_query="q",
        fused_candidates=[
            RetrievalCandidate(chunk_id="a", text="alpha", score=0.2, retrieval_signals={"rrf_score": 0.2}),
            RetrievalCandidate(chunk_id="b", text="beta", score=0.1, retrieval_signals={"rrf_score": 0.1}),
        ],
    )
    config = RetrievalPipelineConfig(enable_rerank=True, rerank_top_k=10, final_top_k=5)
    result = stage.run(state, config)
    assert result.status == "degraded"
    assert result.metadata.get("rerank_applied") is False
    assert all(
        (c.retrieval_signals or {}).get("rerank_score") is None for c in state.final_candidates
    )
