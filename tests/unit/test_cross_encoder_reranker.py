"""Unit tests for Phase 4 cross-encoder reranking."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from tests.unit.retrieval_auth_helpers import ADMIN_ACTOR, SECURITY_ALLOW

from src.features.retrieval.reranking.config import RerankerConfig
from src.features.retrieval.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut, RetrieveRequest
from src.features.retrieval.application.retrieval_service import RetrievalService


def _chunk(chunk_id: str, text: str, *, rrf: float = 0.03, vector: float = 0.9, bm25: float = 12.0) -> ChunkOut:
    return ChunkOut(
        chunk_id=chunk_id,
        id=chunk_id,
        text=text,
        doc_name="doc.pdf",
        section_name="Section",
        page=1,
        score=rrf,
        source="doc.pdf",
        graph_context=[],
        highlight_spans=[],
        metadata={"document_id": f"doc-{chunk_id}"},
        grounding_score=0.42,
        quality_score=0.55,
        retrieval_signals={
            "vector_score": vector,
            "bm25_score": bm25,
            "rrf_score": rrf,
            "score": rrf,
        },
    )


class _BatchTrackingReranker:
    def __init__(self, batch_size_hint: int = 2):
        self.batch_size_hint = batch_size_hint
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [4.0 if "second" in text.lower() else 0.1 for _query, text in pairs]


def test_cross_encoder_reranker_reorders_by_score() -> None:
    model = _BatchTrackingReranker()
    reranker = CrossEncoderReranker(lambda _path: model, config=RerankerConfig(enabled=True, batch_size=8))
    candidates = [
        _chunk("first", "First less relevant result", rrf=0.04),
        _chunk("second", "Second more relevant result", rrf=0.02),
    ]
    result = reranker.rerank("more relevant", candidates, top_k=2, top_k_before=2)
    assert result.applied is True
    assert [item.chunk_id for item in result.chunks] == ["second", "first"]
    assert result.chunks[0].retrieval_signals["rerank_score"] == pytest.approx(4.0)
    assert result.chunks[0].retrieval_signals["vector_score"] == pytest.approx(0.9)
    assert result.chunks[0].retrieval_signals["bm25_score"] == pytest.approx(12.0)
    assert result.chunks[0].retrieval_signals["rrf_score"] == pytest.approx(0.02)
    assert result.chunks[0].grounding_score == pytest.approx(0.42)
    assert result.chunks[0].quality_score == pytest.approx(0.55)
    assert result.chunks[0].score != result.chunks[0].grounding_score


def test_cross_encoder_reranker_disabled_preserves_order() -> None:
    reranker = CrossEncoderReranker(lambda _path: None, config=RerankerConfig(enabled=False))
    candidates = [_chunk("a", "A"), _chunk("b", "B")]
    result = reranker.rerank("query", candidates, top_k=2)
    assert result.applied is False
    assert [item.chunk_id for item in result.chunks] == ["a", "b"]


def test_cross_encoder_reranker_batches_inference() -> None:
    model = _BatchTrackingReranker()
    reranker = CrossEncoderReranker(
        lambda _path: model,
        config=RerankerConfig(enabled=True, batch_size=2, top_k_before=5),
    )
    candidates = [_chunk(f"c{i}", f"chunk {i}") for i in range(5)]
    result = reranker.rerank("query", candidates, top_k=5, top_k_before=5)
    assert result.applied is True
    assert len(model.calls) == 3
    assert sum(len(batch) for batch in model.calls) == 5
    assert result.metrics.batch_count == 3


def test_cross_encoder_reranker_empty_and_single_candidate() -> None:
    model = _BatchTrackingReranker()
    reranker = CrossEncoderReranker(lambda _path: model, config=RerankerConfig(enabled=True))
    empty = reranker.rerank("query", [], top_k=5)
    assert empty.applied is False
    assert empty.chunks == []

    single = reranker.rerank("query", [_chunk("only", "only chunk")], top_k=1)
    assert single.applied is True
    assert len(single.chunks) == 1
    assert model.calls == [[("query", "only chunk")]]


class _FakeQuery:
    def __init__(self, objects):
        self.objects = objects

    def hybrid(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)

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


def test_hybrid_retrieve_with_reranker_changes_order_only() -> None:
    objects = [
        _fake_obj("first", "First less relevant but still more relevant result", 0.95),
        _fake_obj("second", "Second more relevant result", 0.20),
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
        query="more relevant",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="hybrid",
        use_rerank=True,
        use_production_pipeline=False,
    )
    fake_reranker = _BatchTrackingReranker()

    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(service, "_get_reranker", return_value=fake_reranker),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={
                "weaviate_collection": "TestCollection",
                "settings": {"reranking": True, "lexical_composition": True},
            },
        ),
    ):
        response = asyncio.run(service.retrieve(body))

    assert "alpha_weighted_fusion" in response.pipeline_stages_executed
    assert "rerank" in response.pipeline_stages_executed
    assert response.pipeline_trace.get("rerank_ms", 0) >= 0
    assert [item.id for item in response.results] == ["second", "first"]
    assert response.results[0].retrieval_signals["score"] is not None
    assert response.results[0].retrieval_signals["rerank_score"] == pytest.approx(4.0)


def test_vector_and_keyword_modes_compatible_with_reranker() -> None:
    service = RetrievalService()
    service.ready = True
    objects = [_fake_obj("v1", "Vector relevant content", 0.9)]
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.2]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    fake_reranker = MagicMock()
    fake_reranker.predict.return_value = [3.5]

    for mode in ("vector", "keyword"):
        body = RetrieveRequest(
            query="relevant",
            repository_id="12345678-1234-1234-1234-123456789012",
            search_mode=mode,
            use_rerank=True,
            use_production_pipeline=False,
        )
        with (
            patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
            patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
            patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
            patch.object(service, "_collection_exists", return_value=True),
            patch.object(service, "_get_reranker", return_value=fake_reranker),
            patch.object(
                service,
                "_resolve_repository_context",
                return_value={"weaviate_collection": "TestCollection", "settings": {"reranking": True}},
            ),
        ):
            response = asyncio.run(service.retrieve(body))
        assert "rerank" in response.pipeline_stages_executed
        assert response.results[0].retrieval_signals["rerank_score"] == pytest.approx(3.5)
