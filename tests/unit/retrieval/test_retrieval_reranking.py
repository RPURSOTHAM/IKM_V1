from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest
from src.features.retrieval.application.retrieval_service import RetrievalService


class _FakeReranker:
    def predict(self, pairs):
        return [0.1 if "first" in text.lower() else 4.0 for _query, text in pairs]


class _FakeQuery:
    def __init__(self, objects):
        self.objects = objects

    def hybrid(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)

    def bm25(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)

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
        },
        metadata=SimpleNamespace(score=score, distance=None),
    )


async def _run_retrieve(*, use_rerank: bool | None, repo_reranking: bool = True):
    objects = [
        _fake_obj("first", "First more relevant but lower rerank result", 0.95),
        _fake_obj("second", "Second more relevant result", 0.20),
    ]
    collection = _FakeCollection(objects)
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(collection)
    service._model_error = None
    service._query_vector = lambda *_args, **_kwargs: [0.1, 0.2]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    request_kwargs = {
        "query": "more relevant",
        "repository_id": "12345678-1234-1234-1234-123456789012",
    }
    if use_rerank is not None:
        request_kwargs["use_rerank"] = use_rerank
    body = RetrieveRequest(**request_kwargs)

    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=None),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={
                "weaviate_collection": "TestCollection",
                "settings": {
                    "reranking": repo_reranking,
                    "lexical_composition": True,
                    "reranker_model": "fake-reranker",
                },
            },
        ),
        patch.object(service, "_get_reranker", return_value=_FakeReranker()),
    ):
        return await service.retrieve(body)


def test_retrieve_reranks_when_requested() -> None:
    without_rerank = asyncio.run(_run_retrieve(use_rerank=False))
    with_rerank = asyncio.run(_run_retrieve(use_rerank=True))

    assert [item.id for item in without_rerank.results] == ["first", "second"]
    assert [item.id for item in with_rerank.results] == ["second", "first"]
    assert "rerank" not in without_rerank.pipeline_stages_executed
    assert "rerank" in with_rerank.pipeline_stages_executed
    assert with_rerank.results[0].retrieval_signals["rerank_score"] == 4.0
    assert "original_score" in with_rerank.results[0].retrieval_signals
    assert with_rerank.results[0].retrieval_signals["original_score"] > 0


def test_repository_reranking_default_enables_rerank() -> None:
    response = asyncio.run(_run_retrieve(use_rerank=None, repo_reranking=True))

    assert [item.id for item in response.results] == ["second", "first"]
    assert "rerank" in response.pipeline_stages_executed


def test_retrieve_requires_repository_scope() -> None:
    service = RetrievalService()
    service.ready = True
    service._client = object()
    body = RetrieveRequest(query="cleaning procedure")
    cfg = SimpleNamespace(enable_retrieval=True)

    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=None),
    ):
        response = asyncio.run(service.retrieve(body))

    assert response.results == []
    assert response.grounding_summary["skipped"] == "repository_id is required for repository-scoped retrieval"


def test_repo_scoped_retrieve_returns_empty_without_query_evidence() -> None:
    objects = [
        _fake_obj("first", "Deviation management and product disposition content", 0.95),
    ]
    collection = _FakeCollection(objects)
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(collection)
    service._model_error = None
    service._query_vector = lambda *_args, **_kwargs: [0.1, 0.2]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    body = RetrieveRequest(
        query="cleaning validation",
        repository_id="12345678-1234-1234-1234-123456789012",
    )

    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=None),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={
                "weaviate_collection": "TestCollection",
                "settings": {"lexical_composition": True},
            },
        ),
    ):
        response = asyncio.run(service.retrieve(body))

    assert response.results == []
    assert response.grounding_summary["skipped"] == "no relevant evidence found in selected repository"
