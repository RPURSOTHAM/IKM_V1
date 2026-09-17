"""Unit tests for Phase 3 retrieval strategies (vector, BM25, hybrid RRF)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from tests.unit.retrieval_auth_helpers import ADMIN_ACTOR, SECURITY_ALLOW

from src.features.authorization.application.authorization_service import AuthenticatedUser
from src.features.authentication.domain.authentication_exceptions import AuthorizationError
from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest
from src.features.retrieval.application.retrieval_service import RetrievalService
from src.features.retrieval.strategies import (
    Bm25RetrievalStrategy,
    HybridRetrievalStrategy,
    RetrievalStrategyContext,
    VectorRetrievalStrategy,
)
from src.features.retrieval.strategies.result_merger import merge_candidates_by_chunk_id
from src.features.retrieval.strategies.normalization import min_max_normalize
from src.features.retrieval.domain.models import RetrievalCandidate
from src.features.retrieval.strategies.rrf import reciprocal_rank_fusion


def _candidate(chunk_id: str, text: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        text=text,
        doc_name="doc.pdf",
        section_name="Section",
        page=1,
        score=score,
        metadata={"document_id": f"doc-{chunk_id}"},
    )


def test_min_max_normalization_scales_to_unit_interval() -> None:
    normalized = min_max_normalize([0.2, 0.8, None])
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(1.0)
    assert normalized[2] is None


def test_merge_candidates_preserves_both_scores() -> None:
    merged = merge_candidates_by_chunk_id(
        [_candidate("a", "A", 0.9), _candidate("b", "B", 0.5)],
        [_candidate("b", "B", 12.0), _candidate("c", "C", 8.0)],
    )
    by_id = {item.chunk_id: item for item in merged}
    assert by_id["b"].vector_score == pytest.approx(0.5)
    assert by_id["b"].bm25_score == pytest.approx(12.0)
    assert len(merged) == 3


def test_rrf_orders_shared_chunk_ahead() -> None:
    dense = [_candidate("a", "A", 0.9), _candidate("b", "B", 0.8)]
    sparse = [_candidate("b", "B", 0.7), _candidate("c", "C", 0.6)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60, source_names=["vector", "bm25"])
    assert [item.chunk_id for item in fused[:3]] == ["b", "a", "c"]


def test_hybrid_strategy_merges_duplicates_and_sets_rrf_score() -> None:
    def fetch(*, mode: str, query_text: str, query_vector, common):
        if mode == "vector":
            return [_candidate("shared", "Shared chunk", 0.95), _candidate("v-only", "Vector only", 0.5)]
        return [_candidate("shared", "Shared chunk", 18.0), _candidate("b-only", "BM25 only", 9.0)]

    ctx = RetrievalStrategyContext(
        repository_id="repo-1",
        query="batch release",
        top_k=3,
        collection=MagicMock(),
        query_vector=[0.1, 0.2],
        common={"limit": 10},
        hybrid_alpha=0.75,
        citation_lookup={},
        query_terms=["batch"],
        min_score=0.0,
        require_evidence=False,
        use_rrf=True,
        fetch_candidates=fetch,
    )
    result = HybridRetrievalStrategy().search(ctx)
    assert result.strategy == "hybrid"
    assert result.metrics.vector_candidates == 2
    assert result.metrics.bm25_candidates == 2
    assert result.metrics.merged_candidates == 3
    assert "reciprocal_rank_fusion" in result.pipeline_stages
    top = result.chunks[0]
    assert top.retrieval_signals["vector_score"] == pytest.approx(0.95)
    assert top.retrieval_signals["bm25_score"] == pytest.approx(18.0)
    assert "normalized_vector_score" in top.retrieval_signals
    assert "normalized_bm25_score" in top.retrieval_signals
    assert "rrf_score" in top.retrieval_signals
    assert "rerank_score" not in top.retrieval_signals


def test_vector_strategy_returns_vector_scores_only() -> None:
    def fetch(*, mode: str, query_text: str, query_vector, common):
        assert mode == "vector"
        return [_candidate("v1", "Vector hit", 0.88)]

    ctx = RetrievalStrategyContext(
        repository_id="repo-1",
        query="vector query",
        top_k=5,
        collection=MagicMock(),
        query_vector=[0.1],
        common={"limit": 5},
        hybrid_alpha=0.75,
        citation_lookup={},
        query_terms=[],
        min_score=0.0,
        require_evidence=False,
        fetch_candidates=fetch,
    )
    result = VectorRetrievalStrategy().search(ctx)
    assert result.pipeline_stages == ["embedding", "near_vector"]
    assert result.chunks[0].retrieval_signals["vector_score"] == pytest.approx(0.88)


def test_bm25_strategy_delegates_to_keyword_fetch() -> None:
    def fetch(*, mode: str, query_text: str, query_vector, common):
        assert mode == "keyword"
        assert query_vector is None
        return [_candidate("b1", "BM25 hit", 14.2)]

    ctx = RetrievalStrategyContext(
        repository_id="repo-1",
        query="keyword query",
        top_k=5,
        collection=MagicMock(),
        query_vector=None,
        common={"limit": 5},
        hybrid_alpha=0.75,
        citation_lookup={},
        query_terms=[],
        min_score=0.0,
        require_evidence=False,
        fetch_candidates=fetch,
    )
    result = Bm25RetrievalStrategy().search(ctx)
    assert result.pipeline_stages == ["bm25"]
    assert result.chunks[0].retrieval_signals["bm25_score"] == pytest.approx(14.2)


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


def test_retrieve_hybrid_uses_strategy_alpha_by_default() -> None:
    objects = [
        _fake_obj("shared", "Shared chunk appears in both lists", 0.95),
        _fake_obj("vector-only", "Vector only chunk", 0.70),
        _fake_obj("bm25-only", "BM25 only chunk", 0.40),
    ]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
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
        query="shared chunk",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="hybrid",
        use_rerank=False,
        use_production_pipeline=False,
    )

    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={
                "weaviate_collection": "TestCollection",
                "settings": {"reranking": False, "lexical_composition": True},
            },
        ),
    ):
        response = asyncio.run(service.retrieve(body))

    assert "alpha_weighted_fusion" in response.pipeline_stages_executed
    assert "cross_encoder_reranker" not in response.pipeline_stages_executed
    assert response.grounding_summary["strategy"] == "hybrid"
    assert response.grounding_summary["vector_candidates"] >= 1
    assert response.grounding_summary["bm25_candidates"] >= 1
    assert response.pipeline_trace["merge_ms"] >= 0
    assert response.results[0].retrieval_signals.get("hybrid_alpha") == 0.75
    assert response.results[0].retrieval_signals.get("fusion_method") == "alpha_weighted_fusion"


def test_retrieve_vector_mode_strategy() -> None:
    objects = [_fake_obj("v1", "Vector result", 0.91)]
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
        query="vector",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="vector",
        use_rerank=False,
        use_production_pipeline=False,
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {"reranking": False}},
        ),
    ):
        response = asyncio.run(service.retrieve(body))
    assert "embedding" in response.pipeline_stages_executed
    assert "near_vector" in response.pipeline_stages_executed
    assert "bm25" not in response.pipeline_stages_executed
    assert response.pipeline_stages_executed[0] == "query_rewrite"


def test_retrieve_keyword_mode_strategy() -> None:
    objects = [_fake_obj("b1", "BM25 result", 0.81)]
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
        query="keyword",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="keyword",
        use_production_pipeline=False,
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {}},
        ),
    ):
        response = asyncio.run(service.retrieve(body))
    assert response.pipeline_stages_executed == ["bm25"]


def test_retrieve_hybrid_records_metrics_for_large_candidate_pool() -> None:
    objects = [_fake_obj(f"c{i}", f"chunk {i}", 0.5 + i * 0.01) for i in range(50)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.1]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    body = RetrieveRequest(
        query="chunk",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="hybrid",
        top_k=20,
        candidate_k=50,
        use_production_pipeline=False,
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {}},
        ),
    ):
        response = asyncio.run(service.retrieve(body))
    assert response.grounding_summary["returned_candidates"] <= 20
    assert response.grounding_summary["merged_candidates"] >= 1


@patch("src.features.users.application.user_service.get_platform_security_service")
@patch("src.application.consumer_api.context.get_current_user_from_context")
def test_retrieve_hybrid_enforces_repository_authorization(
    mock_actor: MagicMock,
    mock_security: MagicMock,
) -> None:
    actor = AuthenticatedUser(user_id="outsider", platform_role=None, auth_method="jwt")
    mock_actor.return_value = actor
    mock_security.return_value.check_retrieval_access.side_effect = AuthorizationError("No repository access")

    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection([]))
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    body = RetrieveRequest(
        query="blocked",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="hybrid",
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {}},
        ),
        pytest.raises(AuthorizationError),
    ):
        asyncio.run(service.retrieve(body))


@patch("src.features.retrieval.application.retrieval_service._apply_retrieval_dlp")
def test_retrieve_hybrid_applies_dlp(mock_dlp: MagicMock) -> None:
    from src.features.retrieval.schemas.retrieval_schemas import ChunkOut

    objects = [_fake_obj("c1", "SSN 123-45-6789", 0.9)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.1]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    mock_dlp.side_effect = lambda chunks: [
        chunk.model_copy(update={"text": "[REDACTED]"}) for chunk in chunks
    ]
    body = RetrieveRequest(
        query="ssn",
        repository_id="12345678-1234-1234-1234-123456789012",
        search_mode="keyword",
        use_production_pipeline=False,
    )
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=ADMIN_ACTOR),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SECURITY_ALLOW),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {}},
        ),
    ):
        response = asyncio.run(service.retrieve(body))
    mock_dlp.assert_called_once()
    assert response.results[0].text == "[REDACTED]"
