"""Dedicated RC regression tests for retrieval/catalog audit findings."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from tests.unit.retrieval_auth_helpers import ADMIN_ACTOR, SECURITY_ALLOW

from src.features.retrieval.application.document_catalog import (
    aggregate_indexed_metadata,
    record_to_catalog_item,
)
from src.features.retrieval.application.pipeline.stages import HeuristicQueryRewriter
from src.features.retrieval.schemas.retrieval_schemas import DocumentSearchRequest, RetrieveRequest
from src.features.retrieval.scoring.quality_scorer import (
    compute_grounding_score,
    compute_highlight_spans,
    compute_quality_flags,
    compute_quality_score,
)
from src.features.retrieval.application.retrieval_service import RetrievalService
from src.features.retrieval.strategies.result_merger import rank_merged_by_alpha
from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
from src.features.retrieval.domain.models import PipelineState, RetrievalCandidate


REPO_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
REPO_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _fake_obj(chunk_id: str, text: str, score: float, *, repository_id: str = REPO_A, document_id: str = "doc-1"):
    return SimpleNamespace(
        uuid=chunk_id,
        properties={
            "chunk_id": chunk_id,
            "doc_name": "synthetic.pdf",
            "document_name": "synthetic.pdf",
            "document_id": document_id,
            "repository_id": repository_id,
            "section_name": "1.0 PURPOSE",
            "page": 1,
            "line_start": 1,
            "line_end": 2,
            "text": text,
        },
        metadata=SimpleNamespace(distance=None, score=score),
    )


class _FakeCollection:
    def __init__(self, objects):
        self.objects = objects
        self.query = SimpleNamespace(
            near_vector=lambda **_k: SimpleNamespace(objects=objects),
            bm25=lambda **_k: SimpleNamespace(objects=objects),
            hybrid=lambda **_k: SimpleNamespace(objects=objects),
            fetch_objects=lambda **_k: SimpleNamespace(objects=objects),
        )


class _FakeClient:
    def __init__(self, collection):
        self._collection = collection
        self.collections = SimpleNamespace(
            get=lambda *_a, **_k: collection,
            exists=lambda *_a, **_k: True,
        )

    def is_ready(self):
        return True


def _cfg(**overrides):
    base = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _retrieve(service: RetrievalService, body: RetrieveRequest, *, repo_settings=None):
    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=_cfg()),
        patch(
            "src.application.consumer_api.context.get_current_user_from_context",
            return_value=ADMIN_ACTOR,
        ),
        patch(
            "src.features.users.application.user_service.get_platform_security_service",
            return_value=SECURITY_ALLOW,
        ),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={
                "weaviate_collection": "TestCollection",
                "settings": repo_settings
                or {"retrieval_search_mode": "hybrid", "reranking": False, "lexical_composition": True},
            },
        ),
    ):
        return asyncio.run(service.retrieve(body))


def test_rc1_search_mode_vector() -> None:
    objects = [_fake_obj("v1", "1.0 PURPOSE cleaning controls", 0.9)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.1, 0.2]
    body = RetrieveRequest(
        query="PURPOSE",
        repository_id=REPO_A,
        search_mode="vector",
        use_rerank=False,
        use_production_pipeline=False,
        debug_retrieval=True,
    )
    response = _retrieve(service, body)
    assert response.retrieval_summary["strategy"] == "vector"
    assert response.retrieval_summary.get("bm25_candidates", 0) in {0, None}
    assert "bm25" not in response.pipeline_stages_executed
    assert float(response.pipeline_trace.get("bm25_search_ms") or 0.0) == 0.0


def test_rc1_search_mode_keyword() -> None:
    objects = [_fake_obj("k1", "1.0 PURPOSE cleaning controls", 0.8)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    body = RetrieveRequest(
        query="PURPOSE",
        repository_id=REPO_A,
        search_mode="keyword",
        use_rerank=False,
        use_production_pipeline=False,
        debug_retrieval=True,
    )
    response = _retrieve(service, body)
    assert response.retrieval_summary["strategy"] == "keyword"
    assert response.retrieval_summary.get("vector_candidates", 0) in {0, None}
    assert "near_vector" not in response.pipeline_stages_executed
    assert float(response.pipeline_trace.get("vector_search_ms") or 0.0) == 0.0


def test_rc1_search_mode_hybrid() -> None:
    objects = [_fake_obj("h1", "1.0 PURPOSE cleaning controls", 0.85)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.1, 0.2]
    body = RetrieveRequest(
        query="PURPOSE",
        repository_id=REPO_A,
        search_mode="hybrid",
        use_rerank=False,
        use_production_pipeline=False,
        debug_retrieval=True,
    )
    response = _retrieve(service, body)
    assert response.retrieval_summary["strategy"] == "hybrid"
    assert response.retrieval_summary["vector_candidates"] >= 1
    assert response.retrieval_summary["bm25_candidates"] >= 1


def test_rc2_hybrid_alpha_default() -> None:
    req = RetrieveRequest(query="q", repository_id=REPO_A, search_mode="hybrid")
    assert req.hybrid_alpha is None
    assert _cfg().hybrid_alpha == 0.75
    aliased = RetrieveRequest(query="q", repository_id=REPO_A, search_mode="hybrid", alpha=0.25)
    assert aliased.hybrid_alpha == 0.25


def test_rc2_hybrid_alpha_0() -> None:
    from src.features.retrieval.strategies.result_merger import apply_normalized_scores
    from src.features.retrieval.strategies.models import MergedCandidate

    items = [
        MergedCandidate(
            chunk_id="a",
            document_id="d1",
            text="A",
            page=1,
            section="s",
            metadata={},
            doc_name="d",
            vector_score=1.0,
            bm25_score=0.1,
        ),
        MergedCandidate(
            chunk_id="b",
            document_id="d1",
            text="B",
            page=1,
            section="s",
            metadata={},
            doc_name="d",
            vector_score=0.1,
            bm25_score=1.0,
        ),
    ]
    apply_normalized_scores(items)
    ranked = rank_merged_by_alpha(items, alpha=0.0)
    assert ranked[0][0].chunk_id == "b"


def test_rc2_hybrid_alpha_1() -> None:
    from src.features.retrieval.strategies.result_merger import apply_normalized_scores
    from src.features.retrieval.strategies.models import MergedCandidate

    items = [
        MergedCandidate(
            chunk_id="a",
            document_id="d1",
            text="A",
            page=1,
            section="s",
            metadata={},
            doc_name="d",
            vector_score=1.0,
            bm25_score=0.1,
        ),
        MergedCandidate(
            chunk_id="b",
            document_id="d1",
            text="B",
            page=1,
            section="s",
            metadata={},
            doc_name="d",
            vector_score=0.1,
            bm25_score=1.0,
        ),
    ]
    apply_normalized_scores(items)
    ranked = rank_merged_by_alpha(items, alpha=1.0)
    assert ranked[0][0].chunk_id == "a"


def test_rc3_catalog_indexed_chunk_count() -> None:
    stats = aggregate_indexed_metadata(
        [
            SimpleNamespace(
                properties={"document_id": "doc-1", "doc_name": "uuid.pdf", "original_file_name": "orig.pdf"},
                vector={"default": [0.1]},
            ),
            SimpleNamespace(
                properties={"document_id": "doc-1", "doc_name": "uuid.pdf", "original_file_name": "orig.pdf"},
                vector={"default": [0.2]},
            ),
            SimpleNamespace(
                properties={"document_id": "doc-1", "doc_name": "uuid.pdf", "original_file_name": "orig.pdf"},
                vector=None,
            ),
            SimpleNamespace(
                properties={"document_id": "doc-1", "doc_name": "uuid.pdf", "original_file_name": "orig.pdf"},
                vector={"default": [0.3]},
            ),
        ]
    )
    item = record_to_catalog_item(
        {
            "document_id": "doc-1",
            "document_name": "uuid.pdf",
            "original_file_name": "orig.pdf",
            "document_type": "pdf",
            "status": "completed",
            "repository_id": REPO_A,
            "collection_name": "C",
            "metadata": {},
        },
        indexed_stats=stats,
    )
    assert item.indexed.chunk_count == 4
    assert item.indexed.embedded_chunk_count == 3
    assert item.indexed.indexed is True


@patch.object(RetrievalService, "_resolve_collection_context", return_value=("CollectionA", None, None))
@patch.object(RetrievalService, "_retrieval_enabled", return_value=True)
@patch.object(RetrievalService, "_check_retrieval_authorization")
@patch.object(RetrievalService, "_load_catalog_records")
def test_rc4_document_search_repository_scope(
    mock_load: MagicMock,
    _mock_auth: MagicMock,
    _mock_enabled: MagicMock,
    _mock_resolve: MagicMock,
) -> None:
    mock_load.return_value = (
        [
            {
                "document_id": "doc-a",
                "document_name": "a.txt",
                "original_file_name": "a.txt",
                "document_type": "txt",
                "status": "completed",
                "repository_id": REPO_A,
                "collection_name": "A",
                "metadata": {},
            }
        ],
        1,
    )
    service = RetrievalService()
    service.ready = True
    service._client = MagicMock()
    service._collection_exists = MagicMock(return_value=False)
    body = DocumentSearchRequest(query="A_MARKER", repository_id=REPO_A)
    asyncio.run(service.search_documents_catalog(body))
    assert mock_load.call_args.kwargs["intake_filters"]["repository_id"] == REPO_A


def test_rc5_quality_flags_generated() -> None:
    flags = compute_quality_flags(
        text="",
        metadata={},
        evidence_terms=[],
        query_terms=["purpose"],
    )
    assert "missing_text" in flags
    assert "weak_match" in flags
    assert "missing_metadata" in flags
    good = compute_quality_flags(
        text="This procedure defines cleaning controls for manufacturing areas with enough text.",
        metadata={"page": 1, "line_start": 1, "section": "PURPOSE"},
        evidence_terms=["purpose"],
        query_terms=["purpose"],
        retrieval_score=0.9,
        grounding_score=0.9,
    )
    assert good == []


def test_rc6_grounding_score_independent() -> None:
    high = compute_grounding_score(
        retrieval_score=0.9,
        evidence_terms=["purpose", "cleaning"],
        query_terms=["purpose", "cleaning"],
        text="PURPOSE cleaning controls",
        metadata={"page": 1, "section": "PURPOSE"},
    )
    low = compute_grounding_score(
        retrieval_score=0.9,
        evidence_terms=[],
        query_terms=["purpose", "cleaning"],
        text="unrelated content without the query terms",
        metadata={},
    )
    assert high != low
    assert high != 0.9
    assert low != 0.9
    assert low < high


def test_rc8_retrieve_repository_id() -> None:
    objects = [_fake_obj("c1", "1.0 PURPOSE cleaning controls", 0.9, repository_id=REPO_A)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.1, 0.2]
    body = RetrieveRequest(
        query="PURPOSE",
        repository_id=REPO_A,
        search_mode="vector",
        use_rerank=False,
        use_production_pipeline=False,
    )
    response = _retrieve(service, body)
    assert response.results
    assert response.results[0].repository_id == REPO_A


def test_query_expansion_meaningful() -> None:
    rewriter = HeuristicQueryRewriter()
    rewriter._qwen_rewrite = lambda _q, _m: ["cleaning SOP", "sanitation procedure"]
    state = PipelineState(original_query="cleaning procedure")
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert result.status == "ok"
    assert state.expanded_queries
    assert "cleaning procedure" not in {q.lower() for q in state.expanded_queries} or len(state.expanded_queries) > 1
    assert any(q.lower() != "cleaning procedure" for q in state.expanded_queries)
    disabled = PipelineState(original_query="cleaning procedure")
    off = rewriter.run(disabled, RetrievalPipelineConfig(enable_query_rewrite=False))
    assert off.status == "skipped"
    assert disabled.expanded_queries == []


def test_highlight_spans_generated() -> None:
    spans = compute_highlight_spans("1.0 PURPOSE This procedure defines cleaning", ["PURPOSE", "cleaning"])
    assert spans
    text = "1.0 PURPOSE This procedure defines cleaning"
    for start, end in spans:
        assert 0 <= start < end <= len(text)
        assert text[start:end].lower() in {"purpose", "cleaning"}
    assert compute_highlight_spans(text, ["zzznomatch"]) == []


def test_subquery_default_pipeline() -> None:
    service = RetrievalService()
    _effective, _exp, subs, meta = service._prepare_query_rewrite(
        query_text="What is the purpose of the procedure and what are the approval requirements?",
        expand_query=True,
    )
    assert len(subs) >= 2
    assert meta.get("expansion_status") in {"expanded", "unavailable", "unchanged"}

    rewriter = HeuristicQueryRewriter()
    state = PipelineState(
        original_query="What is the purpose of the procedure and what are the approval requirements?",
    )
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert result.status == "ok"
    assert len(state.sub_queries) >= 2


def test_quality_score_not_alias_of_retrieval_score() -> None:
    score = 0.9
    quality = compute_quality_score(
        retrieval_score=score,
        text="short",
        metadata={},
        evidence_terms=[],
        query_terms=["purpose"],
    )
    assert quality != score
    assert quality < score


def test_rc1_production_pipeline_honors_search_mode_vector() -> None:
    """Production path must not execute BM25 when search_mode=vector."""
    objects = [_fake_obj("pv1", "1.0 PURPOSE cleaning controls", 0.9)]
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection(objects))
    service._query_vector = lambda *_a, **_k: [0.1, 0.2]
    calls: list[str] = []

    original = service._fetch_mode_candidates

    def _track(collection, *, mode, **kwargs):
        calls.append(mode)
        return original(collection, mode=mode, **kwargs)

    service._fetch_mode_candidates = _track  # type: ignore[method-assign]
    body = RetrieveRequest(
        query="PURPOSE",
        repository_id=REPO_A,
        search_mode="vector",
        use_rerank=False,
        use_production_pipeline=True,
        expand_query=False,
        debug_retrieval=True,
    )
    response = _retrieve(service, body)
    assert "keyword" not in calls
    assert "vector" in calls
    assert response.retrieval_summary is None or response.retrieval_summary.get("strategy") in {None, "vector", "hybrid"}


def test_rc6_rerank_does_not_alias_grounding_quality() -> None:
    from src.features.retrieval.reranking.cross_encoder_reranker import CrossEncoderReranker
    from src.features.retrieval.reranking.config import RerankerConfig
    from src.features.retrieval.schemas.retrieval_schemas import ChunkOut

    class _Model:
        def predict(self, pairs):
            return [3.0 for _ in pairs]

    chunk = ChunkOut(
        chunk_id="r1",
        id="r1",
        text="PURPOSE cleaning controls for manufacturing areas with enough content.",
        doc_name="doc.pdf",
        section_name="PURPOSE",
        page=1,
        score=0.2,
        source="doc.pdf",
        graph_context=[],
        highlight_spans=[],
        metadata={"page": 1, "section": "PURPOSE", "query_evidence_terms": ["purpose"]},
        grounding_score=0.77,
        quality_score=0.66,
        retrieval_signals={"score": 0.2, "rrf_score": 0.2},
    )
    reranker = CrossEncoderReranker(lambda _path: _Model(), config=RerankerConfig(enabled=True))
    result = reranker.rerank("PURPOSE", [chunk], top_k=1, top_k_before=5)
    assert result.applied
    assert result.chunks[0].grounding_score == 0.77
    assert result.chunks[0].quality_score == 0.66
    assert result.chunks[0].score != 0.77
