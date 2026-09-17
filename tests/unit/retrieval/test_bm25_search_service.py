"""Unit tests for standalone BM25 search service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.features.retrieval.strategies.bm25.executor import bm25_score_from_object, execute_bm25
from src.features.retrieval.strategies.bm25.bm25_search_service import Bm25SearchService
from src.features.retrieval.application.retrieval_service import RetrievalService


class _FakeBm25Query:
    def __init__(self, objects):
        self._objects = objects
        self.last_kwargs = {}

    def bm25(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(objects=self._objects)


def _chunk_obj(chunk_id: str, text: str, *, document_id: str = "doc-1", score: float = 9.5):
    return SimpleNamespace(
        uuid=chunk_id,
        properties={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": text,
            "section": "Batch Release",
            "section_name": "Batch Release",
            "page": 4,
            "category": "procedure",
        },
        metadata=SimpleNamespace(score=score, distance=None),
    )


def test_execute_bm25_delegates_to_collection() -> None:
    collection = MagicMock()
    collection.query = _FakeBm25Query([_chunk_obj("c1", "batch release")])
    result = execute_bm25(collection, "batch release", limit=5)
    assert len(result.objects) == 1
    assert collection.query.last_kwargs["query"] == "batch release"
    assert collection.query.last_kwargs["limit"] == 5


def test_bm25_score_from_object_uses_metadata_score() -> None:
    obj = _chunk_obj("c1", "text", score=18.9)
    assert bm25_score_from_object(obj) == pytest.approx(18.9)


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_search_bm25_returns_results(mock_get_repo: MagicMock, _resolve: MagicMock) -> None:
    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    mock_get_repo.return_value = repo_service

    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True
    collection = MagicMock()
    collection.query = _FakeBm25Query(
        [
            _chunk_obj("c1", "batch release criteria", document_id="doc-abc"),
            _chunk_obj("c2", "unrelated", document_id="doc-other", score=1.2),
        ]
    )
    fake_client.collections.get.return_value = collection

    service = Bm25SearchService()
    service._client = fake_client

    payload = service.search_bm25("repo-1", "batch release", top_k=20)

    assert payload["repository_id"] == "repo-1"
    assert payload["query"] == "batch release"
    assert payload["result_count"] == 2
    assert payload["results"][0]["chunk_id"] == "c1"
    assert payload["results"][0]["bm25_score"] == pytest.approx(9.5)
    assert payload["metrics"]["chunks_returned"] == 2
    assert payload["metrics"]["total_ms"] >= 0


@patch("src.infrastructure.vector_store.shared_weaviate.chunk_index_sync.purge_document_chunks_from_collection")
@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_delete_document_purges_chunks(
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
    mock_purge: MagicMock,
) -> None:
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True
    fake_client.collections.get.return_value = MagicMock()
    mock_purge.return_value = {"attempted": True, "matches": 3, "successful": 3, "failed": 0}

    service = Bm25SearchService()
    service._client = fake_client
    payload = service.delete_document(
        repository_id="repo-1",
        document_id="doc-1",
        document_name="doc-1.pdf",
    )

    assert payload["matches"] == 3
    mock_purge.assert_called_once()


@patch("src.infrastructure.vector_store.shared_weaviate.chunk_index_sync.purge_document_chunks_from_collection")
@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_index_document_prepares_by_purging(
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
    mock_purge: MagicMock,
) -> None:
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True
    fake_client.collections.get.return_value = MagicMock()
    mock_purge.return_value = {"attempted": True, "matches": 1, "successful": 1, "failed": 0}

    service = Bm25SearchService()
    service._client = fake_client
    payload = service.index_document(
        repository_id="repo-1",
        document_id="doc-1",
        document_name="doc-1.pdf",
    )
    assert payload["prepared_for_indexing"] is True


@patch("src.features.repositories.infrastructure.weaviate_admin.delete_collection")
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_delete_repository_removes_collection(mock_get_repo: MagicMock, mock_delete: MagicMock) -> None:
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
    }
    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True

    service = Bm25SearchService()
    service._client = fake_client
    payload = service.delete_repository("repo-1")

    assert payload["deleted"] is True
    mock_delete.assert_called_once_with("RepoCollection")


def test_retrieval_service_keyword_uses_bm25_executor() -> None:
    bm25 = MagicMock()
    bm25.execute_bm25_on_collection.return_value = SimpleNamespace(objects=[])
    service = RetrievalService(enabled=False, bm25_service=bm25)
    collection = MagicMock()
    service._execute_collection_query(
        collection,
        query_text="LOT-12345",
        query_vector=None,
        search_mode="keyword",
        common={"limit": 10},
        hybrid_alpha=0.75,
    )
    bm25.execute_bm25_on_collection.assert_called_once()


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_search_bm25_exact_document_id_in_query(mock_get_repo: MagicMock, _resolve: MagicMock) -> None:
    document_id = "030d8045-6b2a-4f1e-9c3d-8a7b6c5d4e3f"
    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    mock_get_repo.return_value = repo_service

    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True
    collection = MagicMock()
    collection.query = _FakeBm25Query([_chunk_obj("c1", "payload", document_id=document_id)])
    fake_client.collections.get.return_value = collection

    service = Bm25SearchService()
    service._client = fake_client
    payload = service.search_bm25("repo-1", document_id, top_k=5)

    assert payload["result_count"] == 1
    assert payload["results"][0]["document_id"] == document_id


@patch("src.features.repositories.infrastructure.weaviate_admin.list_unique_documents")
@patch("src.features.repositories.infrastructure.weaviate_admin.purge_document_chunks")
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_rebuild_repository_purges_all_documents(
    mock_get_repo: MagicMock,
    mock_purge: MagicMock,
    mock_list: MagicMock,
) -> None:
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    mock_list.return_value = {"documents": ["a.pdf", "b.pdf"]}

    service = Bm25SearchService()
    payload = service.rebuild_repository("repo-1")

    assert payload["documents_purged"] == 2
    assert mock_purge.call_count == 2


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_search_bm25_large_top_k(mock_get_repo: MagicMock, _resolve: MagicMock) -> None:
    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    mock_get_repo.return_value = repo_service

    objects = [_chunk_obj(f"c{i}", f"chunk {i}") for i in range(100)]
    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True
    collection = MagicMock()
    collection.query = _FakeBm25Query(objects)
    fake_client.collections.get.return_value = collection

    service = Bm25SearchService()
    service._client = fake_client
    payload = service.search_bm25("repo-1", "chunk", top_k=100)

    assert payload["result_count"] == 100
    assert payload["metrics"]["chunks_examined"] == 100
