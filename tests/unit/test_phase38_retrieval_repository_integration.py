"""Phase 3.8 — retrieval repository validation and settings consumption."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.shared.errors import DmsServiceError
from src.features.repositories.domain.repository_exceptions import NotFoundError, RepositoryInactiveError
from src.features.retrieval.application.repository_gate import (
    load_active_repository_context,
    retrieval_settings_from_context,
)
from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest
from src.features.retrieval.application.retrieval_service import RetrievalService


def _active_context(**overrides) -> dict:
    base = {
        "repository_id": "repo-1",
        "name": "PlantDocs",
        "weaviate_collection": "PlantDocs",
        "default_tenant_id": None,
        "status": "active",
        "settings_locked": True,
        "settings": {
            "retrieval_search_mode": "hybrid",
            "lexical_composition": True,
            "reranking": True,
            "embedding_model": {
                "provider": "local",
                "model_id": "bge-base-en",
                "local_model_dir": "bge-base-en",
            },
            "chunk_size": 512,
        },
        "processing_hints": {},
    }
    base.update(overrides)
    if "settings" in overrides:
        base["settings"] = overrides["settings"]
    return base


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_load_active_repository_context_success(mock_get_repo: MagicMock) -> None:
    mock_get_repo.return_value.validate_repository_active.return_value = _active_context()
    ctx = load_active_repository_context("repo-1")
    assert ctx["status"] == "active"
    assert ctx["weaviate_collection"] == "PlantDocs"
    assert retrieval_settings_from_context(ctx)["retrieval_search_mode"] == "hybrid"


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_load_active_rejects_unknown(mock_get_repo: MagicMock) -> None:
    mock_get_repo.return_value.validate_repository_active.side_effect = NotFoundError(
        "Repository not found.",
        details={"repository_id": "missing"},
    )
    with pytest.raises(DmsServiceError) as exc:
        load_active_repository_context("missing")
    assert exc.value.code == "repository_not_found"
    assert exc.value.http_status == 404


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_load_active_rejects_archived(mock_get_repo: MagicMock) -> None:
    mock_get_repo.return_value.validate_repository_active.side_effect = RepositoryInactiveError(
        "Repository is not active.",
        details={"repository_id": "repo-1", "status": "archived"},
    )
    with pytest.raises(DmsServiceError) as exc:
        load_active_repository_context("repo-1")
    assert exc.value.code == "repository_inactive"
    assert exc.value.http_status == 409


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_load_active_rejects_configuring(mock_get_repo: MagicMock) -> None:
    mock_get_repo.return_value.validate_repository_active.side_effect = RepositoryInactiveError(
        "Repository is not active.",
        details={"repository_id": "repo-1", "status": "configuring"},
    )
    with pytest.raises(DmsServiceError) as exc:
        load_active_repository_context("repo-1")
    assert exc.value.code == "repository_inactive"


def test_effective_search_mode_request_overrides_repository_default() -> None:
    service = RetrievalService()
    body = RetrieveRequest(query="q", repository_id="11111111-1111-4111-8111-111111111111", search_mode="hybrid")

    assert service._effective_search_mode(body, {"retrieval_search_mode": "vector"}) == "hybrid"
    assert service._effective_search_mode(body, {"retrieval_search_mode": "keyword"}) == "hybrid"
    assert service._effective_search_mode(body, {"retrieval_search_mode": "hybrid"}) == "hybrid"

    vector_body = RetrieveRequest(
        query="q",
        repository_id="11111111-1111-4111-8111-111111111111",
        search_mode="vector",
    )
    assert service._effective_search_mode(vector_body, {"retrieval_search_mode": "hybrid"}) == "vector"

    keyword_body = RetrieveRequest(
        query="q",
        repository_id="11111111-1111-4111-8111-111111111111",
        search_mode="keyword",
    )
    assert service._effective_search_mode(keyword_body, {"retrieval_search_mode": "hybrid"}) == "keyword"

    default_body = RetrieveRequest(
        query="q",
        repository_id="11111111-1111-4111-8111-111111111111",
    )
    assert service._effective_search_mode(default_body, {"retrieval_search_mode": "vector"}) == "vector"
    # lexical disabled forces vector for hybrid/keyword
    assert (
        service._effective_search_mode(
            body,
            {"retrieval_search_mode": "hybrid", "lexical_composition": False},
        )
        == "vector"
    )


def test_effective_search_mode_falls_back_without_repo_settings() -> None:
    service = RetrievalService()
    body = RetrieveRequest(
        query="q",
        repository_id="11111111-1111-4111-8111-111111111111",
        search_mode="vector",
    )
    assert service._effective_search_mode(body, None) == "vector"


@patch("src.features.retrieval.application.repository_gate.load_active_repository_context")
def test_resolve_repository_context_requires_active(mock_load: MagicMock) -> None:
    mock_load.return_value = _active_context()
    service = RetrievalService()
    body = RetrieveRequest(query="q", repository_id="11111111-1111-4111-8111-111111111111")
    ctx = service._resolve_repository_context(body)
    assert ctx is not None
    assert ctx["weaviate_collection"] == "PlantDocs"
    mock_load.assert_called_once_with("11111111-1111-4111-8111-111111111111")


def test_resolve_repository_context_none_without_repository_id() -> None:
    service = RetrievalService()
    body = RetrieveRequest(query="q", search_all_repositories=True)
    # search_all may still require scope depending on validators — use empty optional path
    body_no_repo = RetrieveRequest.model_construct(query="q", repository_id=None, search_mode="hybrid")
    assert service._resolve_repository_context(body_no_repo) is None


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_bm25_resolve_uses_active_gate(mock_get_repo: MagicMock) -> None:
    from src.features.retrieval.strategies.bm25.bm25_search_service import Bm25SearchService

    mock_get_repo.return_value.validate_repository_active.return_value = _active_context()
    ctx = Bm25SearchService._resolve_repository("repo-1")
    assert ctx["weaviate_collection"] == "PlantDocs"
    mock_get_repo.return_value.validate_repository_active.assert_called_once_with("repo-1")


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_bm25_rejects_inactive(mock_get_repo: MagicMock) -> None:
    from src.features.retrieval.strategies.bm25.bm25_search_service import Bm25SearchService

    mock_get_repo.return_value.validate_repository_active.side_effect = RepositoryInactiveError(
        "Repository is not active.",
        details={"status": "archived"},
    )
    with pytest.raises(DmsServiceError) as exc:
        Bm25SearchService._resolve_repository("repo-1")
    assert exc.value.code == "repository_inactive"


def test_collection_name_from_repo_context() -> None:
    service = RetrievalService()
    body = RetrieveRequest(query="q", repository_id="11111111-1111-4111-8111-111111111111")
    name = service._collection_name(
        body,
        {"weaviate_collection": "PlantDocs_Collection"},
    )
    assert name == "PlantDocs_Collection"
