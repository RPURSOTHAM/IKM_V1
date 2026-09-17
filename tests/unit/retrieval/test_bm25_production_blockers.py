"""Production blocker coverage for standalone BM25 search."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.features.authorization.application.authorization_service import AuthenticatedUser
from src.features.authentication.domain.authentication_exceptions import AuthorizationError
from src.features.repositories.domain.repository_exceptions import ServiceUnavailableError
from src.features.repositories.application.repository_service import RepositoryService
from src.features.retrieval.strategies.bm25.bm25_search_service import Bm25SearchService
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut


class _FakeBm25Query:
    def __init__(self, objects):
        self._objects = objects

    def bm25(self, **kwargs):
        return SimpleNamespace(objects=self._objects)


def _chunk_obj(chunk_id: str, text: str, *, document_id: str = "doc-1", score: float = 9.5):
    return SimpleNamespace(
        uuid=chunk_id,
        properties={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": text,
            "section": "Section",
            "page": 1,
        },
        metadata=SimpleNamespace(score=score, distance=None),
    )


def _bm25_service_with_results(text: str = "SSN 123-45-6789") -> Bm25SearchService:
    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True
    collection = MagicMock()
    collection.query = _FakeBm25Query([_chunk_obj("c1", text)])
    fake_client.collections.get.return_value = collection

    service = Bm25SearchService()
    service._client = fake_client
    return service


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
@patch("src.features.users.application.user_service.get_platform_security_service")
@patch("src.application.consumer_api.context.get_current_user_from_context")
def test_search_bm25_authorized_request(
    mock_actor: MagicMock,
    mock_security: MagicMock,
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
) -> None:
    actor = AuthenticatedUser(user_id="user-1", platform_role="administrator", auth_method="jwt")
    mock_actor.return_value = actor
    security = MagicMock()
    security.check_retrieval_access.return_value = "consumer"
    mock_security.return_value = security

    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }

    service = _bm25_service_with_results("authorized hit")
    payload = service.search_bm25("repo-1", "authorized query", top_k=5)

    security.check_retrieval_access.assert_called_once_with(actor, "repo-1")
    assert payload["result_count"] == 1


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
@patch("src.features.users.application.user_service.get_platform_security_service")
@patch("src.application.consumer_api.context.get_current_user_from_context")
def test_search_bm25_unauthorized_request(
    mock_actor: MagicMock,
    mock_security: MagicMock,
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
) -> None:
    actor = AuthenticatedUser(user_id="outsider", platform_role=None, auth_method="jwt")
    mock_actor.return_value = actor
    mock_security.return_value.check_retrieval_access.side_effect = AuthorizationError(
        "No repository access for this user"
    )

    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }

    service = _bm25_service_with_results()
    with pytest.raises(AuthorizationError):
        service.search_bm25("repo-1", "blocked query", top_k=5)


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
@patch("src.features.users.application.user_service.get_platform_security_service")
@patch("src.application.consumer_api.context.get_current_user_from_context")
def test_search_bm25_repository_isolation(
    mock_actor: MagicMock,
    mock_security: MagicMock,
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
) -> None:
    actor = AuthenticatedUser(user_id="user-1", platform_role=None, auth_method="jwt")
    mock_actor.return_value = actor
    security = MagicMock()

    def _check(access_actor, repository_id):
        if repository_id != "repo-owned":
            raise AuthorizationError("No repository access for this user")
        return "consumer"

    security.check_retrieval_access.side_effect = _check
    mock_security.return_value = security
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }

    service = _bm25_service_with_results()
    with pytest.raises(AuthorizationError):
        service.search_bm25("repo-other", "cross repo query", top_k=5)


@patch("src.features.retrieval.application.retrieval_service._apply_retrieval_dlp")
@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
@patch("src.application.consumer_api.context.get_current_user_from_context", return_value=None)
def test_search_bm25_applies_retrieval_dlp(
    _mock_actor: MagicMock,
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
    mock_dlp: MagicMock,
) -> None:
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    mock_dlp.return_value = [
        ChunkOut(
            chunk_id="c1",
            id="c1",
            text="[REDACTED]",
            doc_name="doc-1",
            section_name="Section",
            page=1,
            score=9.5,
            source="bm25",
            graph_context=[],
            highlight_spans=[],
        )
    ]

    service = _bm25_service_with_results("SSN 123-45-6789")
    payload = service.search_bm25("repo-1", "sensitive", top_k=5)

    mock_dlp.assert_called_once()
    assert payload["results"][0]["text"] == "[REDACTED]"


@patch("src.features.users.infrastructure.user_repository.get_platform_security_store")
@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
@patch("src.application.consumer_api.context.get_current_user_from_context")
def test_search_bm25_records_audit_event(
    mock_actor: MagicMock,
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
    mock_store: MagicMock,
) -> None:
    actor = AuthenticatedUser(
        user_id="auditor",
        platform_role=None,
        auth_method="api_key",
        is_admin_api_key=True,
    )
    mock_actor.return_value = actor
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }
    sec_store = MagicMock()
    mock_store.return_value = sec_store

    service = _bm25_service_with_results("audit hit")
    service.search_bm25("repo-1", "audit query", top_k=5)

    sec_store.insert_audit_event.assert_called_once()
    event = sec_store.insert_audit_event.call_args.args[0]
    assert event["event_type"] == "retrieval.search_executed"
    assert event["repository_id"] == "repo-1"
    assert event["new_value_json"]["query"] == "audit query"
    assert event["new_value_json"]["result_count"] == 1
    assert "latency_ms" in event["new_value_json"]


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
@patch("src.application.consumer_api.context.get_current_user_from_context", return_value=None)
def test_search_bm25_repository_not_found(
    _mock_actor: MagicMock,
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
) -> None:
    from src.features.repositories.domain.repository_exceptions import NotFoundError as RepoNotFoundError

    mock_get_repo.return_value.validate_repository_active.side_effect = RepoNotFoundError(
        "Repository not found.",
        details={"repository_id": "missing-repo"},
    )

    service = Bm25SearchService()
    service._client = MagicMock()
    service._client.is_ready.return_value = True

    from src.shared.errors import DmsServiceError

    with pytest.raises(DmsServiceError) as exc_info:
        service.search_bm25("missing-repo", "query", top_k=5)
    assert exc_info.value.http_status == 404


@patch("src.features.retrieval.strategies.bm25.bm25_search_service.resolve_collection", side_effect=lambda col, _tenant: col)
@patch("src.features.repositories.application.repository_service.get_repository_service")
@patch("src.application.consumer_api.context.get_current_user_from_context", return_value=None)
def test_search_bm25_empty_repository(
    _mock_actor: MagicMock,
    mock_get_repo: MagicMock,
    _resolve: MagicMock,
) -> None:
    mock_get_repo.return_value.validate_repository_active.return_value = {
        "weaviate_collection": "RepoCollection",
        "default_tenant_id": None,
    }

    fake_client = MagicMock()
    fake_client.is_ready.return_value = True
    fake_client.collections.exists.return_value = True
    collection = MagicMock()
    collection.query = _FakeBm25Query([])
    fake_client.collections.get.return_value = collection

    service = Bm25SearchService()
    service._client = fake_client
    payload = service.search_bm25("repo-1", "empty index", top_k=10)

    assert payload["result_count"] == 0
    assert payload["results"] == []


@patch("src.features.repositories.infrastructure.weaviate_admin.purge_document_chunks")
def test_purge_weaviate_chunks_passes_document_id_and_name(mock_purge: MagicMock) -> None:
    import importlib

    from src.features.documents.application.document_purge_service import purge_weaviate_chunks

    mock_purge.return_value = {"successful": 2, "matches": 2, "failed": 0}
    backend_mod = importlib.import_module("src.features.repositories.infrastructure.weaviate_backend")
    mock_backend = MagicMock()
    mock_backend.ready = True

    with patch.object(backend_mod, "backend", mock_backend):
        payload = purge_weaviate_chunks(
            collection_name="RepoCollection",
            document_name="file.pdf",
            document_id="doc-uuid",
        )

    assert payload["attempted"] is True
    mock_purge.assert_called_once_with(
        "RepoCollection",
        "file.pdf",
        document_id="doc-uuid",
        tenant=None,
    )


@patch("src.features.repositories.infrastructure.weaviate_admin.delete_collection")
def test_delete_repository_raises_on_weaviate_failure(mock_delete: MagicMock) -> None:
    import importlib

    mock_delete.side_effect = RuntimeError("weaviate unavailable")

    store = MagicMock()
    record = MagicMock()
    record.repository_id = "repo-1"
    record.name = "test"
    record.weaviate_collection = "TestCollectionDelFail"
    store.get_by_id.return_value = record
    store.count_documents.return_value = 0

    service = RepositoryService(store=store)
    backend_mod = importlib.import_module("src.features.repositories.infrastructure.weaviate_backend")
    mock_backend = MagicMock()
    mock_backend.ready = True
    client = MagicMock()
    client.collections.exists.return_value = True
    mock_backend.require_client.return_value = client

    with patch.object(backend_mod, "backend", mock_backend):
        with pytest.raises(ServiceUnavailableError) as exc_info:
            service.delete_repository("repo-1")

    assert exc_info.value.details.get("collection_name") == "TestCollectionDelFail"
    store.delete_repository.assert_not_called()


def test_bm25_http_route_propagates_authorization_error() -> None:
    from src.application.consumer_api.main import create_app
    from src.features.retrieval.api.retrieval_routes import _bm25_svc
    from unittest.mock import patch

    service = MagicMock()
    service.search_bm25.side_effect = AuthorizationError("No repository access for this user")

    app = create_app()
    app.dependency_overrides[_bm25_svc] = lambda: service
    client = TestClient(app)
    with patch("src.application.consumer_api.auth_middleware.requires_retrieval_auth", return_value=False):
        response = client.post(
            "/api/v1/repositories/00000000-0000-4000-8000-000000000001/search/bm25",
            json={"query": "test", "top_k": 5},
        )
    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == "forbidden"
