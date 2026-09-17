"""Repository service delete/read behaviour with invalid draft settings."""

from __future__ import annotations

import uuid

import pytest

from src.features.repositories.domain.repository_exceptions import InvalidSettingsError
from src.features.repositories.application.repository_service import RepositoryService
from src.features.repositories.application.repository_settings_service import merge_settings, resolve_for_repository


@pytest.fixture
def repo_service(monkeypatch: pytest.MonkeyPatch) -> RepositoryService:
    from tests.unit.repositories.test_repository_service_operations import _FakeStore

    store = _FakeStore()
    svc = RepositoryService(store=store)

    class _DocTypes:
        def initialize(self) -> None:
            return None

        def bootstrap_repository(self, repository_id: str, *, created_by: str | None = None) -> str:
            return "basic-type-id"

        def delete_repository_document_types(self, repository_id: str) -> None:
            return None

    monkeypatch.setattr(
        "src.features.document_types.application.document_type_service.get_document_type_service",
        lambda: _DocTypes(),
    )
    monkeypatch.setattr(
        "src.features.repositories.application.owner_profiles.load_platform_display_names",
        lambda _ids: {},
    )
    return svc


def _unique_name() -> str:
    return f"unit-delete-{uuid.uuid4().hex[:10]}"


def test_get_repository_uses_lenient_settings_for_invalid_stored_values(
    repo_service: RepositoryService,
) -> None:
    store = repo_service._require_store()
    owner = "admin"
    created = repo_service.create_repository({"name": _unique_name(), "owner_user_id": owner})
    repo_id = created["repository_id"]

    invalid_settings = {
        "embedding_model": {
            "provider": "local",
            "model_id": "missing-on-disk-model",
            "local_model_dir": "missing-on-disk-model",
        },
        "chunk_size": 512,
        "chunk_overlap": 9999,
    }
    store.upsert_settings(repo_id, invalid_settings)

    with pytest.raises(InvalidSettingsError):
        merge_settings(invalid_settings, strict_local=True)

    payload = repo_service.get_repository(repo_id)
    assert payload["repository_id"] == repo_id
    assert payload["status"] == "configuring"
    assert payload["document_count"] == 0
    assert payload["settings"]["embedding_model"]["model_id"] == "missing-on-disk-model"

    deleted = repo_service.delete_repository(repo_id)
    assert deleted["deleted"] is True
    assert store.get_by_id(repo_id) is None


def test_delete_active_repository_with_zero_documents(
    repo_service: RepositoryService,
) -> None:
    owner = "admin"
    name = _unique_name()
    created = repo_service.create_repository(
        {
            "name": name,
            "owner_user_id": owner,
            "settings": {
                "chunking_strategy": "section-based",
                "chunk_size": 512,
                "chunk_overlap": 2,
                "embedding_model": {
                    "provider": "local",
                    "model_id": "all-MiniLM-L6-v2",
                    "local_model_dir": "all-MiniLM-L6-v2",
                },
            },
        }
    )
    repo_id = created["repository_id"]
    repo_service.update_settings(repo_id, {"status": "active"})

    payload = repo_service.get_repository(repo_id)
    assert payload["status"] == "active"
    assert payload["document_count"] == 0

    deleted = repo_service.delete_repository(repo_id)
    assert deleted["deleted"] is True
