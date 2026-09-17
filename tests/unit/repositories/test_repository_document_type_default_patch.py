from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from src.features.repositories.domain.repository_exceptions import SettingsLockedError
from src.features.repositories.domain.repository import RepositoryRecord
from src.features.repositories.application.repository_service import RepositoryService


class _FakeRepositoryStore:
    def __init__(self, record: RepositoryRecord, settings: dict):
        self._record = record
        self._settings = dict(settings)

    def ping(self) -> bool:
        return True

    def get_by_id(self, repository_id: str) -> RepositoryRecord | None:
        if repository_id != self._record.repository_id:
            return None
        return replace(self._record)

    def count_documents(self, repository_id: str) -> int:
        return 2

    def get_settings(self, repository_id: str) -> dict:
        return dict(self._settings) if repository_id == self._record.repository_id else {}

    def upsert_settings(self, repository_id: str, settings: dict) -> dict:
        if repository_id == self._record.repository_id:
            self._settings = dict(settings)
        return dict(self._settings)


def _active_repo_record(repository_id: str = "repo-1") -> RepositoryRecord:
    now = datetime.utcnow()
    return RepositoryRecord(
        repository_id=repository_id,
        name="unit-repo",
        owner_user_id="admin",
        weaviate_collection="Unit_repo",
        default_tenant_id=None,
        status="active",
        created_at=now,
        updated_at=now,
        settings_locked_at=now,
        owner_user_name="Admin",
    )


def test_document_type_default_patch_allowed_on_active_repository(monkeypatch) -> None:
    record = _active_repo_record()
    store = _FakeRepositoryStore(record, {"document_type_id": "basic-id", "chunk_size": 512})
    service = RepositoryService(store=store)

    def _assert(_repository_id: str, document_type_id: str) -> None:
        assert document_type_id == "sop-id"

    monkeypatch.setattr(service, "_assert_repository_document_type", _assert)

    payload = service.update_settings(record.repository_id, {"document_type_id": "sop-id"})
    settings = payload.get("settings") or {}
    assert settings.get("document_type_id") == "sop-id"
    assert store.get_settings(record.repository_id)["document_type_id"] == "sop-id"


def test_other_settings_patch_still_locked_on_active_repository() -> None:
    record = _active_repo_record()
    store = _FakeRepositoryStore(record, {"chunk_size": 512})
    service = RepositoryService(store=store)

    with pytest.raises(SettingsLockedError) as exc:
        service.update_settings(record.repository_id, {"chunk_size": 700})

    assert exc.value.code == "settings_read_only"


def test_is_document_type_default_patch_requires_only_document_type_id() -> None:
    assert RepositoryService._is_document_type_default_patch({"document_type_id": "abc"}) is True
    assert RepositoryService._is_document_type_default_patch({"document_type_id": ""}) is False
    assert RepositoryService._is_document_type_default_patch({"document_type_id": "abc", "chunk_size": 512}) is False
