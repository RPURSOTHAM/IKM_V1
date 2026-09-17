from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from src.features.repositories.domain.repository_exceptions import InvalidSettingsError
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
        return 0

    def get_settings(self, repository_id: str) -> dict:
        return dict(self._settings) if repository_id == self._record.repository_id else {}

    def upsert_settings(self, repository_id: str, settings: dict) -> dict:
        if repository_id == self._record.repository_id:
            self._settings = dict(settings)
        return dict(self._settings)


def _repo_record(repository_id: str = "repo-1") -> RepositoryRecord:
    now = datetime.utcnow()
    return RepositoryRecord(
        repository_id=repository_id,
        name="unit-repo",
        owner_user_id="admin",
        weaviate_collection="Unit_repo",
        default_tenant_id=None,
        status="configuring",
        created_at=now,
        updated_at=now,
        settings_locked_at=None,
        owner_user_name="Admin",
    )


def test_patch_key_fields_breaks_deadlock_and_persists() -> None:
    record = _repo_record()
    store = _FakeRepositoryStore(
        record,
        {
            "key_field_extraction_enabled": True,
            "key_field_extraction": True,
            "key_fields": [],
        },
    )
    service = RepositoryService(store=store)

    response = service.patch_repository_key_fields(
        record.repository_id,
        [
            {"name": "document_number", "type": "string", "required": True},
            {"name": "effective_date", "type": "date", "required": False},
        ],
    )

    assert response["repository_id"] == record.repository_id
    assert response["count"] == 2
    stored = store.get_settings(record.repository_id)
    assert len(stored.get("key_fields") or []) == 2


def test_disable_key_field_extraction_recovery_patch_succeeds() -> None:
    record = _repo_record()
    store = _FakeRepositoryStore(
        record,
        {
            "key_field_extraction_enabled": True,
            "key_field_extraction": True,
            "key_fields": [],
        },
    )
    service = RepositoryService(store=store)

    payload = service.update_settings(record.repository_id, {"key_field_extraction_enabled": False})
    settings = payload.get("settings") or {}
    assert settings.get("key_field_extraction_enabled") is False
    assert settings.get("key_field_extraction") is False


def test_patch_key_fields_ignores_unrelated_invalid_legacy_settings() -> None:
    record = _repo_record()
    store = _FakeRepositoryStore(
        record,
        {
            "chunk_size": 150,
            "key_field_extraction_enabled": True,
            "key_field_extraction": True,
            "key_fields": [],
        },
    )
    service = RepositoryService(store=store)

    response = service.patch_repository_key_fields(
        record.repository_id,
        [{"name": "version", "type": "string", "required": False}],
    )
    assert response["count"] == 1
    stored = store.get_settings(record.repository_id)
    assert (stored.get("key_fields") or [])[0]["name"] == "version"


def test_enable_extraction_with_empty_key_fields_fails() -> None:
    record = _repo_record()
    store = _FakeRepositoryStore(
        record,
        {
            "chunk_size": 512,
            "chunk_overlap": 2,
            "key_field_extraction_enabled": False,
            "key_field_extraction": False,
            "key_fields": [],
        },
    )
    service = RepositoryService(store=store)

    with pytest.raises(InvalidSettingsError):
        service.update_settings(record.repository_id, {"key_field_extraction_enabled": True})


def test_enable_extraction_repairs_unrelated_legacy_chunk_size() -> None:
    record = _repo_record()
    store = _FakeRepositoryStore(
        record,
        {
            "chunk_size": 150,
            "chunk_overlap": 2,
            "document_type_id": "basic-document-type",
            "key_field_extraction_enabled": False,
            "key_field_extraction": False,
            "key_fields": [],
        },
    )
    service = RepositoryService(store=store)

    payload = service.update_settings(
        record.repository_id,
        {
            "key_field_extraction": True,
            "key_field_extraction_enabled": True,
        },
    )
    settings = payload.get("settings") or {}
    stored = store.get_settings(record.repository_id)

    assert settings["key_field_extraction"] is True
    assert settings["key_field_extraction_enabled"] is True
    assert stored["key_field_extraction"] is True
    assert stored["key_field_extraction_enabled"] is True
    assert stored["chunk_size"] >= 200
