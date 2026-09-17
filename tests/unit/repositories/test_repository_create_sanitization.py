from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from src.features.repositories.domain.repository_exceptions import InvalidSettingsError
from src.features.repositories.domain.repository import RepositoryRecord
from src.features.repositories.application.repository_service import RepositoryService


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class _FakeStore:
    def __init__(self) -> None:
        self.records: dict[str, RepositoryRecord] = {}
        self.saved_settings: dict[str, dict[str, Any]] = {}
        self._seq = 0

    def ping(self) -> bool:
        return True

    def validate_name(self, name: str) -> str:
        trimmed = str(name).strip()
        if not trimmed:
            raise ValueError("name required")
        return trimmed

    def get_by_name(self, name: str) -> RepositoryRecord | None:
        for record in self.records.values():
            if record.name == name:
                return record
        return None

    def get_by_collection(self, collection: str) -> RepositoryRecord | None:
        for record in self.records.values():
            if record.weaviate_collection == collection:
                return record
        return None

    def list_repositories(
        self,
        *,
        owner_user_id: str | None = None,
        status: str | None = None,
    ) -> list[RepositoryRecord]:
        items = list(self.records.values())
        if owner_user_id:
            items = [r for r in items if r.owner_user_id == owner_user_id]
        if status:
            items = [r for r in items if r.status == status]
        return items

    def insert_repository(
        self,
        *,
        name: str,
        owner_user_id: str,
        owner_user_name: str | None = None,
        weaviate_collection: str,
        default_tenant_id: str | None,
        status: str = "configuring",
        repository_id: str | None = None,
    ) -> RepositoryRecord:
        self._seq += 1
        repo_id = repository_id or f"repo-{self._seq}"
        now = _utc_now()
        record = RepositoryRecord(
            repository_id=repo_id,
            name=name,
            owner_user_id=owner_user_id,
            owner_user_name=owner_user_name,
            weaviate_collection=weaviate_collection,
            default_tenant_id=default_tenant_id,
            status=status,
            created_at=now,
            updated_at=now,
            settings_locked_at=None,
        )
        self.records[repo_id] = record
        return record

    def create_repository_with_settings(
        self,
        *,
        name: str,
        owner_user_id: str,
        weaviate_collection: str,
        settings: dict[str, Any] | None = None,
        owner_user_name: str | None = None,
        default_tenant_id: str | None = None,
        status: str = "configuring",
        repository_id: str | None = None,
        settings_locked_at=None,
    ):
        from src.features.repositories.domain.repository import RepositorySettings

        record = self.insert_repository(
            name=name,
            owner_user_id=owner_user_id,
            owner_user_name=owner_user_name,
            weaviate_collection=weaviate_collection,
            default_tenant_id=default_tenant_id,
            status=status,
            repository_id=repository_id,
        )
        saved = self.upsert_settings(record.repository_id, dict(settings or {}))
        return record, RepositorySettings(
            repository_id=record.repository_id,
            settings=saved,
            updated_at=record.updated_at,
        )

    def delete_repository(self, repository_id: str) -> bool:
        self.saved_settings.pop(repository_id, None)
        return self.records.pop(repository_id, None) is not None

    def upsert_settings(self, repository_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        self.saved_settings[repository_id] = dict(settings)
        return settings

    def count_documents(self, repository_id: str) -> int:
        return 0


@dataclass
class _FakeDocTypeService:
    basic_type_id: str = "basic-type-id"

    def initialize(self) -> None:
        return None

    def bootstrap_repository(self, repository_id: str, *, created_by: str | None = None) -> str:
        return self.basic_type_id


def test_normalize_create_settings_strips_swagger_placeholders() -> None:
    service = RepositoryService()
    payload_settings = {
        "embedding_model": {
            "provider": "local",
            "model_id": "string",
            "local_model_dir": "string",
        },
        "extraction_model": {
            "provider": "string",
            "model_id": "string",
        },
        "chunking_strategy": "",
    }
    normalized = service._normalize_create_settings(payload_settings)
    assert "embedding_model" not in normalized
    assert "extraction_model" not in normalized
    assert "chunking_strategy" not in normalized


def test_create_repository_minimal_payload_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore()
    service = RepositoryService(store=store)
    validate_calls: list[dict[str, Any]] = []

    def _fake_validate_embedding_model(embedding_model: dict[str, Any], *, strict_local: bool = True) -> dict[str, Any]:
        validate_calls.append(dict(embedding_model))
        return embedding_model

    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.validate_embedding_model",
        _fake_validate_embedding_model,
    )
    monkeypatch.setattr(
        "src.features.document_types.application.document_type_service.get_document_type_service",
        lambda: _FakeDocTypeService(),
    )
    monkeypatch.setattr(
        "src.features.users.infrastructure.user_repository.get_platform_security_store",
        lambda: None,
    )
    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.resolve_for_repository",
        lambda settings: {
            "embedding_model": {
                "provider": "local",
                "model_id": "bge-base-en",
                "local_model_dir": "bge-base-en",
            },
            "document_type_id": settings.get("document_type_id"),
        },
    )

    created = service.create_repository({"name": "repo-minimal"})
    assert created["owner_user_id"] == "system"
    assert created["settings"]["embedding_model"]["model_id"] == "bge-base-en"
    assert created["settings"]["document_type_id"] == "basic-type-id"
    assert validate_calls == []


def test_create_repository_swagger_placeholder_embedding_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    service = RepositoryService(store=store)
    validate_calls: list[dict[str, Any]] = []

    def _fake_validate_embedding_model(embedding_model: dict[str, Any], *, strict_local: bool = True) -> dict[str, Any]:
        validate_calls.append(dict(embedding_model))
        return embedding_model

    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.validate_embedding_model",
        _fake_validate_embedding_model,
    )
    monkeypatch.setattr(
        "src.features.document_types.application.document_type_service.get_document_type_service",
        lambda: _FakeDocTypeService(),
    )
    monkeypatch.setattr(
        "src.features.users.infrastructure.user_repository.get_platform_security_store",
        lambda: None,
    )
    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.resolve_for_repository",
        lambda settings: {"embedding_model": {"provider": "local", "model_id": "bge-base-en", "local_model_dir": "bge-base-en"}},
    )

    created = service.create_repository(
        {
            "name": "repo-swagger",
            "owner_user_id": "string",
            "settings": {
                "embedding_model": {
                    "provider": "local",
                    "model_id": "string",
                    "local_model_dir": "string",
                }
            },
        }
    )
    repo_id = created["repository_id"]
    persisted = store.saved_settings[repo_id]
    assert "embedding_model" not in persisted
    assert persisted["document_type_id"] == "basic-type-id"
    assert validate_calls == []


def test_create_repository_invalid_embedding_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    service = RepositoryService(store=store)

    def _fake_validate_embedding_model(embedding_model: dict[str, Any], *, strict_local: bool = True) -> dict[str, Any]:
        raise InvalidSettingsError("invalid embedding")

    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.validate_embedding_model",
        _fake_validate_embedding_model,
    )

    with pytest.raises(InvalidSettingsError, match="invalid embedding"):
        service.create_repository(
            {
                "name": "repo-invalid",
                "settings": {
                    "embedding_model": {
                        "provider": "local",
                        "model_id": "definitely-missing-model",
                        "local_model_dir": "definitely-missing-model",
                    }
                },
            }
        )
