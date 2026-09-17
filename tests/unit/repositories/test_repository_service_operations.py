"""Phase 3.4 — RepositoryService business operation facade tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest

from src.features.repositories.domain.repository_exceptions import (
    DatabaseUnavailable,
    ImmutableRepositorySetting,
    InvalidRepositoryStatusTransition,
    NotFoundError,
    RepositoryAlreadyExists,
    RepositoryCannotBeDeleted,
    RepositoryNotFound,
)
from src.features.repositories.domain.repository import RepositoryRecord, RepositorySettings
from src.features.repositories.application.repository_service import RepositoryService


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class _FakeStore:
    def __init__(self) -> None:
        self.records: dict[str, RepositoryRecord] = {}
        self.settings: dict[str, dict[str, Any]] = {}
        self.doc_counts: dict[str, int] = {}

    def ping(self) -> bool:
        return True

    def get_by_id(self, repository_id: str) -> RepositoryRecord | None:
        record = self.records.get(repository_id)
        return replace(record) if record else None

    def get_by_collection(self, collection: str) -> RepositoryRecord | None:
        for record in self.records.values():
            if record.weaviate_collection == collection:
                return replace(record)
        return None

    def list_repositories(
        self,
        *,
        owner_user_id: str | None = None,
        status: str | None = None,
    ) -> list[RepositoryRecord]:
        items = [replace(r) for r in self.records.values()]
        if owner_user_id:
            items = [r for r in items if r.owner_user_id == owner_user_id]
        if status:
            items = [r for r in items if r.status == status]
        return items

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
        now = _utc_now()
        repo_id = repository_id or f"repo-{len(self.records) + 1}"
        record = RepositoryRecord(
            repository_id=repo_id,
            name=name,
            owner_user_id=owner_user_id,
            owner_user_name=owner_user_name or "Owner",
            weaviate_collection=weaviate_collection,
            default_tenant_id=default_tenant_id,
            status=status,
            created_at=now,
            updated_at=now,
            settings_locked_at=settings_locked_at,
        )
        self.records[repo_id] = record
        saved = self.upsert_settings(repo_id, dict(settings or {}))
        return record, RepositorySettings(repository_id=repo_id, settings=saved, updated_at=now)

    def get_settings(self, repository_id: str) -> dict[str, Any]:
        return dict(self.settings.get(repository_id) or {})

    def upsert_settings(self, repository_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        self.settings[repository_id] = dict(settings)
        return dict(settings)

    def update_repository_status(self, repository_id: str, status: str) -> RepositoryRecord:
        record = self.records.get(repository_id)
        if record is None:
            raise RepositoryNotFound("missing", details={"repository_id": repository_id})
        updated = replace(record, status=status, updated_at=_utc_now())
        self.records[repository_id] = updated
        return replace(updated)

    def archive_repository(self, repository_id: str) -> RepositoryRecord:
        return self.update_repository_status(repository_id, "archived")

    def lock_settings(self, repository_id: str) -> None:
        record = self.records.get(repository_id)
        if record is None:
            raise RepositoryNotFound("missing", details={"repository_id": repository_id})
        self.records[repository_id] = replace(record, settings_locked_at=_utc_now())

    def update_owner_user_name(self, repository_id: str, owner_user_name: str | None) -> None:
        self.update_repository(repository_id, owner_user_name=owner_user_name)

    def update_repository(self, repository_id: str, **kwargs) -> RepositoryRecord:
        record = self.records.get(repository_id)
        if record is None:
            raise RepositoryNotFound("missing", details={"repository_id": repository_id})
        updates = {k: v for k, v in kwargs.items() if v is not ...}
        updated = replace(record, updated_at=_utc_now(), **updates)
        self.records[repository_id] = updated
        return replace(updated)

    def count_documents(self, repository_id: str) -> int:
        return int(self.doc_counts.get(repository_id, 0))

    def delete_repository(self, repository_id: str) -> bool:
        self.settings.pop(repository_id, None)
        return self.records.pop(repository_id, None) is not None


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> RepositoryService:
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


def test_create_get_list_delete_workflow(service: RepositoryService) -> None:
    created = service.create_repository({"name": "OpsRepoOne", "owner_user_id": "u1"})
    repo_id = created["repository_id"]
    assert created["status"] == "configuring"
    assert created["name"] == "OpsRepoOne"

    loaded = service.get_repository(repo_id)
    assert loaded["repository_id"] == repo_id

    listed = service.list_repositories(owner_user_id="u1")
    assert listed["count"] == 1
    assert listed["repositories"][0]["name"] == "OpsRepoOne"

    deleted = service.delete_repository(repo_id)
    assert deleted["deleted"] is True
    with pytest.raises(NotFoundError):
        service.get_repository(repo_id)


def test_activate_archive_reactivate(service: RepositoryService) -> None:
    created = service.create_repository(
        {
            "name": "LifecycleRepo",
            "owner_user_id": "u1",
            "settings": {"chunk_size": 512, "chunk_overlap": 2},
        }
    )
    repo_id = created["repository_id"]

    active = service.activate_repository(repo_id)
    assert active["status"] == "active"
    assert active["settings_locked"] is True

    archived = service.archive_repository(repo_id)
    assert archived["status"] == "archived"

    with pytest.raises(NotFoundError):
        service.reactivate_repository("00000000-0000-4000-8000-000000000099")

    reactivated = service.reactivate_repository(repo_id)
    assert reactivated["status"] == "active"


def test_reactivate_requires_archived(service: RepositoryService) -> None:
    created = service.create_repository({"name": "NotArchivedYet", "owner_user_id": "u1"})
    with pytest.raises(InvalidRepositoryStatusTransition):
        service.reactivate_repository(created["repository_id"])


def test_update_repository_editable_and_immutable(service: RepositoryService) -> None:
    created = service.create_repository({"name": "EditableRepo", "owner_user_id": "u1"})
    repo_id = created["repository_id"]

    updated = service.update_repository(
        repo_id,
        {"owner_user_name": "Plant Owner", "default_tenant_id": "tenant-a"},
    )
    assert updated["owner_user_name"] == "Plant Owner"
    assert updated["default_tenant_id"] == "tenant-a"

    with pytest.raises(ImmutableRepositorySetting):
        service.update_repository(repo_id, {"name": "OtherName"})

    with pytest.raises(InvalidRepositoryStatusTransition):
        service.update_repository(repo_id, {"status": "active"})


def test_update_repository_settings_rejects_status(service: RepositoryService) -> None:
    created = service.create_repository({"name": "SettingsOnly", "owner_user_id": "u1"})
    with pytest.raises(InvalidRepositoryStatusTransition):
        service.update_repository_settings(created["repository_id"], {"status": "active"})

    payload = service.update_repository_settings(
        created["repository_id"],
        {"chunk_size": 640, "chunk_overlap": 4},
    )
    assert payload["settings"]["chunk_size"] == 640


def test_delete_blocked_when_documents_exist(service: RepositoryService) -> None:
    created = service.create_repository({"name": "HasDocs", "owner_user_id": "u1"})
    repo_id = created["repository_id"]
    service._require_store().doc_counts[repo_id] = 3
    with pytest.raises(RepositoryCannotBeDeleted):
        service.delete_repository(repo_id)


def test_persistence_duplicate_mapped_to_business_exception(
    service: RepositoryService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.features.repositories.domain.repository_exceptions import DuplicateRepository

    store = service._require_store()

    def _dup(**_kwargs):
        raise DuplicateRepository("dup", details={"name": "X"})

    monkeypatch.setattr(store, "create_repository_with_settings", _dup)
    with pytest.raises(RepositoryAlreadyExists):
        service.create_repository({"name": "DupMapped", "owner_user_id": "u1"})


def test_persistence_unavailable_mapped(service: RepositoryService, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.features.repositories.domain.repository_exceptions import ServiceUnavailableError

    store = service._require_store()

    def _down(_repository_id: str):
        raise DatabaseUnavailable("down")

    monkeypatch.setattr(store, "get_by_id", _down)
    with pytest.raises(ServiceUnavailableError):
        service.get_repository("any")
