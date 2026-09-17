"""Unit tests for repository Consumer API HTTP routes (Phase 3.6).

Uses FastAPI TestClient + in-memory fake store — no live DMS required.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.application.consumer_api.errors.handlers import register_exception_handlers
from src.features.repositories.api.repository_routes import router as repository_router
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
        record = self.records[repository_id]
        updated = replace(record, status=status, updated_at=_utc_now())
        self.records[repository_id] = updated
        return replace(updated)

    def archive_repository(self, repository_id: str) -> RepositoryRecord:
        return self.update_repository_status(repository_id, "archived")

    def lock_settings(self, repository_id: str) -> None:
        record = self.records[repository_id]
        self.records[repository_id] = replace(record, settings_locked_at=_utc_now())

    def update_owner_user_name(self, repository_id: str, owner_user_name: str | None) -> None:
        self.update_repository(repository_id, owner_user_name=owner_user_name)

    def update_repository(self, repository_id: str, **kwargs) -> RepositoryRecord:
        record = self.records[repository_id]
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
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    store = _FakeStore()
    service = RepositoryService(store=store)

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
    monkeypatch.setattr(
        "src.features.users.infrastructure.user_repository.get_platform_security_store",
        lambda: None,
    )
    monkeypatch.setattr(
        "src.features.repositories.api.repository_routes.get_repository_service",
        lambda: service,
    )
    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.get_repository_service",
        lambda: service,
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(repository_router, prefix="/api/v1")
    client = TestClient(app)
    client.store = store  # type: ignore[attr-defined]
    client.service = service  # type: ignore[attr-defined]
    return client


def _create(client: TestClient, name: str = "ApiRepoOne") -> dict[str, Any]:
    response = client.post(
        "/api/v1/repositories",
        json={
            "name": name,
            "owner_user_id": "u1",
            "settings": {"chunk_size": 512, "chunk_overlap": 2},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_create_list_get_delete(api_client: TestClient) -> None:
    created = _create(api_client, "CrudRepo")
    repo_id = created["repository_id"]
    assert created["status"] == "configuring"

    listed = api_client.get("/api/v1/repositories")
    assert listed.status_code == 200
    assert listed.json()["count"] >= 1

    got = api_client.get(f"/api/v1/repositories/{repo_id}")
    assert got.status_code == 200
    assert got.json()["name"] == "CrudRepo"

    deleted = api_client.delete(f"/api/v1/repositories/{repo_id}")
    assert deleted.status_code == 204

    missing = api_client.get(f"/api/v1/repositories/{repo_id}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


def test_update_repository_patch(api_client: TestClient) -> None:
    created = _create(api_client, "PatchRepo")
    repo_id = created["repository_id"]

    updated = api_client.patch(
        f"/api/v1/repositories/{repo_id}",
        json={"owner_user_name": "Plant Owner", "default_tenant_id": "t1"},
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["owner_user_name"] == "Plant Owner"
    assert body["default_tenant_id"] == "t1"

    rejected = api_client.patch(
        f"/api/v1/repositories/{repo_id}",
        json={"name": "Nope"},
    )
    # Pydantic rejects unknown fields or service rejects — either 422 or 409/400
    assert rejected.status_code in {400, 409, 422}


def test_activate_archive_reactivate(api_client: TestClient) -> None:
    created = _create(api_client, "LifecycleRepo")
    repo_id = created["repository_id"]

    active = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert active.status_code == 200
    assert active.json()["status"] == "active"
    assert active.json()["settings_locked"] is True

    archived = api_client.post(f"/api/v1/repositories/{repo_id}/archive")
    assert archived.status_code == 200
    assert archived.json()["status"] == "archived"

    reactivated = api_client.post(f"/api/v1/repositories/{repo_id}/reactivate")
    assert reactivated.status_code == 200
    assert reactivated.json()["status"] == "active"


def test_settings_get_and_patch(api_client: TestClient) -> None:
    created = _create(api_client, "SettingsRepo")
    repo_id = created["repository_id"]

    patched = api_client.patch(
        f"/api/v1/repositories/{repo_id}/settings",
        json={"chunk_size": 640, "chunk_overlap": 4, "reranking": True},
    )
    assert patched.status_code == 200
    assert patched.json()["settings"]["chunk_size"] == 640

    fetched = api_client.get(f"/api/v1/repositories/{repo_id}/settings")
    assert fetched.status_code == 200
    assert fetched.json()["settings"]["chunk_size"] == 640


def test_duplicate_name_conflict(api_client: TestClient) -> None:
    _create(api_client, "DupRepo")
    again = api_client.post(
        "/api/v1/repositories",
        json={"name": "DupRepo", "owner_user_id": "u1"},
    )
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "duplicate_name"


def test_openapi_includes_lifecycle_paths(api_client: TestClient) -> None:
    schema = api_client.get("/openapi.json").json()
    paths = schema.get("paths") or {}
    assert "/api/v1/repositories/{repository_id}/activate" in paths
    assert "/api/v1/repositories/{repository_id}/archive" in paths
    assert "/api/v1/repositories/{repository_id}/reactivate" in paths
    assert "patch" in (paths.get("/api/v1/repositories/{repository_id}") or {})
