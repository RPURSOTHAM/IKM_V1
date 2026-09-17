"""Business change audit trail — mutation persistence and /audit/changes exposure."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.features.document_types.domain.models import (
    DocumentTypeRecord,
    KeyFieldDefinitionRecord,
    MetadataFieldRecord,
)
from src.features.document_types.application.document_type_service import DocumentTypeService
from src.features.repositories.domain.repository_exceptions import RepositoryNotFound
from src.features.repositories.domain.repository import RepositoryRecord, RepositorySettings
from src.features.repositories.application.repository_service import RepositoryService
from src.features.observability.audit.domain.audit_models import AuditEventRecord, AuditQuery
from src.features.observability.audit.infrastructure.audit_repository import AuditRepository
from src.features.observability.audit.application.audit_service import AuditService
from src.features.observability.audit.application.change_tracker import get_deep_changes
from src.features.observability.middleware.request_context import apply_synthetic_context, reset_context


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class _FakeRepoStore:
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

    def list_repositories(self, *, owner_user_id: str | None = None, status: str | None = None):
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
def repo_service(monkeypatch: pytest.MonkeyPatch) -> RepositoryService:
    store = _FakeRepoStore()
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
        "src.features.users.infrastructure.user_repository.get_platform_security_store",
        lambda: None,
    )
    monkeypatch.setattr(
        "src.features.repositories.application.owner_profiles.load_platform_display_names",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.resolve_for_repository",
        lambda settings: dict(settings or {}),
    )
    return svc


class TestRepositoryBusinessChangeAudit:
    def test_activate_repository_audits_status_change(self, repo_service: RepositoryService) -> None:
        created = repo_service.create_repository(
            {"name": "ActivateAuditRepo", "owner_user_id": "u1", "settings": {"chunk_size": 512}}
        )
        repo_id = created["repository_id"]
        with patch.object(RepositoryService, "_audit_repository_change") as audit:
            result = repo_service.activate_repository(repo_id)
            assert result["status"] == "active"
            audit.assert_called()
            kwargs = audit.call_args.kwargs
            assert audit.call_args.args[0] == "REPOSITORY_ACTIVATED"
            assert kwargs["old"] == {"status": "configuring"}
            assert kwargs["new"] == {"status": "active"}
            assert kwargs["action"] == "activate"

    def test_activate_noop_when_already_active_skips_audit(self, repo_service: RepositoryService) -> None:
        created = repo_service.create_repository(
            {"name": "AlreadyActiveRepo", "owner_user_id": "u1", "settings": {"chunk_size": 512}}
        )
        repo_id = created["repository_id"]
        repo_service.activate_repository(repo_id)
        with patch.object(RepositoryService, "_audit_repository_change") as audit:
            repo_service.activate_repository(repo_id)
            audit.assert_not_called()

    def test_update_repository_audits_changed_fields_only(self, repo_service: RepositoryService) -> None:
        created = repo_service.create_repository({"name": "UpdateAuditRepo", "owner_user_id": "u1"})
        repo_id = created["repository_id"]
        with patch.object(RepositoryService, "_audit_repository_change") as audit:
            repo_service.update_repository(
                repo_id,
                {"owner_user_name": "Plant Owner", "default_tenant_id": "tenant-a"},
            )
            assert audit.call_args.args[0] == "REPOSITORY_UPDATED"
            assert audit.call_args.kwargs["old"]["owner_user_name"] == "Owner"
            assert audit.call_args.kwargs["new"]["owner_user_name"] == "Plant Owner"
            assert audit.call_args.kwargs["new"]["default_tenant_id"] == "tenant-a"

    def test_unchanged_repository_update_skips_persist(self, repo_service: RepositoryService) -> None:
        created = repo_service.create_repository({"name": "NoChangeRepo", "owner_user_id": "u1"})
        repo_id = created["repository_id"]
        with patch("src.features.observability.audit.application.business_audit.get_audit_service") as get_svc:
            svc = MagicMock()
            get_svc.return_value = svc
            repo_service.update_repository(repo_id, {})
            svc.record_change.assert_not_called()

    def test_settings_update_audits_deep_diff(self, repo_service: RepositoryService) -> None:
        created = repo_service.create_repository(
            {"name": "SettingsAuditRepo", "owner_user_id": "u1", "settings": {"chunk_size": 512}}
        )
        repo_id = created["repository_id"]
        with patch.object(RepositoryService, "_audit_repository_change") as audit:
            repo_service.update_settings(repo_id, {"chunk_size": 1024})
            assert audit.call_args.args[0] == "REPOSITORY_SETTINGS_UPDATED"
            changes = get_deep_changes(audit.call_args.kwargs["old"], audit.call_args.kwargs["new"])
            fields = {c["field"] for c in changes}
            assert "chunk_size" in fields


class TestDocumentTypeBusinessChangeAudit:
    def test_document_type_create_update_delete_audit(self) -> None:
        now = _utc_now()
        store = MagicMock()
        existing = DocumentTypeRecord(
            document_type_id="dt-1",
            repository_id="repo-1",
            name="SOP",
            code=None,
            description="old",
            parent_document_type_id="basic",
            depth_level=1,
            is_system=False,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        updated = replace(existing, description="new", updated_at=_utc_now())
        store.ping.return_value = True
        store.get_type_by_id.return_value = existing

        types_domain = MagicMock()
        types_domain.create_type.return_value = existing
        types_domain.update.return_value = updated
        types_domain.delete.return_value = None

        service = DocumentTypeService(store=store)
        service._types = types_domain
        service._store = store
        with patch.object(DocumentTypeService, "_audit_type_change") as audit:
            service.create_repository_document_type("repo-1", {"name": "SOP"})
            assert audit.call_args_list[0].args[0] == "DOCUMENT_TYPE_CREATED"
            assert audit.call_args_list[0].kwargs["action"] == "create"

            service.update_type("dt-1", {"description": "new"})
            assert audit.call_args_list[1].args[0] == "DOCUMENT_TYPE_UPDATED"
            assert audit.call_args_list[1].kwargs["old"]["description"] == "old"
            assert audit.call_args_list[1].kwargs["new"]["description"] == "new"

            service.delete_type("dt-1")
            assert audit.call_args_list[2].args[0] == "DOCUMENT_TYPE_DELETED"
            assert audit.call_args_list[2].kwargs["new"] == {}

    def test_key_field_mutation_audit(self) -> None:
        now = _utc_now()
        field = KeyFieldDefinitionRecord(
            id="kf-1",
            document_type_id="dt-1",
            field_name="revision",
            field_type="string",
            required=False,
            default_value=None,
            description=None,
            created_at=now,
            updated_at=now,
        )
        updated = replace(field, required=True, updated_at=_utc_now())
        store = MagicMock()
        store.ping.return_value = True
        store.get_key_field_by_id.return_value = field
        kf = MagicMock()
        kf.create_field.return_value = field
        kf.update_field.return_value = updated
        kf.delete_field.return_value = None
        service = DocumentTypeService(store=store)
        service._key_fields = kf
        service._store = store
        with patch.object(DocumentTypeService, "_audit_type_change") as audit:
            service.create_key_field("dt-1", {"field_name": "revision"})
            assert audit.call_args_list[0].args[0] == "DOCUMENT_KEY_FIELD_CREATED"
            service.update_key_field("kf-1", {"required": True})
            assert audit.call_args_list[1].args[0] == "DOCUMENT_KEY_FIELD_UPDATED"
            assert audit.call_args_list[1].kwargs["new"]["required"] is True
            service.delete_key_field("kf-1")
            assert audit.call_args_list[2].args[0] == "DOCUMENT_KEY_FIELD_DELETED"

    def test_metadata_field_mutation_audit(self) -> None:
        now = _utc_now()
        field = MetadataFieldRecord(
            metadata_field_id="mf-1",
            document_type_id="dt-1",
            field_name="author",
            display_label="Author",
            data_type="string",
            required=False,
            default_value=None,
            enum_values=None,
            max_length=None,
            is_system=False,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        updated = replace(field, display_label="Document Author", updated_at=_utc_now())
        store = MagicMock()
        store.ping.return_value = True
        store.get_field_by_id.return_value = field
        md = MagicMock()
        md.create_field.return_value = field
        md.update_field.return_value = updated
        md.delete_field.return_value = None
        service = DocumentTypeService(store=store)
        service._metadata = md
        service._store = store
        with patch.object(DocumentTypeService, "_audit_type_change") as audit:
            service.create_metadata_field(
                "dt-1",
                {"field_name": "author", "display_label": "Author", "data_type": "string"},
            )
            assert audit.call_args_list[0].args[0] == "DOCUMENT_METADATA_FIELD_CREATED"
            service.update_metadata_field("dt-1", "mf-1", {"display_label": "Document Author"})
            assert audit.call_args_list[1].args[0] == "DOCUMENT_METADATA_FIELD_UPDATED"
            service.delete_metadata_field("dt-1", "mf-1")
            assert audit.call_args_list[2].args[0] == "DOCUMENT_METADATA_FIELD_DELETED"

    def test_unchanged_type_update_skips_persist(self) -> None:
        now = _utc_now()
        existing = DocumentTypeRecord(
            document_type_id="dt-1",
            repository_id="repo-1",
            name="SOP",
            code=None,
            description="same",
            parent_document_type_id="basic",
            depth_level=1,
            is_system=False,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        store = MagicMock()
        store.ping.return_value = True
        store.get_type_by_id.return_value = existing
        types_domain = MagicMock()
        types_domain.update.return_value = replace(existing, updated_at=_utc_now())
        service = DocumentTypeService(store=store)
        service._types = types_domain
        service._store = store
        with patch("src.features.observability.audit.application.business_audit.get_audit_service") as get_svc:
            audit_svc = MagicMock()
            get_svc.return_value = audit_svc
            service.update_type("dt-1", {"description": "same"})
            audit_svc.record_change.assert_called_once()
            # Empty deep-diff is skipped inside record_change — verify via real service path.
            real = AuditService()
            with patch.object(real, "_persist") as persist:
                real.record_change(
                    "DOCUMENT_TYPE_UPDATED",
                    entity_type="document_type",
                    entity_id="dt-1",
                    old={"description": "same"},
                    new={"description": "same"},
                )
                persist.assert_not_called()


class TestAuditChangesEndpointBacking:
    def test_record_change_persists_non_empty_changes_sync(self) -> None:
        svc = AuditService()
        tokens = apply_synthetic_context(user_id="alice", application_name="test-app")
        try:
            with patch.object(svc, "_persist") as persist:
                with patch.object(svc, "_async_persist") as async_persist:
                    svc.record_change(
                        "REPOSITORY_ACTIVATED",
                        entity_type="repository",
                        entity_id="repo-1",
                        old={"status": "configuring"},
                        new={"status": "active"},
                        category="repository",
                        action="activate",
                        source="API",
                    )
                    persist.assert_called_once()
                    async_persist.assert_not_called()
                    event: AuditEventRecord = persist.call_args.args[0]
                    assert event.event_type == "REPOSITORY_ACTIVATED"
                    assert event.user_id == "alice"
                    assert event.changes is not None
                    assert event.changes[0]["field"] == "status"
                    assert event.changes[0]["old"] == "configuring"
                    assert event.changes[0]["new"] == "active"
        finally:
            reset_context(tokens)

    def test_list_changes_requires_non_empty_changes_column(self) -> None:
        """Mirrors the SQL filter used by GET /api/v1/audit/changes."""
        query = AuditQuery(page=1, page_size=50)
        repo = AuditRepository(db=MagicMock())  # type: ignore[arg-type]
        with patch.object(repo, "list_events", return_value=([], 0)) as list_events:
            repo.list_changes(query)
            assert query.has_changes is True
            list_events.assert_called_once_with(query)

    def test_technical_record_still_async_without_changes(self) -> None:
        svc = AuditService()
        with patch.object(svc, "_async_persist") as async_persist:
            with patch.object(svc, "_persist") as persist:
                svc.record("APPLICATION_REQUEST", action="search", category="application")
                async_persist.assert_called_once()
                persist.assert_not_called()
                event: AuditEventRecord = async_persist.call_args.args[0]
                assert event.changes is None

    def test_list_audit_changes_returns_503_when_store_unavailable(self) -> None:
        from fastapi import HTTPException

        from src.features.observability.audit.api.audit_routes import list_audit_changes

        with patch(
            "src.features.observability.audit.api.audit_routes.get_audit_repository",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc:
                list_audit_changes()
            assert exc.value.status_code == 503
            assert "unavailable" in str(exc.value.detail).lower()
