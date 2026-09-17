"""Unit tests for Phase 3 repository document-type migration and metadata exposure."""

from __future__ import annotations

import uuid

import pytest

from src.features.document_types.infrastructure.document_type_migration import migrate_repository_document_types
from src.features.documents.application.document_metadata_service import (
    resolve_document_metadata_bundle,
    resolve_key_field_metadata_bundle,
)
from src.features.document_types.application.document_type_service import DocumentTypeService
from src.features.repositories.application.repository_service import RepositoryService


@pytest.fixture
def repo_service() -> RepositoryService:
    service = RepositoryService()
    service._require_store()
    return service


@pytest.fixture
def doc_type_service() -> DocumentTypeService:
    service = DocumentTypeService()
    service._require_store()
    return service


def _unique_name() -> str:
    return f"unit-phase3-{uuid.uuid4().hex[:10]}"


def test_migration_bootstraps_basic_and_migrates_key_fields(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = repo_service.create_repository({"name": _unique_name(), "owner_user_id": "admin"})
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    # Simulate legacy Phase 2 key_fields on settings.
    updated = repo_service.update_settings(
        repository_id,
        {
            "key_field_extraction_enabled": True,
            "key_fields": [
                {
                    "name": "custom_owner",
                    "type": "string",
                    "required": False,
                    "description": "Owner from legacy settings",
                },
                {
                    "name": "document_number",
                    "type": "string",
                    "required": True,
                    "description": "Already on Basic — should skip",
                },
            ],
        },
    )
    settings = dict(updated.get("settings") or {})

    result = migrate_repository_document_types(
        repository_id,
        dry_run=False,
        settings=settings,
        dt_service=doc_type_service,
        update_settings=lambda rid, patch: repo_service.update_settings(rid, patch),
    )
    assert result["basic_created"] is False
    assert result["fields_migrated"] >= 1
    assert result["fields_skipped"] >= 1

    fields = doc_type_service.list_key_fields(basic_id)
    names = {field["field_name"] for field in fields["fields"]}
    assert "custom_owner" in names
    assert "document_number" in names

    # Idempotent second run should not duplicate.
    settings = dict(repo_service.get_settings(repository_id).get("settings") or {})
    second = migrate_repository_document_types(
        repository_id,
        dry_run=False,
        settings=settings,
        dt_service=doc_type_service,
        update_settings=lambda rid, patch: repo_service.update_settings(rid, patch),
    )
    assert second["fields_migrated"] == 0


def test_metadata_bundle_exposes_effective_fields(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = repo_service.create_repository({"name": _unique_name(), "owner_user_id": "admin"})
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]
    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    doc_type_service.create_key_field(
        sop["document_type_id"],
        {"field_name": "sop_owner", "field_type": "string"},
    )

    record = {
        "document_id": str(uuid.uuid4()),
        "document_type_id": sop["document_type_id"],
        "document_type_name": "SOP",
        "metadata": {
            "document_type_id": sop["document_type_id"],
            "document_type_name": "SOP",
        },
        "processing": {
            "document_type_id": sop["document_type_id"],
            "document_type_name": "SOP",
        },
    }
    key_bundle = resolve_key_field_metadata_bundle(record)
    assert key_bundle["document_type_id"] == sop["document_type_id"]
    assert key_bundle["document_type_name"] == "SOP"
    effective_names = {field["field_name"] for field in key_bundle["effective_fields"]}
    assert "document_number" in effective_names
    assert "sop_owner" in effective_names
    assert "extracted_metadata" in key_bundle

    bundle = resolve_document_metadata_bundle(record)
    assert bundle["document_type_id"] == sop["document_type_id"]
    assert bundle["document_type_name"] == "SOP"
    assert isinstance(bundle["effective_fields"], list)
    assert isinstance(bundle["extracted_metadata"], dict)
