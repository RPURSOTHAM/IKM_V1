"""Unit tests for Phase 3.3 business validation (pure validators + exceptions)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.features.repositories.domain.repository_policy import (
    assert_activation,
    assert_delete_allowed,
    assert_status_transition,
    assert_update_patch,
    normalize_repository_name,
    validate_activation,
    validate_delete,
    validate_settings_business,
    validate_status_transition,
)
from src.features.repositories.domain.repository_exceptions import (
    ImmutableRepositorySetting,
    InvalidRepositoryName,
    InvalidRepositoryStatusTransition,
    RepositoryActivationFailed,
    RepositoryCannotBeDeleted,
)
from src.features.repositories.domain.repository import RepositoryRecord


def _record(**overrides) -> RepositoryRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None)
    base = dict(
        repository_id="11111111-1111-4111-8111-111111111111",
        name="PlantDocs",
        owner_user_id="owner-1",
        weaviate_collection="PlantDocs",
        default_tenant_id=None,
        status="configuring",
        created_at=now,
        updated_at=now,
        settings_locked_at=None,
    )
    base.update(overrides)
    return RepositoryRecord(**base)


def test_normalize_repository_name_trims_and_rejects_reserved() -> None:
    assert normalize_repository_name("  Valid_Name  ") == "Valid_Name"
    with pytest.raises(InvalidRepositoryName):
        normalize_repository_name("system")
    with pytest.raises(InvalidRepositoryName):
        normalize_repository_name("")


def test_status_transitions_phase2() -> None:
    assert validate_status_transition("configuring", "approved").is_valid
    assert validate_status_transition("configuring", "active").is_valid
    assert validate_status_transition("approved", "active").is_valid
    assert validate_status_transition("active", "archived").is_valid
    assert validate_status_transition("archived", "active").is_valid  # reactivation
    assert not validate_status_transition("active", "configuring").is_valid
    assert not validate_status_transition("archived", "configuring").is_valid
    with pytest.raises(InvalidRepositoryStatusTransition):
        assert_status_transition("active", "configuring")


def test_activation_from_configuring_allowed() -> None:
    record = _record(status="configuring")
    assert validate_activation(record, {"chunk_size": 512}).is_valid
    assert_activation(record, {"chunk_size": 512})


def test_reactivation_from_archived_allowed() -> None:
    """Phase 2: archived → active is reactivation (settings already locked)."""
    record = _record(status="archived")
    assert validate_activation(record, {}).is_valid
    assert_activation(record, {})
    assert validate_status_transition("archived", "active").is_valid


def test_activation_from_active_rejected() -> None:
    record = _record(status="active")
    with pytest.raises(RepositoryActivationFailed):
        assert_activation(record, {"chunk_size": 512})


def test_update_rejects_immutable_and_locked_settings() -> None:
    active = _record(status="active", settings_locked_at=datetime(2026, 1, 2))
    with pytest.raises(ImmutableRepositorySetting):
        assert_update_patch(active, {"weaviate_collection": "Other"})
    with pytest.raises(ImmutableRepositorySetting):
        assert_update_patch(active, {"chunk_size": 400})
    # lifecycle archive still allowed
    assert_update_patch(active, {"status": "archived"})


def test_delete_validation_suggests_archive_when_docs_exist() -> None:
    record = _record(status="active")
    result = validate_delete(record, document_count=3)
    assert not result.is_valid
    assert result.errors[0].details.get("suggested_action") == "archive"
    with pytest.raises(RepositoryCannotBeDeleted):
        assert_delete_allowed(record, document_count=3)
    assert_delete_allowed(record, document_count=0)


def test_settings_business_rules() -> None:
    assert validate_settings_business({"chunk_size": 512, "chunk_overlap": 2}).is_valid
    assert not validate_settings_business({"chunk_size": 50}).is_valid
    assert not validate_settings_business({"retrieval_search_mode": "magic"}).is_valid
    assert not validate_settings_business(
        {"key_field_extraction_enabled": True, "key_fields": []},
        require_activation_fields=True,
    ).is_valid
    assert not validate_settings_business(
        {"embedding_model": {"provider": "local"}},
    ).is_valid
