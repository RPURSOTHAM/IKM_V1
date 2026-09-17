"""Unit tests for Phase 3.1 repository domain models and model-level validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.features.repositories.domain.constants import (
    DEFAULT_REPOSITORY_STATUS,
    REPOSITORY_STATUS_VALUES,
    STATUS_ACTIVE,
    STATUS_CONFIGURING,
)
from src.features.repositories.domain.repository import (
    Repository,
    RepositoryOwner,
    RepositoryRecord,
    RepositorySettings,
    RepositoryStatus,
    RepositoryValidationResult,
    default_status,
    parse_repository_id,
    repository_to_dict,
)
from src.features.repositories.domain.repository_validator import (
    validate_owner_user_id,
    validate_repository_id,
    validate_repository_name,
    validate_repository_record,
    validate_repository_settings_shape,
    validate_repository_status,
)


def _sample_record(**overrides) -> RepositoryRecord:
    base = dict(
        repository_id="11111111-1111-4111-8111-111111111111",
        name="PlantDocs",
        owner_user_id="user-1",
        weaviate_collection="PlantDocs",
        default_tenant_id=None,
        status=STATUS_CONFIGURING,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None),
    )
    base.update(overrides)
    return RepositoryRecord(**base)


def test_repository_status_enum_matches_phase2() -> None:
    assert {s.value for s in RepositoryStatus} == REPOSITORY_STATUS_VALUES
    assert "deleted" not in REPOSITORY_STATUS_VALUES
    assert RepositoryStatus.CONFIGURING.value == "configuring"
    assert RepositoryStatus.ACTIVE.value == "active"
    assert default_status() == DEFAULT_REPOSITORY_STATUS == STATUS_CONFIGURING


def test_repository_alias_is_record() -> None:
    assert Repository is RepositoryRecord


def test_parse_repository_id() -> None:
    assert parse_repository_id("22222222-2222-4222-8222-222222222222") == (
        "22222222-2222-4222-8222-222222222222"
    )
    with pytest.raises(ValueError):
        parse_repository_id("not-a-uuid")


def test_validate_repository_name_rules() -> None:
    assert validate_repository_name("Valid_Name-1").is_valid
    assert not validate_repository_name("").is_valid
    assert not validate_repository_name(" bad").is_valid
    assert not validate_repository_name("-bad").is_valid
    assert not validate_repository_name("system").is_valid
    assert "reserved_name" in validate_repository_name("admin").error_codes()


def test_validate_status_and_owner() -> None:
    assert validate_repository_status(STATUS_ACTIVE).is_valid
    assert not validate_repository_status("deleted").is_valid
    assert validate_owner_user_id("owner-1").is_valid
    assert not validate_owner_user_id("").is_valid


def test_validate_repository_record_ok() -> None:
    result = validate_repository_record(_sample_record())
    assert result.is_valid


def test_validate_repository_record_bad_id() -> None:
    result = validate_repository_record(_sample_record(repository_id="x"))
    assert not result.is_valid
    assert "invalid_repository_id" in result.error_codes()


def test_repository_settings_shape() -> None:
    settings = RepositorySettings(
        repository_id="11111111-1111-4111-8111-111111111111",
        settings={"chunk_size": 512},
    )
    assert validate_repository_settings_shape(settings).is_valid
    assert not validate_repository_settings_shape([1, 2, 3]).is_valid  # type: ignore[arg-type]


def test_owner_and_validation_result_helpers() -> None:
    owner = RepositoryOwner(user_id="u1", user_name="Ada")
    assert owner.user_id == "u1"
    ok = RepositoryValidationResult.ok()
    assert ok.is_valid
    failed = RepositoryValidationResult.failed([])
    # empty errors list still marks failed factory — ensure helper sets is_valid False
    assert failed.is_valid is False


def test_repository_to_dict_still_works() -> None:
    payload = repository_to_dict(_sample_record(status=STATUS_ACTIVE), document_count=0)
    assert payload["status"] == "active"
    assert payload["document_count"] == 0
    assert payload["settings_locked"] is False


def test_validate_repository_id() -> None:
    assert validate_repository_id("11111111-1111-4111-8111-111111111111").is_valid
    assert not validate_repository_id("").is_valid
