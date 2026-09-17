"""Unit tests for repository owner display-name resolution."""

from __future__ import annotations

from datetime import datetime, timezone

from src.features.repositories.domain.repository import RepositoryRecord, repository_to_dict
from src.features.repositories.application.owner_profiles import resolve_owner_user_name


def _record(**overrides) -> RepositoryRecord:
    now = datetime.now(timezone.utc)
    base = {
        "repository_id": "repo-1",
        "name": "demo",
        "owner_user_id": "user-1",
        "weaviate_collection": "demo",
        "default_tenant_id": None,
        "status": "configuring",
        "created_at": now,
        "updated_at": now,
        "settings_locked_at": None,
        "owner_user_name": None,
    }
    base.update(overrides)
    return RepositoryRecord(**base)


def test_resolve_owner_user_name_prefers_stored_value() -> None:
    name = resolve_owner_user_name(
        "user-1",
        stored_name="Stored Owner",
        platform_names={"user-1": "Platform Owner"},
    )
    assert name == "Stored Owner"


def test_resolve_owner_user_name_uses_platform_lookup() -> None:
    name = resolve_owner_user_name(
        "user-1",
        platform_names={"user-1": "Platform Owner"},
    )
    assert name == "Platform Owner"


def test_resolve_owner_user_name_ignores_uuid_stored_value() -> None:
    owner_id = "aead2e1d-6f59-4395-8c60-7605abe5ff10"
    name = resolve_owner_user_name(
        owner_id,
        stored_name=owner_id,
        platform_names={owner_id: "Jane Doe"},
    )
    assert name == "Jane Doe"


def test_repository_to_dict_includes_owner_fields() -> None:
    payload = repository_to_dict(
        _record(owner_user_name="Jane Doe"),
        owner_user_name="Jane Doe",
    )
    assert payload["owner_user_id"] == "user-1"
    assert payload["owner_user_name"] == "Jane Doe"


def test_repository_to_dict_always_includes_owner_user_name_key() -> None:
    payload = repository_to_dict(_record(owner_user_name=None))
    assert "owner_user_name" in payload
    assert payload["owner_user_name"] is None
    assert "settings_locked_at" in payload
    assert payload["settings_locked_at"] is None


def test_repository_to_dict_honors_explicit_null_owner_user_name() -> None:
    owner_id = "aead2e1d-6f59-4395-8c60-7605abe5ff10"
    payload = repository_to_dict(
        _record(owner_user_id=owner_id, owner_user_name=owner_id),
        owner_user_name=None,
    )
    assert payload["owner_user_name"] is None
