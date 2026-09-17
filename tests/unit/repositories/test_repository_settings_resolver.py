"""Unit tests for repository settings resolution."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.features.repositories.domain.repository_exceptions import InvalidSettingsError, NotFoundError
from src.features.repositories.domain.repository import RepositoryRecord
from src.features.repositories.application.repository_settings_service import (
    EffectiveRepositorySettings,
    RepositorySettingsNotFound,
    SettingsResolver,
    merge_settings,
    merge_settings_lenient,
    resolve_for_repository,
    resolve_repository_context,
)


def test_merge_settings_lenient_accepts_invalid_partial_embedding() -> None:
    partial = {
        "embedding_model": {
            "provider": "local",
            "model_id": "missing-on-disk-model",
            "local_model_dir": "missing-on-disk-model",
        },
        "chunk_overlap": 9999,
        "chunk_size": 512,
    }
    resolved = merge_settings_lenient(partial)
    assert resolved["embedding_model"]["model_id"] == "missing-on-disk-model"
    assert resolved["chunk_overlap"] == 9999


def test_merge_settings_strict_rejects_invalid_embedding() -> None:
    partial = {
        "embedding_model": {
            "provider": "local",
            "model_id": "missing-on-disk-model",
            "local_model_dir": "missing-on-disk-model",
        },
    }
    with pytest.raises(InvalidSettingsError):
        merge_settings(partial)


def test_resolve_for_repository_allows_missing_local_model_dir() -> None:
    stored = {
        "embedding_model": {
            "provider": "local",
            "model_id": "missing-on-disk-model",
            "local_model_dir": "missing-on-disk-model",
        },
        "chunk_size": 512,
        "chunk_overlap": 2,
        "chunking_strategy": "section-based",
    }
    with pytest.raises(InvalidSettingsError):
        merge_settings(stored, strict_local=True)
    resolved = resolve_for_repository(stored)
    assert resolved["embedding_model"]["local_model_dir"] == "missing-on-disk-model"


def test_settings_resolver_resolve_applies_defaults_and_overrides() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    record = RepositoryRecord(
        repository_id="11111111-1111-4111-8111-111111111111",
        name="ResolverRepo",
        owner_user_id="owner",
        weaviate_collection="ResolverRepo",
        default_tenant_id=None,
        status="active",
        created_at=now,
        updated_at=now,
        settings_locked_at=now,
    )
    stored = {
        "chunk_size": 640,
        "chunk_overlap": 3,
        "retrieval_search_mode": "vector",
        "reranking": True,
        "template_extraction": True,
        "embedding_model": {
            "provider": "local",
            "model_id": "bge-base-en",
            "local_model_dir": "bge-base-en",
        },
    }
    effective = SettingsResolver().resolve(record, stored)
    assert isinstance(effective, EffectiveRepositorySettings)
    assert effective.settings["chunk_size"] == 640
    assert effective.settings["retrieval_search_mode"] == "vector"
    assert effective.settings["reranking"] is True
    assert effective.processing_hints["template_extraction"] is True
    assert effective.retrieval()["retrieval_search_mode"] == "vector"
    assert effective.to_dict()["settings_locked"] is True


def test_settings_resolver_not_found_maps_via_context_helper() -> None:
    class _EmptyStore:
        def get_by_id(self, repository_id: str):
            return None

        def get_settings(self, repository_id: str):
            return {}

    with pytest.raises(RepositorySettingsNotFound):
        SettingsResolver(store=_EmptyStore()).resolve_by_repository_id("missing")

    with pytest.raises(NotFoundError):
        resolve_repository_context("missing", store=_EmptyStore())
