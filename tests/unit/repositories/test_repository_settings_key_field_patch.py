from __future__ import annotations

import pytest

from src.features.repositories.schemas.repository_schemas import UpdateRepositorySettingsRequest
from src.features.repositories.application.repository_settings_service import merge_settings, processing_hints_from_settings
from src.features.repositories.domain.repository_exceptions import InvalidSettingsError


def test_update_repository_settings_request_accepts_key_field_extraction_flag() -> None:
    body = UpdateRepositorySettingsRequest(key_field_extraction=True)
    payload = body.model_dump(exclude_none=True)
    assert payload["key_field_extraction"] is True


def test_update_repository_settings_request_accepts_strict_key_field_page_scope() -> None:
    body = UpdateRepositorySettingsRequest(strict_key_field_page_scope=True)
    payload = body.model_dump(exclude_none=True)
    assert payload["strict_key_field_page_scope"] is True


def test_processing_hints_include_content_intelligence_flags() -> None:
    hints = processing_hints_from_settings(
        {
            "content_intelligence": True,
            "intelligent_extraction": True,
            "metadata_extraction": True,
        }
    )
    assert hints["content_intelligence"] is True
    assert hints["intelligent_extraction"] is True
    assert hints["intelligence_extraction"] is True
    assert "intelligence_extraction" in hints["enabled_processor_types"]


def test_update_repository_settings_request_accepts_key_field_extraction_enabled_alias() -> None:
    body = UpdateRepositorySettingsRequest(key_field_extraction_enabled=True)
    payload = body.model_dump(exclude_none=True)
    assert payload["key_field_extraction_enabled"] is True


def test_merge_settings_normalizes_key_field_extraction_aliases() -> None:
    merged = merge_settings(
        {
            "key_field_extraction_enabled": True,
            "key_fields": [{"name": "document_number", "type": "string", "required": True}],
        }
    )
    assert merged["key_field_extraction"] is True
    assert merged["key_field_extraction_enabled"] is True


def test_merge_settings_rejects_duplicate_key_field_names() -> None:
    with pytest.raises(InvalidSettingsError):
        merge_settings(
            {
                "key_fields": [
                    {"name": "document_number", "type": "string"},
                    {"name": "Document_Number", "type": "string"},
                ]
            }
        )


def test_merge_settings_requires_key_fields_when_extraction_enabled() -> None:
    with pytest.raises(InvalidSettingsError):
        merge_settings({"key_field_extraction_enabled": True, "key_fields": []})
