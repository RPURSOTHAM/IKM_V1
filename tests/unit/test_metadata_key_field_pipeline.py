"""Focused tests for metadata/key-field extraction enablement and API resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.features.documents.application.document_metadata_service import resolve_key_fields
from src.features.documents.application.document_service import DocumentReceiverService
from src.features.document_processing.shared_processor.types import ProcessorType, enabled_processor_types


def test_enabled_processor_types_includes_metadata_when_flag_true() -> None:
    types = enabled_processor_types({"metadata_extraction": True})
    assert ProcessorType.METADATA_EXTRACTION.value in types
    assert ProcessorType.CHUNKING_VECTORIZING.value in types


def test_enabled_processor_types_excludes_when_flags_false() -> None:
    types = enabled_processor_types(
        {
            "metadata_extraction": False,
            "key_field_extraction": False,
            "key_field_extraction_enabled": False,
            "validation_enabled": True,
        }
    )
    assert types == [ProcessorType.CHUNKING_VECTORIZING.value]


def test_apply_extraction_enablement_defaults_metadata_on_when_unset() -> None:
    service = DocumentReceiverService()
    updated = service._apply_extraction_enablement(
        {"document_type_id": "dtype-1"},
        stored_settings={},
    )
    assert updated["metadata_extraction"] is True


def test_apply_extraction_enablement_respects_explicit_metadata_false() -> None:
    service = DocumentReceiverService()
    updated = service._apply_extraction_enablement(
        {"document_type_id": "dtype-1", "metadata_extraction": True},
        stored_settings={"metadata_extraction": False},
    )
    assert updated["metadata_extraction"] is False


def test_apply_extraction_enablement_auto_enables_key_fields_when_configured() -> None:
    service = DocumentReceiverService()
    configured = [{"name": "document_title", "required": True}]
    with patch(
        "src.features.document_types.application.processing_key_fields.resolve_key_fields_for_type",
        return_value=configured,
    ):
        updated = service._apply_extraction_enablement(
            {"document_type_id": "dtype-1"},
            stored_settings={"document_type_id": "dtype-1"},
        )
    assert updated["key_field_extraction"] is True
    assert updated["key_field_extraction_enabled"] is True
    types = enabled_processor_types(updated)
    assert ProcessorType.KEY_FIELD_EXTRACTION.value in types
    assert ProcessorType.METADATA_EXTRACTION.value in types


def test_apply_extraction_enablement_respects_explicit_key_field_false() -> None:
    service = DocumentReceiverService()
    with patch(
        "src.features.document_types.application.processing_key_fields.resolve_key_fields_for_type",
        return_value=[{"name": "document_title"}],
    ):
        updated = service._apply_extraction_enablement(
            {"document_type_id": "dtype-1"},
            stored_settings={"key_field_extraction_enabled": False},
        )
    assert updated["key_field_extraction"] is False
    assert updated["key_field_extraction_enabled"] is False
    types = enabled_processor_types(updated)
    assert ProcessorType.KEY_FIELD_EXTRACTION.value not in types


def test_resolve_key_fields_falls_back_to_intake_cache() -> None:
    record = {
        "document_id": "doc-1",
        "metadata": {
            "document_type_id": "dtype-1",
            "extracted_metadata": {"document_title": "SOP Cleaning"},
            "key_field_metadata": {
                "source": "key_field_extraction",
                "completed_at": "2026-01-01T00:00:00",
                "fields": [
                    {"field_name": "document_title", "value": "SOP Cleaning"},
                ],
            },
        },
    }
    with patch(
        "src.features.documents.application.document_metadata_service.fetch_key_fields_from_neo4j",
        return_value=None,
    ):
        resolved = resolve_key_fields(record)
    assert resolved["source"] == "key_field_extraction"
    assert "document_title" in resolved["fields"]
    assert resolved["fields"]["document_title"]["value"] == "SOP Cleaning"


def test_resolve_key_fields_unknown_fields_not_invented() -> None:
    record = {"document_id": "doc-2", "metadata": {}}
    with patch(
        "src.features.documents.application.document_metadata_service.fetch_key_fields_from_neo4j",
        return_value=None,
    ):
        resolved = resolve_key_fields(record)
    assert resolved["source"] == "none"
    assert resolved["fields"] == {}


def test_attach_key_fields_loads_when_enabled() -> None:
    configured = [{"name": "version", "required": False}]
    with patch(
        "src.features.document_types.application.processing_key_fields.resolve_key_fields_for_type",
        return_value=configured,
    ):
        updated = DocumentReceiverService._attach_key_fields(
            {
                "document_type_id": "dtype-1",
                "key_field_extraction_enabled": True,
            }
        )
    assert updated["key_fields"] == configured


def test_attach_key_fields_skips_when_disabled() -> None:
    with patch(
        "src.features.document_types.application.processing_key_fields.resolve_key_fields_for_type"
    ) as resolve_mock:
        updated = DocumentReceiverService._attach_key_fields(
            {
                "document_type_id": "dtype-1",
                "key_field_extraction_enabled": False,
                "key_field_extraction": False,
            }
        )
    resolve_mock.assert_not_called()
    assert "key_fields" not in updated or not updated.get("key_fields")


def test_resolve_type_metadata_completed_without_fields_is_not_pending() -> None:
    from src.features.documents.application.document_metadata_service import resolve_type_metadata

    record = {
        "document_id": "doc-3",
        "processing": {"document_type_id": "dtype-1"},
        "metadata": {
            # Sticky pre-processing placeholder must not win over COMPLETED.
            "type_metadata": {"source": "pending", "fields": []},
        },
    }
    job = {
        "scheduling_metadata": {
            "processor_results": {
                "metadata_extraction": {
                    "status": "COMPLETED",
                    "completed_at": "2026-01-01T00:00:00",
                    "document_metadata": {
                        "document_type_id": "dtype-1",
                        "extracted_fields": [],
                        "extracted_field_count": 0,
                        "skipped": True,
                    },
                }
            }
        }
    }
    with patch(
        "src.features.documents.application.document_metadata_service.fetch_type_metadata_from_neo4j",
        return_value=None,
    ):
        resolved = resolve_type_metadata(record, job=job)
    assert resolved["processor_status"] == "COMPLETED"
    assert resolved["source"] == "metadata_extraction"
    assert resolved["fields"] == []


def test_validation_requires_key_field_extraction() -> None:
    types = enabled_processor_types(
        {
            "validation_enabled": True,
            "key_field_extraction": False,
            "key_field_extraction_enabled": False,
        }
    )
    assert ProcessorType.DOCUMENT_VALIDATION.value not in types

    types_on = enabled_processor_types(
        {
            "validation_enabled": True,
            "key_field_extraction": True,
        }
    )
    assert ProcessorType.DOCUMENT_VALIDATION.value in types_on
    assert ProcessorType.KEY_FIELD_EXTRACTION.value in types_on
