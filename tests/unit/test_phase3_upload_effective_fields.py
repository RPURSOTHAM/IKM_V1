"""Unit tests for Phase 3 upload effective-field attachment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.features.documents.application.document_service import DocumentReceiverService


def test_attach_key_fields_resolves_effective_fields_even_when_extraction_disabled() -> None:
    processing = {
        "document_type_id": "type-1",
        "key_field_extraction_enabled": False,
    }
    with patch(
        "src.features.document_types.application.document_type_service.get_document_type_service"
    ) as get_svc:
        svc = MagicMock()
        svc.resolve_effective_fields.return_value = {
            "document_type_id": "type-1",
            "fields": [
                {"field_name": "document_number", "field_type": "string", "inherited": True},
                {"field_name": "equipment_id", "field_type": "string", "inherited": False},
            ],
            "count": 2,
        }
        get_svc.return_value = svc
        updated = DocumentReceiverService._attach_key_fields(processing)

    assert len(updated["effective_fields"]) == 2
    assert updated["effective_fields"][0]["field_name"] == "document_number"
    assert "key_fields" not in updated or not updated.get("key_fields")


def test_attach_key_fields_uses_document_type_when_repo_key_fields_empty() -> None:
    processing = {
        "document_type_id": "type-1",
        "key_field_extraction_enabled": True,
        "key_fields": [],
    }
    with patch(
        "src.features.document_types.application.document_type_service.get_document_type_service"
    ) as get_svc, patch(
        "src.features.document_types.application.processing_key_fields.resolve_key_fields_for_type",
        return_value=[{"field_name": "document_number", "data_type": "string"}],
    ):
        svc = MagicMock()
        svc.resolve_effective_fields.return_value = {
            "fields": [{"field_name": "document_number", "field_type": "string"}],
            "count": 1,
        }
        get_svc.return_value = svc
        updated = DocumentReceiverService._attach_key_fields(processing)

    assert updated["effective_fields"]
    assert updated["key_fields"][0]["field_name"] == "document_number"


def test_persist_document_type_metadata_copies_effective_fields() -> None:
    service = DocumentReceiverService.__new__(DocumentReceiverService)
    record = {
        "document_id": "doc-1",
        "processing": {
            "document_type_id": "type-1",
            "document_type_name": "SOP",
            "effective_fields": [{"field_name": "version", "type": "string"}],
        },
        "metadata": {},
    }
    service._persist_document_type_metadata(record)
    assert record["metadata"]["document_type_id"] == "type-1"
    assert record["metadata"]["document_type_name"] == "SOP"
    assert record["metadata"]["effective_fields"][0]["field_name"] == "version"
    assert record["document_type_id"] == "type-1"
    assert record["document_type_name"] == "SOP"
