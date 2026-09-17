"""Unit tests for Phase 4 document validation engine."""

from __future__ import annotations

from src.features.document_validation.application.validation_pipeline import (
    map_validation_status_to_document_status,
    run_document_validation,
    validate_value_type,
)
from src.features.document_processing.shared_processor.types import ProcessorType, enabled_processor_types, processor_dependencies_met


def test_missing_required_field_fails() -> None:
    result = run_document_validation(
        effective_fields=[
            {"field_name": "document_number", "type": "string", "required": True},
            {"field_name": "effective_date", "type": "date", "required": True},
        ],
        extracted_fields={"document_number": {"value": "SOP-1", "confidence": 0.9}},
    )
    assert result["status"] == "FAIL"
    assert result["document_status"] == "INVALID"
    assert result["missing_required_fields"] == ["effective_date"]


def test_invalid_date_field_fails() -> None:
    result = run_document_validation(
        effective_fields=[{"field_name": "effective_date", "type": "date", "required": False}],
        extracted_fields={"effective_date": {"value": "abc123", "confidence": 0.95}},
    )
    assert result["status"] == "FAIL"
    assert result["invalid_fields"] == [
        {"field": "effective_date", "reason": "invalid_date", "value": "abc123"}
    ]


def test_invalid_integer_field_fails() -> None:
    assert validate_value_type("12.5", "integer") == "invalid_integer"
    result = run_document_validation(
        effective_fields=[{"field_name": "batch_count", "type": "integer", "required": False}],
        extracted_fields={"batch_count": {"value": "not-a-number", "confidence": 0.9}},
    )
    assert result["status"] == "FAIL"
    assert result["invalid_fields"][0]["reason"] == "invalid_integer"


def test_low_confidence_warning() -> None:
    result = run_document_validation(
        effective_fields=[{"field_name": "document_title", "type": "string", "required": False}],
        extracted_fields={"document_title": {"value": "Title", "confidence": 0.72}},
        confidence_threshold=0.85,
    )
    assert result["status"] == "WARNING"
    assert result["document_status"] == "WARNING"
    assert result["low_confidence_fields"] == [{"field": "document_title", "confidence": 0.72}]
    assert not result["missing_required_fields"]
    assert not result["invalid_fields"]


def test_pass_scenario() -> None:
    result = run_document_validation(
        effective_fields=[
            {"field_name": "document_number", "type": "string", "required": True},
            {"field_name": "effective_date", "type": "date", "required": True},
        ],
        extracted_fields={
            "document_number": {"value": "MSOP-001", "confidence": 0.93},
            "effective_date": {"value": "2026-01-01", "confidence": 0.91},
        },
    )
    assert result["status"] == "PASS"
    assert result["document_status"] == "VALID"
    assert result["missing_required_fields"] == []
    assert result["invalid_fields"] == []
    assert result["low_confidence_fields"] == []


def test_inherited_and_local_fields_validated() -> None:
    result = run_document_validation(
        effective_fields=[
            {
                "field_name": "document_number",
                "type": "string",
                "required": True,
                "inherited": True,
            },
            {
                "field_name": "equipment_id",
                "type": "string",
                "required": True,
                "inherited": False,
            },
        ],
        extracted_fields={
            "document_number": {"value": "MSOP-001", "confidence": 0.9},
            # equipment_id missing -> fail
        },
    )
    assert result["status"] == "FAIL"
    assert result["missing_required_fields"] == ["equipment_id"]


def test_local_field_type_validation() -> None:
    result = run_document_validation(
        effective_fields=[
            {"field_name": "equipment_id", "type": "string", "required": False},
            {"field_name": "batch_count", "type": "integer", "required": False},
        ],
        extracted_fields={
            "equipment_id": {"value": "EQ-1", "confidence": 0.9},
            "batch_count": {"value": "3", "confidence": 0.9},
        },
    )
    assert result["status"] == "PASS"


def test_status_mapping() -> None:
    assert map_validation_status_to_document_status("PASS") == "VALID"
    assert map_validation_status_to_document_status("WARNING") == "WARNING"
    assert map_validation_status_to_document_status("FAIL") == "INVALID"


def test_enabled_processor_types_includes_validation() -> None:
    enabled = enabled_processor_types(
        {
            "key_field_extraction_enabled": True,
            "validation_enabled": True,
        }
    )
    assert ProcessorType.KEY_FIELD_EXTRACTION.value in enabled
    assert ProcessorType.DOCUMENT_VALIDATION.value in enabled
    assert enabled.index(ProcessorType.KEY_FIELD_EXTRACTION.value) < enabled.index(
        ProcessorType.DOCUMENT_VALIDATION.value
    )


def test_validation_disabled_not_scheduled() -> None:
    enabled = enabled_processor_types(
        {
            "key_field_extraction_enabled": True,
            "validation_enabled": False,
        }
    )
    assert ProcessorType.DOCUMENT_VALIDATION.value not in enabled


def test_validation_requires_key_field_extraction() -> None:
    enabled = enabled_processor_types({"validation_enabled": True})
    assert ProcessorType.DOCUMENT_VALIDATION.value not in enabled


def test_processor_dependencies_for_validation() -> None:
    assert processor_dependencies_met("document_validation", {}) is False
    assert (
        processor_dependencies_met(
            "document_validation",
            {"key_field_extraction": {"status": "COMPLETED"}},
        )
        is True
    )


def test_empty_string_required_is_missing() -> None:
    result = run_document_validation(
        effective_fields=[{"field_name": "document_number", "type": "string", "required": True}],
        extracted_fields={"document_number": {"value": "   ", "confidence": 0.99}},
    )
    assert result["status"] == "FAIL"
    assert "document_number" in result["missing_required_fields"]
