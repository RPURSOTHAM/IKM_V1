"""Unit tests for Phase 4 validation metadata API exposure."""

from __future__ import annotations

from src.features.documents.application.document_metadata_service import (
    resolve_document_metadata_bundle,
    resolve_validation_bundle,
)


def test_metadata_bundle_exposes_validation() -> None:
    record = {
        "document_id": "doc-1",
        "document_type_id": "type-1",
        "document_type_name": "Manufacturing SOP",
        "metadata": {
            "document_type_id": "type-1",
            "document_type_name": "Manufacturing SOP",
            "extracted_metadata": {"document_number": "MSOP-1"},
            "validation": {
                "status": "WARNING",
                "document_status": "WARNING",
                "missing_required_fields": [],
                "invalid_fields": [],
                "low_confidence_fields": [{"field": "title", "confidence": 0.72}],
            },
            "validation_status": "WARNING",
        },
        "processing": {"document_type_id": "type-1", "validation_enabled": True},
    }
    bundle = resolve_document_metadata_bundle(record)
    assert "validation" in bundle
    assert bundle["validation"]["status"] == "WARNING"
    assert bundle["validation"]["low_confidence_fields"][0]["field"] == "title"
    assert bundle["validation_status"] == "WARNING"


def test_resolve_validation_bundle_from_job() -> None:
    record = {"document_id": "doc-2", "metadata": {}}
    job = {
        "scheduling_metadata": {
            "processor_results": {
                "document_validation": {
                    "status": "COMPLETED",
                    "completed_at": "2026-07-23T12:00:00",
                    "document_metadata": {
                        "status": "PASS",
                        "document_status": "VALID",
                        "missing_required_fields": [],
                        "invalid_fields": [],
                        "low_confidence_fields": [],
                        "confidence_threshold": 0.85,
                    },
                }
            }
        }
    }
    validation = resolve_validation_bundle(record, job=job)
    assert validation["status"] == "PASS"
    assert validation["document_status"] == "VALID"
