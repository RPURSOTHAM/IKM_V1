"""Unit test for Phase 4 document validation processor persistence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.features.document_processing.core.contract import ProcessRequest
from src.features.document_processing.processors.document_validation import DocumentValidationProcessor


def test_validation_artifact_persistence() -> None:
    request = ProcessRequest(
        processor_type="document_validation",
        document_id="doc-val-1",
        document_name="doc-val-1.pdf",
        document_type_id="type-1",
        document_type_name="Manufacturing SOP",
        effective_fields=[
            {"field_name": "document_number", "type": "string", "required": True},
            {"field_name": "equipment_id", "type": "string", "required": False},
        ],
        validation_confidence_threshold=0.85,
    )
    artifact = {
        "artifact_payload": {
            "document_type_id": "type-1",
            "source_pages": [1, 2, 12],
            "fields": {
                "document_number": {"value": "MSOP-1", "confidence": 0.9},
                "equipment_id": {"value": "EQ-1", "confidence": 0.7},
            },
        }
    }
    with patch(
        "src.features.document_processing.processors.document_validation._wait_for_key_field_artifact",
        return_value=artifact,
    ), patch(
        "src.features.document_processing.processors.document_validation.store_document_graph",
        return_value="neo4j://DocumentArtifact/doc-val-1/document_validation",
    ) as store:
        result = DocumentValidationProcessor().run(
            request,
            Path("doc-val-1.pdf"),
            set_status=lambda s, p: None,
            check_stop=lambda: False,
        )

    assert result.processor_type == "document_validation"
    assert result.result_location.endswith("/document_validation")
    assert result.document_metadata["status"] == "WARNING"
    assert result.document_metadata["document_status"] == "WARNING"
    assert result.document_metadata["low_confidence_fields"][0]["field"] == "equipment_id"
    stored_payload = store.call_args.kwargs["payload"]
    assert stored_payload["status"] == "WARNING"
    assert store.call_args.kwargs["processor_type"] == "document_validation"
