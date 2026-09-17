"""Regression: reject Swagger placeholder / unknown metadata fields before persistence."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.documents.application.document_service import DocumentReceiverService


def test_update_document_metadata_rejects_unknown_fields() -> None:
    service = DocumentReceiverService()
    store = MagicMock()
    store.get_document.return_value = {
        "document_id": "doc-1",
        "repository_id": "repo-1",
        "metadata": {"document_type_id": "type-1", "repository_id": "repo-1"},
        "document_type_id": "type-1",
    }
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    with patch.object(
        DocumentReceiverService,
        "_allowed_effective_field_names",
        return_value={"document_title", "version"},
    ), patch(
        "src.application.consumer_api.context.get_current_user_from_context",
        return_value=None,
    ):
        with pytest.raises(HTTPException) as exc:
            service.update_document_metadata("doc-1", {"additionalProp1": {}})

    assert exc.value.status_code == 400
    assert exc.value.detail["message"] == "Unknown metadata fields"
    assert exc.value.detail["fields"] == ["additionalProp1"]
    store.update_document_metadata.assert_not_called()


def test_update_document_metadata_accepts_effective_fields() -> None:
    service = DocumentReceiverService()
    store = MagicMock()
    record = {
        "document_id": "doc-1",
        "repository_id": "repo-1",
        "metadata": {"document_type_id": "type-1", "repository_id": "repo-1"},
        "document_type_id": "type-1",
        "document_type_name": "SOP",
    }
    updated = {
        **record,
        "metadata": {
            "document_type_id": "type-1",
            "repository_id": "repo-1",
            "document_title": "Updated SOP",
        },
    }
    store.get_document.return_value = record
    store.update_document_metadata.return_value = updated
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service.document_response = MagicMock(return_value=MagicMock(model_dump=lambda: {"ok": True}))  # type: ignore[method-assign]

    with patch.object(
        DocumentReceiverService,
        "_allowed_effective_field_names",
        return_value={"document_title", "version"},
    ), patch(
        "src.application.consumer_api.context.get_current_user_from_context",
        return_value=None,
    ):
        service.update_document_metadata("doc-1", {"document_title": "Updated SOP"})

    store.update_document_metadata.assert_called_once_with(
        "doc-1",
        {"document_title": "Updated SOP"},
        merge=True,
    )


def test_record_metadata_edits_skips_unknown_fields() -> None:
    from src.features.human_review.application.review_service import DocumentReviewService

    service = DocumentReviewService.__new__(DocumentReviewService)
    store = MagicMock()
    store.get_open_review_for_document.return_value = None
    store.get_latest_review_for_document.return_value = None
    service._require_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service.rerun_validation = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._allowed_metadata_field_names = MagicMock(  # type: ignore[method-assign]
        return_value={"document_title", "document_number", "version", "effective_date"}
    )

    out = service.record_metadata_edits(
        document_id="doc-1",
        changes=[
            {"field_name": "additionalProp1", "old_value": None, "new_value": {}},
            {"field_name": "document_title", "old_value": "A", "new_value": "B"},
        ],
        modified_by="tester",
        allowed_field_names={"document_title"},
    )

    assert out is None or store.insert_audit.called
    assert store.insert_audit.call_count == 1
    assert store.insert_audit.call_args.kwargs["field_name"] == "document_title"


def test_review_history_filters_invalid_metadata_edits_and_counts_valid_only() -> None:
    from src.features.human_review.application.review_service import DocumentReviewService

    service = DocumentReviewService.__new__(DocumentReviewService)
    store = MagicMock()
    store.list_audit_for_document.return_value = [
        {"action": "review_created", "field_name": None},
        {"action": "metadata_edit", "field_name": "additionalProp1", "new_value": {}},
        {"action": "metadata_edit", "field_name": "document_title", "new_value": "Updated SOP Title"},
        {"action": "metadata_edit", "field_name": "document_number", "new_value": "SOP-1"},
        {"action": "metadata_edit", "field_name": "version", "new_value": "2"},
        {"action": "metadata_edit", "field_name": "effective_date", "new_value": "2026-01-01"},
        {"action": "approve", "field_name": None},
    ]
    service._require_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service._require_read = MagicMock()  # type: ignore[method-assign]
    service._allowed_metadata_field_names = MagicMock(  # type: ignore[method-assign]
        return_value={"document_title", "document_number", "version", "effective_date", "owner"}
    )

    with patch(
        "src.features.documents.application.document_service.DocumentReceiverService.get_document_record",
        return_value={"document_id": "doc-1", "repository_id": "repo-1", "metadata": {}},
    ):
        # Bypass constructor wiring by calling unbound method with prepared service.
        out = DocumentReviewService.get_review_history(service, "doc-1")

    assert out["count"] == 6
    field_names = [row.get("field_name") for row in out["history"] if row.get("action") == "metadata_edit"]
    assert field_names == ["document_title", "document_number", "version", "effective_date"]
    assert "additionalProp1" not in field_names


def test_insert_audit_rejects_additional_prop_field() -> None:
    from src.features.human_review.domain.review_exceptions import ValidationError
    from src.features.human_review.infrastructure.review_repository import DocumentReviewStore

    store = DocumentReviewStore.__new__(DocumentReviewStore)
    with pytest.raises(ValidationError) as exc:
        store.insert_audit(
            document_id="doc-1",
            action="metadata_edit",
            field_name="additionalProp1",
            old_value=None,
            new_value={},
            modified_by="tester",
            allowed_field_names={"document_title"},
        )
    assert exc.value.details["fields"] == ["additionalProp1"]
