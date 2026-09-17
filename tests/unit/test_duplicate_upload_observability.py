"""Duplicate upload rejection must record audit + metrics and keep HTTP 409 shape."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.documents.application.document_service import DocumentReceiverService


def _existing_duplicate(**overrides: Any) -> dict[str, Any]:
    record = {
        "document_id": "doc-existing-1",
        "document_name": "duplicate.pdf",
        "original_file_name": "duplicate.pdf",
        "status": "queued",
        "repository_id": "repo-dup-1",
        "content_hash": "abcdef0123456789ffff",
        "metadata": {"content_hash": "abcdef0123456789ffff"},
    }
    record.update(overrides)
    return record


def test_raise_duplicate_upload_records_audit_and_metrics() -> None:
    existing = _existing_duplicate()
    audit = MagicMock()
    metrics = MagicMock()

    with (
        patch(
            "src.features.observability.audit.application.audit_service.get_audit_service",
            return_value=audit,
        ),
        patch(
            "src.features.observability.metrics.application.metrics_service.get_metrics_service",
            return_value=metrics,
        ),
        pytest.raises(HTTPException) as raised,
    ):
        DocumentReceiverService._raise_duplicate_upload(existing)

    exc = raised.value
    assert exc.status_code == 409
    assert isinstance(exc.detail, dict)
    assert exc.detail["code"] == "duplicate_document"
    assert exc.detail["existing_document_id"] == existing["document_id"]

    audit.record.assert_called_once()
    audit_kwargs = audit.record.call_args
    assert audit_kwargs.args[0] == "DOCUMENT_UPLOAD_DUPLICATE_REJECTED"
    assert audit_kwargs.kwargs["category"] == "document"
    assert audit_kwargs.kwargs["action"] == "upload_duplicate"
    meta = audit_kwargs.kwargs["metadata"]
    assert meta["existing_document_id"] == existing["document_id"]
    assert meta["existing_status"] == "queued"
    assert meta["scope"] == "repository"
    assert meta["content_hash_prefix"] == "abcdef012345"
    assert meta["filename"] == "duplicate.pdf"
    assert "content_hash" not in meta or meta.get("content_hash") is None

    metrics.record.assert_called_once()
    metric_kwargs = metrics.record.call_args
    assert metric_kwargs.args[0] == "document_upload_duplicate"
    assert metric_kwargs.kwargs["status"] == "failure"


def test_raise_duplicate_upload_collection_scope_without_hash() -> None:
    existing = _existing_duplicate(
        repository_id=None,
        content_hash=None,
        metadata={},
        original_file_name=None,
        document_name="coll-doc.pdf",
    )
    audit = MagicMock()
    metrics = MagicMock()

    with (
        patch(
            "src.features.observability.audit.application.audit_service.get_audit_service",
            return_value=audit,
        ),
        patch(
            "src.features.observability.metrics.application.metrics_service.get_metrics_service",
            return_value=metrics,
        ),
        pytest.raises(HTTPException) as raised,
    ):
        DocumentReceiverService._raise_duplicate_upload(existing)

    assert raised.value.status_code == 409
    assert raised.value.detail["code"] == "duplicate_document"
    meta = audit.record.call_args.kwargs["metadata"]
    assert meta["scope"] == "collection"
    assert "content_hash_prefix" not in meta
    assert meta["filename"] == "coll-doc.pdf"
