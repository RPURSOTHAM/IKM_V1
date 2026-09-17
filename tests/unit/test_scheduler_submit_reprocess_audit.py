"""Submit/reprocess should emit processor audit hooks after successful queueing."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.features.documents.application.document_service import DocumentReceiverService
from src.shared.errors.service_errors import DmsServiceError


def _sample_record(**overrides: Any) -> dict[str, Any]:
    record = {
        "document_id": "doc-submit-1",
        "document_name": "sample.pdf",
        "original_file_name": "sample.pdf",
        "document_type": "pdf",
        "repository_type": "local",
        "repository_path": "/tmp/sample.pdf",
        "collection_name": "default",
        "tenant_id": None,
        "status": "uploaded",
        "upload_timestamp": 1.0,
        "repository_id": "repo-submit-1",
        "repository_name": "Demo Repository",
        "metadata": {},
    }
    record.update(overrides)
    return record


@pytest.mark.asyncio
async def test_submit_document_records_audit_after_success() -> None:
    svc = DocumentReceiverService(publisher=MagicMock())
    record = _sample_record()
    store = MagicMock()
    store.get_document.return_value = record
    js = MagicMock()

    with (
        patch.object(svc, "get_store", return_value=store),
        patch.object(svc, "_job_store", return_value=js),
        patch.object(svc, "submit_to_queue") as submit,
        patch.object(svc, "document_response", return_value=MagicMock()),
        patch(
            "src.features.observability.hooks.processor_hooks.record_document_submitted"
        ) as submitted,
    ):
        await svc.submit_document(record["document_id"])

    submit.assert_called_once_with(record)
    submitted.assert_called_once_with(
        record["document_id"],
        repository_id=record["repository_id"],
        document_name=record["document_name"],
        repository_name=record["repository_name"],
    )


@pytest.mark.asyncio
async def test_submit_document_skips_audit_on_queue_failure() -> None:
    svc = DocumentReceiverService(publisher=MagicMock())
    record = _sample_record()
    store = MagicMock()
    store.get_document.return_value = record
    js = MagicMock()

    with (
        patch.object(svc, "get_store", return_value=store),
        patch.object(svc, "_job_store", return_value=js),
        patch.object(svc, "submit_to_queue", side_effect=RuntimeError("queue down")),
        patch(
            "src.features.observability.hooks.processor_hooks.record_document_submitted"
        ) as submitted,
        pytest.raises(DmsServiceError),
    ):
        await svc.submit_document(record["document_id"])

    submitted.assert_not_called()


@pytest.mark.asyncio
async def test_reprocess_document_records_audit_after_success() -> None:
    svc = DocumentReceiverService(publisher=MagicMock())
    record = _sample_record(document_id="doc-reprocess-1")
    store = MagicMock()
    store.get_document.return_value = record
    js = MagicMock()

    with (
        patch.object(svc, "get_store", return_value=store),
        patch.object(svc, "_job_store", return_value=js),
        patch.object(svc, "submit_to_queue") as submit,
        patch.object(svc, "document_response", return_value=MagicMock()),
        patch(
            "src.features.observability.hooks.processor_hooks.record_document_reprocessed"
        ) as reprocessed,
    ):
        await svc.reprocess_document(record["document_id"])

    submit.assert_called_once_with(record)
    js.update_for_reprocess.assert_called_once_with(record["document_id"])
    reprocessed.assert_called_once_with(
        record["document_id"],
        repository_id=record["repository_id"],
        document_name=record["document_name"],
        repository_name=record["repository_name"],
    )


@pytest.mark.asyncio
async def test_reprocess_document_skips_audit_on_queue_failure() -> None:
    svc = DocumentReceiverService(publisher=MagicMock())
    record = _sample_record(document_id="doc-reprocess-2")
    store = MagicMock()
    store.get_document.return_value = record
    js = MagicMock()

    with (
        patch.object(svc, "get_store", return_value=store),
        patch.object(svc, "_job_store", return_value=js),
        patch.object(svc, "submit_to_queue", side_effect=RuntimeError("queue down")),
        patch(
            "src.features.observability.hooks.processor_hooks.record_document_reprocessed"
        ) as reprocessed,
        pytest.raises(DmsServiceError),
    ):
        await svc.reprocess_document(record["document_id"])

    reprocessed.assert_not_called()
