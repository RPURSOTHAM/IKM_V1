from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from src.features.documents.application.document_service import DocumentReceiverService


def test_validate_document_type_rejects_unknown_id() -> None:
    service = DocumentReceiverService()
    store = MagicMock()
    store.get_type_by_id.return_value = None
    service._document_type_store = MagicMock(return_value=store)  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as exc:
        service._validate_document_type("unknown")

    assert exc.value.status_code == 400
    assert exc.value.detail == {"error": "Invalid document_type_id"}


def test_validate_document_type_rejects_inactive_type() -> None:
    service = DocumentReceiverService()
    store = MagicMock()
    store.get_type_by_id.return_value = SimpleNamespace(
        document_type_id="type-1",
        name="Type 1",
        repository_id="repo-1",
        is_active=False,
    )
    service._document_type_store = MagicMock(return_value=store)  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as exc:
        service._validate_document_type("type-1", repository_id="repo-1")

    assert exc.value.status_code == 400
    assert exc.value.detail == {"error": "Invalid document_type_id"}


def test_validate_document_type_accepts_active_matching_type() -> None:
    service = DocumentReceiverService()
    store = MagicMock()
    store.get_type_by_id.return_value = SimpleNamespace(
        document_type_id="type-1",
        name="Type 1",
        repository_id="repo-1",
        is_active=True,
    )
    service._document_type_store = MagicMock(return_value=store)  # type: ignore[method-assign]

    resolved_id, resolved_name = service._validate_document_type("type-1", repository_id="repo-1")
    assert resolved_id == "type-1"
    assert resolved_name == "Type 1"


def test_validate_document_type_returns_503_when_store_lookup_fails() -> None:
    service = DocumentReceiverService()
    store = MagicMock()
    store.get_type_by_id.side_effect = RuntimeError("db offline")
    service._document_type_store = MagicMock(return_value=store)  # type: ignore[method-assign]

    with pytest.raises(HTTPException) as exc:
        service._validate_document_type("type-1")

    assert exc.value.status_code == 503
    assert exc.value.detail["error"] == "Document type validation unavailable"


def test_resolve_document_type_propagates_non_400_errors() -> None:
    service = DocumentReceiverService()
    service._validate_document_type = MagicMock(  # type: ignore[method-assign]
        side_effect=HTTPException(status_code=503, detail="Document type store unavailable")
    )

    with pytest.raises(HTTPException) as exc:
        service._resolve_document_type(
            repository_id="repo-1",
            document_type_id=None,
            processing={"document_type_id": "type-1"},
        )

    assert exc.value.status_code == 503


@pytest.mark.anyio
async def test_upload_duplicate_cleans_up_document_instance() -> None:
    service = DocumentReceiverService()
    live = SimpleNamespace(
        max_files_per_request=5,
        upload_dir=".",
        allowed_extensions=[".txt", ".pdf", ".docx"],
        repository_type="local",
        default_collection_name="Document",
        default_tenant_id=None,
    )
    service._live_settings = MagicMock(return_value=live)  # type: ignore[method-assign]
    service._require_active_repository = MagicMock()  # type: ignore[method-assign]
    service._validate_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._resolve_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    service._save_upload_file = AsyncMock(return_value=("hash-1", 100))  # type: ignore[method-assign]
    service._apply_repository_context = MagicMock()  # type: ignore[method-assign]
    service._purge_replaceable_duplicates = MagicMock()  # type: ignore[method-assign]
    service._find_duplicate_document = MagicMock(  # type: ignore[method-assign]
        return_value={"document_id": "existing-doc", "document_name": "existing.txt"}
    )
    service._delete_document_instance = MagicMock()  # type: ignore[method-assign]
    service._raise_duplicate_upload = MagicMock(  # type: ignore[method-assign]
        side_effect=HTTPException(status_code=409, detail={"error": "duplicate"})
    )

    upload = UploadFile(filename="sample.txt", file=io.BytesIO(b"sample"))

    with patch(
        "src.features.security.compliance.compliance_validator.check_document_compliance",
        return_value=None,
    ):
        with pytest.raises(HTTPException) as exc:
            await service.upload_documents(
                [upload],
                repository_id="repo-1",
                document_type_id="type-1",
                submit_for_processing=False,
            )

    assert exc.value.status_code == 409
    service._delete_document_instance.assert_called_once()


@pytest.mark.anyio
async def test_upload_forwards_document_type_context_to_compliance_scan() -> None:
    service = DocumentReceiverService()
    live = SimpleNamespace(
        max_files_per_request=5,
        upload_dir=".",
        allowed_extensions=[".txt", ".pdf", ".docx"],
        repository_type="local",
        default_collection_name="Document",
        default_tenant_id=None,
    )
    service._live_settings = MagicMock(return_value=live)  # type: ignore[method-assign]
    service._require_active_repository = MagicMock()  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    service._save_upload_file = AsyncMock(return_value=("hash-1", 100))  # type: ignore[method-assign]
    service._validate_document_type = MagicMock(return_value=("type-1", "SOP"))  # type: ignore[method-assign]
    service._resolve_document_type = MagicMock(return_value=("type-1", "SOP"))  # type: ignore[method-assign]
    service._apply_repository_context = MagicMock()  # type: ignore[method-assign]
    service._purge_replaceable_duplicates = MagicMock()  # type: ignore[method-assign]
    service._find_duplicate_document = MagicMock(  # type: ignore[method-assign]
        return_value={"document_id": "existing-doc", "document_name": "existing.txt"}
    )
    service._delete_document_instance = MagicMock()  # type: ignore[method-assign]
    service._raise_duplicate_upload = MagicMock(  # type: ignore[method-assign]
        side_effect=HTTPException(status_code=409, detail={"error": "duplicate"})
    )

    upload = UploadFile(filename="sample.txt", file=io.BytesIO(b"sample"))

    with patch(
        "src.features.security.compliance.compliance_validator.check_document_compliance",
        return_value=None,
    ) as compliance_mock:
        with pytest.raises(HTTPException):
            await service.upload_documents(
                [upload],
                repository_id="repo-1",
                document_type_id="type-1",
                submit_for_processing=False,
            )

    compliance_mock.assert_called_once()
    _, kwargs = compliance_mock.call_args
    assert kwargs["repository_id"] == "repo-1"
    assert kwargs["document_type_id"] == "type-1"
    assert kwargs["document_type_name"] == "SOP"
