"""Regression tests for repository-scoped duplicate upload detection."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
from src.features.documents.application.document_service import DocumentReceiverService


def _store_with_docs(docs: list[dict[str, Any]]) -> MetadataStore:
    path = Path(tempfile.mkdtemp()) / "metadata.json"
    store = MetadataStore(str(path))
    for doc in docs:
        store.insert_document(doc)
    return store


REPO_A = "11111111-1111-4111-8111-111111111111"
REPO_B = "22222222-2222-4222-8222-222222222222"


def _doc(
    *,
    document_id: str,
    content_hash: str,
    status: str = "uploaded",
    repository_id: str = REPO_A,
    document_name: str | None = None,
    content_text_hash: str | None = None,
) -> dict[str, Any]:
    name = document_name or f"{document_id}.txt"
    return {
        "document_id": document_id,
        "document_name": name,
        "original_file_name": name,
        "document_type": "txt",
        "repository_type": "local",
        "repository_path": f"/tmp/{name}",
        "collection_name": "TestCollection",
        "tenant_id": None,
        "status": status,
        "upload_timestamp": 1.0,
        "content_hash": content_hash,
        "content_text_hash": content_text_hash,
        "repository_id": repository_id,
        "metadata": {
            "content_hash": content_hash,
            "content_text_hash": content_text_hash,
            "repository_id": repository_id,
        },
    }


@pytest.mark.parametrize(
    "status",
    ["uploaded", "queued", "processing", "completed", "received", "failed", "human_review"],
)
def test_find_duplicate_blocks_active_statuses(status: str) -> None:
    content_hash = "abc123" * 5 + "abcd"
    store = _store_with_docs([_doc(document_id="existing-1", content_hash=content_hash, status=status)])
    service = DocumentReceiverService()
    with patch.object(service, "get_store", return_value=store):
        duplicate = service._find_duplicate_document(
            content_hash=content_hash,
            repository_id=REPO_A,
            collection_name="TestCollection",
            tenant_id=None,
        )
    assert duplicate is not None
    assert duplicate["document_id"] == "existing-1"


def test_find_duplicate_blocks_human_review() -> None:
    content_hash = "def456" * 5 + "defg"
    store = _store_with_docs(
        [_doc(document_id="review-1", content_hash=content_hash, status="human_review")]
    )
    service = DocumentReceiverService()
    with patch.object(service, "get_store", return_value=store):
        duplicate = service._find_duplicate_document(
            content_hash=content_hash,
            repository_id=REPO_A,
            collection_name="TestCollection",
            tenant_id=None,
        )
    assert duplicate is not None
    assert duplicate["document_id"] == "review-1"


def test_find_duplicate_scoped_to_repository() -> None:
    content_hash = "hash789" * 5 + "hash"
    store = _store_with_docs(
        [
            _doc(document_id="repo-a-doc", content_hash=content_hash, repository_id=REPO_A),
            _doc(document_id="repo-b-doc", content_hash=content_hash, repository_id=REPO_B),
        ]
    )
    service = DocumentReceiverService()
    with patch.object(service, "get_store", return_value=store):
        dup_a = service._find_duplicate_document(
            content_hash=content_hash,
            repository_id=REPO_A,
            collection_name="TestCollection",
            tenant_id=None,
        )
        dup_b = service._find_duplicate_document(
            content_hash=content_hash,
            repository_id=REPO_B,
            collection_name="TestCollection",
            tenant_id=None,
        )
    assert dup_a["document_id"] == "repo-a-doc"
    assert dup_b["document_id"] == "repo-b-doc"


def test_find_duplicate_renamed_file_same_hash() -> None:
    content_hash = "same000" * 5 + "same"
    store = _store_with_docs(
        [_doc(document_id="doc-1", content_hash=content_hash, document_name="first-name.txt")]
    )
    service = DocumentReceiverService()
    with patch.object(service, "get_store", return_value=store):
        duplicate = service._find_duplicate_document(
            content_hash=content_hash,
            repository_id=REPO_A,
            collection_name="TestCollection",
            tenant_id=None,
        )
    assert duplicate is not None


def test_insert_document_if_not_duplicate_race_safe() -> None:
    path = Path(tempfile.mkdtemp()) / "metadata.json"
    store = MetadataStore(str(path))
    existing = _doc(document_id="existing", content_hash="race" * 10, status="completed")
    store.insert_document(existing)
    incoming = _doc(document_id="incoming", content_hash="race" * 10, document_name="incoming.txt")
    conflict = store.insert_document_if_not_duplicate(
        incoming,
        content_hash="race" * 10,
        repository_id=REPO_A,
        non_blocking_statuses=DocumentReceiverService._NON_BLOCKING_DUPLICATE_STATUSES,
    )
    assert conflict is not None
    assert conflict["document_id"] == "existing"
    assert store.get_document("incoming") is None


def test_raise_duplicate_upload_is_http_409() -> None:
    existing = _doc(document_id="dup", content_hash="x" * 64)
    with pytest.raises(HTTPException) as exc:
        DocumentReceiverService._raise_duplicate_upload(existing)
    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "duplicate_document"


@pytest.mark.anyio
async def test_batch_upload_accepts_unique_and_rejects_duplicates() -> None:
    """Multi-file upload must keep unique files and skip duplicates (partial success)."""
    import io
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import UploadFile

    from src.features.documents.schemas.document_schemas import DocumentResponse

    service = DocumentReceiverService()
    live = SimpleNamespace(
        max_files_per_request=20,
        upload_dir=tempfile.mkdtemp(),
        allowed_extensions=[".txt", ".pdf", ".docx"],
        repository_type="local",
        default_collection_name="Document",
        default_tenant_id=None,
    )
    job_store = MagicMock()
    job_store.ensure_received_from_record.return_value = "job-1"
    store = MagicMock()
    store.insert_document_if_not_duplicate.return_value = None
    store.get_document.return_value = None

    service._live_settings = MagicMock(return_value=live)  # type: ignore[method-assign]
    service._require_active_repository = MagicMock()  # type: ignore[method-assign]
    service._validate_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._resolve_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service._save_upload_file = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            ("hash-new-1", 10),
            ("hash-dup", 10),
            ("hash-new-2", 10),
            ("hash-dup-2", 10),
        ]
    )
    service._apply_repository_context = MagicMock()  # type: ignore[method-assign]
    service._purge_replaceable_duplicates = MagicMock()  # type: ignore[method-assign]
    service._delete_document_instance = MagicMock()  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]

    def _response(record: dict[str, Any]) -> DocumentResponse:
        return DocumentResponse(
            document_id=str(record["document_id"]),
            document_name=str(record["document_name"]),
            original_file_name=str(record["original_file_name"]),
            document_type=str(record["document_type"]),
            repository_type=str(record["repository_type"]),
            repository_path=str(record["repository_path"]),
            collection_name=str(record.get("collection_name") or "Document"),
            status=str(record.get("status") or "uploaded"),
            upload_timestamp=float(record.get("upload_timestamp") or 1.0),
            metadata=dict(record.get("metadata") or {}),
        )

    service.document_response = MagicMock(side_effect=_response)  # type: ignore[method-assign]
    service._find_duplicate_document = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            None,
            {
                "document_id": "existing-1",
                "original_file_name": "dup-a.txt",
                "status": "completed",
                "repository_id": REPO_A,
            },
            None,
            {
                "document_id": "existing-2",
                "original_file_name": "dup-b.txt",
                "status": "completed",
                "repository_id": REPO_A,
            },
        ]
    )

    files = [
        UploadFile(filename="new-a.txt", file=io.BytesIO(b"a")),
        UploadFile(filename="dup-a.txt", file=io.BytesIO(b"dup1")),
        UploadFile(filename="new-b.txt", file=io.BytesIO(b"b")),
        UploadFile(filename="dup-b.txt", file=io.BytesIO(b"dup2")),
    ]

    with (
        patch(
            "src.features.security.compliance.compliance_validator.check_document_compliance",
            return_value=None,
        ),
        patch(
            "src.features.security.compliance.compliance_validator.security_scan_requires_review",
            return_value=False,
        ),
        patch(
            "src.features.observability.audit.application.audit_service.get_audit_service",
            return_value=MagicMock(),
        ),
        patch(
            "src.features.observability.metrics.application.metrics_service.get_metrics_service",
            return_value=MagicMock(),
        ),
    ):
        result = await service.upload_documents(
            files,
            repository_id=REPO_A,
            submit_for_processing=False,
        )

    assert result.uploaded_count == 2
    assert result.rejected_count == 2
    assert len(result.documents) == 2
    assert len(result.rejected) == 2
    assert {d.original_file_name for d in result.documents} == {"new-a.txt", "new-b.txt"}
    assert {r.filename for r in result.rejected} == {"dup-a.txt", "dup-b.txt"}
    assert all(r.code == "duplicate_document" for r in result.rejected)


@pytest.mark.anyio
async def test_batch_upload_rejects_identical_files_in_same_request() -> None:
    """Second identical file in one multipart upload is rejected (even if first is human_review)."""
    import io
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import UploadFile

    from src.features.documents.schemas.document_schemas import DocumentResponse

    service = DocumentReceiverService()
    live = SimpleNamespace(
        max_files_per_request=20,
        upload_dir=tempfile.mkdtemp(),
        allowed_extensions=[".txt", ".pdf", ".docx"],
        repository_type="local",
        default_collection_name="Document",
        default_tenant_id=None,
    )
    job_store = MagicMock()
    job_store.ensure_received_from_record.return_value = "job-1"
    store = MagicMock()
    store.insert_document_if_not_duplicate.return_value = None
    store.get_document.return_value = None

    service._live_settings = MagicMock(return_value=live)  # type: ignore[method-assign]
    service._require_active_repository = MagicMock()  # type: ignore[method-assign]
    service._validate_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._resolve_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    # Same content hash for both files.
    service._save_upload_file = AsyncMock(return_value=("same-hash-batch", 10))  # type: ignore[method-assign]
    service._apply_repository_context = MagicMock()  # type: ignore[method-assign]
    service._purge_replaceable_duplicates = MagicMock()  # type: ignore[method-assign]
    service._delete_document_instance = MagicMock()  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service._find_duplicate_document = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._register_human_review_queue_entry = MagicMock()  # type: ignore[method-assign]

    def _response(record: dict[str, Any]) -> DocumentResponse:
        return DocumentResponse(
            document_id=str(record["document_id"]),
            document_name=str(record["document_name"]),
            original_file_name=str(record["original_file_name"]),
            document_type=str(record["document_type"]),
            repository_type=str(record["repository_type"]),
            repository_path=str(record["repository_path"]),
            collection_name=str(record.get("collection_name") or "Document"),
            status=str(record.get("status") or "uploaded"),
            upload_timestamp=float(record.get("upload_timestamp") or 1.0),
            metadata=dict(record.get("metadata") or {}),
        )

    service.document_response = MagicMock(side_effect=_response)  # type: ignore[method-assign]

    files = [
        UploadFile(filename="first.txt", file=io.BytesIO(b"same")),
        UploadFile(filename="second.txt", file=io.BytesIO(b"same")),
    ]

    with (
        patch(
            "src.features.security.compliance.compliance_validator.check_document_compliance",
            return_value={
                "status": "human_review",
                "severity": "medium",
                "requires_human_review": True,
                "dlp_decision": "HUMAN_REVIEW",
            },
        ),
        patch(
            "src.features.security.compliance.compliance_validator.security_scan_requires_review",
            return_value=True,
        ),
        patch(
            "src.features.observability.audit.application.audit_service.get_audit_service",
            return_value=MagicMock(),
        ),
        patch(
            "src.features.observability.metrics.application.metrics_service.get_metrics_service",
            return_value=MagicMock(),
        ),
    ):
        result = await service.upload_documents(
            files,
            repository_id=REPO_A,
            submit_for_processing=False,
        )

    assert result.uploaded_count == 1
    assert result.rejected_count == 1
    assert result.documents[0].original_file_name == "first.txt"
    assert result.rejected[0].filename == "second.txt"
    assert result.rejected[0].code == "duplicate_document"


@pytest.mark.anyio
async def test_batch_upload_continues_after_compliance_failure() -> None:
    """Multi-file upload keeps good files when another file is compliance-blocked."""
    import io
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from fastapi import UploadFile

    from src.features.documents.schemas.document_schemas import DocumentResponse
    from src.shared.errors.service_errors import DmsServiceError

    service = DocumentReceiverService()
    live = SimpleNamespace(
        max_files_per_request=20,
        upload_dir=tempfile.mkdtemp(),
        allowed_extensions=[".txt", ".pdf", ".docx"],
        repository_type="local",
        default_collection_name="Document",
        default_tenant_id=None,
    )
    job_store = MagicMock()
    job_store.ensure_received_from_record.return_value = "job-1"
    store = MagicMock()
    store.insert_document_if_not_duplicate.return_value = None
    store.get_document.return_value = None

    service._live_settings = MagicMock(return_value=live)  # type: ignore[method-assign]
    service._require_active_repository = MagicMock()  # type: ignore[method-assign]
    service._validate_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._resolve_document_type = MagicMock(return_value=("type-1", "Type 1"))  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service._save_upload_file = AsyncMock(  # type: ignore[method-assign]
        side_effect=[("hash-ok", 10), ("hash-blocked", 10)]
    )
    service._apply_repository_context = MagicMock()  # type: ignore[method-assign]
    service._purge_replaceable_duplicates = MagicMock()  # type: ignore[method-assign]
    service._delete_document_instance = MagicMock()  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service._find_duplicate_document = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._register_human_review_queue_entry = MagicMock()  # type: ignore[method-assign]

    def _response(record: dict[str, Any]) -> DocumentResponse:
        return DocumentResponse(
            document_id=str(record["document_id"]),
            document_name=str(record["document_name"]),
            original_file_name=str(record["original_file_name"]),
            document_type=str(record["document_type"]),
            repository_type=str(record["repository_type"]),
            repository_path=str(record["repository_path"]),
            collection_name=str(record.get("collection_name") or "Document"),
            status=str(record.get("status") or "uploaded"),
            upload_timestamp=float(record.get("upload_timestamp") or 1.0),
            metadata=dict(record.get("metadata") or {}),
        )

    service.document_response = MagicMock(side_effect=_response)  # type: ignore[method-assign]

    files = [
        UploadFile(filename="good.txt", file=io.BytesIO(b"ok")),
        UploadFile(filename="blocked.txt", file=io.BytesIO(b"ssn")),
    ]

    call_count = {"n": 0}

    def _compliance(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise DmsServiceError(
                "Compliance violation: SSN",
                component="document_receiver_service",
                code="compliance_violation",
                http_status=400,
                user_message="Upload Blocked: SSN detected",
            )
        return {"status": "allow"}

    with (
        patch(
            "src.features.security.compliance.compliance_validator.check_document_compliance",
            side_effect=_compliance,
        ),
        patch(
            "src.features.security.compliance.compliance_validator.security_scan_requires_review",
            return_value=False,
        ),
        patch(
            "src.features.observability.audit.application.audit_service.get_audit_service",
            return_value=MagicMock(),
        ),
        patch(
            "src.features.observability.metrics.application.metrics_service.get_metrics_service",
            return_value=MagicMock(),
        ),
    ):
        result = await service.upload_documents(
            files,
            repository_id=REPO_A,
            submit_for_processing=False,
        )

    assert result.uploaded_count == 1
    assert result.rejected_count == 1
    assert result.documents[0].original_file_name == "good.txt"
    assert result.rejected[0].filename == "blocked.txt"
    assert result.rejected[0].code == "compliance_violation"


def test_normalized_text_hash_matches_across_whitespace() -> None:
    from src.features.documents.application.content_fingerprint import hash_normalized_text

    body = (
        "This annexure records the official change control number for the quality "
        "document and states that the change control number is 197902 in revision history."
    )
    a = hash_normalized_text(body)
    b = hash_normalized_text(body.upper().replace("  ", "\n\n"))
    assert a is not None
    assert a == b


def test_find_duplicate_by_normalized_text_hash() -> None:
    text_hash = "text" * 16
    store = _store_with_docs(
        [
            _doc(
                document_id="pdf-1",
                content_hash="pdf-bytes-hash-aaaa",
                content_text_hash=text_hash,
                document_name="same.pdf",
            )
        ]
    )
    service = DocumentReceiverService()
    with patch.object(service, "get_store", return_value=store):
        duplicate = service._find_duplicate_document(
            content_hash="docx-bytes-hash-bbbb",
            content_text_hash=text_hash,
            repository_id=REPO_A,
            collection_name="TestCollection",
            tenant_id=None,
        )
    assert duplicate is not None
    assert duplicate["document_id"] == "pdf-1"
