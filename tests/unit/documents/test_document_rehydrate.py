"""Tests for rehydrating intake metadata from MySQL jobs after JSON store loss."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.features.documents.application.document_service import DocumentReceiverService


@pytest.fixture
def receiver(tmp_path: Path) -> DocumentReceiverService:
    service = DocumentReceiverService()
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    live = MagicMock()
    live.upload_dir = str(upload_dir)
    live.repository_type = "local"
    live.default_collection_name = "Document"
    live.default_tenant_id = "default"
    live.allowed_extensions = [".pdf", ".docx", ".doc", ".txt"]
    service._live_settings = MagicMock(return_value=live)  # type: ignore[method-assign]

    store = MagicMock()
    store.get_document.return_value = None
    store.upsert_document.side_effect = lambda record: dict(record)
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    return service


def test_rehydrate_document_from_job_when_file_exists(receiver: DocumentReceiverService, tmp_path: Path) -> None:
    document_id = "7b05d9c4-1111-2222-3333-444455556666"
    stored_name = f"{document_id}.pdf"
    upload_path = tmp_path / "uploads" / stored_name
    upload_path.write_bytes(b"%PDF-1.4 test")

    job = {
        "job_id": "job-1",
        "document_id": document_id,
        "document_name": stored_name,
        "source_location": str(upload_path),
        "collection_name": "SOP",
        "tenant_id": "tenant-1",
        "status": "COMPLETED",
        "scheduling_metadata": {
            "processor_results": {
                "chunking_vectorizing": {
                    "status": "COMPLETED",
                    "document_metadata": {
                        "original_file_name": "Equipment SOP.pdf",
                        "file_size_bytes": 1234,
                        "chunk_count": 5,
                    },
                }
            }
        },
        "created_at": "2026-06-06T10:00:00",
        "completed_at": "2026-06-06T10:05:00",
    }

    with patch(
        "src.features.repositories.infrastructure.repository_repository.get_repository_store",
        return_value=MagicMock(get_repository_id_for_document=MagicMock(return_value="repo-1")),
    ):
        restored = receiver._try_rehydrate_document_from_job(document_id, job)

    assert restored is not None
    assert restored["original_file_name"] == "Equipment SOP.pdf"
    assert restored["repository_path"] == str(upload_path)
    receiver.get_store().upsert_document.assert_called_once()


def test_repository_entry_from_job_uses_completed_status(receiver: DocumentReceiverService) -> None:
    document_id = "doc-abc"
    job = {
        "document_id": document_id,
        "document_name": f"{document_id}.docx",
        "status": "COMPLETED",
        "scheduling_metadata": {
            "processor_results": {
                "chunking_vectorizing": {
                    "document_metadata": {"original_file_name": "URS Template.docx"},
                }
            }
        },
        "created_at": "2026-06-06T10:00:00",
    }
    entry = receiver._repository_entry_from_job(document_id, job)
    assert entry["status"] == "completed"
    assert entry["original_file_name"] == "URS Template.docx"


def test_rehydrate_document_from_disk_when_upload_exists(receiver: DocumentReceiverService, tmp_path: Path) -> None:
    document_id = "7b05d9c4-d67c-47ab-a89f-1f4c24316972"
    upload_path = tmp_path / "uploads" / f"{document_id}.pdf"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(b"%PDF-1.4 test")

    with patch(
        "src.features.documents.application.document_service.DocumentReceiverService._repository_weaviate_context",
        return_value=("SOP_REPO", "tenant-1"),
    ), patch(
        "src.features.documents.application.document_metadata_service.weaviate_stats_for_document_id",
        return_value={
            "chunk_count": 36,
            "file_size_bytes": upload_path.stat().st_size,
            "doc_name": "Equipment SOP.pdf",
            "indexing_status": "indexed",
        },
    ), patch(
        "src.features.repositories.infrastructure.repository_repository.get_repository_store",
        return_value=MagicMock(get_repository_id_for_document=MagicMock(return_value="repo-1")),
    ):
        restored = receiver._try_rehydrate_document_from_disk(document_id)

    assert restored is not None
    assert restored["original_file_name"] == "Equipment SOP.pdf"
    assert restored["status"] == "completed"
    receiver.get_store().upsert_document.assert_called_once()


def test_repository_entry_from_indexed_orphan_uses_weaviate_stats(receiver: DocumentReceiverService) -> None:
    document_id = "7b05d9c4-d67c-47ab-a89f-1f4c24316972"
    with patch.object(receiver, "_repository_weaviate_context", return_value=("SOP_REPO", "tenant-1")), patch.object(
        receiver, "_resolve_upload_path_without_job", return_value=None
    ), patch(
        "src.features.documents.application.document_metadata_service.weaviate_stats_for_document_id",
        return_value={
            "chunk_count": 36,
            "file_size_bytes": 737433,
            "doc_name": "Equipment SOP.pdf",
            "indexing_status": "indexed",
        },
    ), patch(
        "src.features.repositories.infrastructure.repository_repository.get_repository_store",
        return_value=MagicMock(get_repository_id_for_document=MagicMock(return_value="repo-1")),
    ):
        entry = receiver._repository_entry_from_indexed_orphan(document_id)

    assert entry is not None
    assert entry["status"] == "completed"
    assert entry["original_file_name"] == "Equipment SOP.pdf"
    assert entry["processing"]["job_status"] == "COMPLETED"
