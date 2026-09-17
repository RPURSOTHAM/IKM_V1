"""Repository-wide document purge covers intake-only and linked documents."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.documents.application.document_service import DocumentReceiverService
from src.features.repositories.application.repository_service import RepositoryService


@pytest.fixture
def repo_service() -> RepositoryService:
    return RepositoryService()


def test_collect_document_ids_merges_links_and_intake(repo_service: RepositoryService) -> None:
    repository_id = "2e3678dd-dd2e-476d-a615-ba0890d0a71f"
    linked_id = "deb0595c-5531-4fd6-bf3c-0ed44ffbe735"
    intake_only_id = "82928aa9-126e-4e51-92af-81b227ced13c"

    mock_store = MagicMock()
    mock_store.get_by_id.return_value = MagicMock(repository_id=repository_id)
    mock_store.list_document_ids.return_value = [linked_id]
    mock_store.get_repository_id_for_document.return_value = None

    mock_meta_store = MagicMock()
    mock_meta_store.list_documents.return_value = [
        {"document_id": linked_id, "repository_id": repository_id},
        {"document_id": intake_only_id, "repository_id": repository_id},
    ]

    mock_receiver = MagicMock()
    mock_receiver.get_store.return_value = mock_meta_store

    with patch.object(repo_service, "_require_store", return_value=mock_store), patch(
        "src.features.documents.application.document_service.DocumentReceiverService",
        return_value=mock_receiver,
    ), patch(
        "src.infrastructure.database.document_jobs.get_document_job_store",
        return_value=None,
    ):
        ids = repo_service.collect_document_ids_for_repository(repository_id)

    assert set(ids) == {linked_id, intake_only_id}


def test_purge_document_for_repository_deletes_intake_without_link() -> None:
    receiver = DocumentReceiverService()
    repository_id = "9ec82c0b-9576-40a7-b694-048b1e26597d"
    document_id = "82928aa9-126e-4e51-92af-81b227ced13c"
    record = {
        "document_id": document_id,
        "repository_id": repository_id,
        "metadata": {"repository_id": repository_id},
        "document_name": "Cleaning Procedure.pdf",
        "repository_type": "local",
    }

    with patch.object(receiver, "get_store") as get_store, patch.object(
        receiver,
        "delete_document_cascade",
        return_value={"document_id": document_id, "deleted": True},
    ) as cascade, patch.object(
        receiver,
        "_document_linked_to_repository",
        return_value=False,
    ):
        get_store.return_value.get_document.return_value = record
        result = receiver.purge_document_for_repository(document_id, repository_id)

    assert result["deleted"] is True
    cascade.assert_called_once_with(document_id, delete_file=True)


def test_purge_document_for_repository_rejects_other_repository() -> None:
    receiver = DocumentReceiverService()
    record = {
        "document_id": "doc-1",
        "repository_id": "other-repo",
        "metadata": {},
    }

    with patch.object(receiver, "get_store") as get_store:
        get_store.return_value.get_document.return_value = record
        with pytest.raises(HTTPException) as exc:
            receiver.purge_document_for_repository("doc-1", "target-repo")

    assert exc.value.status_code == 404


def test_purge_all_documents_reports_remaining(repo_service: RepositoryService) -> None:
    repository_id = "2e3678dd-dd2e-476d-a615-ba0890d0a71f"
    mock_store = MagicMock()
    mock_store.get_by_id.return_value = MagicMock(repository_id=repository_id)

    collect_calls = {"count": 0}

    def _collect(_repository_id: str, *, limit: int = 5000) -> list[str]:
        collect_calls["count"] += 1
        if collect_calls["count"] == 1:
            return ["doc-a"]
        return []

    mock_receiver = MagicMock()
    mock_receiver.purge_document_for_repository.return_value = {"deleted": True}

    with patch.object(repo_service, "_require_store", return_value=mock_store), patch.object(
        repo_service,
        "collect_document_ids_for_repository",
        side_effect=_collect,
    ), patch(
        "src.features.documents.application.document_service.DocumentReceiverService",
        return_value=mock_receiver,
    ):
        result = repo_service.purge_all_documents(repository_id)

    assert result["deleted"] == 1
    assert result["failed"] == 0
    assert result["remaining"] == 0
    mock_receiver.purge_document_for_repository.assert_called_once_with(
        "doc-a",
        repository_id,
        delete_file=True,
    )
