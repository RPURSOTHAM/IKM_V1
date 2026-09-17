"""Tests for deleting documents whose intake metadata record is missing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.documents.application.document_service import DocumentReceiverService


@pytest.fixture
def receiver() -> DocumentReceiverService:
    service = DocumentReceiverService()
    service.get_store = MagicMock(return_value=MagicMock(get_document=MagicMock(return_value=None)))
    return service


def test_delete_missing_document_cleans_orphan_link(receiver: DocumentReceiverService) -> None:
    with patch.object(
        DocumentReceiverService,
        "_document_linked_to_repository",
        return_value=True,
    ), patch(
        "src.features.documents.application.document_service.purge_repository_link",
        create=True,
    ) as purge_link, patch(
        "src.features.documents.application.document_purge_service.purge_repository_link",
        return_value={"attempted": True, "removed": True},
    ), patch(
        "src.features.documents.application.document_purge_service.purge_document_job",
        return_value={"attempted": True, "deleted": True},
    ):
        result = receiver.delete_document_cascade(
            "doc-orphan-1",
            repository_id="repo-1",
        )

    assert result["deleted"] is True
    assert result["orphan_cleanup"] is True


def test_delete_missing_document_not_linked_raises(receiver: DocumentReceiverService) -> None:
    with patch.object(
        DocumentReceiverService,
        "_document_linked_to_repository",
        return_value=False,
    ):
        with pytest.raises(HTTPException) as exc:
            receiver.delete_document_cascade("doc-orphan-2", repository_id="repo-1")
    assert exc.value.status_code == 404
