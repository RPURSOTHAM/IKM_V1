"""Unit tests for repository-scoped document chunk listing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.repositories.domain.repository_exceptions import NotFoundError
from src.features.retrieval.application.document_chunks import (
    format_document_chunk,
    sort_document_chunks,
)
from src.features.retrieval.application.retrieval_service import RetrievalService


def _chunk_entry(
    *,
    chunk_id: str,
    page: int,
    section_path: str,
    line_start: int,
    text: str,
    embedding_version: str = "bge-base-en@1",
) -> dict:
    return {
        "uuid": chunk_id,
        "properties": {
            "chunk_id": chunk_id,
            "document_id": "doc-1",
            "page": page,
            "section_path": section_path,
            "section_name": section_path.split(" ", 1)[-1],
            "line_start": line_start,
            "line_end": line_start + 3,
            "text": text,
            "embedding_version": embedding_version,
            "category": "procedure",
        },
    }


def test_sort_document_chunks_orders_by_page_section_and_line() -> None:
    raw = [
        _chunk_entry(chunk_id="c3", page=2, section_path="2 Batch Release", line_start=5, text="third"),
        _chunk_entry(chunk_id="c1", page=1, section_path="1 Introduction", line_start=1, text="first"),
        _chunk_entry(chunk_id="c2", page=1, section_path="1 Introduction", line_start=8, text="second"),
    ]
    ordered = sort_document_chunks(raw)
    assert [item["properties"]["chunk_id"] for item in ordered] == ["c1", "c2", "c3"]


def test_format_document_chunk_includes_metadata_and_embedding_flag() -> None:
    entry = _chunk_entry(
        chunk_id="c1",
        page=3,
        section_path="1 Scope",
        line_start=4,
        text="hello world",
    )
    formatted = format_document_chunk(entry, document_id="doc-1", chunk_index=7)
    assert formatted["chunk_id"] == "c1"
    assert formatted["document_id"] == "doc-1"
    assert formatted["page"] == 3
    assert formatted["section"] == "Scope"
    assert formatted["chunk_index"] == 7
    assert formatted["text"] == "hello world"
    assert formatted["token_count"] == 2
    assert formatted["embedding_exists"] is True
    assert formatted["metadata"]["line_start"] == 4
    assert formatted["metadata"]["category"] == "procedure"


@patch("src.features.retrieval.application.document_chunks.fetch_repository_document_chunks")
@patch("src.features.documents.application.document_service.DocumentReceiverService")
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_get_document_chunks_existing_document(
    mock_get_repo_service: MagicMock,
    mock_receiver_cls: MagicMock,
    mock_fetch: MagicMock,
) -> None:
    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {
        "weaviate_collection": "repo_collection",
        "default_tenant_id": "tenant-a",
    }
    repo_service.collect_document_ids_for_repository.return_value = ["doc-1"]
    mock_get_repo_service.return_value = repo_service

    receiver = MagicMock()
    receiver.get_document_record.return_value = {
        "document_id": "doc-1",
        "document_name": "doc-1.pdf",
        "collection_name": "repo_collection",
        "tenant_id": "tenant-a",
    }
    mock_receiver_cls.return_value = receiver

    mock_fetch.return_value = [
        {
            "chunk_id": "c1",
            "document_id": "doc-1",
            "page": 1,
            "section": "Intro",
            "chunk_index": 0,
            "text": "alpha",
            "token_count": 1,
            "embedding_exists": True,
            "metadata": {},
        }
    ]

    service = RetrievalService()
    payload = service.get_document_chunks("repo-1", "doc-1")

    assert payload["repository_id"] == "repo-1"
    assert payload["document_id"] == "doc-1"
    assert payload["chunk_count"] == 1
    assert payload["chunks"][0]["chunk_id"] == "c1"
    mock_fetch.assert_called_once_with(
        collection_name="repo_collection",
        tenant="tenant-a",
        document_id="doc-1",
        document_name="doc-1.pdf",
    )


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_get_document_chunks_missing_repository(mock_get_repo_service: MagicMock) -> None:
    from src.shared.errors import DmsServiceError

    repo_service = MagicMock()
    repo_service.validate_repository_active.side_effect = NotFoundError(
        "Repository not found.",
        details={"repository_id": "missing-repo"},
    )
    mock_get_repo_service.return_value = repo_service

    service = RetrievalService()
    with pytest.raises(DmsServiceError) as exc:
        service.get_document_chunks("missing-repo", "doc-1")
    assert exc.value.code == "repository_not_found"


@patch("src.features.documents.application.document_service.DocumentReceiverService")
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_get_document_chunks_missing_document(
    mock_get_repo_service: MagicMock,
    mock_receiver_cls: MagicMock,
) -> None:
    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {"weaviate_collection": "repo_collection"}
    repo_service.collect_document_ids_for_repository.return_value = ["doc-1"]
    mock_get_repo_service.return_value = repo_service

    receiver = MagicMock()
    receiver.get_document_record.side_effect = HTTPException(status_code=404, detail="Document not found")
    mock_receiver_cls.return_value = receiver

    service = RetrievalService()
    with pytest.raises(NotFoundError, match="Document not found"):
        service.get_document_chunks("repo-1", "doc-1")


@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_get_document_chunks_document_not_in_repository(
    mock_get_repo_service: MagicMock,
) -> None:
    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {"weaviate_collection": "repo_collection"}
    repo_service.collect_document_ids_for_repository.return_value = ["other-doc"]
    mock_get_repo_service.return_value = repo_service

    service = RetrievalService()
    with pytest.raises(NotFoundError, match="not linked"):
        service.get_document_chunks("repo-1", "doc-1")


@patch("src.features.retrieval.application.document_chunks.fetch_repository_document_chunks")
@patch("src.features.documents.application.document_service.DocumentReceiverService")
@patch("src.features.repositories.application.repository_service.get_repository_service")
def test_get_document_chunks_zero_chunks(
    mock_get_repo_service: MagicMock,
    mock_receiver_cls: MagicMock,
    mock_fetch: MagicMock,
) -> None:
    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {
        "weaviate_collection": "repo_collection",
        "default_tenant_id": None,
    }
    repo_service.collect_document_ids_for_repository.return_value = ["doc-1"]
    mock_get_repo_service.return_value = repo_service

    receiver = MagicMock()
    receiver.get_document_record.return_value = {
        "document_id": "doc-1",
        "document_name": "doc-1.pdf",
        "collection_name": "repo_collection",
    }
    mock_receiver_cls.return_value = receiver
    mock_fetch.return_value = []

    service = RetrievalService()
    payload = service.get_document_chunks("repo-1", "doc-1")

    assert payload["chunk_count"] == 0
    assert payload["chunks"] == []


@patch("src.features.retrieval.application.document_chunks.weaviate_admin.fetch_all_document_chunks")
def test_fetch_all_document_chunks_large_document(mock_fetch_all: MagicMock) -> None:
    from src.features.retrieval.application.document_chunks import fetch_repository_document_chunks

    mock_fetch_all.return_value = [
        {
            "uuid": f"c{i}",
            "properties": {
                "chunk_id": f"c{i}",
                "document_id": "doc-1",
                "page": i // 100 + 1,
                "section_path": f"Section {i // 100}",
                "section_name": f"Section {i // 100}",
                "line_start": i,
                "text": f"chunk {i}",
                "embedding_version": "bge-base-en@1",
            },
        }
        for i in range(1200)
    ]

    chunks = fetch_repository_document_chunks(
        collection_name="repo_collection",
        tenant=None,
        document_id="doc-1",
        document_name="doc-1.pdf",
    )

    assert len(chunks) == 1200
    assert chunks[0]["chunk_index"] == 0
    assert chunks[-1]["chunk_index"] == 1199
    assert chunks[0]["page"] <= chunks[-1]["page"]
    mock_fetch_all.assert_called_once()
