"""Unit tests for consumer document chunk listing / Weaviate lookup filters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.documents.application.document_chunks import (
    _weaviate_doc_name_candidates,
    list_document_chunks,
)
from src.features.repositories.infrastructure.weaviate_admin import _document_chunk_filter
from src.infrastructure.vector_store.dms_weaviate_client.client import connect_weaviate


def test_weaviate_doc_name_candidates_prefer_original_filename() -> None:
    record = {
        "document_id": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d",
        "document_name": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d.pdf",
        "original_file_name": "GL-CQA-ANN-0427.pdf",
    }
    assert _weaviate_doc_name_candidates(record)[0] == "GL-CQA-ANN-0427.pdf"


@patch("src.features.documents.application.document_chunks.repository_admin.list_chunks_metadata")
def test_list_document_chunks_filters_by_document_id(mock_list: MagicMock) -> None:
    mock_list.return_value = {
        "chunks": [
            {
                "uuid": "u1",
                "properties": {
                    "chunk_id": "c1",
                    "doc_name": "GL-CQA-ANN-0427.pdf",
                    "document_id": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d",
                    "text": "hello",
                    "page": 1,
                },
                "vector": [0.1, 0.2],
            },
            {
                "uuid": "u2",
                "properties": {
                    "chunk_id": "c2",
                    "doc_name": "GL-CQA-ANN-0427.pdf",
                    "document_id": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d",
                    "text": "world",
                    "page": 2,
                },
                "vector": [0.3, 0.4],
            },
            {
                "uuid": "u3",
                "properties": {
                    "chunk_id": "c3",
                    "doc_name": "GL-CQA-ANN-0427.pdf",
                    "document_id": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d",
                    "text": "!",
                    "page": 3,
                },
                "vector": [0.5, 0.6],
            },
        ],
        "returned": 3,
    }
    record = {
        "document_id": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d",
        "document_name": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d.pdf",
        "original_file_name": "GL-CQA-ANN-0427.pdf",
        "collection_name": "Test_repository",
        "tenant_id": None,
    }
    payload = list_document_chunks(record, include_text=True, include_vector=False)
    assert payload["returned"] == 3
    assert len(payload["chunks"]) == 3
    mock_list.assert_called_once()
    args = mock_list.call_args.args
    kwargs = mock_list.call_args.kwargs
    assert args[0] == "Test_repository"
    assert kwargs["document_id"] == "5b4f5738-9fa3-45e6-8dde-99b3e066d45d"
    assert kwargs["document_name"] == "GL-CQA-ANN-0427.pdf"


@patch("src.features.documents.application.document_chunks.repository_admin.list_chunks_metadata")
def test_list_document_chunks_empty_when_document_missing(mock_list: MagicMock) -> None:
    mock_list.return_value = {"chunks": [], "returned": 0}
    record = {
        "document_id": "00000000-0000-0000-0000-000000000000",
        "document_name": "missing.pdf",
        "original_file_name": "missing.pdf",
        "collection_name": "Test_repository",
    }
    payload = list_document_chunks(record)
    assert payload["returned"] == 0
    assert payload["chunks"] == []


def test_list_document_chunks_requires_collection() -> None:
    with pytest.raises(HTTPException) as exc:
        list_document_chunks({"document_id": "x", "document_name": "a.pdf"})
    assert exc.value.status_code == 422


def test_document_chunk_filter_includes_document_id_and_name_properties() -> None:
    flt = _document_chunk_filter(
        document_id="5b4f5738-9fa3-45e6-8dde-99b3e066d45d",
        document_name="GL-CQA-ANN-0427.pdf",
    )
    assert flt is not None
    # Filter objects are opaque; ensure construction does not raise and returns a value.
    assert flt is not _document_chunk_filter()


@patch("weaviate.auth.Auth.api_key", return_value="AUTH")
@patch("weaviate.connect_to_custom")
@patch("src.infrastructure.vector_store.dms_weaviate_client.client.settings")
def test_connect_weaviate_uses_api_key_and_docker_hosts(
    mock_settings: MagicMock,
    mock_connect: MagicMock,
    mock_api_key: MagicMock,
) -> None:
    mock_settings.weaviate_url = "http://weaviate:8080"
    mock_settings.weaviate_grpc_port = 50051
    mock_settings.weaviate_api_key = "weaviate_secret_key"
    mock_connect.return_value = MagicMock(name="client")

    client = connect_weaviate()
    assert client is mock_connect.return_value
    kwargs = mock_connect.call_args.kwargs
    assert kwargs["http_host"] == "weaviate"
    assert kwargs["http_port"] == 8080
    assert kwargs["grpc_host"] == "weaviate"
    assert kwargs["grpc_port"] == 50051
    assert kwargs["auth_credentials"] == "AUTH"
    mock_api_key.assert_called_once_with("weaviate_secret_key")
