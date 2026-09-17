"""Document catalog indexed-count and repository-scoped search regression tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.features.retrieval.application.document_catalog import (
    aggregate_indexed_metadata,
    build_catalog_records_index,
    record_to_catalog_item,
)
from src.features.retrieval.schemas.retrieval_schemas import DocumentSearchRequest
from src.features.retrieval.application.retrieval_service import RetrievalService


REPO_A = "11111111-1111-4111-8111-111111111111"
REPO_B = "22222222-2222-4222-8222-222222222222"


def _chunk_obj(**props):
    return SimpleNamespace(
        properties=props,
        vector={"default": [0.1, 0.2]},
    )


def test_aggregate_indexed_metadata_keys_by_document_id() -> None:
    stats = aggregate_indexed_metadata(
        [
            _chunk_obj(
                document_id="doc-uuid-1",
                doc_name="stored-name.txt",
                document_name="original.txt",
            ),
            _chunk_obj(
                document_id="doc-uuid-1",
                doc_name="stored-name.txt",
                document_name="original.txt",
            ),
        ]
    )
    assert stats["doc-uuid-1"]["chunk_count"] == 2
    assert stats["doc-uuid-1"]["embedded_chunk_count"] == 2
    assert stats["stored-name.txt"]["chunk_count"] == 2


def test_record_to_catalog_item_resolves_by_document_id() -> None:
    record = {
        "document_id": "doc-uuid-1",
        "document_name": "stored-name.txt",
        "original_file_name": "original.txt",
        "document_type": "txt",
        "status": "completed",
        "repository_id": REPO_A,
        "collection_name": "RepoA",
        "metadata": {},
    }
    indexed_stats = aggregate_indexed_metadata(
        [_chunk_obj(document_id="doc-uuid-1", doc_name="different-weaviate-key.txt")]
    )
    item = record_to_catalog_item(record, indexed_stats=indexed_stats)
    assert item.indexed.chunk_count == 1
    assert item.indexed.embedded_chunk_count == 1
    assert item.indexed.indexed is True


def test_same_filename_different_repositories_do_not_collide() -> None:
    stats = aggregate_indexed_metadata(
        [
            _chunk_obj(document_id="doc-a", doc_name="shared.txt", repository_id=REPO_A),
            _chunk_obj(document_id="doc-b", doc_name="shared.txt", repository_id=REPO_B),
        ]
    )
    record_a = {
        "document_id": "doc-a",
        "document_name": "shared.txt",
        "original_file_name": "shared.txt",
        "document_type": "txt",
        "status": "completed",
        "repository_id": REPO_A,
        "collection_name": "A",
        "metadata": {},
    }
    record_b = {
        "document_id": "doc-b",
        "document_name": "shared.txt",
        "original_file_name": "shared.txt",
        "document_type": "txt",
        "status": "completed",
        "repository_id": REPO_B,
        "collection_name": "B",
        "metadata": {},
    }
    assert record_to_catalog_item(record_a, indexed_stats=stats).indexed.chunk_count == 1
    assert record_to_catalog_item(record_b, indexed_stats=stats).indexed.chunk_count == 1


@patch.object(RetrievalService, "_resolve_collection_context", return_value=("CollectionA", None, None))
@patch.object(RetrievalService, "_retrieval_enabled", return_value=True)
@patch.object(RetrievalService, "_check_retrieval_authorization")
@patch.object(RetrievalService, "_load_catalog_records")
def test_search_documents_catalog_applies_repository_id_filter(
    mock_load: MagicMock,
    _mock_auth: MagicMock,
    _mock_enabled: MagicMock,
    _mock_resolve: MagicMock,
) -> None:
    repo_a_record = {
        "document_id": "doc-a",
        "document_name": "a.txt",
        "original_file_name": "a.txt",
        "document_type": "txt",
        "status": "completed",
        "repository_id": REPO_A,
        "collection_name": "A",
        "metadata": {},
    }
    mock_load.return_value = ([repo_a_record], 1)
    service = RetrievalService()
    service.ready = True
    service._client = MagicMock()
    service._collection_exists = MagicMock(return_value=False)

    import asyncio

    body = DocumentSearchRequest(
        query="A_MARKER_UNIQUE_TEXT",
        repository_id=REPO_A,
    )
    asyncio.run(service.search_documents_catalog(body))

    intake_filters = mock_load.call_args.kwargs["intake_filters"]
    assert intake_filters["repository_id"] == REPO_A


def test_build_catalog_records_index_supports_multiple_keys() -> None:
    record = {
        "document_id": "doc-1",
        "document_name": "stored.txt",
        "original_file_name": "original.txt",
    }
    index = build_catalog_records_index([record])
    assert index["doc-1"] is record
    assert index["stored.txt"] is record
    assert index["original.txt"] is record
