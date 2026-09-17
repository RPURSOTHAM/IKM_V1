"""Redis render-cache purge regression tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.features.documents.application.document_purge_service import purge_all_artifacts


def test_purge_all_artifacts_includes_redis_render_cache() -> None:
    record = {
        "document_id": "doc-123",
        "document_name": "doc.txt",
        "collection_name": "Test",
        "repository_type": "local",
        "repository_path": "/tmp/missing.txt",
    }
    with (
        patch(
            "src.features.documents.application.document_purge_service.purge_weaviate_chunks",
            return_value={"attempted": True},
        ),
        patch(
            "src.features.documents.application.document_purge_service.purge_neo4j_document",
            return_value={"attempted": True},
        ),
        patch(
            "src.features.documents.application.document_purge_service.purge_document_type_mysql",
            return_value={"attempted": False},
        ),
        patch(
            "src.features.documents.application.document_purge_service.purge_repository_link",
            return_value={"attempted": True},
        ),
        patch(
            "src.features.documents.application.document_purge_service.purge_document_job",
            return_value={"attempted": True},
        ),
        patch(
            "src.features.documents.application.document_purge_service.purge_redis_render_cache",
            return_value={"attempted": True, "deleted": 1},
        ) as mock_redis,
    ):
        result = purge_all_artifacts(record, delete_file=False)
    mock_redis.assert_called_once_with("doc-123")
    assert result["redis_render_cache"]["deleted"] == 1
