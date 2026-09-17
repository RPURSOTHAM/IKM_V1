"""Phase 3.7 — upload repository validation and settings propagation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.features.documents.application.document_service import DocumentReceiverService
from src.features.repositories.domain.repository_exceptions import NotFoundError, RepositoryInactiveError


def _service() -> DocumentReceiverService:
    service = DocumentReceiverService.__new__(DocumentReceiverService)
    return service


def test_require_active_repository_rejects_unknown() -> None:
    service = _service()
    repo_service = MagicMock()
    repo_service.validate_repository_active.side_effect = NotFoundError(
        "Repository not found.",
        details={"repository_id": "missing"},
    )
    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ):
        with pytest.raises(NotFoundError):
            service._require_active_repository("missing")


def test_require_active_repository_rejects_inactive_and_archived() -> None:
    service = _service()
    repo_service = MagicMock()
    repo_service.validate_repository_active.side_effect = RepositoryInactiveError(
        "Repository is not active.",
        details={"repository_id": "repo-1", "status": "archived"},
    )
    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ):
        with pytest.raises(RepositoryInactiveError) as exc:
            service._require_active_repository("repo-1")
    assert exc.value.details.get("status") == "archived"


def test_require_active_repository_allows_none_for_legacy_uploads() -> None:
    service = _service()
    assert service._require_active_repository(None) is None
    assert service._require_active_repository("") is None


def test_apply_repository_context_stores_effective_settings_and_hints() -> None:
    service = _service()
    service._bind_document_type = MagicMock()  # type: ignore[method-assign]
    record: dict = {"document_id": "doc-1", "metadata": {}}

    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {
        "weaviate_collection": "PlantDocs",
        "default_tenant_id": "t1",
        "status": "active",
        "settings": {
            "chunk_size": 640,
            "chunk_overlap": 3,
            "retrieval_search_mode": "hybrid",
            "template_extraction": True,
            "embedding_model": {"provider": "local", "model_id": "bge-base-en"},
        },
        "processing_hints": {
            "chunk_size": 640,
            "chunk_overlap_sentences": 3,
            "model_name": "bge-base-en",
            "template_extraction": True,
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
        },
    }
    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ), patch(
        "src.application.consumer_api.context.get_current_user_from_context",
        return_value=None,
    ):
        service._apply_repository_context(
            record,
            repository_id="repo-1",
            collection_name=None,
            tenant_id=None,
            link_document=False,
            audit_upload=False,
        )

    assert record["repository_id"] == "repo-1"
    assert record["effective_settings"]["chunk_size"] == 640
    assert record["processing"]["enabled_processor_types"] == [
        "chunking_vectorizing",
        "template_extraction",
    ]
    assert "template_extraction" in record["processing"]["enabled_processor_types"]
    assert record["metadata"]["repository_id"] == "repo-1"


def test_queue_payload_includes_repository_settings_and_repository_id() -> None:
    record = {
        "document_id": "doc-1",
        "document_name": "doc-1.pdf",
        "collection_name": "PlantDocs",
        "tenant_id": None,
        "repository_id": "repo-1",
        "processing": {
            "chunk_size": 512,
            "chunk_overlap_sentences": 2,
            "model_name": "bge-base-en",
            "model_dir": "bge-base-en",
            "chunking_strategy": "section-based",
            "enabled_processor_types": ["chunking_vectorizing"],
            "template_extraction": False,
            "key_fields": [{"name": "document_number"}],
        },
        "metadata": {},
    }
    payload = DocumentReceiverService.queue_payload(record)
    assert payload["repository_id"] == "repo-1"
    assert payload["chunk_size"] == 512
    assert payload["repository_settings"]["chunk_size"] == 512
    assert payload["repository_settings"]["embedding_model_name"] == "bge-base-en"
    assert payload["repository_settings"]["key_fields"] == [{"name": "document_number"}]
    assert payload["enabled_processor_types"] == ["chunking_vectorizing"]


def test_seed_processor_plan_includes_processing_hints_snapshot() -> None:
    from src.infrastructure.database.document_jobs import DocumentJobStore

    store = DocumentJobStore.__new__(DocumentJobStore)
    store._scheduling_metadata = MagicMock(return_value={})  # type: ignore[method-assign]
    store._write_scheduling_metadata = MagicMock()  # type: ignore[method-assign]
    store.seed_processor_plan(
        "doc-1",
        ["chunking_vectorizing"],
        context={
            "repository_id": "repo-1",
            "processing_hints": {"chunk_size": 512, "model_name": "bge-base-en"},
            "chunk_size": 512,
        },
    )
    written = store._write_scheduling_metadata.call_args.args[1]
    assert written["repository_id"] == "repo-1"
    assert written["processing_hints"]["chunk_size"] == 512
    assert written["chunk_size"] == 512
