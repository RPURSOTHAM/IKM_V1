"""Regression: repository template_extraction=true schedules both processor jobs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.features.documents.application.document_service import DocumentReceiverService
from src.features.document_processing.shared_processor.types import ProcessorType, enabled_processor_types


def _service(*, publisher: MagicMock | None = None) -> DocumentReceiverService:
    service = DocumentReceiverService.__new__(DocumentReceiverService)
    pub = publisher or MagicMock()
    pub.queue_name = "document_jobs"
    service._publisher = pub
    service._ensure_publisher = MagicMock(return_value=pub)  # type: ignore[method-assign]
    return service


def test_enabled_processor_types_includes_template_extraction() -> None:
    enabled = enabled_processor_types({"template_extraction": True})
    assert ProcessorType.CHUNKING_VECTORIZING.value in enabled
    assert ProcessorType.TEMPLATE_EXTRACTION.value in enabled


def test_enabled_processor_types_accepts_string_true() -> None:
    enabled = enabled_processor_types({"template_extraction": "true"})
    assert ProcessorType.TEMPLATE_EXTRACTION.value in enabled


def test_compute_enabled_includes_template_when_enabled() -> None:
    enabled = DocumentReceiverService._compute_enabled_processor_types({"template_extraction": True})
    assert enabled == ["chunking_vectorizing", "template_extraction"]


def test_queue_payload_includes_template_extraction_flag() -> None:
    record = {
        "document_id": "doc-te",
        "document_name": "doc-te.docx",
        "original_file_name": "sample.docx",
        "collection_name": "RepoCollection",
        "tenant_id": None,
        "repository_id": "repo-1",
        "metadata": {"repository_id": "repo-1"},
        "processing": {
            "template_extraction": True,
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
        },
    }
    payload = DocumentReceiverService.queue_payload(record)
    assert payload["template_extraction"] is True
    assert "template_extraction" in payload["enabled_processor_types"]


def test_submit_to_queue_publishes_chunking_and_template() -> None:
    publisher = MagicMock()
    publisher.queue_name = "document_jobs"
    service = _service(publisher=publisher)
    store = MagicMock()
    job_store = MagicMock()
    job_store.get_by_document_id.return_value = {
        "job_id": "job-1",
        "status": "RECEIVED",
        "scheduling_metadata": {"enabled_processor_types": ["chunking_vectorizing", "template_extraction"]},
    }
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service._processing_allowed_after_security_review = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._resolve_record_document_type = MagicMock(return_value=(None, None))  # type: ignore[method-assign]
    service._apply_resolved_document_type = MagicMock()  # type: ignore[method-assign]
    service._attach_metadata_fields = MagicMock(side_effect=lambda p: dict(p))  # type: ignore[method-assign]
    service._attach_key_fields = MagicMock(side_effect=lambda p: dict(p))  # type: ignore[method-assign]
    service._persist_document_type_metadata = MagicMock()  # type: ignore[method-assign]
    service._preserve_repository_id = MagicMock(side_effect=lambda record, rid: rid or record.get("repository_id"))  # type: ignore[method-assign]

    record = {
        "document_id": "doc-te",
        "document_name": "doc-te.docx",
        "original_file_name": "sample.docx",
        "collection_name": "RepoCollection",
        "tenant_id": None,
        "repository_id": "repo-1",
        "metadata": {"repository_id": "repo-1"},
        "processing": {"template_extraction": True},
        "status": "uploaded",
    }
    resolved = {
        "settings": {"template_extraction": True, "key_field_extraction_enabled": False},
        "processing_hints": {
            "template_extraction": True,
            "key_field_extraction": False,
            "key_field_extraction_enabled": False,
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
        },
    }
    repo_service = MagicMock()
    repo_service.resolve_settings.return_value = resolved

    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ):
        service.submit_to_queue(record)

    published_types = [call.args[0].get("processor_type") for call in publisher.publish_document_job.call_args_list]
    assert published_types == ["chunking_vectorizing", "template_extraction"]
    job_store.seed_processor_plan.assert_called_once()
    seed_args = job_store.seed_processor_plan.call_args
    assert seed_args.args[1] == ["chunking_vectorizing", "template_extraction"]
    assert seed_args.kwargs["context"]["template_extraction"] is True


def test_submit_forces_template_when_missing_from_enabled_list() -> None:
    publisher = MagicMock()
    publisher.queue_name = "document_jobs"
    service = _service(publisher=publisher)
    store = MagicMock()
    job_store = MagicMock()
    job_store.get_by_document_id.return_value = {"job_id": "job-1", "status": "RECEIVED", "scheduling_metadata": {}}
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service._processing_allowed_after_security_review = MagicMock(return_value=True)  # type: ignore[method-assign]
    service._resolve_record_document_type = MagicMock(return_value=(None, None))  # type: ignore[method-assign]
    service._apply_resolved_document_type = MagicMock()  # type: ignore[method-assign]
    service._attach_metadata_fields = MagicMock(side_effect=lambda p: dict(p))  # type: ignore[method-assign]
    service._attach_key_fields = MagicMock(side_effect=lambda p: dict(p))  # type: ignore[method-assign]
    service._persist_document_type_metadata = MagicMock()  # type: ignore[method-assign]
    service._preserve_repository_id = MagicMock(side_effect=lambda record, rid: rid or record.get("repository_id"))  # type: ignore[method-assign]
    # Simulate a bad compute that drops TE even though the flag is set.
    service._compute_enabled_processor_types = MagicMock(return_value=["chunking_vectorizing"])  # type: ignore[method-assign]

    record = {
        "document_id": "doc-te-force",
        "document_name": "doc-te.docx",
        "collection_name": "RepoCollection",
        "repository_id": "repo-1",
        "metadata": {"repository_id": "repo-1"},
        "processing": {},
        "status": "uploaded",
    }
    resolved = {
        "settings": {"template_extraction": True},
        "processing_hints": {
            "template_extraction": True,
            "enabled_processor_types": ["chunking_vectorizing"],
        },
    }
    repo_service = MagicMock()
    repo_service.resolve_settings.return_value = resolved

    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ):
        service.submit_to_queue(record)

    published_types = [call.args[0].get("processor_type") for call in publisher.publish_document_job.call_args_list]
    assert published_types == ["chunking_vectorizing", "template_extraction"]
