"""Regression tests: SOP/custom uploads schedule key_field_extraction from repository settings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.features.documents.application.document_service import DocumentReceiverService
from src.features.document_processing.shared_processor.types import ProcessorType


def _service(*, publisher: MagicMock | None = None) -> DocumentReceiverService:
    service = DocumentReceiverService.__new__(DocumentReceiverService)
    pub = publisher or MagicMock()
    pub.queue_name = "document_jobs"
    service._publisher = pub
    service._ensure_publisher = MagicMock(return_value=pub)  # type: ignore[method-assign]
    return service


def _base_record(
    *,
    repository_id: str | None = "repo-1",
    document_type_id: str = "type-basic",
    document_type_name: str = "Basic",
    processing: dict | None = None,
) -> dict:
    record = {
        "document_id": "doc-1",
        "document_name": "doc-1.txt",
        "original_file_name": "sample.txt",
        "collection_name": "RepoCollection",
        "tenant_id": None,
        "repository_id": repository_id,
        "document_type_id": document_type_id,
        "document_type_name": document_type_name,
        "metadata": {"repository_id": repository_id} if repository_id else {},
        "processing": processing
        or {
            "document_type_id": document_type_id,
            "document_type_name": document_type_name,
            "key_field_extraction": True,
            "key_field_extraction_enabled": True,
            "enabled_processor_types": ["chunking_vectorizing", "key_field_extraction"],
            "key_fields": [],
        },
    }
    return record


def test_compute_enabled_includes_key_field_when_enabled() -> None:
    enabled = DocumentReceiverService._compute_enabled_processor_types(
        {"key_field_extraction_enabled": True}
    )
    assert ProcessorType.KEY_FIELD_EXTRACTION.value in enabled
    assert ProcessorType.CHUNKING_VECTORIZING.value in enabled


def test_compute_enabled_excludes_key_field_when_disabled() -> None:
    enabled = DocumentReceiverService._compute_enabled_processor_types(
        {"key_field_extraction_enabled": False, "key_field_extraction": False}
    )
    assert ProcessorType.KEY_FIELD_EXTRACTION.value not in enabled


def test_recover_repository_id_from_document_type() -> None:
    service = _service()
    store = MagicMock()
    store.get_type_by_id.return_value = SimpleNamespace(repository_id="repo-from-type")
    service._document_type_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    assert service._repository_id_from_document_type("type-sop") == "repo-from-type"


def test_preserve_repository_id_never_overwrites_with_null() -> None:
    service = _service()
    record = _base_record(repository_id="repo-1")
    with patch.object(service, "_recover_repository_id", return_value=None):
        preserved = service._preserve_repository_id(record, None)
    assert preserved == "repo-1"
    assert record["repository_id"] == "repo-1"
    assert record["metadata"]["repository_id"] == "repo-1"


def test_queue_payload_includes_repository_id_from_metadata() -> None:
    record = _base_record(repository_id=None)
    record["metadata"]["repository_id"] = "repo-meta"
    record["processing"]["repository_id"] = "repo-meta"
    payload = DocumentReceiverService.queue_payload(record)
    assert payload["repository_id"] == "repo-meta"
    assert "key_field_extraction" in payload["enabled_processor_types"]


@pytest.mark.parametrize(
    ("document_type_id", "document_type_name"),
    [
        ("type-basic", "Basic"),
        ("type-sop", "SOP"),
        ("type-custom", "Custom"),
    ],
)
def test_submit_to_queue_schedules_key_field_for_document_types(
    document_type_id: str,
    document_type_name: str,
) -> None:
    publisher = MagicMock()
    service = _service(publisher=publisher)
    record = _base_record(
        document_type_id=document_type_id,
        document_type_name=document_type_name,
        processing={
            "document_type_id": document_type_id,
            "document_type_name": document_type_name,
            # Simulate stale/incomplete prior processing without enabled list.
            "key_fields": [],
        },
    )
    job_store = MagicMock()
    job_store.get_by_document_id.return_value = {
        "job_id": "job-1",
        "status": "RECEIVED",
        "scheduling_metadata": {},
    }
    store = MagicMock()
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    service._resolve_record_document_type = MagicMock(  # type: ignore[method-assign]
        return_value=(document_type_id, document_type_name)
    )
    service._upsert_document_instance = MagicMock()  # type: ignore[method-assign]
    service._attach_metadata_fields = DocumentReceiverService._attach_metadata_fields  # type: ignore[method-assign]
    service._attach_key_fields = staticmethod(lambda processing: dict(processing))  # type: ignore[method-assign]
    service._persist_document_type_metadata = MagicMock()  # type: ignore[method-assign]

    repo_service = MagicMock()
    repo_service.resolve_settings.return_value = {
        "settings": {
            "key_field_extraction_enabled": True,
            "key_field_extraction": True,
            "document_type_id": "type-basic",
        },
        "processing_hints": {
            "key_field_extraction": True,
            "key_field_extraction_enabled": True,
            "enabled_processor_types": ["chunking_vectorizing", "key_field_extraction"],
            "document_type_id": "type-basic",
            "document_type_name": "Basic",
            "key_fields": [],
        },
    }

    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ), patch(
        "src.features.document_processing.shared_processor.types.processor_dependencies_met",
        return_value=True,
    ):
        service.submit_to_queue(record)

    assert record["repository_id"] == "repo-1"
    assert record["processing"]["document_type_id"] == document_type_id
    assert record["processing"]["document_type_name"] == document_type_name
    assert "key_field_extraction" in record["processing"]["enabled_processor_types"]
    seeded_enabled = job_store.seed_processor_plan.call_args.args[1]
    assert "key_field_extraction" in seeded_enabled
    context = job_store.seed_processor_plan.call_args.kwargs["context"]
    assert context["repository_id"] == "repo-1"
    assert context["document_type_id"] == document_type_id
    assert context["key_field_extraction_enabled"] is True
    published_types = {call.args[0]["processor_type"] for call in publisher.publish_document_job.call_args_list}
    assert "chunking_vectorizing" in published_types
    assert "key_field_extraction" in published_types


def test_submit_to_queue_skips_key_field_when_repository_disabled() -> None:
    service = _service()
    record = _base_record(
        document_type_id="type-sop",
        document_type_name="SOP",
        processing={"document_type_id": "type-sop", "document_type_name": "SOP"},
    )
    job_store = MagicMock()
    job_store.get_by_document_id.return_value = {"job_id": "job-1", "status": "RECEIVED", "scheduling_metadata": {}}
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    service._resolve_record_document_type = MagicMock(return_value=("type-sop", "SOP"))  # type: ignore[method-assign]
    service._upsert_document_instance = MagicMock()  # type: ignore[method-assign]
    service._attach_key_fields = staticmethod(lambda processing: dict(processing))  # type: ignore[method-assign]
    service._persist_document_type_metadata = MagicMock()  # type: ignore[method-assign]

    repo_service = MagicMock()
    repo_service.resolve_settings.return_value = {
        "settings": {
            "key_field_extraction_enabled": False,
            "key_field_extraction": False,
            "metadata_extraction": False,
        },
        "processing_hints": {
            "key_field_extraction": False,
            "key_field_extraction_enabled": False,
            "metadata_extraction": False,
            "enabled_processor_types": ["chunking_vectorizing"],
            "document_type_id": "type-basic",
        },
    }
    repo_store = MagicMock()
    repo_store.get_settings.return_value = {
        "key_field_extraction_enabled": False,
        "key_field_extraction": False,
        "metadata_extraction": False,
    }
    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ), patch(
        "src.features.repositories.infrastructure.repository_repository.get_repository_store",
        return_value=repo_store,
    ), patch(
        "src.features.document_processing.shared_processor.types.processor_dependencies_met",
        return_value=True,
    ):
        service.submit_to_queue(record)

    assert record["processing"]["enabled_processor_types"] == ["chunking_vectorizing"]
    assert "key_field_extraction" not in record["processing"]["enabled_processor_types"]
    published_types = {
        call.args[0]["processor_type"] for call in service.publisher.publish_document_job.call_args_list
    }
    assert published_types == {"chunking_vectorizing"}


def test_submit_to_queue_recovers_repository_id_from_document_type_when_missing() -> None:
    service = _service()
    record = _base_record(repository_id=None)
    record["metadata"] = {"document_type_id": "type-sop"}
    record["processing"] = {"document_type_id": "type-sop", "document_type_name": "SOP"}
    job_store = MagicMock()
    job_store.get_by_document_id.return_value = {"job_id": "job-1", "status": "RECEIVED", "scheduling_metadata": {}}
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    service._resolve_record_document_type = MagicMock(return_value=("type-sop", "SOP"))  # type: ignore[method-assign]
    service._repository_id_from_document_type = MagicMock(return_value="repo-recovered")  # type: ignore[method-assign]
    service._document_type_store = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._upsert_document_instance = MagicMock()  # type: ignore[method-assign]
    service._attach_key_fields = staticmethod(lambda processing: dict(processing))  # type: ignore[method-assign]
    service._persist_document_type_metadata = MagicMock()  # type: ignore[method-assign]

    repo_service = MagicMock()
    # Force fallback to document-type repository recovery (no linked repository_document).
    repo_service.get_repository_id_for_document.return_value = None
    repo_service.resolve_settings.return_value = {
        "settings": {"key_field_extraction_enabled": True},
        "processing_hints": {
            "key_field_extraction": True,
            "key_field_extraction_enabled": True,
            "enabled_processor_types": ["chunking_vectorizing", "key_field_extraction"],
        },
    }
    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ), patch(
        "src.features.document_processing.shared_processor.types.processor_dependencies_met",
        return_value=True,
    ), patch(
        "src.features.repositories.infrastructure.repository_repository.get_repository_store",
        return_value=None,
    ):
        service.submit_to_queue(record)

    assert record["repository_id"] == "repo-recovered"
    assert "key_field_extraction" in record["processing"]["enabled_processor_types"]
    repo_service.resolve_settings.assert_called_once_with("repo-recovered")


def test_submit_to_queue_handles_settings_lookup_failure_without_dropping_type() -> None:
    service = _service()
    record = _base_record(
        document_type_id="type-sop",
        document_type_name="SOP",
        processing={
            "document_type_id": "type-sop",
            "document_type_name": "SOP",
            "key_field_extraction_enabled": True,
            "key_field_extraction": True,
            "enabled_processor_types": ["chunking_vectorizing", "key_field_extraction"],
        },
    )
    job_store = MagicMock()
    job_store.get_by_document_id.return_value = {"job_id": "job-1", "status": "RECEIVED", "scheduling_metadata": {}}
    service._job_store = MagicMock(return_value=job_store)  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    service._resolve_record_document_type = MagicMock(return_value=("type-sop", "SOP"))  # type: ignore[method-assign]
    service._upsert_document_instance = MagicMock()  # type: ignore[method-assign]
    service._attach_key_fields = staticmethod(lambda processing: dict(processing))  # type: ignore[method-assign]
    service._persist_document_type_metadata = MagicMock()  # type: ignore[method-assign]

    repo_service = MagicMock()
    repo_service.resolve_settings.side_effect = RuntimeError("settings unavailable")
    with patch(
        "src.features.repositories.application.repository_service.get_repository_service",
        return_value=repo_service,
    ), patch(
        "src.features.document_processing.shared_processor.types.processor_dependencies_met",
        return_value=True,
    ):
        service.submit_to_queue(record)

    assert record["repository_id"] == "repo-1"
    assert record["processing"]["document_type_id"] == "type-sop"
    assert "key_field_extraction" in record["processing"]["enabled_processor_types"]


def test_apply_repository_context_recovers_repo_from_type_when_form_omits_repository() -> None:
    service = _service()
    record = {"document_id": "doc-1", "metadata": {}}
    service._repository_id_from_document_type = MagicMock(return_value="repo-from-type")  # type: ignore[method-assign]
    service._bind_document_type = MagicMock()  # type: ignore[method-assign]

    repo_service = MagicMock()
    repo_service.validate_repository_active.return_value = {
        "weaviate_collection": "FromTypeRepo",
        "default_tenant_id": None,
        "processing_hints": {
            "key_field_extraction_enabled": True,
            "enabled_processor_types": ["chunking_vectorizing", "key_field_extraction"],
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
            repository_id=None,
            collection_name=None,
            tenant_id=None,
            document_type_id="type-sop",
            link_document=False,
            audit_upload=False,
        )

    assert record["repository_id"] == "repo-from-type"
    assert record["collection_name"] == "FromTypeRepo"
    assert "key_field_extraction" in record["processing"]["enabled_processor_types"]
    service._bind_document_type.assert_called_once()


def test_seed_processor_plan_persists_scheduling_context() -> None:
    from src.infrastructure.database.document_jobs import DocumentJobStore

    store = DocumentJobStore.__new__(DocumentJobStore)
    store._scheduling_metadata = MagicMock(return_value={})  # type: ignore[method-assign]
    store._write_scheduling_metadata = MagicMock()  # type: ignore[method-assign]
    store.seed_processor_plan(
        "doc-1",
        ["chunking_vectorizing", "key_field_extraction"],
        context={
            "repository_id": "repo-1",
            "document_type_id": "type-sop",
            "key_field_extraction_enabled": True,
        },
    )
    written = store._write_scheduling_metadata.call_args.args[1]
    assert written["enabled_processor_types"] == ["chunking_vectorizing", "key_field_extraction"]
    assert written["repository_id"] == "repo-1"
    assert written["document_type_id"] == "type-sop"
    assert written["key_field_extraction_enabled"] is True
