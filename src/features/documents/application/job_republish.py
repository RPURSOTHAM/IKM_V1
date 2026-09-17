"""Lightweight RabbitMQ republish helpers for scheduler recovery (no FastAPI/Pydantic)."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
from src.features.documents.infrastructure.message_publisher import QueuePublisher
from src.features.configuration.platform_settings import get_settings
from src.infrastructure.database.document_jobs import get_document_job_store

logger = logging.getLogger(__name__)


def _normalize_optional_id(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def queue_payload_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Build a processor queue payload from an intake record (scheduler-safe)."""
    processing = record.get("processing") or {}
    metadata = record.get("metadata") or {}
    resolved_document_type_id = (
        processing.get("document_type_id")
        or record.get("document_type_id")
        or metadata.get("document_type_id")
    )
    resolved_document_type_name = (
        processing.get("document_type_name")
        or record.get("document_type_name")
        or metadata.get("document_type_name")
    )
    payload: dict[str, Any] = {
        "document_id": record["document_id"],
        "document_name": record["document_name"],
        "original_file_name": record.get("original_file_name")
        or (record.get("metadata") or {}).get("original_file_name"),
        "collection_name": record["collection_name"],
        "tenant_id": record.get("tenant_id"),
        "key_fields": processing.get("key_fields") or [],
        "effective_fields": processing.get("effective_fields")
        or (record.get("metadata") or {}).get("effective_fields")
        or [],
    }
    repository_path = record.get("repository_path") or (
        metadata.get("repository_path") if isinstance(metadata, dict) else None
    )
    if repository_path:
        payload["document_path"] = str(repository_path)
    if resolved_document_type_id:
        payload["document_type_id"] = resolved_document_type_id
    if resolved_document_type_name:
        payload["document_type_name"] = resolved_document_type_name
    repository_id = (
        _normalize_optional_id(record.get("repository_id"))
        or _normalize_optional_id(metadata.get("repository_id"))
        or _normalize_optional_id(processing.get("repository_id"))
    )
    if repository_id:
        payload["repository_id"] = repository_id
    for key in (
        "chunk_size",
        "chunk_overlap_sentences",
        "chunking_strategy",
        "chunking_config",
        "model_name",
        "model_dir",
        "citation_retainment",
        "key_field_extraction",
        "key_field_extraction_enabled",
        "validation_enabled",
        "validation_confidence_threshold",
        "document_type_id",
        "document_type_name",
        "metadata_fields",
        "effective_fields",
        "extraction_model",
        "enabled_processor_types",
        "metadata_extraction",
        "template_extraction",
        "reference_document_extraction",
        "conversion_for_rendering",
    ):
        if processing.get(key) is not None:
            payload[key] = processing[key]
    payload["strict_key_field_page_scope"] = bool(processing.get("strict_key_field_page_scope") or False)
    return payload


def _publisher_from_settings() -> QueuePublisher:
    settings = get_settings()
    return QueuePublisher(
        settings.rabbitmq_host,
        settings.rabbitmq_port,
        settings.rabbitmq_user,
        settings.rabbitmq_pass,
        settings.rabbitmq_queue_name,
    )


def initial_processor_types_for_upload(enabled: Sequence[str]) -> list[str]:
    """Return processor types to publish on first intake (chain the rest after completion)."""
    from src.features.document_processing.shared_processor.types import ProcessorType

    normalized = [str(pt).strip().lower() for pt in enabled if str(pt).strip()]
    if ProcessorType.MEDIA_TRANSCRIPTION.value in normalized:
        return [ProcessorType.MEDIA_TRANSCRIPTION.value]
    if ProcessorType.CHUNKING_VECTORIZING.value in normalized:
        return [ProcessorType.CHUNKING_VECTORIZING.value]
    return [normalized[0]] if normalized else [ProcessorType.CHUNKING_VECTORIZING.value]


def enqueue_ready_processors(document_id: str) -> list[str]:
    """Publish processor types whose dependencies are satisfied and outcomes are pending."""
    js = get_document_job_store()
    if not js:
        return []
    record = get_intake_record(document_id)
    if not record:
        return []
    from src.features.security.compliance.compliance_validator import (
        processing_allowed_after_security_review,
    )

    if not processing_allowed_after_security_review(record):
        return []
    job = js.get_by_document_id(document_id)
    meta = (job or {}).get("scheduling_metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    enabled = list(meta.get("enabled_processor_types") or [])
    results = meta.get("processor_results") or {}
    if not isinstance(results, dict):
        results = {}
    from src.features.document_processing.shared_processor.types import processor_dependencies_met

    ready: list[str] = []
    for processor_type in enabled:
        pt = str(processor_type).strip().lower()
        if not pt:
            continue
        outcome = (results.get(pt) or {}).get("status")
        if str(outcome or "").upper() in {"COMPLETED", "FAILED"}:
            continue
        if processor_dependencies_met(pt, results, enabled_types=enabled):
            ready.append(pt)
    if not ready:
        return []
    # Publish every processor whose dependencies are satisfied (parallel slots).
    republish_processor_types(document_id, ready)
    return ready


def get_intake_record(document_id: str) -> dict[str, Any] | None:
    settings = get_settings()
    return MetadataStore(settings.db_path).get_document(document_id)


def submit_record_to_queue(record: dict[str, Any]) -> None:
    """Publish all enabled processor types for a RECEIVED orphan recovery path.

    Documents pending Security Human Review are skipped until Allow / Mask and Allow.
    """
    from src.features.security.compliance.compliance_validator import (
        processing_allowed_after_security_review,
    )

    document_id = str(record.get("document_id") or "")
    if not processing_allowed_after_security_review(record):
        logger.info(
            "Skipping RECEIVED queue recovery for document_id=%s: pending human review",
            document_id,
        )
        # Keep intake status on human_review so clients do not show queued.
        try:
            MetadataStore(get_settings().db_path).update_document_status(document_id, "human_review")
        except Exception:
            logger.debug("Could not reaffirm human_review status for %s", document_id, exc_info=True)
        return

    processing = dict(record.get("processing") or {})
    enabled = list(processing.get("enabled_processor_types") or ["chunking_vectorizing"])
    publisher = _publisher_from_settings()
    base = queue_payload_from_record(record)
    for processor_type in initial_processor_types_for_upload(enabled):
        payload = dict(base)
        payload["processor_type"] = processor_type
        payload["enabled_processor_types"] = enabled
        publisher.publish_document_job(payload)
    js = get_document_job_store()
    if js:
        js.update_status_queued(str(record["document_id"]))
    MetadataStore(get_settings().db_path).update_document_status(
        str(record["document_id"]),
        "queued",
    )


def republish_processor_types(document_id: str, processor_types: Sequence[str]) -> None:
    """Re-queue specific processor jobs (scheduler recovery for multi-processor plans)."""
    types = [str(pt).strip() for pt in processor_types if str(pt).strip()]
    if not types:
        return
    record = get_intake_record(document_id)
    if not record:
        logger.warning("Republish skipped: intake record missing document_id=%s", document_id)
        return
    from src.features.security.compliance.compliance_validator import (
        processing_allowed_after_security_review,
    )

    if not processing_allowed_after_security_review(record):
        logger.info(
            "Republish skipped for document_id=%s: pending human review",
            document_id,
        )
        return
    processing = dict(record.get("processing") or {})
    enabled = list(processing.get("enabled_processor_types") or [])
    if not enabled:
        js = get_document_job_store()
        job = js.get_by_document_id(document_id) if js else None
        meta = (job or {}).get("scheduling_metadata") or {}
        enabled = list(meta.get("enabled_processor_types") or ["chunking_vectorizing"])
        processing["enabled_processor_types"] = enabled
        record["processing"] = processing
    publisher = _publisher_from_settings()
    base = queue_payload_from_record(record)
    published: list[str] = []
    for processor_type in types:
        if enabled and processor_type not in enabled:
            continue
        payload = dict(base)
        payload["processor_type"] = processor_type
        payload["enabled_processor_types"] = enabled
        publisher.publish_document_job(payload)
        published.append(processor_type)
        logger.info(
            "Republished document_id=%s for processor_type=%s (recovery)",
            document_id,
            processor_type,
        )
    if not published:
        return
    js = get_document_job_store()
    if js:
        job = js.get_by_document_id(document_id)
        if job and str(job.get("status") or "").strip().upper() == "FAILED":
            js.update_status_queued(document_id)
        js.mark_processor_republished(document_id, published)
        js.heartbeat(document_id, notes=f"republished:{','.join(published)}")
