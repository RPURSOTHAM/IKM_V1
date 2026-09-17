from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from fastapi import HTTPException, UploadFile
except ImportError:  # pragma: no cover - scheduler image has no FastAPI
    # Allow DocumentReceiverService.republish_processor_types from the scheduler
    # without pulling in the full API dependency set.
    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: Any = None):
            self.status_code = status_code
            self.detail = detail
            super().__init__(str(detail))

    UploadFile = Any  # type: ignore[misc,assignment]

from src.shared.errors import COMPONENT_DOCUMENT_RECEIVER, raise_service_error
from src.shared.errors.service_errors import DmsServiceError
from src.infrastructure.database.document_jobs import get_document_job_store
from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
from src.features.documents.infrastructure.message_publisher import QueuePublisher
from src.features.documents.schemas.document_schemas import (
    DocumentJobStatusResponse,
    DocumentRegisterRequest,
    DocumentResponse,
    RepositoryConfigRequest,
    UploadRejection,
    UploadResponse,
)
from src.features.configuration.platform_settings import ApiSettings, get_settings

_logger = logging.getLogger(__name__)


def _dms_async_upload_finalize_enabled() -> bool:
    return str(os.getenv("DMS_ASYNC_UPLOAD_FINALIZE", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pipeline_trace(
    stage: str,
    *,
    document_id: str | None = None,
    job_id: str | None = None,
    current_status: str | None = None,
    next_status: str | None = None,
    reason: str | None = None,
    **extra: Any,
) -> None:
    payload = {
        "stage": stage,
        "document_id": document_id,
        "job_id": job_id,
        "current_status": current_status,
        "next_status": next_status,
        "reason": reason,
        **{k: v for k, v in extra.items() if v is not None},
    }
    _logger.info("PIPELINE_TRACE %s", payload)

_EXTENSION_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain; charset=utf-8",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "wma": "audio/x-ms-wma",
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "mpeg": "video/mpeg",
    "m4v": "video/x-m4v",
    "wmv": "video/x-ms-wmv",
}

_CIH_MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v", ".wmv",
}
_CIH_PPT_EXTENSIONS = {".pptx", ".ppt"}


class DocumentReceiverService:
    """Document intake: metadata persistence, MySQL job rows, and RabbitMQ queue submission."""

    def __init__(self, api_settings: ApiSettings | None = None, publisher: QueuePublisher | None = None):
        bootstrap = api_settings or get_settings()
        self._bootstrap = bootstrap
        self._publisher = publisher or QueuePublisher(
            bootstrap.rabbitmq_host,
            bootstrap.rabbitmq_port,
            bootstrap.rabbitmq_user,
            bootstrap.rabbitmq_pass,
            bootstrap.rabbitmq_queue_name,
        )
        self._store: MetadataStore | None = None

    def _live_settings(self) -> ApiSettings:
        return get_settings()

    def _ensure_publisher(self) -> QueuePublisher:
        live = self._live_settings()
        if (
            self._publisher.host != live.rabbitmq_host
            or self._publisher.port != live.rabbitmq_port
            or self._publisher.user != live.rabbitmq_user
            or self._publisher.password != live.rabbitmq_pass
            or self._publisher.queue_name != live.rabbitmq_queue_name
        ):
            self._publisher = QueuePublisher(
                live.rabbitmq_host,
                live.rabbitmq_port,
                live.rabbitmq_user,
                live.rabbitmq_pass,
                live.rabbitmq_queue_name,
            )
        return self._publisher

    @property
    def publisher(self) -> QueuePublisher:
        return self._ensure_publisher()

    def get_store(self) -> MetadataStore:
        if self._store is None:
            self._store = MetadataStore(self._bootstrap.db_path)
        return self._store

    def init_store(self) -> None:
        self.get_store()

    @staticmethod
    def _job_store():
        return get_document_job_store()

    @staticmethod
    def document_status_from_job(job_status: str) -> str:
        key = (job_status or "").strip().upper()
        mapping = {
            "RECEIVED": "received",
            "QUEUED": "queued",
            "DISPATCHING": "dispatching",
            "ASSIGNED": "assigned",
            "HUMAN_REVIEW": "human_review",
            "STARTED": "starting",
            "IN_PROGRESS": "processing",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "TIMED_OUT": "failed",
            "QUEUE_FAILED": "failed",
            "STOPPED": "stopped",
        }
        return mapping.get(key, key.lower() or "unknown")

    def merge_job_into_record(self, record: dict[str, Any], job: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(record)
        if not job:
            return merged
        from src.infrastructure.database.document_jobs import effective_job_status

        merged["job_id"] = job.get("job_id")
        merged["batch_id"] = job.get("batch_id")
        job_status = self.document_status_from_job(effective_job_status(job))
        metadata = merged.get("metadata") if isinstance(merged.get("metadata"), dict) else {}
        try:
            from src.features.security.compliance.compliance_validator import (
                document_pending_human_review,
                processing_allowed_after_security_review,
            )

            pending_review = document_pending_human_review(merged)
            allowed_after_review = (
                processing_allowed_after_security_review(merged) if pending_review else True
            )
        except Exception:
            intake_status = str(merged.get("status") or "").lower()
            pending_review = intake_status == "human_review" or bool(
                metadata.get("requires_human_review")
            )
            allowed_after_review = not pending_review

        if pending_review and not allowed_after_review:
            # Hold visible status on human_review for any job state until Allow/Mask.
            # Otherwise orphan recovery / mid-flight jobs hide the review requirement.
            merged["status"] = "human_review"
            if job.get("error_details"):
                merged["error_details"] = job["error_details"]
            return merged

        merged["status"] = job_status
        if job.get("error_details"):
            merged["error_details"] = job["error_details"]
        elif isinstance(job.get("scheduling_metadata"), dict):
            chunk = (job["scheduling_metadata"].get("processor_results") or {}).get("chunking_vectorizing") or {}
            if isinstance(chunk, dict) and chunk.get("error_details"):
                merged["error_details"] = chunk["error_details"]
        elif job_status == "completed":
            merged["error_details"] = None

        completed_ts = self._job_timestamp(job, "completed_at")
        if completed_ts is not None:
            merged["processing_completion_timestamp"] = completed_ts
        elif job_status not in {"completed", "failed", "stopped"}:
            # Keep any prior completion stamp only for terminal jobs; clear for
            # in-flight states so clients do not see a stale completion time.
            pass

        # Propagate scheduling plan + per-processor outcomes from the shared
        # document_job row into document.metadata so GET /documents/{id} always
        # shows the enabled plan and sibling results (e.g. template_extraction
        # COMPLETED while chunking FAILED).
        sched = job.get("scheduling_metadata") if isinstance(job.get("scheduling_metadata"), dict) else {}
        results = sched.get("processor_results") if isinstance(sched.get("processor_results"), dict) else None
        meta_out = dict(metadata) if isinstance(metadata, dict) else {}
        meta_changed = False
        if isinstance(sched.get("enabled_processor_types"), list):
            meta_out["enabled_processor_types"] = list(sched.get("enabled_processor_types") or [])
            meta_changed = True
        if "template_extraction" in sched and not isinstance(meta_out.get("template_extraction"), dict):
            meta_out["template_extraction_enabled"] = bool(sched.get("template_extraction"))
            meta_changed = True
        if results:
            meta_out["processor_results"] = results
            template = results.get("template_extraction")
            if isinstance(template, dict) and str(template.get("status") or "").upper() == "COMPLETED":
                meta_out["template_extraction"] = {
                    "status": "COMPLETED",
                    "result_location": template.get("result_location"),
                    "completed_at": template.get("completed_at"),
                    "source": "template_extraction",
                }
            meta_changed = True
        if meta_changed:
            merged["metadata"] = meta_out
        return merged

    def document_response(
        self,
        record: dict[str, Any],
        *,
        prefetched: dict[str, dict[str, Any]] | None = None,
    ) -> DocumentResponse:
        js = self._job_store()
        job: dict[str, Any] | None = None
        if js:
            if prefetched is not None:
                job = prefetched.get(record["document_id"])
            else:
                job = js.get_by_document_id(record["document_id"])
        merged = self.merge_job_into_record(dict(record), job)
        merged = self._enrich_document_type_fields(merged)
        if not merged.get("validation_status"):
            meta = merged.get("metadata") if isinstance(merged.get("metadata"), dict) else {}
            merged["validation_status"] = meta.get("validation_status") or (
                (meta.get("validation") or {}).get("document_status")
                if isinstance(meta.get("validation"), dict)
                else None
            )
        payload = {k: v for k, v in merged.items() if k in DocumentResponse.model_fields}
        processing = merged.get("processing") if isinstance(merged.get("processing"), dict) else {}
        _pipeline_trace(
            "response_generation",
            document_id=str(merged.get("document_id") or ""),
            repository_id=merged.get("repository_id") or (merged.get("metadata") or {}).get("repository_id"),
            effective_document_type_id=merged.get("document_type_id") or processing.get("document_type_id"),
            enabled_processor_types=processing.get("enabled_processor_types"),
            key_field_extraction_enabled=processing.get("key_field_extraction_enabled"),
        )
        return DocumentResponse(**payload)

    @staticmethod
    def _document_type_store():
        from src.features.document_types.infrastructure.document_type_repository import get_document_type_store

        return get_document_type_store()

    @staticmethod
    def _normalize_optional_id(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text or text.lower() in {"null", "none", "undefined", "string"}:
            return None
        return text

    def _repository_id_from_document_type(self, document_type_id: str | None) -> str | None:
        type_id = self._normalize_optional_id(document_type_id)
        if not type_id:
            return None
        try:
            store = self._document_type_store()
            if store is None:
                return None
            doc_type = store.get_type_by_id(type_id)
            if doc_type is None:
                return None
            return self._normalize_optional_id(getattr(doc_type, "repository_id", None))
        except Exception:
            _logger.debug(
                "Could not resolve repository_id from document_type_id=%s",
                type_id,
                exc_info=True,
            )
            return None

    def _recover_repository_id(self, record: dict[str, Any]) -> str | None:
        """Preserve repository_id from record/metadata/instance/type; never invent null over a value."""
        metadata = record.get("metadata") or {}
        processing = record.get("processing") or {}
        candidates = (
            record.get("repository_id"),
            metadata.get("repository_id") if isinstance(metadata, dict) else None,
            processing.get("repository_id") if isinstance(processing, dict) else None,
        )
        for candidate in candidates:
            recovered = self._normalize_optional_id(candidate)
            if recovered:
                return recovered

        document_id = self._normalize_optional_id(record.get("document_id"))
        if document_id:
            try:
                from src.features.repositories.application.repository_service import get_repository_service

                linked = self._normalize_optional_id(get_repository_service().get_repository_id_for_document(document_id))
                if linked:
                    return linked
            except Exception:
                _logger.debug(
                    "Could not recover repository_id from repository_document for document_id=%s",
                    document_id,
                    exc_info=True,
                )
            try:
                store = self._document_type_store()
                if store is not None:
                    instance = store.get_document_instance(document_id)
                    linked = self._normalize_optional_id((instance or {}).get("repository_id"))
                    if linked:
                        return linked
            except Exception:
                _logger.debug(
                    "Could not recover repository_id from document_instance for document_id=%s",
                    document_id,
                    exc_info=True,
                )

        document_type_id = self._normalize_optional_id(
            (processing.get("document_type_id") if isinstance(processing, dict) else None)
            or record.get("document_type_id")
            or (metadata.get("document_type_id") if isinstance(metadata, dict) else None)
        )
        return self._repository_id_from_document_type(document_type_id)

    def _preserve_repository_id(self, record: dict[str, Any], repository_id: str | None) -> str | None:
        previous = self._normalize_optional_id(record.get("repository_id"))
        recovered = (
            self._normalize_optional_id(repository_id)
            or previous
            or self._recover_repository_id(record)
        )
        if not recovered:
            return None
        if previous and previous != recovered:
            _logger.warning(
                "repository_id mismatch for document_id=%s previous=%s recovered=%s; keeping previous",
                record.get("document_id"),
                previous,
                recovered,
            )
            recovered = previous
        record["repository_id"] = recovered
        metadata = dict(record.get("metadata") or {})
        metadata["repository_id"] = recovered
        record["metadata"] = metadata
        processing = dict(record.get("processing") or {})
        processing["repository_id"] = recovered
        record["processing"] = processing
        return recovered

    @staticmethod
    def _compute_enabled_processor_types(processing: dict[str, Any] | None) -> list[str]:
        from src.features.document_processing.shared_processor.types import enabled_processor_types

        computed = list(enabled_processor_types(processing or {}))
        if not computed:
            return ["chunking_vectorizing"]
        return computed

    def _ensure_cih_media_processor(
        self,
        record: dict[str, Any],
        enabled: list[str],
        processing: dict[str, Any],
    ) -> list[str]:
        """Queue media_transcription ahead of chunking for audio/video uploads."""
        filename = str(
            record.get("original_file_name")
            or record.get("document_name")
            or (record.get("metadata") or {}).get("original_file_name")
            or ""
        )
        ext = Path(filename).suffix.lower()
        if not ext:
            ext = "." + str(record.get("document_type") or "").lower().lstrip(".")
        if ext not in _CIH_MEDIA_EXTENSIONS:
            return enabled
        processing["media_transcription"] = True
        ordered = ["media_transcription"] + [pt for pt in enabled if pt != "media_transcription"]
        return ordered

    @staticmethod
    def _processing_flag_enabled(processing: dict[str, Any] | None, *flags: str) -> bool:
        from src.features.document_processing.shared_processor.types import _setting_flag_enabled

        data = processing or {}
        return any(_setting_flag_enabled(data, flag) for flag in flags)

    @staticmethod
    def _stored_bool(stored: dict[str, Any] | None, *keys: str) -> bool | None:
        """Return explicit stored bool for the first present key, else None if unset."""
        if not isinstance(stored, dict):
            return None
        for key in keys:
            if key not in stored:
                continue
            value = stored.get(key)
            if value is None:
                continue
            if value is True or value == 1:
                return True
            if value is False or value == 0:
                return False
            if isinstance(value, str) and value.strip():
                return value.strip().lower() in {"1", "true", "yes", "on"}
        return None

    def _apply_extraction_enablement(
        self,
        processing: dict[str, Any],
        *,
        stored_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enable metadata/key-field processors when configured fields exist.

        Explicit stored repository flags always win. When flags were never stored
        (inherited False defaults), enable:
        - metadata_extraction (matches create-UI default)
        - key_field_extraction when the document type defines key fields
        """
        updated = dict(processing or {})
        stored = dict(stored_settings or {})

        stored_metadata = self._stored_bool(stored, "metadata_extraction")
        if stored_metadata is None:
            updated["metadata_extraction"] = True
        else:
            updated["metadata_extraction"] = stored_metadata

        stored_kfe = self._stored_bool(
            stored,
            "key_field_extraction_enabled",
            "key_field_extraction",
        )
        if stored_kfe is False:
            updated["key_field_extraction"] = False
            updated["key_field_extraction_enabled"] = False
            return updated
        if stored_kfe is True:
            updated["key_field_extraction"] = True
            updated["key_field_extraction_enabled"] = True
            return updated

        has_key_fields = bool(updated.get("key_fields"))
        document_type_id = str(updated.get("document_type_id") or "").strip() or None
        if not has_key_fields and document_type_id:
            try:
                from src.features.document_types.application.processing_key_fields import (
                    resolve_key_fields_for_type,
                )

                has_key_fields = bool(resolve_key_fields_for_type(document_type_id))
            except Exception:
                has_key_fields = False
        if has_key_fields:
            updated["key_field_extraction"] = True
            updated["key_field_extraction_enabled"] = True
        return updated

    def _validate_document_type(
        self,
        document_type_id: str,
        *,
        repository_id: str | None = None,
    ) -> tuple[str, str | None]:
        candidate = (document_type_id or "").strip()
        if not candidate:
            raise HTTPException(status_code=400, detail={"error": "Invalid document_type_id"})
        try:
            store = self._document_type_store()
        except Exception as exc:
            _logger.exception("Document type store initialization failed during upload validation")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Document type validation unavailable",
                    "details": str(exc),
                },
            ) from exc
        if store is None:
            raise HTTPException(status_code=503, detail="Document type store is not configured.")
        try:
            doc_type = store.get_type_by_id(candidate)
        except Exception as exc:
            _logger.exception("Document type lookup failed for document_type_id=%s", candidate)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Document type validation unavailable",
                    "details": str(exc),
                },
            ) from exc
        if doc_type is None:
            raise HTTPException(status_code=400, detail={"error": "Invalid document_type_id"})
        if not bool(getattr(doc_type, "is_active", True)):
            raise HTTPException(status_code=400, detail={"error": "Invalid document_type_id"})
        if repository_id and doc_type.repository_id and str(doc_type.repository_id) != str(repository_id):
            raise HTTPException(status_code=400, detail={"error": "Invalid document_type_id"})
        return str(doc_type.document_type_id), str(doc_type.name or "").strip() or None

    def _resolve_document_type(
        self,
        *,
        repository_id: str | None,
        document_type_id: str | None,
        processing: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None]:
        if document_type_id:
            return self._validate_document_type(document_type_id, repository_id=repository_id)

        processing_type_id = str((processing or {}).get("document_type_id") or "").strip()
        if processing_type_id:
            try:
                return self._validate_document_type(processing_type_id, repository_id=repository_id)
            except HTTPException as exc:
                if exc.status_code != 400:
                    raise
                _logger.debug(
                    "Ignoring invalid repository default document_type_id=%s for repository_id=%s",
                    processing_type_id,
                    repository_id,
                    exc_info=True,
                )

        if repository_id:
            store = self._document_type_store()
            if store:
                basic = store.get_basic_type_for_repository(str(repository_id))
                if basic is not None:
                    return str(basic.document_type_id), str(basic.name or "").strip() or None
        return None, None

    def _upsert_document_instance(
        self,
        *,
        document_id: str,
        document_type_id: str,
        repository_id: str | None,
        tenant_id: str | None,
    ) -> None:
        store = self._document_type_store()
        if store is None:
            return
        store.upsert_document_instance(
            document_id,
            document_type_id,
            repository_id=repository_id,
            tenant_id=tenant_id,
        )

    def _delete_document_instance(self, document_id: str) -> None:
        store = self._document_type_store()
        if store is None:
            return
        try:
            store.delete_document_data(document_id)
        except Exception:
            _logger.debug("Could not delete document_instance for document_id=%s", document_id, exc_info=True)

    def _resolve_record_document_type(self, record: dict[str, Any]) -> tuple[str | None, str | None]:
        processing = dict(record.get("processing") or {})
        metadata = dict(record.get("metadata") or {})
        document_id = str(record.get("document_id") or "").strip()
        repository_id = str(record.get("repository_id") or metadata.get("repository_id") or "").strip() or None

        resolved_type_id = (
            str(processing.get("document_type_id") or "").strip()
            or str(record.get("document_type_id") or "").strip()
            or str(metadata.get("document_type_id") or "").strip()
        ) or None
        resolved_type_name = (
            str(processing.get("document_type_name") or "").strip()
            or str(record.get("document_type_name") or "").strip()
            or str(metadata.get("document_type_name") or "").strip()
        ) or None

        if not resolved_type_id and document_id:
            try:
                store = self._document_type_store()
                if store:
                    instance = store.get_document_instance(document_id)
                    if instance:
                        resolved_type_id = str(instance.get("document_type_id") or "").strip() or None
            except Exception:
                _logger.debug("Could not read document_instance for document_id=%s", document_id, exc_info=True)

        if not resolved_type_id:
            fallback_type_id, fallback_type_name = self._resolve_document_type(
                repository_id=repository_id,
                document_type_id=None,
                processing=processing,
            )
            resolved_type_id = fallback_type_id
            resolved_type_name = resolved_type_name or fallback_type_name

        if resolved_type_id and not resolved_type_name:
            try:
                store = self._document_type_store()
                if store:
                    doc_type = store.get_type_by_id(resolved_type_id)
                    if doc_type is not None:
                        resolved_type_name = str(doc_type.name or "").strip() or None
            except Exception:
                _logger.debug(
                    "Could not resolve document type name for document_type_id=%s",
                    resolved_type_id,
                    exc_info=True,
                )

        return resolved_type_id, resolved_type_name

    def _apply_resolved_document_type(
        self,
        record: dict[str, Any],
        *,
        document_type_id: str | None,
        document_type_name: str | None,
    ) -> None:
        if not document_type_id:
            return
        processing = dict(record.get("processing") or {})
        metadata = dict(record.get("metadata") or {})
        processing["document_type_id"] = document_type_id
        metadata["document_type_id"] = document_type_id
        record["document_type_id"] = document_type_id
        if document_type_name:
            processing["document_type_name"] = document_type_name
            metadata["document_type_name"] = document_type_name
            record["document_type_name"] = document_type_name
        record["processing"] = processing
        record["metadata"] = metadata

    def _enrich_document_type_fields(self, record: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(record)
        metadata = dict(enriched.get("metadata") or {})
        processing = dict(enriched.get("processing") or {})
        doc_id = str(enriched.get("document_id") or "").strip()
        document_type_id = (
            str(enriched.get("document_type_id") or "").strip()
            or str(metadata.get("document_type_id") or "").strip()
            or str(processing.get("document_type_id") or "").strip()
        )
        instance: dict[str, Any] | None = None
        if not document_type_id and doc_id:
            try:
                store = self._document_type_store()
                if store:
                    instance = store.get_document_instance(doc_id)
                    document_type_id = str((instance or {}).get("document_type_id") or "").strip()
            except Exception:
                _logger.debug("Could not resolve document_instance for document_id=%s", doc_id, exc_info=True)

        document_type_name = str(enriched.get("document_type_name") or "").strip() or None
        if not document_type_name:
            document_type_name = str(metadata.get("document_type_name") or "").strip() or None
        if not document_type_name and document_type_id:
            try:
                store = self._document_type_store()
                if store:
                    doc_type = store.get_type_by_id(document_type_id)
                    if doc_type is not None:
                        document_type_name = str(doc_type.name or "").strip() or None
            except Exception:
                _logger.debug(
                    "Could not resolve document type name for document_type_id=%s",
                    document_type_id,
                    exc_info=True,
                )

        if document_type_id:
            enriched["document_type_id"] = document_type_id
            metadata["document_type_id"] = document_type_id
            processing["document_type_id"] = document_type_id
        if document_type_name:
            enriched["document_type_name"] = document_type_name
            metadata["document_type_name"] = document_type_name
            processing.setdefault("document_type_name", document_type_name)
        if instance:
            if not enriched.get("repository_id") and instance.get("repository_id"):
                enriched["repository_id"] = instance.get("repository_id")
            if not enriched.get("tenant_id") and instance.get("tenant_id"):
                enriched["tenant_id"] = instance.get("tenant_id")
        if metadata:
            enriched["metadata"] = metadata
        if processing:
            enriched["processing"] = processing
        return enriched

    def processing_summary_from_job(self, job: dict[str, Any] | None) -> dict[str, Any]:
        if not job:
            return {}
        from src.infrastructure.database.document_jobs import effective_job_status

        meta = job.get("scheduling_metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        return {
            "job_status": effective_job_status(job) or job.get("status"),
            "processor_id": job.get("processor_id"),
            "enabled_processor_types": meta.get("enabled_processor_types"),
            "processor_results": meta.get("processor_results"),
            "processor_assignments": meta.get("processor_assignments"),
            "current_dispatch": meta.get("current_dispatch"),
        }

    @staticmethod
    def _job_timestamp(job: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            raw = job.get(key)
            if raw is None or raw == "":
                continue
            if isinstance(raw, (int, float)):
                return float(raw)
            if hasattr(raw, "timestamp"):
                try:
                    return float(raw.timestamp())  # type: ignore[union-attr]
                except (TypeError, ValueError, OSError):
                    continue
            if isinstance(raw, str):
                from datetime import datetime

                try:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
        return None

    def _resolve_job_source_path(self, document_id: str, job: dict[str, Any]) -> Path | None:
        candidates: list[Path] = []
        source_location = str(job.get("source_location") or "").strip()
        if source_location:
            candidates.append(Path(source_location))
        stored_name = str(job.get("document_name") or "").strip()
        if stored_name:
            candidates.append(Path(self._live_settings().upload_dir) / stored_name)
        candidates.append(Path(self._live_settings().upload_dir) / document_id)
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _resolve_upload_path_without_job(self, document_id: str) -> Path | None:
        upload_dir = Path(self._live_settings().upload_dir)
        candidates: list[Path] = [upload_dir / document_id]
        for ext in self._live_settings().allowed_extensions:
            normalized = str(ext).lstrip(".")
            if normalized:
                candidates.append(upload_dir / f"{document_id}.{normalized}")
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.is_file():
                return path
        return None

    def _repository_weaviate_context(self, repository_id: str | None) -> tuple[str | None, str | None]:
        if not repository_id:
            return None, None
        try:
            from src.features.repositories.application.repository_settings_service import resolve_repository_context

            resolved = resolve_repository_context(str(repository_id))
        except Exception:
            return None, None
        collection = str(resolved.get("weaviate_collection") or "").strip() or None
        tenant = resolved.get("default_tenant_id")
        if tenant is None:
            tenant = self._live_settings().default_tenant_id
        return collection, str(tenant) if tenant else None

    def _require_active_repository(self, repository_id: str | None) -> dict[str, Any] | None:
        """Validate repository exists and is ACTIVE before accepting an upload.

        Uses RepositoryService (SettingsResolver underneath). Returns the resolved
        repository context dict, or None when no repository_id was provided
        (legacy uploads without a repository remain supported).
        """
        repository_id = self._normalize_optional_id(repository_id)
        if not repository_id:
            return None
        from src.features.repositories.application.repository_service import get_repository_service

        return get_repository_service().validate_repository_active(repository_id)

    def _try_rehydrate_document_from_disk(self, document_id: str) -> dict[str, Any] | None:
        """Rebuild intake metadata from an on-disk upload when JSON metadata was lost."""
        path = self._resolve_upload_path_without_job(document_id)
        if path is None:
            return None

        from src.features.documents.application.document_metadata_service import (
            is_internal_storage_filename,
            weaviate_stats_for_document_id,
        )
        from src.features.repositories.application.repository_service import get_repository_service

        repository_id: str | None = get_repository_service().get_repository_id_for_document(document_id)

        collection_name = self._live_settings().default_collection_name
        tenant_id = self._live_settings().default_tenant_id
        if repository_id:
            resolved_collection, resolved_tenant = self._repository_weaviate_context(repository_id)
            if resolved_collection:
                collection_name = resolved_collection
            if resolved_tenant:
                tenant_id = resolved_tenant

        file_size = int(path.stat().st_size)
        weaviate_stats = (
            weaviate_stats_for_document_id(
                str(collection_name or ""),
                document_id,
                tenant=str(tenant_id) if tenant_id else None,
                file_size_bytes=file_size,
            )
            if collection_name
            else {}
        )

        stored_name = path.name
        display_name = stored_name
        doc_name = str(weaviate_stats.get("doc_name") or "").strip()
        if doc_name and not is_internal_storage_filename(doc_name, document_id):
            display_name = doc_name

        document_type = path.suffix.lstrip(".").lower() or "bin"
        chunk_count = weaviate_stats.get("chunk_count")
        record: dict[str, Any] = {
            "document_id": document_id,
            "document_name": stored_name,
            "original_file_name": display_name,
            "document_type": document_type,
            "repository_type": self._live_settings().repository_type,
            "repository_path": str(path),
            "collection_name": collection_name,
            "tenant_id": tenant_id,
            "status": "completed" if chunk_count else "received",
            "upload_timestamp": path.stat().st_mtime,
            "metadata": {
                "original_file_name": display_name,
                "document_info": {
                    "file_size_bytes": file_size,
                    "chunk_count": chunk_count,
                    "line_count": weaviate_stats.get("line_count"),
                    "indexing_status": weaviate_stats.get("indexing_status"),
                },
            },
        }
        if repository_id:
            record["repository_id"] = repository_id
            record["metadata"]["repository_id"] = repository_id

        restored = self.get_store().upsert_document(record)
        _logger.info("Rehydrated intake metadata for document_id=%s from on-disk upload", document_id)
        return restored

    def _repository_entry_from_indexed_orphan(self, document_id: str) -> dict[str, Any] | None:
        """List-view payload when only Weaviate chunks and repository links remain."""
        from src.features.documents.application.document_metadata_service import (
            is_internal_storage_filename,
            weaviate_stats_for_document_id,
        )
        from src.features.repositories.application.repository_service import get_repository_service

        repository_id: str | None = get_repository_service().get_repository_id_for_document(document_id)

        collection_name, tenant_id = self._repository_weaviate_context(repository_id)
        if not collection_name:
            return None

        upload_path = self._resolve_upload_path_without_job(document_id)
        file_size = int(upload_path.stat().st_size) if upload_path is not None else None
        weaviate_stats = weaviate_stats_for_document_id(
            collection_name,
            document_id,
            tenant=tenant_id,
            file_size_bytes=file_size,
        )
        chunk_count = int(weaviate_stats.get("chunk_count") or 0)
        if chunk_count <= 0:
            return None

        doc_name = str(weaviate_stats.get("doc_name") or document_id).strip()
        display_name = doc_name
        if is_internal_storage_filename(display_name, document_id):
            display_name = f"Indexed document ({document_id[:8]}…)"

        chunk_meta = {
            "original_file_name": display_name,
            "file_size_bytes": weaviate_stats.get("file_size_bytes") or file_size,
            "chunk_count": chunk_count,
            "line_count": weaviate_stats.get("line_count"),
            "collection_name": collection_name,
            "tenant_id": tenant_id,
        }
        processing = {
            "job_status": "COMPLETED",
            "processor_results": {
                "chunking_vectorizing": {
                    "status": "COMPLETED",
                    "document_metadata": chunk_meta,
                }
            },
        }
        return {
            "document_id": document_id,
            "document_name": doc_name,
            "original_file_name": display_name,
            "status": "completed",
            "repository_id": repository_id,
            "linked": True,
            "processing": processing,
            "metadata": {"document_info": chunk_meta, "original_file_name": display_name},
            "upload_timestamp": upload_path.stat().st_mtime if upload_path is not None else None,
            "processing_completion_timestamp": None,
            "error_details": None,
        }

    def _try_rehydrate_document_from_job(self, document_id: str, job: dict[str, Any]) -> dict[str, Any] | None:
        """Rebuild JSON intake metadata from a MySQL job row and on-disk upload file."""
        path = self._resolve_job_source_path(document_id, job)
        if path is None:
            return None

        from src.features.documents.application.document_metadata_service import (
            resolve_display_filename_from_job,
        )
        from src.features.repositories.application.repository_service import get_repository_service

        stored_name = str(job.get("document_name") or path.name).strip()
        document_type = path.suffix.lstrip(".") or self.safe_extension(path.name)
        display_name = resolve_display_filename_from_job(job, document_id=document_id) or stored_name
        repository_id: str | None = get_repository_service().get_repository_id_for_document(document_id)

        upload_ts = self._job_timestamp(job, "created_at") or time.time()
        completed_ts = self._job_timestamp(job, "completed_at", "updated_at")
        from src.infrastructure.database.document_jobs import effective_job_status

        record: dict[str, Any] = {
            "document_id": document_id,
            "document_name": stored_name,
            "original_file_name": display_name,
            "document_type": document_type,
            "repository_type": self._live_settings().repository_type,
            "repository_path": str(path),
            "collection_name": job.get("collection_name") or self._live_settings().default_collection_name,
            "tenant_id": job.get("tenant_id") or self._live_settings().default_tenant_id,
            "status": self.document_status_from_job(effective_job_status(job)),
            "upload_timestamp": upload_ts,
            "metadata": {
                "job_id": job.get("job_id"),
                "batch_id": job.get("batch_id"),
                "original_file_name": display_name,
            },
        }
        if repository_id:
            record["repository_id"] = repository_id
            record["metadata"]["repository_id"] = repository_id
        if completed_ts is not None:
            record["processing_completion_timestamp"] = completed_ts
        if job.get("error_details"):
            record["error_details"] = job.get("error_details")

        restored = self.get_store().upsert_document(record)
        _logger.info("Rehydrated intake metadata for document_id=%s from MySQL job", document_id)
        return restored

    def _repository_entry_from_job(self, document_id: str, job: dict[str, Any]) -> dict[str, Any]:
        """List-view payload when intake JSON is gone but the MySQL job row still exists."""
        from src.features.documents.application.document_metadata_service import (
            resolve_display_filename_from_job,
        )
        from src.infrastructure.database.document_jobs import effective_job_status

        processing = self.processing_summary_from_job(job)
        display_name = resolve_display_filename_from_job(job, document_id=document_id)
        stored_name = str(job.get("document_name") or document_id)
        status = self.document_status_from_job(effective_job_status(job))
        return {
            "document_id": document_id,
            "document_name": stored_name,
            "original_file_name": display_name,
            "status": status,
            "linked": True,
            "processing": processing,
            "metadata": {},
            "upload_timestamp": self._job_timestamp(job, "created_at"),
            "processing_completion_timestamp": self._job_timestamp(job, "completed_at"),
            "error_details": job.get("error_details"),
        }

    def repository_document_entry(
        self,
        document_id: str,
        *,
        prefetched_jobs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        job = (prefetched_jobs or {}).get(document_id)
        if job is None:
            js = self._job_store()
            job = js.get_by_document_id(document_id) if js else None

        record = self.get_store().get_document(document_id)
        if not record and job:
            record = self._try_rehydrate_document_from_job(document_id, job)
        if not record:
            record = self._try_rehydrate_document_from_disk(document_id)
        if not record:
            if job:
                return self._repository_entry_from_job(document_id, job)
            orphan = self._repository_entry_from_indexed_orphan(document_id)
            if orphan:
                return orphan
            processing = self.processing_summary_from_job(job)
            from src.features.repositories.application.repository_service import get_repository_service

            repository_id = get_repository_service().get_repository_id_for_document(document_id)
            return {
                "document_id": document_id,
                "document_name": document_id,
                "original_file_name": None,
                "status": "missing",
                "repository_id": repository_id,
                "linked": True,
                "processing": processing,
                "metadata": {},
                "error_details": "Document metadata is unavailable in DMS; remove this stale record.",
            }

        doc = self.document_response(record, prefetched=prefetched_jobs)
        entry = doc.model_dump()
        from src.features.documents.application.document_metadata_service import (
            resolve_display_filename,
        )

        display_name = resolve_display_filename(record, job=job)
        if display_name:
            entry["original_file_name"] = display_name
        if job is None:
            js = self._job_store()
            job = js.get_by_document_id(document_id) if js else None
        entry["processing"] = self.processing_summary_from_job(job)
        return entry

    def safe_extension(self, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        allowed = self._live_settings().allowed_extensions
        if ext not in allowed:
            media_extensions = set(_CIH_MEDIA_EXTENSIONS)
            if ext in media_extensions or ext in _CIH_PPT_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "unsupported_media_type",
                        "message": (
                            f"Uploads with extension {ext} are not enabled. "
                            f"Allowed types: {', '.join(sorted(allowed))}."
                        ),
                        "extension": ext,
                        "allowed_extensions": sorted(allowed),
                    },
                )
            raise HTTPException(status_code=400, detail=f"Unsupported document type: {ext}")
        return ext.lstrip(".")

    @staticmethod
    def media_type_for_document(document_type: str, filename: str) -> str:
        ext = document_type.lower().lstrip(".")
        if ext in _EXTENSION_MEDIA_TYPES:
            return _EXTENSION_MEDIA_TYPES[ext]
        guessed, _ = mimetypes.guess_type(filename)
        return guessed or "application/octet-stream"

    async def _save_upload_file(self, upload: UploadFile, destination: Path) -> tuple[str, int]:
        live = self._live_settings()
        max_bytes = live.max_upload_bytes
        total = 0
        digest = hashlib.sha256()
        try:
            with destination.open("wb") as output:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum upload size of {max_bytes} bytes.",
                        )
                    digest.update(chunk)
                    output.write(chunk)
        except HTTPException:
            if destination.exists():
                destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return digest.hexdigest(), total

    # No status is skippable: a document already in the repository (including
    # human_review) blocks another upload of the same bytes or wording.
    _NON_BLOCKING_DUPLICATE_STATUSES: frozenset[str] = frozenset()

    def _find_duplicate_document(
        self,
        *,
        content_hash: str,
        repository_id: str | None,
        collection_name: str | None,
        tenant_id: str | None,
        exclude_document_id: str | None = None,
        content_text_hash: str | None = None,
    ) -> dict[str, Any] | None:
        if not content_hash and not content_text_hash:
            return None
        store = self.get_store()
        if hasattr(store, "find_duplicate_by_content_hash"):
            return store.find_duplicate_by_content_hash(
                content_hash,
                content_text_hash=content_text_hash,
                repository_id=repository_id,
                collection_name=collection_name if repository_id is None else None,
                tenant_id=tenant_id if repository_id is None else None,
                exclude_document_id=exclude_document_id,
                non_blocking_statuses=self._NON_BLOCKING_DUPLICATE_STATUSES,
            )
        candidates = store.list_documents(
            repository_id=repository_id,
            collection_name=collection_name if repository_id is None else None,
            tenant_id=tenant_id if repository_id is None else None,
            limit=1_000_000,
        )
        for record in candidates:
            if exclude_document_id and str(record.get("document_id") or "") == str(exclude_document_id):
                continue
            status = str(record.get("status") or "").strip().lower()
            if status in self._NON_BLOCKING_DUPLICATE_STATUSES:
                continue
            metadata = record.get("metadata") or {}
            existing_hash = record.get("content_hash") or metadata.get("content_hash")
            existing_text = record.get("content_text_hash") or metadata.get("content_text_hash")
            if content_hash and existing_hash == content_hash:
                return record
            if content_text_hash and existing_text and existing_text == content_text_hash:
                return record
        return None

    def _purge_replaceable_duplicates(
        self,
        *,
        content_hash: str,
        repository_id: str | None,
        collection_name: str | None,
        tenant_id: str | None,
        exclude_document_id: str | None = None,
    ) -> None:
        """Remove stalled same-hash uploads (e.g. human_review) so a re-upload can proceed."""
        if not content_hash:
            return
        store = self.get_store()
        candidates = store.list_documents(
            repository_id=repository_id,
            collection_name=collection_name if repository_id is None else None,
            tenant_id=tenant_id if repository_id is None else None,
            limit=1_000_000,
        )
        for record in candidates:
            doc_id = str(record.get("document_id") or "")
            if not doc_id or (exclude_document_id and doc_id == str(exclude_document_id)):
                continue
            status = str(record.get("status") or "").strip().lower()
            if status not in self._NON_BLOCKING_DUPLICATE_STATUSES:
                continue
            metadata = record.get("metadata") or {}
            existing_hash = record.get("content_hash") or metadata.get("content_hash")
            if existing_hash != content_hash:
                continue
            try:
                path = Path(str(record.get("repository_path") or ""))
                if path.is_file():
                    path.unlink(missing_ok=True)
            except Exception:
                _logger.debug("Could not delete stalled duplicate file for document_id=%s", doc_id, exc_info=True)
            try:
                store.delete_document(doc_id)
            except Exception:
                _logger.debug("Could not delete stalled duplicate metadata for document_id=%s", doc_id, exc_info=True)
            try:
                self._delete_document_instance(doc_id)
            except Exception:
                _logger.debug("Could not delete stalled duplicate instance for document_id=%s", doc_id, exc_info=True)
            _logger.info(
                "Purged stalled duplicate document_id=%s status=%s to allow re-upload of content_hash=%s",
                doc_id,
                status,
                content_hash[:12],
            )

    @staticmethod
    def _duplicate_detail(existing: dict[str, Any]) -> dict[str, Any]:
        scope = "repository" if existing.get("repository_id") else "collection"
        return {
            "code": "duplicate_document",
            "message": f"An identical document already exists in this {scope}.",
            "existing_document_id": existing.get("document_id"),
            "existing_document_name": existing.get("original_file_name") or existing.get("document_name"),
            "existing_status": existing.get("status"),
            "existing_repository_id": existing.get("repository_id"),
        }

    @classmethod
    def _record_duplicate_rejection(
        cls,
        existing: dict[str, Any],
        *,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Record observability for a duplicate upload and return the rejection payload."""
        detail = cls._duplicate_detail(existing)
        content_hash = existing.get("content_hash") or (existing.get("metadata") or {}).get("content_hash")
        display_name = filename or detail.get("existing_document_name")
        try:
            from src.features.observability.audit.application.audit_service import get_audit_service
            from src.features.observability.metrics.application.metrics_service import get_metrics_service

            meta: dict[str, Any] = {
                "existing_document_id": existing.get("document_id"),
                "existing_status": existing.get("status"),
                "scope": "repository" if existing.get("repository_id") else "collection",
            }
            if content_hash:
                meta["content_hash_prefix"] = str(content_hash)[:12]
            if display_name:
                meta["filename"] = display_name
            get_audit_service().record(
                "DOCUMENT_UPLOAD_DUPLICATE_REJECTED",
                category="document",
                action="upload_duplicate",
                source="API",
                entity_type="document",
                entity_id=existing.get("document_id"),
                document_name=display_name,
                status="failure",
                metadata=meta,
            )
            get_metrics_service().record(
                "document_upload_duplicate",
                status="failure",
                document_id=existing.get("document_id"),
                repository_id=existing.get("repository_id"),
                metadata=meta,
            )
        except Exception:
            _logger.debug("Duplicate-upload observability failed", exc_info=True)
        return detail

    @classmethod
    def _raise_duplicate_upload(cls, existing: dict[str, Any]) -> None:
        detail = cls._record_duplicate_rejection(existing)
        raise HTTPException(status_code=409, detail=detail)

    @classmethod
    def _rejection_from_duplicate(
        cls,
        existing: dict[str, Any],
        *,
        filename: str | None = None,
    ) -> UploadRejection:
        detail = cls._record_duplicate_rejection(existing, filename=filename)
        return UploadRejection(
            filename=filename,
            code=str(detail.get("code") or "duplicate_document"),
            message=str(detail.get("message") or "Duplicate document."),
            existing_document_id=detail.get("existing_document_id"),
            existing_document_name=detail.get("existing_document_name"),
            existing_status=detail.get("existing_status"),
            existing_repository_id=detail.get("existing_repository_id"),
        )

    @staticmethod
    def _rejection_from_error(
        *,
        filename: str | None,
        code: str,
        message: str,
    ) -> UploadRejection:
        return UploadRejection(
            filename=filename,
            code=code,
            message=message,
        )

    @classmethod
    def _rejection_from_exception(
        cls,
        exc: BaseException,
        *,
        filename: str | None,
    ) -> UploadRejection:
        """Map a per-file failure into an UploadRejection for multi-file batches."""
        if isinstance(exc, DmsServiceError):
            return cls._rejection_from_error(
                filename=filename,
                code=str(exc.code or "upload_failed"),
                message=str(exc.user_message or exc.reason or "Upload failed."),
            )
        if isinstance(exc, HTTPException):
            detail = exc.detail
            if isinstance(detail, dict):
                code = str(detail.get("code") or f"http_{exc.status_code}")
                message = str(
                    detail.get("message")
                    or detail.get("error")
                    or detail.get("detail")
                    or detail
                )
            else:
                code = f"http_{exc.status_code}"
                message = str(detail)
            return cls._rejection_from_error(
                filename=filename,
                code=code,
                message=message,
            )
        return cls._rejection_from_error(
            filename=filename,
            code="upload_failed",
            message=str(exc) or "Upload failed.",
        )

    def resolve_original_file(self, document_id: str) -> tuple[Path, str, str]:
        """Return local path, download filename, and media type for an ingested document."""
        record = self.get_store().get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found")
        repository_type = str(record.get("repository_type") or "local")
        if repository_type != "local":
            raise HTTPException(
                status_code=501,
                detail=f"Original file download is not implemented for repository type '{repository_type}'.",
            )
        path = Path(str(record.get("repository_path") or ""))
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Original file is not available on disk")
        filename = str(record.get("original_file_name") or record.get("document_name") or path.name)
        document_type = str(record.get("document_type") or Path(filename).suffix.lstrip("."))
        media_type = self.media_type_for_document(document_type, filename)
        return path, filename, media_type

    def get_document_preview(self, document_id: str) -> dict[str, Any]:
        """Resolve read-only preview content, converting the document when required."""
        from src.features.documents.application.document_metadata_service import (
            check_document_access,
            resolve_document_info,
        )
        from src.features.documents.application.document_preview_service import resolve_document_preview

        record = self.get_document_record(document_id)
        check_document_access(record)
        job = self._job_store().get_by_document_id(document_id) if self._job_store() else None
        document_info = resolve_document_info(record, job=job)

        path, filename, media_type = self.resolve_original_file(document_id)
        return resolve_document_preview(
            record,
            media_type=media_type,
            path=path,
            filename=filename,
            document_info=document_info,
        )

    def get_document_info(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_metadata_service import (
            check_document_access,
            resolve_document_info,
        )

        record = self.get_document_record(document_id)
        check_document_access(record)
        job = self._job_store().get_by_document_id(document_id) if self._job_store() else None
        return resolve_document_info(record, job=job)

    def get_type_metadata(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_metadata_service import (
            check_document_access,
            resolve_type_metadata,
        )

        record = self.get_document_record(document_id)
        check_document_access(record)
        job = self._job_store().get_by_document_id(document_id) if self._job_store() else None
        return resolve_type_metadata(record, job=job)

    def get_key_fields(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_metadata_service import (
            check_document_access,
            resolve_key_fields,
        )

        record = self.get_document_record(document_id)
        check_document_access(record)
        return resolve_key_fields(record)

    def get_document_template(self, document_id: str) -> dict[str, Any]:
        """Return stored template extraction output for a document."""
        from fastapi import HTTPException

        from src.features.documents.application.document_metadata_service import check_document_access
        from src.infrastructure.document_databases.neo4j_store import (
            fetch_document_artifact,
            fetch_template_graph,
        )

        record = self.get_document_record(document_id)
        check_document_access(record)

        template = fetch_template_graph(document_id)
        source = "document_template"
        if not template:
            artifact = fetch_document_artifact(document_id, "template_extraction")
            if artifact:
                template = artifact
                source = "document_artifact"
        if not template:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "template_not_found",
                    "message": "Template extraction result was not found for this document.",
                    "document_id": document_id,
                },
            )

        envelope = template.pop("_template", None) if isinstance(template, dict) else None
        artifact_meta = template.pop("_artifact", None) if isinstance(template, dict) else None
        result_location = None
        if isinstance(envelope, dict):
            result_location = envelope.get("result_location")
        if not result_location:
            result_location = f"neo4j://DocumentTemplate/{document_id}" if source == "document_template" else (
                f"neo4j://DocumentArtifact/{document_id}/template_extraction"
            )
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        stored = metadata.get("template_extraction") if isinstance(metadata.get("template_extraction"), dict) else {}
        return {
            "document_id": document_id,
            "source": source,
            "result_location": stored.get("result_location") or result_location,
            "status": stored.get("status"),
            "template": template,
            "graph": envelope,
            "artifact": artifact_meta,
        }

    def start_template_extraction(self, document_id: str) -> dict[str, Any]:
        """Queue the template_extraction processor for an existing document."""
        from fastapi import HTTPException

        from src.features.documents.application.document_metadata_service import check_document_access

        record = self.get_document_record(document_id)
        check_document_access(record)
        # Stale metadata.requires_human_review / security_scan.status can remain after
        # reviewer Allow. Use the same enqueue gate as submit_to_queue.
        if not self._processing_allowed_after_security_review(record):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "human_review_pending",
                    "message": "Template extraction is deferred until human review is decided.",
                    "document_id": document_id,
                },
            )
        js = self._job_store()
        if not js:
            raise HTTPException(status_code=503, detail="PostgreSQL document jobs are not configured")

        processing = dict(record.get("processing") or {})
        processing["template_extraction"] = True
        enabled = self._compute_enabled_processor_types(processing)
        if "template_extraction" not in enabled:
            enabled = list(enabled) + ["template_extraction"]
        processing["enabled_processor_types"] = enabled
        record["processing"] = processing
        try:
            self.get_store().update_document_metadata(
                document_id,
                {"template_extraction_enabled": True, "enabled_processor_types": enabled},
                merge=True,
            )
        except Exception:
            _logger.debug(
                "Could not persist template_extraction flag for document_id=%s",
                document_id,
                exc_info=True,
            )

        payload = self.queue_payload(record)
        payload["processor_type"] = "template_extraction"
        payload["enabled_processor_types"] = enabled
        payload["template_extraction"] = True
        self.publisher.publish_document_job(payload)
        js.ensure_received_from_record(record, batch_id=None)
        try:
            js.mark_processor_republished(document_id, ["template_extraction"])
        except Exception:
            _logger.debug("Could not mark template_extraction republished for %s", document_id, exc_info=True)
        js.update_status_queued(document_id)
        self.get_store().update_document_status(document_id, "queued", queued_at=time.time())
        return {
            "document_id": document_id,
            "queued": True,
            "processor_type": "template_extraction",
            "status": "QUEUED",
            "message": (
                "Template extraction queued. Poll GET /api/v1/documents/{id}/status "
                "then GET /api/v1/documents/{id}/template for the stored result."
            ),
        }

    @staticmethod
    def _document_type_id_from_record(record: dict[str, Any]) -> str | None:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        processing = record.get("processing") if isinstance(record.get("processing"), dict) else {}
        return (
            DocumentReceiverService._normalize_optional_id(record.get("document_type_id"))
            or DocumentReceiverService._normalize_optional_id(metadata.get("document_type_id"))
            or DocumentReceiverService._normalize_optional_id(processing.get("document_type_id"))
        )

    def _allowed_effective_field_names(self, document_type_id: str | None) -> set[str]:
        type_id = self._normalize_optional_id(document_type_id)
        if not type_id:
            return set()
        try:
            from src.features.document_types.application.document_type_service import get_document_type_service

            bundle = get_document_type_service().resolve_effective_fields(type_id)
        except Exception:
            _logger.debug(
                "Could not resolve effective fields for document_type_id=%s",
                type_id,
                exc_info=True,
            )
            return set()
        names: set[str] = set()
        for field in bundle.get("fields") or []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("field_name") or field.get("name") or "").strip()
            if name:
                names.add(name)
        return names

    @staticmethod
    def _validate_known_metadata_fields(fields: dict[str, Any], allowed: set[str]) -> None:
        unknown = sorted(
            {
                str(key).strip()
                for key in (fields or {})
                if str(key or "").strip() and str(key).strip() not in allowed
            }
        )
        if unknown:
            raise HTTPException(
                status_code=400,
                detail={"message": "Unknown metadata fields", "fields": unknown},
            )

    def update_key_fields(
        self,
        document_id: str,
        fields: dict[str, Any],
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(fields, dict) or not fields:
            raise HTTPException(status_code=422, detail="fields must contain at least one key.")

        from src.application.consumer_api.context import get_current_user_from_context, get_request_id
        from src.features.documents.application.document_metadata_service import (
            check_document_access,
            fetch_key_field_artifact_from_neo4j,
            resolve_key_fields,
        )
        from src.infrastructure.document_databases.neo4j_store import store_document_graph

        record = self.get_document_record(document_id)
        check_document_access(record)

        document_type_id = self._document_type_id_from_record(record)
        allowed_fields = self._allowed_effective_field_names(document_type_id)
        # Reject Swagger placeholders / unknown keys before any Neo4j or audit writes.
        self._validate_known_metadata_fields(fields, allowed_fields)

        actor = get_current_user_from_context()
        repository_id = str(record.get("repository_id") or (record.get("metadata") or {}).get("repository_id") or "").strip() or None
        if actor is not None and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous":
            if repository_id:
                from src.features.users.application.user_service import get_platform_security_service

                get_platform_security_service().check_upload_access(actor, str(repository_id))

        existing = fetch_key_field_artifact_from_neo4j(document_id) or {}
        artifact_payload = existing.get("artifact_payload") if isinstance(existing.get("artifact_payload"), dict) else {}
        existing_fields = artifact_payload.get("fields") if isinstance(artifact_payload, dict) else {}
        if not isinstance(existing_fields, dict):
            existing_fields = {}

        # Keep previously stored valid fields; drop unknown keys already persisted by older bugs.
        updated_fields: dict[str, Any] = {
            str(k): v for k, v in existing_fields.items() if str(k).strip() in allowed_fields
        }
        now_iso = datetime.now(timezone.utc).isoformat()
        updated_by = (
            str(actor.user_id)
            if actor is not None and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous"
            else "system"
        )
        history_entries: list[dict[str, Any]] = []
        for field_name_raw, new_value in fields.items():
            field_name = str(field_name_raw or "").strip()
            if not field_name or field_name not in allowed_fields:
                continue
            prior = updated_fields.get(field_name)
            if isinstance(prior, dict):
                old_value = prior.get("value")
                prior_confidence = prior.get("confidence")
            else:
                old_value = prior
                prior_confidence = None
            if old_value == new_value:
                continue
            new_entry = {
                "value": new_value,
                "confidence": prior_confidence,
                "source": "manual_override",
                "updated_at": now_iso,
                "updated_by": updated_by,
                "change_reason": reason,
            }
            updated_fields[field_name] = new_entry
            history_entries.append(
                {
                    "field_name": field_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "updated_at": now_iso,
                    "updated_by": updated_by,
                    "change_reason": reason,
                }
            )

        # Second safety check: never audit or persist unknown field names.
        history_entries = [
            entry for entry in history_entries if str(entry.get("field_name") or "").strip() in allowed_fields
        ]
        if not history_entries:
            return resolve_key_fields(record)

        payload = {
            **(artifact_payload if isinstance(artifact_payload, dict) else {}),
            "document_type_id": (
                existing.get("document_type_id")
                or document_type_id
                or record.get("document_type_id")
                or (record.get("metadata") or {}).get("document_type_id")
                or (record.get("processing") or {}).get("document_type_id")
            ),
            "document_type_name": str(
                record.get("document_type_name")
                or (record.get("metadata") or {}).get("document_type_name")
                or (record.get("processing") or {}).get("document_type_name")
                or ""
            ).strip()
            or None,
            "source": "manual_override",
            "fields": updated_fields,
        }
        existing_history = payload.get("history")
        if not isinstance(existing_history, list):
            existing_history = []
        cleaned_history = [
            entry
            for entry in existing_history
            if isinstance(entry, dict) and str(entry.get("field_name") or "").strip() in allowed_fields
        ]
        payload["history"] = cleaned_history + history_entries
        store_document_graph(
            document_id=document_id,
            repository_id=repository_id,
            processor_type="key_field_extraction",
            payload=payload,
        )

        try:
            from src.features.audit.application.security_audit_service import AuditService
            from src.features.users.infrastructure.user_repository import get_platform_security_store

            sec_store = get_platform_security_store()
            if sec_store:
                audit_service = AuditService(sec_store)
                for change in history_entries:
                    field_name = str(change.get("field_name") or "").strip()
                    if field_name not in allowed_fields:
                        continue
                    audit_service.record(
                        event_category="key_field_management",
                        event_type="key_field.updated",
                        action="update",
                        outcome="success",
                        actor=actor if actor and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous" else None,
                        repository_id=repository_id,
                        document_id=document_id,
                        resource_type="key_field",
                        resource_id=field_name,
                        old_value={"field_name": field_name, "value": change.get("old_value")},
                        new_value={"field_name": field_name, "value": change.get("new_value")},
                        change_reason=reason,
                        request_id=get_request_id(),
                    )
        except Exception:
            _logger.exception("Failed to write key field audit events for document_id=%s", document_id)

        # Phase 5: review audit trail + automatic validation re-run after metadata edits.
        try:
            from src.features.human_review.application.review_service import get_document_review_service

            get_document_review_service().record_metadata_edits(
                document_id=document_id,
                changes=history_entries,
                modified_by=updated_by,
                reason=reason,
                allowed_field_names=allowed_fields,
            )
        except Exception:
            _logger.exception("Failed to record review metadata edits for document_id=%s", document_id)

        refreshed = self.get_document_record(document_id)
        return resolve_key_fields(refreshed)

    def get_key_fields_history(self, document_id: str, *, limit: int = 200) -> dict[str, Any]:
        from src.features.documents.application.document_metadata_service import check_document_access

        record = self.get_document_record(document_id)
        check_document_access(record)
        repository_id = str(record.get("repository_id") or (record.get("metadata") or {}).get("repository_id") or "").strip() or None
        allowed_fields = self._allowed_effective_field_names(self._document_type_id_from_record(record))
        history: list[dict[str, Any]] = []
        try:
            from src.features.users.infrastructure.user_repository import get_platform_security_store

            sec_store = get_platform_security_store()
            if sec_store:
                rows = sec_store.list_audit_events(
                    repository_id=repository_id,
                    category="key_field_management",
                    event_type="key_field.updated",
                    document_id=document_id,
                    limit=max(limit * 3, limit),
                    ascending=True,
                )
                history = [
                    {
                        "audit_id": row.get("audit_id"),
                        "event_time": row.get("event_time"),
                        "document_id": row.get("document_id"),
                        "field_name": ((row.get("new_value_json") or {}).get("field_name") or (row.get("old_value_json") or {}).get("field_name")),
                        "old_value": (row.get("old_value_json") or {}).get("value"),
                        "new_value": (row.get("new_value_json") or {}).get("value"),
                        "actor_user_id": row.get("actor_user_id"),
                        "change_reason": row.get("change_reason"),
                    }
                    for row in rows
                ]
        except Exception:
            _logger.exception("Failed to load key field history for document_id=%s", document_id)
        if allowed_fields:
            history = [
                entry
                for entry in history
                if str(entry.get("field_name") or "").strip() in allowed_fields
            ]
        else:
            history = [
                entry
                for entry in history
                if not str(entry.get("field_name") or "").strip().lower().startswith("additionalprop")
            ]
        history = history[: max(1, limit)]
        return {"document_id": document_id, "history": history, "count": len(history)}

    def get_document_metadata_bundle(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_metadata_service import (
            check_document_access,
            resolve_document_metadata_bundle,
        )

        record = self.get_document_record(document_id)
        check_document_access(record)
        job = self._job_store().get_by_document_id(document_id) if self._job_store() else None
        return resolve_document_metadata_bundle(record, job=job)

    def get_document_validation(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_metadata_service import (
            check_document_access,
            resolve_validation_bundle,
        )

        record = self.get_document_record(document_id)
        check_document_access(record)
        job = self._job_store().get_by_document_id(document_id) if self._job_store() else None
        validation = resolve_validation_bundle(record, job=job)
        return {
            "document_id": document_id,
            "status": validation.get("status"),
            "document_status": validation.get("document_status"),
            "missing_required_fields": validation.get("missing_required_fields") or [],
            "invalid_fields": validation.get("invalid_fields") or [],
            "low_confidence_fields": validation.get("low_confidence_fields") or [],
            "confidence_threshold": validation.get("confidence_threshold"),
            "source": validation.get("source"),
            "completed_at": validation.get("completed_at"),
        }

    def get_document_chunks(
        self,
        document_id: str,
        *,
        limit: int = 500,
        offset: int = 0,
        include_text: bool = True,
        include_vector: bool = True,
    ) -> dict[str, Any]:
        """Return Weaviate chunks (text and optional vectors) for a document."""
        from src.features.documents.application.document_chunks import list_document_chunks
        from src.features.documents.application.document_metadata_service import check_document_access

        record = self.get_document_record(document_id)
        check_document_access(record)

        payload = list_document_chunks(
            record,
            limit=max(1, min(limit, 2000)),
            offset=max(0, offset),
            include_text=include_text,
            include_vector=include_vector,
        )
        document_info = self.get_document_info(document_id)
        payload["document_info"] = document_info
        return payload

    def get_document_extraction(self, document_id: str) -> dict[str, Any]:
        """Return extracted pages/slides/transcript sections for CIH."""
        from fastapi import HTTPException

        from src.features.documents.application.document_chunks import list_document_chunks
        from src.features.documents.application.document_extraction_cache import (
            build_extraction_payload,
            live_extraction_reload_enabled,
            load_extraction_sidecar,
            sections_from_blocks,
            sections_from_chunk_rows,
        )
        from src.features.documents.application.document_metadata_service import check_document_access
        from src.features.document_processing.loaders.document_text import load_document_blocks
        from src.features.document_processing.media.cih_media import is_media_suffix, load_transcript_sidecar
        from src.infrastructure.document_databases.neo4j_store import fetch_document_artifact

        record = self.get_document_record(document_id)
        check_document_access(record)
        path, filename, _media_type = self.resolve_original_file(document_id)
        suffix = Path(filename).suffix.lower() or path.suffix.lower()

        sections: list[dict[str, Any]] = []
        source = "document_blocks"
        cached = load_extraction_sidecar(path)
        if cached and cached.get("sections"):
            return cached
        if is_media_suffix(suffix):
            artifact = fetch_document_artifact(document_id, "media_transcription") or load_transcript_sidecar(path)
            if artifact:
                source = "media_transcription"
                for seg in artifact.get("segments") or []:
                    if not isinstance(seg, dict):
                        continue
                    text = str(seg.get("text") or "").strip()
                    if not text:
                        continue
                    sections.append(
                        {
                            "index": seg.get("index") or (len(sections) + 1),
                            "page": seg.get("index") or (len(sections) + 1),
                            "text": text,
                            "start": seg.get("start"),
                            "end": seg.get("end"),
                            "kind": "transcript",
                        }
                    )
                if not sections and str(artifact.get("full_text") or "").strip():
                    sections.append(
                        {
                            "index": 1,
                            "page": 1,
                            "text": str(artifact.get("full_text")).strip(),
                            "kind": "transcript",
                        }
                    )
            # Media extraction is transcript-backed only. Do not fall through to
            # load_document_blocks (that raises ValueError → opaque 500).
            if not sections:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Extracted content not found. Wait until processing status is COMPLETED "
                        "and media_transcription has finished."
                    ),
                )

        if not sections and not live_extraction_reload_enabled():
            try:
                chunk_payload = list_document_chunks(
                    record,
                    limit=5000,
                    offset=0,
                    include_text=True,
                    include_vector=False,
                )
                chunk_sections = sections_from_chunk_rows(
                    chunk_payload.get("chunks") or [],
                    suffix=suffix,
                )
                if chunk_sections:
                    return build_extraction_payload(
                        document_id=document_id,
                        filename=filename,
                        suffix=suffix,
                        sections=chunk_sections,
                        source="indexed_chunks",
                    )
            except HTTPException:
                pass
            except Exception:
                _logger.debug(
                    "Chunk-backed extraction fallback failed for document_id=%s",
                    document_id,
                    exc_info=True,
                )

        if not sections:
            try:
                blocks, meta = load_document_blocks(path, mask_sensitive=False, document_id=document_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            sections = sections_from_blocks(blocks, suffix=suffix)
            source = str(meta.get("file_extension") or suffix.lstrip(".") or source)

        return build_extraction_payload(
            document_id=document_id,
            filename=filename,
            suffix=suffix,
            sections=sections,
            source=source,
        )

    def get_document_transcript(self, document_id: str) -> dict[str, Any]:
        """Return timed transcript for audio/video; empty payload for other types."""
        from fastapi import HTTPException

        from src.features.documents.application.document_metadata_service import check_document_access
        from src.features.document_processing.media.cih_media import is_media_suffix, load_transcript_sidecar
        from src.infrastructure.document_databases.neo4j_store import fetch_document_artifact

        record = self.get_document_record(document_id)
        check_document_access(record)
        path, filename, _media_type = self.resolve_original_file(document_id)
        suffix = Path(filename).suffix.lower() or path.suffix.lower()
        if not is_media_suffix(suffix):
            return {
                "document_id": document_id,
                "original_file_name": filename,
                "available": False,
                "reason": "not_audio_or_video",
                "segments": [],
                "full_text": "",
            }

        artifact = fetch_document_artifact(document_id, "media_transcription")
        if not artifact:
            artifact = load_transcript_sidecar(path)
        if not artifact:
            raise HTTPException(
                status_code=404,
                detail="Transcript not found. Wait until media_transcription status is COMPLETED.",
            )
        return {
            "document_id": document_id,
            "original_file_name": filename,
            "available": True,
            "media_type": artifact.get("media_type"),
            "provider": artifact.get("provider"),
            "model": artifact.get("model"),
            "task": artifact.get("task"),
            "language": artifact.get("language"),
            "detected_language": artifact.get("detected_language"),
            "output_language": artifact.get("output_language"),
            "segment_count": artifact.get("segment_count") or len(artifact.get("segments") or []),
            "segments": artifact.get("segments") or [],
            "full_text": artifact.get("full_text") or "",
        }

    def get_document_record(self, document_id: str) -> dict[str, Any]:
        record = self.get_store().get_document(document_id)
        if record:
            return record
        job = self._job_store().get_by_document_id(document_id) if self._job_store() else None
        if job:
            restored = self._try_rehydrate_document_from_job(document_id, job)
            if restored:
                return restored
        restored = self._try_rehydrate_document_from_disk(document_id)
        if restored:
            return restored
        raise HTTPException(status_code=404, detail="Document not found")

    @staticmethod
    def queue_payload(record: dict[str, Any]) -> dict[str, Any]:
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
        # Scheduler also rebuilds document_path from document_name; include the intake
        # path so processors can resolve the file even if document_name is atypical.
        repository_path = record.get("repository_path") or (metadata.get("repository_path") if isinstance(metadata, dict) else None)
        if repository_path:
            payload["document_path"] = str(repository_path)
        if resolved_document_type_id:
            payload["document_type_id"] = resolved_document_type_id
        if resolved_document_type_name:
            payload["document_type_name"] = resolved_document_type_name
        repository_id = (
            DocumentReceiverService._normalize_optional_id(record.get("repository_id"))
            or DocumentReceiverService._normalize_optional_id(metadata.get("repository_id"))
            or DocumentReceiverService._normalize_optional_id(processing.get("repository_id"))
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
        # Always emit a concrete bool so processor schema validation never sees null.
        payload["strict_key_field_page_scope"] = bool(processing.get("strict_key_field_page_scope") or False)
        # Nested repository_settings for ProcessRequest (flat fields remain for backward compat).
        repo_settings: dict[str, Any] = {}
        for src, dest in (
            ("chunk_size", "chunk_size"),
            ("chunk_overlap_sentences", "chunk_overlap_sentences"),
            ("chunking_strategy", "chunking_strategy"),
            ("chunking_config", "chunking_config"),
            ("citation_retainment", "citation_retainment"),
            ("model_name", "embedding_model_name"),
            ("model_dir", "embedding_model_dir"),
            ("extraction_model", "extraction_model"),
            ("document_type_id", "document_type_id"),
            ("document_type_name", "document_type_name"),
            ("metadata_fields", "metadata_fields"),
            ("key_fields", "key_fields"),
            ("strict_key_field_page_scope", "strict_key_field_page_scope"),
        ):
            if processing.get(src) is not None:
                repo_settings[dest] = processing[src]
        if repo_settings:
            payload["repository_settings"] = repo_settings
        return payload

    @staticmethod
    def _attach_metadata_fields(processing: dict[str, Any]) -> dict[str, Any]:
        updated = dict(processing or {})
        if updated.get("metadata_extraction") and updated.get("document_type_id"):
            try:
                from src.features.document_types.application.processing_fields import resolve_metadata_fields_for_type

                updated["metadata_fields"] = resolve_metadata_fields_for_type(str(updated["document_type_id"]))
            except Exception:
                _logger.debug(
                    "Could not resolve metadata fields for document_type_id=%s",
                    updated.get("document_type_id"),
                    exc_info=True,
                )
        return updated

    @staticmethod
    def _attach_key_fields(processing: dict[str, Any]) -> dict[str, Any]:
        updated = dict(processing or {})
        document_type_id = str(updated.get("document_type_id") or "").strip() or None
        if document_type_id and not updated.get("effective_fields"):
            try:
                from src.features.document_types.application.document_type_service import get_document_type_service

                bundle = get_document_type_service().resolve_effective_fields(document_type_id)
                fields = bundle.get("fields") if isinstance(bundle, dict) else []
                if isinstance(fields, list):
                    updated["effective_fields"] = fields
            except Exception:
                _logger.debug(
                    "Could not resolve effective fields for document_type_id=%s",
                    document_type_id,
                    exc_info=True,
                )
        extraction_enabled = bool(
            updated.get("key_field_extraction")
            if updated.get("key_field_extraction") is not None
            else updated.get("key_field_extraction_enabled")
        )
        if not extraction_enabled:
            return updated
        if updated.get("key_fields"):
            return updated
        if not document_type_id:
            return updated
        try:
            from src.features.document_types.application.processing_key_fields import resolve_key_fields_for_type

            updated["key_fields"] = resolve_key_fields_for_type(document_type_id)
        except Exception:
            _logger.debug(
                "Could not resolve key fields for document_type_id=%s",
                document_type_id,
                exc_info=True,
            )
        return updated

    def _persist_document_type_metadata(self, record: dict[str, Any]) -> None:
        """Persist document type identity and effective fields on document metadata."""
        processing = dict(record.get("processing") or {})
        metadata = dict(record.get("metadata") or {})
        document_type_id = (
            str(processing.get("document_type_id") or "").strip()
            or str(record.get("document_type_id") or "").strip()
            or str(metadata.get("document_type_id") or "").strip()
            or None
        )
        document_type_name = (
            str(processing.get("document_type_name") or "").strip()
            or str(record.get("document_type_name") or "").strip()
            or str(metadata.get("document_type_name") or "").strip()
            or None
        )
        effective_fields = processing.get("effective_fields") or metadata.get("effective_fields") or []
        if not isinstance(effective_fields, list):
            effective_fields = []
        if document_type_id:
            metadata["document_type_id"] = document_type_id
            record["document_type_id"] = document_type_id
        if document_type_name:
            metadata["document_type_name"] = document_type_name
            record["document_type_name"] = document_type_name
        if effective_fields:
            metadata["effective_fields"] = effective_fields
            processing["effective_fields"] = effective_fields
        record["metadata"] = metadata
        record["processing"] = processing

    @staticmethod
    def _processing_allowed_after_security_review(record: dict[str, Any]) -> bool:
        """Block processor queue until reviewer Allow/Mask when DLP flagged human review."""
        from src.features.security.compliance.compliance_validator import (
            processing_allowed_after_security_review,
        )

        return processing_allowed_after_security_review(record)
    @staticmethod
    def _register_human_review_queue_entry(
        record: dict[str, Any],
        security_scan: dict[str, Any] | None,
    ) -> str | None:
        document_id = str(record.get("document_id") or "").strip()
        if not document_id or not isinstance(security_scan, dict):
            return None
        try:
            from src.features.security.review.human_review_queue import ensure_human_review_queue_entry

            review_id = ensure_human_review_queue_entry(
                document_id=document_id,
                document_name=str(record.get("document_name") or record.get("original_file_name") or document_id),
                security_scan=security_scan,
                document_type_id=str(record.get("document_type_id") or "").strip() or None,
                document_type_name=str(record.get("document_type_name") or "").strip() or None,
            )
        except Exception:
            _logger.exception("Failed to register human review queue entry for document_id=%s", document_id)
            return None
        if review_id:
            metadata = dict(record.get("metadata") or {})
            metadata["review_id"] = review_id
            metadata["requires_human_review"] = True
            security_meta = dict(metadata.get("security_scan") or {})
            security_meta["review_id"] = review_id
            metadata["security_scan"] = security_meta
            record["metadata"] = metadata
        return review_id

    def submit_to_queue(self, record: dict[str, Any]) -> None:
        if not self._processing_allowed_after_security_review(record):
            document_id = str(record.get("document_id") or "")
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            security_scan = metadata.get("security_scan") if isinstance(metadata.get("security_scan"), dict) else {}
            self._register_human_review_queue_entry(record, security_scan)
            if document_id:
                self.get_store().update_document_status(document_id, "human_review")
            _pipeline_trace(
                "queue_submission_deferred",
                document_id=document_id or None,
                current_status=str(record.get("status") or "human_review"),
                next_status="human_review",
                reason="pending_human_review",
            )
            return
        repository_id = self._preserve_repository_id(record, record.get("repository_id"))
        prior_processing = dict(record.get("processing") or {})
        prior_enabled = list(prior_processing.get("enabled_processor_types") or [])
        _pipeline_trace(
            "repository_id_received",
            document_id=str(record.get("document_id") or ""),
            repository_id=repository_id,
            prior_enabled_processor_types=prior_enabled or None,
            prior_template_extraction=prior_processing.get("template_extraction"),
            prior_key_field_extraction_enabled=prior_processing.get("key_field_extraction_enabled"),
        )
        resolved_type_id, resolved_type_name = self._resolve_record_document_type(record)
        if not repository_id and resolved_type_id:
            repository_id = self._preserve_repository_id(
                record,
                self._repository_id_from_document_type(resolved_type_id),
            )

        settings_loaded = False
        stored_settings: dict[str, Any] = {}
        if repository_id:
            try:
                from src.features.repositories.application.repository_service import get_repository_service

                # RepositoryService → SettingsResolver (do not read RepositoryStore here).
                repo_service = get_repository_service()
                try:
                    from src.features.repositories.infrastructure.repository_repository import get_repository_store

                    store = get_repository_store()
                    if store is not None:
                        stored_settings = dict(store.get_settings(str(repository_id)) or {})
                except Exception:
                    stored_settings = {}

                resolved = repo_service.resolve_settings(str(repository_id))
                hints = dict(resolved.get("processing_hints") or {})
                if hints:
                    # Preserve upload-selected document type over repository default type.
                    if prior_processing.get("document_type_id"):
                        hints["document_type_id"] = prior_processing.get("document_type_id")
                    if prior_processing.get("document_type_name"):
                        hints["document_type_name"] = prior_processing.get("document_type_name")
                    if prior_processing.get("effective_fields"):
                        hints["effective_fields"] = prior_processing.get("effective_fields")
                    if prior_processing.get("key_fields"):
                        hints["key_fields"] = prior_processing.get("key_fields")
                    record["processing"] = hints
                    settings_loaded = True
                else:
                    record["processing"] = prior_processing
                resolved_settings = (
                    resolved.get("settings") if isinstance(resolved.get("settings"), dict) else {}
                )
                if resolved_settings:
                    record["effective_settings"] = dict(resolved_settings)
                _pipeline_trace(
                    "repository_settings_loaded",
                    document_id=str(record.get("document_id") or ""),
                    repository_id=repository_id,
                    repository_template_extraction=resolved_settings.get("template_extraction"),
                    processing_hints_template_extraction=hints.get("template_extraction"),
                    key_field_extraction_enabled=resolved_settings.get("key_field_extraction_enabled"),
                    enabled_processor_types_before_scheduling=list(
                        (record.get("processing") or {}).get("enabled_processor_types") or []
                    ),
                )
            except Exception as exc:
                _logger.error(
                    "Could not refresh processing hints for repository_id=%s document_id=%s: %s",
                    repository_id,
                    record.get("document_id"),
                    exc,
                    exc_info=True,
                )
                record["processing"] = prior_processing
                _pipeline_trace(
                    "repository_settings_lookup_failure",
                    document_id=str(record.get("document_id") or ""),
                    repository_id=repository_id,
                    reason=str(exc),
                    prior_enabled_processor_types=prior_enabled or None,
                )
        else:
            _logger.error(
                "submit_to_queue missing repository_id for document_id=%s; "
                "key_field_extraction cannot be scheduled from repository settings",
                record.get("document_id"),
            )
            _pipeline_trace(
                "repository_settings_lookup_failure",
                document_id=str(record.get("document_id") or ""),
                reason="missing_repository_id",
                prior_enabled_processor_types=prior_enabled or None,
            )

        if prior_processing.get("document_type_id") and not resolved_type_id:
            resolved_type_id = str(prior_processing["document_type_id"])
        if prior_processing.get("document_type_name") and not resolved_type_name:
            resolved_type_name = str(prior_processing["document_type_name"])

        self._apply_resolved_document_type(
            record,
            document_type_id=resolved_type_id,
            document_type_name=resolved_type_name,
        )
        _pipeline_trace(
            "effective_document_type_resolved",
            document_id=str(record.get("document_id") or ""),
            repository_id=repository_id,
            effective_document_type_id=resolved_type_id,
            effective_document_type_name=resolved_type_name,
            settings_loaded=settings_loaded,
        )
        if resolved_type_id:
            try:
                self._upsert_document_instance(
                    document_id=str(record["document_id"]),
                    document_type_id=str(resolved_type_id),
                    repository_id=str(repository_id) if repository_id else None,
                    tenant_id=str(record.get("tenant_id")) if record.get("tenant_id") else None,
                )
            except Exception:
                _logger.debug("Could not upsert document_instance for document_id=%s", record.get("document_id"), exc_info=True)

        processing = dict(record.get("processing") or {})
        if resolved_type_id:
            processing["document_type_id"] = resolved_type_id
        if resolved_type_name:
            processing["document_type_name"] = resolved_type_name
        processing = self._apply_extraction_enablement(
            processing,
            stored_settings=stored_settings,
        )
        processing = self._attach_metadata_fields(processing)
        processing = self._attach_key_fields(processing)
        # Always recompute from repository flags so document-type binding cannot drop
        # optional processors (key_field_extraction, template_extraction, …).
        enabled = self._compute_enabled_processor_types(processing)
        enabled = self._ensure_cih_media_processor(record, enabled, processing)
        processing["enabled_processor_types"] = enabled
        if repository_id:
            processing["repository_id"] = repository_id
        record["processing"] = processing
        self._preserve_repository_id(record, repository_id)
        self._persist_document_type_metadata(record)
        _pipeline_trace(
            "enabled_processor_types_after_scheduling",
            document_id=str(record.get("document_id") or ""),
            repository_id=repository_id,
            effective_document_type_id=resolved_type_id,
            template_extraction=processing.get("template_extraction"),
            key_field_extraction_enabled=processing.get("key_field_extraction_enabled"),
            key_field_extraction=processing.get("key_field_extraction"),
            enabled_processor_types=enabled,
            key_fields_count=len(processing.get("key_fields") or []),
        )
        try:
            meta_patch: dict[str, Any] = {
                "enabled_processor_types": list(enabled),
                "template_extraction_enabled": self._processing_flag_enabled(processing, "template_extraction"),
            }
            metadata = record.get("metadata") or {}
            for key in ("document_type_id", "document_type_name", "effective_fields", "repository_id"):
                if metadata.get(key) is not None:
                    meta_patch[key] = metadata[key]
            if meta_patch:
                self.get_store().update_document_metadata(
                    record["document_id"],
                    meta_patch,
                    merge=True,
                )
            patch_fields: dict[str, Any] = {}
            if record.get("processing"):
                patch_fields["processing"] = record["processing"]
            if record.get("document_type_id"):
                patch_fields["document_type_id"] = record["document_type_id"]
            if record.get("document_type_name"):
                patch_fields["document_type_name"] = record["document_type_name"]
            if repository_id:
                patch_fields["repository_id"] = repository_id
            if patch_fields:
                self.get_store().patch_document(record["document_id"], patch_fields)
        except Exception:
            _logger.debug(
                "Could not persist document type metadata for document_id=%s",
                record.get("document_id"),
                exc_info=True,
            )
        if processing.get("metadata_extraction") and not processing.get("document_type_id"):
            _logger.warning(
                "metadata_extraction is enabled for document_id=%s but document_type_id is missing; "
                "field extraction will be limited.",
                record.get("document_id"),
            )
        if (
            self._processing_flag_enabled(processing, "key_field_extraction_enabled", "key_field_extraction")
            and "key_field_extraction" not in enabled
        ):
            _logger.error(
                "key_field_extraction enabled for document_id=%s repository_id=%s but missing from "
                "enabled_processor_types=%s; forcing inclusion",
                record.get("document_id"),
                repository_id,
                enabled,
            )
            enabled = list(enabled) + ["key_field_extraction"]
            processing["enabled_processor_types"] = enabled
            record["processing"] = processing
        if self._processing_flag_enabled(processing, "template_extraction") and "template_extraction" not in enabled:
            _logger.error(
                "template_extraction enabled for document_id=%s repository_id=%s but missing from "
                "enabled_processor_types=%s; forcing inclusion",
                record.get("document_id"),
                repository_id,
                enabled,
            )
            enabled = list(enabled) + ["template_extraction"]
            processing["enabled_processor_types"] = enabled
            record["processing"] = processing
            _pipeline_trace(
                "template_extraction_forced_into_enabled_processor_types",
                document_id=str(record.get("document_id") or ""),
                repository_id=repository_id,
                template_extraction=True,
                enabled_processor_types=enabled,
            )
        js = self._job_store()
        if js:
            js.seed_processor_plan(
                record["document_id"],
                enabled,
                context={
                    "repository_id": repository_id,
                    "document_type_id": resolved_type_id,
                    "document_type_name": resolved_type_name,
                    "template_extraction": self._processing_flag_enabled(processing, "template_extraction"),
                    "key_field_extraction": processing.get("key_field_extraction"),
                    "key_field_extraction_enabled": processing.get("key_field_extraction_enabled"),
                    "key_fields": processing.get("key_fields") or [],
                    "effective_fields": processing.get("effective_fields") or [],
                    "collection_name": record.get("collection_name"),
                    "tenant_id": record.get("tenant_id"),
                    # Snapshot of SettingsResolver processing_hints for recovery / diagnostics.
                    "processing_hints": {
                        key: processing.get(key)
                        for key in (
                            "chunk_size",
                            "chunk_overlap_sentences",
                            "chunking_strategy",
                            "chunking_config",
                            "model_name",
                            "model_dir",
                            "citation_retainment",
                            "enabled_processor_types",
                            "retrieval_search_mode",
                            "template_extraction",
                            "key_field_extraction_enabled",
                            "metadata_extraction",
                            "reference_document_extraction",
                            "conversion_for_rendering",
                        )
                        if processing.get(key) is not None
                    },
                    "chunk_size": processing.get("chunk_size"),
                    "chunk_overlap_sentences": processing.get("chunk_overlap_sentences"),
                    "model_name": processing.get("model_name"),
                    "model_dir": processing.get("model_dir"),
                    "chunking_strategy": processing.get("chunking_strategy"),
                },
            )
            row = js.get_by_document_id(str(record["document_id"]))
            _pipeline_trace(
                "processor_plan_seeded",
                document_id=str(record["document_id"]),
                job_id=str((row or {}).get("job_id") or ""),
                current_status=str((row or {}).get("status") or "RECEIVED"),
                next_status=str((row or {}).get("status") or "RECEIVED"),
                repository_id=repository_id,
                effective_document_type_id=resolved_type_id,
                template_extraction=self._processing_flag_enabled(processing, "template_extraction"),
                key_field_extraction_enabled=processing.get("key_field_extraction_enabled"),
                enabled_processor_types=enabled,
                processor_queue_contents=enabled,
                scheduling_metadata_populated=bool((row or {}).get("scheduling_metadata")),
            )
        from src.features.documents.application.job_republish import initial_processor_types_for_upload

        initial_types = initial_processor_types_for_upload(enabled)
        for processor_type in initial_types:
            payload = self.queue_payload(record)
            payload["processor_type"] = processor_type
            payload["enabled_processor_types"] = enabled
            payload["template_extraction"] = self._processing_flag_enabled(processing, "template_extraction")
            _pipeline_trace(
                "rabbitmq_publish_attempt",
                document_id=str(record["document_id"]),
                current_status="RECEIVED",
                next_status="QUEUED",
                processor_type=processor_type,
                template_extraction=self._processing_flag_enabled(processing, "template_extraction"),
                enabled_processor_types=enabled,
                queue_name=self.publisher.queue_name,
            )
            self.publisher.publish_document_job(payload)
            _pipeline_trace(
                "rabbitmq_publish_success",
                document_id=str(record["document_id"]),
                current_status="RECEIVED",
                next_status="QUEUED",
                processor_type=processor_type,
                template_extraction=self._processing_flag_enabled(processing, "template_extraction"),
                enabled_processor_types=enabled,
                queue_name=self.publisher.queue_name,
            )
            _logger.info(
                "Queued document_id=%s for initial processor_type=%s (enabled plan=%s)",
                record["document_id"],
                processor_type,
                enabled,
            )
        if js:
            js.update_status_queued(record["document_id"])
            row = js.get_by_document_id(str(record["document_id"]))
            _pipeline_trace(
                "job_queued",
                document_id=str(record["document_id"]),
                job_id=str((row or {}).get("job_id") or ""),
                current_status="RECEIVED",
                next_status=str((row or {}).get("status") or "QUEUED"),
                reason="update_status_queued",
            )
        self.get_store().update_document_status(record["document_id"], "queued", queued_at=time.time())

    def republish_processor_types(self, document_id: str, processor_types: list[str]) -> None:
        """Re-queue specific processor jobs (scheduler recovery for multi-processor plans)."""
        if not processor_types:
            return
        record = self.get_store().get_document(document_id)
        if not record:
            return
        repository_id = self._preserve_repository_id(record, record.get("repository_id"))
        prior_processing = dict(record.get("processing") or {})
        if repository_id:
            try:
                from src.features.repositories.application.repository_settings_service import resolve_repository_context

                resolved = resolve_repository_context(str(repository_id))
                hints = dict(resolved.get("processing_hints") or prior_processing)
                if prior_processing.get("document_type_id"):
                    hints["document_type_id"] = prior_processing.get("document_type_id")
                if prior_processing.get("document_type_name"):
                    hints["document_type_name"] = prior_processing.get("document_type_name")
                if prior_processing.get("effective_fields"):
                    hints["effective_fields"] = prior_processing.get("effective_fields")
                if prior_processing.get("key_fields"):
                    hints["key_fields"] = prior_processing.get("key_fields")
                record["processing"] = hints
            except Exception:
                pass
        resolved_type_id, resolved_type_name = self._resolve_record_document_type(record)
        self._apply_resolved_document_type(
            record,
            document_type_id=resolved_type_id,
            document_type_name=resolved_type_name,
        )
        processing = self._attach_metadata_fields(record.get("processing") or {})
        processing = self._attach_key_fields(processing)
        enabled = self._compute_enabled_processor_types(processing)
        enabled = self._ensure_cih_media_processor(record, enabled, processing)
        processing["enabled_processor_types"] = enabled
        if repository_id:
            processing["repository_id"] = repository_id
        record["processing"] = processing
        for processor_type in processor_types:
            if processor_type not in enabled:
                continue
            payload = self.queue_payload(record)
            payload["processor_type"] = processor_type
            payload["enabled_processor_types"] = enabled
            self.publisher.publish_document_job(payload)
            _logger.info(
                "Republished document_id=%s for processor_type=%s (recovery)",
                document_id,
                processor_type,
            )

        js = self._job_store()
        if js:
            js.mark_processor_republished(document_id, processor_types)
            js.heartbeat(document_id, notes=f"republished:{','.join(processor_types)}")

    def upsert_repository_config(self, body: RepositoryConfigRequest) -> dict[str, Any]:
        return self.get_store().upsert_repository_config(body.repository_type, body.config)

    def get_repository_config(self) -> dict[str, Any]:
        return self.get_store().get_repository_config(
            self._live_settings().repository_type,
            self._live_settings().upload_dir,
        )

    def _apply_repository_context(
        self,
        record: dict[str, Any],
        *,
        repository_id: str | None,
        collection_name: str | None,
        tenant_id: str | None,
        document_type_id: str | None = None,
        link_document: bool = True,
        audit_upload: bool = True,
    ) -> None:
        repository_id = self._normalize_optional_id(repository_id)
        document_type_id = self._normalize_optional_id(document_type_id)
        if not repository_id and document_type_id:
            repository_id = self._repository_id_from_document_type(document_type_id)
            _pipeline_trace(
                "repository_id_recovered_from_document_type",
                document_id=str(record.get("document_id") or ""),
                repository_id=repository_id,
                effective_document_type_id=document_type_id,
            )
        if not repository_id:
            if document_type_id:
                self._bind_document_type(record, repository_id=None, document_type_id=document_type_id)
            _pipeline_trace(
                "repository_context_skipped",
                document_id=str(record.get("document_id") or ""),
                reason="missing_repository_id",
                effective_document_type_id=document_type_id,
            )
            return
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.users.application.user_service import get_platform_security_service
        from src.features.repositories.application.repository_service import get_repository_service

        actor = get_current_user_from_context()
        from src.features.authentication.domain.authentication_exceptions import AuthenticationError
        from src.features.users.application.user_service import get_platform_security_service

        if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
            raise AuthenticationError("Authentication required")
        get_platform_security_service().check_upload_access(actor, repository_id)

        repo = get_repository_service().validate_repository_active(repository_id)
        record["repository_id"] = repository_id
        clean_collection = self._normalize_optional_id(collection_name)
        clean_tenant = self._normalize_optional_id(tenant_id)
        record["collection_name"] = clean_collection or repo["weaviate_collection"]
        record["tenant_id"] = clean_tenant if clean_tenant is not None else repo.get("default_tenant_id")
        # Effective settings from SettingsResolver (via validate_repository_active → resolve_settings).
        effective = repo.get("settings") if isinstance(repo.get("settings"), dict) else {}
        if effective:
            record["effective_settings"] = dict(effective)
            record.setdefault("metadata", {})["effective_settings"] = {
                key: effective[key]
                for key in (
                    "chunk_size",
                    "chunk_overlap",
                    "chunking_strategy",
                    "embedding_model",
                    "retrieval_search_mode",
                    "reranking",
                    "template_extraction",
                    "key_field_extraction_enabled",
                    "key_field_extraction",
                )
                if key in effective
            }
        record["processing"] = repo.get("processing_hints") or {}
        record.setdefault("metadata", {})["repository_id"] = repository_id
        processing = dict(record.get("processing") or {})
        processing["repository_id"] = repository_id
        record["processing"] = processing
        _pipeline_trace(
            "repository_context_applied",
            document_id=str(record.get("document_id") or ""),
            repository_id=repository_id,
            template_extraction=processing.get("template_extraction"),
            key_field_extraction_enabled=processing.get("key_field_extraction_enabled"),
            enabled_processor_types_before_scheduling=list(processing.get("enabled_processor_types") or []),
            effective_document_type_id=document_type_id,
        )
        self._bind_document_type(record, repository_id=repository_id, document_type_id=document_type_id)
        if link_document:
            get_repository_service().link_document(record["document_id"], repository_id)

        actor = get_current_user_from_context()
        if audit_upload and actor and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous":
            try:
                from src.features.audit.application.security_audit_service import AuditService
                from src.features.users.infrastructure.user_repository import get_platform_security_store

                sec_store = get_platform_security_store()
                if sec_store:
                    AuditService(sec_store).record(
                        event_category="document_intake",
                        event_type="document.upload_accepted",
                        action="create",
                        outcome="success",
                        actor=actor,
                        repository_id=repository_id,
                        document_id=record["document_id"],
                        new_value={
                            "document_name": record.get("original_file_name"),
                            "repository_id": repository_id,
                        },
                    )
            except Exception:
                pass

    def _bind_document_type(
        self,
        record: dict[str, Any],
        *,
        repository_id: str | None,
        document_type_id: str | None,
    ) -> None:
        processing = dict(record.get("processing") or {})
        resolved_type_id, resolved_type_name = self._resolve_document_type(
            repository_id=repository_id,
            document_type_id=document_type_id,
            processing=processing,
        )
        if not resolved_type_id:
            return

        processing["document_type_id"] = str(resolved_type_id)
        if resolved_type_name:
            processing["document_type_name"] = resolved_type_name
        record["processing"] = processing
        record["document_type_id"] = str(resolved_type_id)
        if resolved_type_name:
            record["document_type_name"] = resolved_type_name
        record.setdefault("metadata", {})["document_type_id"] = str(resolved_type_id)
        if resolved_type_name:
            record.setdefault("metadata", {})["document_type_name"] = resolved_type_name
        try:
            self._upsert_document_instance(
                document_id=str(record["document_id"]),
                document_type_id=str(resolved_type_id),
                repository_id=str(repository_id) if repository_id else None,
                tenant_id=str(record.get("tenant_id")) if record.get("tenant_id") else None,
            )
        except Exception:
            _logger.debug("Could not upsert document_instance for document_id=%s", record.get("document_id"), exc_info=True)

    async def upload_documents(
        self,
        files: list[UploadFile],
        *,
        collection_name: str | None = None,
        tenant_id: str | None = None,
        repository_id: str | None = None,
        document_type_id: str | None = None,
        submit_for_processing: bool = True,
    ) -> UploadResponse:
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required.")
        repository_id = self._normalize_optional_id(repository_id)
        document_type_id = self._normalize_optional_id(document_type_id)
        if not repository_id and document_type_id:
            repository_id = self._repository_id_from_document_type(document_type_id)
        # Phase 3.7: reject unknown / inactive / archived repositories before accepting files.
        if repository_id:
            self._require_active_repository(repository_id)
        if document_type_id:
            self._validate_document_type(document_type_id, repository_id=repository_id)
        live = self._live_settings()
        if len(files) > live.max_files_per_request:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Too many files in one request ({len(files)}). "
                    f"Maximum allowed is {live.max_files_per_request}."
                ),
            )
        js = self._job_store()
        if not js:
            raise HTTPException(status_code=503, detail="PostgreSQL document jobs are not configured")
        responses: list[DocumentResponse] = []
        rejected: list[UploadRejection] = []
        batch_id = str(uuid.uuid4())
        total_files = len(files)
        temporary_repository: dict[str, Any] | None = None
        from src.features.repositories.application.temporary_repository_service import (
            ensure_temporary_repository,
            get_temporary_repository_config,
            should_route_to_temporary_repository,
        )

        temp_config = get_temporary_repository_config()
        if temp_config.enabled and should_route_to_temporary_repository(repository_id):
            # One repository is created once for this HTTP upload request, then
            # shared by every file in that request only.
            temporary_repository = ensure_temporary_repository(batch_id=batch_id)
            repository_id = str(temporary_repository["repository_id"])
            document_type_id = None
            collection_name = None
            tenant_id = None
            _pipeline_trace(
                "temporary_repository_batch_routed",
                batch_id=batch_id,
                repository_id=repository_id,
                repository_name=temporary_repository.get("repository_name"),
                total_files=total_files,
                reason="repository_not_selected",
            )
        # Track hashes accepted earlier in this same request so identical files
        # (byte hash or normalized text) in one batch are rejected.
        batch_accepted_hashes: dict[str, dict[str, Any]] = {}
        batch_accepted_text_hashes: dict[str, dict[str, Any]] = {}
        for upload in files:
            filename = (upload.filename or "").strip() or None
            try:
                effective_repository_id = repository_id
                effective_document_type_id = document_type_id
                effective_collection_name = collection_name
                effective_tenant_id = tenant_id
                await self._upload_one_document(
                    upload,
                    responses=responses,
                    rejected=rejected,
                    batch_accepted_hashes=batch_accepted_hashes,
                    batch_accepted_text_hashes=batch_accepted_text_hashes,
                    batch_id=batch_id,
                    total_files=total_files,
                    collection_name=effective_collection_name,
                    tenant_id=effective_tenant_id,
                    repository_id=effective_repository_id,
                    document_type_id=effective_document_type_id,
                    temporary_repository=temporary_repository,
                    submit_for_processing=submit_for_processing,
                    live=live,
                    js=js,
                )
            except (HTTPException, DmsServiceError) as per_file_exc:
                # Single-file uploads keep hard HTTP errors for backward compatibility.
                # Multi-file batches collect the rejection and continue with remaining files.
                if total_files == 1:
                    raise
                rejected.append(
                    self._rejection_from_exception(per_file_exc, filename=filename or upload.filename)
                )
            except Exception as per_file_exc:
                if total_files == 1:
                    raise
                _logger.exception(
                    "Per-file upload failed for %s in batch %s",
                    filename or upload.filename,
                    batch_id,
                )
                rejected.append(
                    self._rejection_from_exception(per_file_exc, filename=filename or upload.filename)
                )
        # Multi-file batch where every file was a duplicate: keep 409 so clients
        # still see a clear conflict when nothing was accepted.
        if total_files > 1 and not responses and rejected:
            if all(item.code == "duplicate_document" for item in rejected):
                first = rejected[0]
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "duplicate_document",
                        "message": (
                            f"All {len(rejected)} file(s) were rejected as duplicates. "
                            f"{first.message}"
                        ),
                        "existing_document_id": first.existing_document_id,
                        "existing_document_name": first.existing_document_name,
                        "existing_status": first.existing_status,
                        "existing_repository_id": first.existing_repository_id,
                        "rejected_count": len(rejected),
                        "rejected": [item.model_dump() for item in rejected],
                    },
                )
        return UploadResponse(
            batch_id=batch_id,
            uploaded_count=len(responses),
            documents=responses,
            rejected_count=len(rejected),
            rejected=rejected,
        )

    async def _finalize_upload_async(
        self,
        *,
        document_id: str,
        destination: Path,
        batch_id: str,
        content_hash: str,
        repository_id: str | None,
        collection_name: str | None,
        tenant_id: str | None,
        document_type_id: str | None,
        resolved_upload_type_id: str | None,
        resolved_upload_type_name: str | None,
    ) -> None:
        """Background: optional security scan, queue submission, then text fingerprint."""
        js = self._job_store()
        store = self.get_store()
        try:
            from src.features.documents.application.content_fingerprint import compute_content_text_hash
            from src.features.security.compliance.compliance_validator import (
                check_document_compliance,
                security_scan_requires_review,
            )

            record = store.get_document(document_id)
            if not record:
                return

            security_scan: dict[str, Any] | None = None
            try:
                security_scan = await asyncio.to_thread(
                    check_document_compliance,
                    destination,
                    repository_id=repository_id,
                    document_id=document_id,
                    document_type_id=resolved_upload_type_id,
                    document_type_name=resolved_upload_type_name,
                )
            except DmsServiceError as compliance_exc:
                destination.unlink(missing_ok=True)
                if js:
                    js.update_status_queue_failed(document_id, str(compliance_exc))
                store.update_document_status(document_id, "blocked", error_details=str(compliance_exc))
                return

            if isinstance(security_scan, dict) and security_scan.get("status"):
                security_meta = {
                    "status": security_scan.get("status"),
                    "severity": security_scan.get("severity"),
                    "risk_level": security_scan.get("risk_level"),
                    "reason": security_scan.get("reason"),
                    "detected_categories": security_scan.get("detected_categories") or [],
                    "dlp_decision": security_scan.get("dlp_decision"),
                    "requires_human_review": security_scan.get("requires_human_review"),
                    "available_reviewer_actions": security_scan.get("available_reviewer_actions") or [],
                }
                metadata_patch = {**record.get("metadata", {}), "security_scan": security_meta}
                store.patch_document(document_id, {"metadata": metadata_patch})
                record["metadata"] = metadata_patch

            if security_scan_requires_review(security_scan if isinstance(security_scan, dict) else None):
                store.update_document_status(document_id, "human_review")
                record = store.get_document(document_id) or record
                record["status"] = "human_review"
                record["metadata"] = {
                    **record.get("metadata", {}),
                    "requires_human_review": True,
                    "upload_finalize": "human_review",
                }
                self._register_human_review_queue_entry(record, security_scan if isinstance(security_scan, dict) else None)
                _pipeline_trace(
                    "queue_submission_deferred",
                    document_id=document_id,
                    current_status="uploaded",
                    next_status="human_review",
                    reason="pending_human_review",
                )
                return

            record = store.get_document(document_id) or record
            record["metadata"] = {**record.get("metadata", {}), "upload_finalize": "queued"}
            await asyncio.to_thread(self.submit_to_queue, record)
            _pipeline_trace(
                "async_finalize_complete",
                document_id=document_id,
                current_status="uploaded",
                next_status="QUEUED",
                reason="submit_to_queue",
            )

            # Text fingerprint is best-effort and must not block queue submission (OCR on large PDFs is slow).
            try:
                content_text_hash = await asyncio.to_thread(compute_content_text_hash, destination)
                if not content_text_hash:
                    store.patch_document(
                        document_id,
                        {"metadata": {**record.get("metadata", {}), "upload_finalize": "complete"}},
                    )
                    return
                duplicate = self._find_duplicate_document(
                    content_hash=content_hash,
                    content_text_hash=content_text_hash,
                    repository_id=repository_id,
                    collection_name=collection_name,
                    tenant_id=tenant_id,
                    exclude_document_id=document_id,
                )
                if duplicate:
                    _pipeline_trace(
                        "async_finalize_text_duplicate_after_queue",
                        document_id=document_id,
                        current_status=record.get("status"),
                        next_status="duplicate_detected",
                        reason="content_text_hash_duplicate",
                    )
                    return
                metadata_patch = {
                    **record.get("metadata", {}),
                    "content_text_hash": content_text_hash,
                    "upload_finalize": "complete",
                }
                store.patch_document(
                    document_id,
                    {
                        "content_text_hash": content_text_hash,
                        "metadata": metadata_patch,
                    },
                )
            except Exception:
                _logger.warning(
                    "async upload text fingerprint failed for %s (processing already queued)",
                    document_id,
                    exc_info=True,
                )
        except Exception as exc:
            _logger.exception("async upload finalize failed for %s", document_id)
            _pipeline_trace(
                "queue_submission_failed",
                document_id=document_id,
                current_status="uploaded",
                next_status="QUEUE_FAILED",
                reason=str(exc),
            )
            if js:
                js.update_status_queue_failed(document_id, str(exc))
            store.update_document_status(document_id, "queue_failed", error_details=str(exc))

    async def _upload_one_document(
        self,
        upload: UploadFile,
        *,
        responses: list[DocumentResponse],
        rejected: list[UploadRejection],
        batch_accepted_hashes: dict[str, dict[str, Any]],
        batch_accepted_text_hashes: dict[str, dict[str, Any]],
        batch_id: str,
        total_files: int,
        collection_name: str | None,
        tenant_id: str | None,
        repository_id: str | None,
        document_type_id: str | None,
        temporary_repository: dict[str, Any] | None,
        submit_for_processing: bool,
        live: Any,
        js: Any,
    ) -> None:
        if not (upload.filename or "").strip():
            raise HTTPException(status_code=400, detail="Each uploaded file must have a filename.")
        document_type = self.safe_extension(upload.filename or "")
        document_id = str(uuid.uuid4())
        _pipeline_trace(
            "upload_received",
            document_id=document_id,
            current_status=None,
            next_status="uploaded",
            filename=upload.filename,
            repository_id=repository_id,
            submit_for_processing=submit_for_processing,
        )
        stored_name = f"{document_id}{Path(upload.filename or '').suffix.lower()}"
        destination = Path(live.upload_dir) / stored_name
        content_hash, file_size = await self._save_upload_file(upload, destination)
        async_finalize = _dms_async_upload_finalize_enabled() and submit_for_processing
        _pipeline_trace(
            "upload_file_saved",
            document_id=document_id,
            current_status=None,
            next_status="uploaded",
            path=str(destination),
            file_size_bytes=file_size,
            async_finalize=async_finalize,
        )

        resolved_upload_type_id, resolved_upload_type_name = self._resolve_document_type(
            repository_id=repository_id,
            document_type_id=document_type_id,
            processing={},
        )
        content_text_hash: str | None = None
        security_scan: dict[str, Any] | None = None
        if async_finalize:
            pending_human_review = False
        else:
            from src.features.documents.application.content_fingerprint import compute_content_text_hash

            content_text_hash = compute_content_text_hash(destination)
            try:
                from src.features.security.compliance.compliance_validator import check_document_compliance

                security_scan = check_document_compliance(
                    destination,
                    repository_id=repository_id,
                    document_id=document_id,
                    document_type_id=resolved_upload_type_id,
                    document_type_name=resolved_upload_type_name,
                )
            except DmsServiceError as compliance_exc:
                destination.unlink(missing_ok=True)
                raise compliance_exc

        record = {
            "document_id": document_id,
            "document_name": stored_name,
            "original_file_name": upload.filename or stored_name,
            "document_type": document_type,
            "repository_type": live.repository_type,
            "repository_path": str(destination),
            "collection_name": collection_name or live.default_collection_name,
            "tenant_id": tenant_id or live.default_tenant_id,
            "status": "uploaded",
            "upload_timestamp": time.time(),
            "content_hash": content_hash,
            "content_text_hash": content_text_hash,
            "file_size_bytes": file_size,
            "metadata": {
                "batch_id": batch_id,
                "original_file_name": upload.filename or stored_name,
                "content_hash": content_hash,
                "content_text_hash": content_text_hash,
                "file_size_bytes": file_size,
            },
        }
        if temporary_repository is not None:
            retention_hours = int(temporary_repository.get("retention_hours") or 24)
            retention_seconds = int(temporary_repository.get("retention_seconds") or (retention_hours * 3600))
            expires_at = datetime.now(timezone.utc).timestamp() + retention_seconds
            record["metadata"]["temporary_repository"] = {
                "enabled": True,
                "repository_name": temporary_repository.get("repository_name"),
                "batch_id": temporary_repository.get("batch_id"),
                "retention_hours": retention_hours,
                "retention_seconds": retention_seconds,
                "expires_at": expires_at,
                "routing_reason": "repository_not_selected",
            }
        if isinstance(security_scan, dict) and security_scan.get("status"):
            record["metadata"]["security_scan"] = {
                "status": security_scan.get("status"),
                "severity": security_scan.get("severity"),
                "risk_level": security_scan.get("risk_level"),
                "reason": security_scan.get("reason"),
                "detected_categories": security_scan.get("detected_categories") or [],
                "dlp_decision": security_scan.get("dlp_decision"),
                "requires_human_review": security_scan.get("requires_human_review"),
                "available_reviewer_actions": security_scan.get("available_reviewer_actions") or [],
            }
        if not async_finalize:
            from src.features.security.compliance.compliance_validator import (
                security_scan_requires_review,
            )

            pending_human_review = security_scan_requires_review(
                security_scan if isinstance(security_scan, dict) else None
            )
        # Resolve repository scope first so duplicate checks are correctly scoped.
        self._apply_repository_context(
            record,
            repository_id=repository_id,
            collection_name=collection_name,
            tenant_id=tenant_id,
            document_type_id=document_type_id,
            link_document=False,
            audit_upload=False,
        )
        # Same-batch duplicate (byte hash or normalized text).
        batch_duplicate = batch_accepted_hashes.get(content_hash)
        if not batch_duplicate and content_text_hash:
            batch_duplicate = batch_accepted_text_hashes.get(content_text_hash)
        if batch_duplicate:
            destination.unlink(missing_ok=True)
            self._delete_document_instance(document_id)
            if total_files == 1:
                self._raise_duplicate_upload(batch_duplicate)
            rejected.append(
                self._rejection_from_duplicate(
                    batch_duplicate,
                    filename=upload.filename or stored_name,
                )
            )
            return
        self._purge_replaceable_duplicates(
            content_hash=content_hash,
            repository_id=record.get("repository_id"),
            collection_name=record.get("collection_name"),
            tenant_id=record.get("tenant_id"),
            exclude_document_id=document_id,
        )
        duplicate = self._find_duplicate_document(
            content_hash=content_hash,
            content_text_hash=content_text_hash,
            repository_id=record.get("repository_id"),
            collection_name=record.get("collection_name"),
            tenant_id=record.get("tenant_id"),
            exclude_document_id=document_id,
        )
        if duplicate:
            destination.unlink(missing_ok=True)
            self._delete_document_instance(document_id)
            # Single-file uploads keep HTTP 409 for backward compatibility.
            # Multi-file batches reject the duplicate and continue with the rest.
            if total_files == 1:
                self._raise_duplicate_upload(duplicate)
            rejected.append(
                self._rejection_from_duplicate(
                    duplicate,
                    filename=upload.filename or stored_name,
                )
            )
            return
        if pending_human_review:
            record["status"] = "human_review"
            record["metadata"]["requires_human_review"] = True
            self._register_human_review_queue_entry(record, security_scan if isinstance(security_scan, dict) else None)
        self._apply_repository_context(
            record,
            repository_id=repository_id,
            collection_name=collection_name,
            tenant_id=tenant_id,
            document_type_id=document_type_id,
        )
        job_id = js.ensure_received_from_record(record, batch_id=batch_id)
        _pipeline_trace(
            "job_inserted",
            document_id=document_id,
            job_id=job_id,
            current_status=None,
            next_status="RECEIVED",
            reason="insert_received/ensure_received_from_record",
        )
        record["metadata"] = {**record.get("metadata", {}), "job_id": job_id, "batch_id": batch_id}
        store = self.get_store()
        if hasattr(store, "insert_document_if_not_duplicate"):
            race_duplicate = store.insert_document_if_not_duplicate(
                record,
                content_hash=content_hash,
                content_text_hash=content_text_hash,
                repository_id=record.get("repository_id"),
                collection_name=record.get("collection_name"),
                tenant_id=record.get("tenant_id"),
                non_blocking_statuses=self._NON_BLOCKING_DUPLICATE_STATUSES,
            )
            if race_duplicate:
                destination.unlink(missing_ok=True)
                self._delete_document_instance(document_id)
                js.delete_by_document_id(document_id)
                if total_files == 1:
                    self._raise_duplicate_upload(race_duplicate)
                rejected.append(
                    self._rejection_from_duplicate(
                        race_duplicate,
                        filename=upload.filename or stored_name,
                    )
                )
                return
        else:
            store.insert_document(record)
        # Remember this hash for later files in the same multipart upload.
        accepted_meta = {
            "document_id": document_id,
            "original_file_name": record.get("original_file_name"),
            "document_name": record.get("document_name"),
            "status": record.get("status"),
            "repository_id": record.get("repository_id"),
            "content_hash": content_hash,
            "content_text_hash": content_text_hash,
        }
        batch_accepted_hashes[content_hash] = accepted_meta
        if content_text_hash:
            batch_accepted_text_hashes[content_text_hash] = accepted_meta
        _pipeline_trace(
            "document_created",
            document_id=document_id,
            job_id=job_id,
            current_status="uploaded",
            next_status="uploaded",
            reason="metadata_store.insert_document",
        )
        if submit_for_processing:
            if pending_human_review:
                _pipeline_trace(
                    "queue_submission_deferred",
                    document_id=document_id,
                    job_id=job_id,
                    current_status="RECEIVED",
                    next_status="human_review",
                    reason="pending_human_review",
                )
            elif async_finalize:
                record["metadata"] = {
                    **record.get("metadata", {}),
                    "upload_finalize": "pending",
                }
                store.patch_document(document_id, {"metadata": record["metadata"]})
                asyncio.create_task(
                    self._finalize_upload_async(
                        document_id=document_id,
                        destination=destination,
                        batch_id=batch_id,
                        content_hash=content_hash,
                        repository_id=record.get("repository_id"),
                        collection_name=record.get("collection_name"),
                        tenant_id=record.get("tenant_id"),
                        document_type_id=document_type_id,
                        resolved_upload_type_id=resolved_upload_type_id,
                        resolved_upload_type_name=resolved_upload_type_name,
                    )
                )
                _pipeline_trace(
                    "async_finalize_scheduled",
                    document_id=document_id,
                    job_id=job_id,
                    current_status="RECEIVED",
                    next_status="uploaded",
                    reason="deferred_security_and_queue",
                )
            else:
                try:
                    await asyncio.to_thread(self.submit_to_queue, record)
                    record = self.get_store().get_document(document_id) or record
                except Exception as exc:
                    _pipeline_trace(
                        "queue_submission_failed",
                        document_id=document_id,
                        job_id=job_id,
                        current_status="RECEIVED",
                        next_status="QUEUE_FAILED",
                        reason=str(exc),
                    )
                    js.update_status_queue_failed(document_id, str(exc))
                    self.get_store().update_document_status(document_id, "queue_failed", error_details=str(exc))
                    record = self.get_store().get_document(document_id) or record
        else:
            _pipeline_trace(
                "queue_submission_skipped",
                document_id=document_id,
                job_id=job_id,
                current_status="RECEIVED",
                next_status="RECEIVED",
                reason="submit_for_processing=false",
            )
        responses.append(self.document_response(record))

    async def register_document(self, body: DocumentRegisterRequest) -> DocumentResponse:
        document_type = self.safe_extension(body.document_name)
        document_id = str(uuid.uuid4())
        batch_id = str(uuid.uuid4())
        js = self._job_store()
        if not js:
            raise HTTPException(status_code=503, detail="PostgreSQL document jobs are not configured")
        live = self._live_settings()

        try:
            from src.features.security.compliance.compliance_validator import check_document_compliance
            file_path = Path(body.repository_path or body.document_name)
            if not file_path.is_absolute():
                file_path = Path(live.upload_dir) / file_path
            resolved_type_id, resolved_type_name = self._resolve_document_type(
                repository_id=body.repository_id,
                document_type_id=body.document_type_id,
                processing={},
            )
            check_document_compliance(
                file_path,
                repository_id=body.repository_id,
                document_type_id=resolved_type_id,
                document_type_name=resolved_type_name,
            )
        except DmsServiceError as compliance_exc:
            raise compliance_exc

        record = {
            "document_id": document_id,
            "document_name": body.document_name,
            "original_file_name": body.original_file_name or body.document_name,
            "document_type": document_type,
            "repository_type": body.repository_type,
            "repository_path": body.repository_path or body.document_name,
            "collection_name": body.collection_name or live.default_collection_name,
            "tenant_id": body.tenant_id or live.default_tenant_id,
            "status": "registered",
            "upload_timestamp": time.time(),
            "metadata": body.metadata,
        }
        self._apply_repository_context(
            record,
            repository_id=body.repository_id,
            collection_name=body.collection_name,
            tenant_id=body.tenant_id,
            document_type_id=body.document_type_id,
        )
        job_id = js.ensure_received_from_record(record, batch_id=batch_id)
        record["metadata"] = {**record.get("metadata", {}), "job_id": job_id, "batch_id": batch_id}
        self.get_store().insert_document(record)
        if body.submit_for_processing:
            try:
                await asyncio.to_thread(self.submit_to_queue, record)
                record = self.get_store().get_document(document_id) or record
            except Exception as exc:
                js.update_status_queue_failed(document_id, str(exc))
                self.get_store().update_document_status(document_id, "queue_failed", error_details=str(exc))
                record = self.get_store().get_document(document_id) or record
        return self.document_response(record)

    async def reprocess_document(self, document_id: str) -> DocumentResponse:
        record = self.get_store().get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found")
        from src.features.documents.application.document_metadata_service import check_document_access

        check_document_access(record)
        js = self._job_store()
        if not js:
            raise HTTPException(status_code=503, detail="PostgreSQL document jobs are not configured")
        js.ensure_received_from_record(record, batch_id=None)
        js.update_for_reprocess(document_id)
        try:
            await asyncio.to_thread(self.submit_to_queue, record)
        except Exception as exc:
            js.update_status_queue_failed(document_id, str(exc))
            self.get_store().update_document_status(document_id, "queue_failed", error_details=str(exc))
            raise_service_error(
                COMPONENT_DOCUMENT_RECEIVER,
                code="queue_submission_failed",
                http_status=503,
                user_message="Document processing could not be queued. Please try again shortly.",
                reason=f"Queue submission failed during reprocess for document_id={document_id}",
                cause=exc,
            )
        refreshed = self.get_store().get_document(document_id) or record
        try:
            from src.features.observability.hooks.processor_hooks import record_document_reprocessed

            record_document_reprocessed(
                document_id,
                repository_id=refreshed.get("repository_id"),
                document_name=refreshed.get("document_name") or refreshed.get("original_file_name"),
                repository_name=refreshed.get("repository_name"),
            )
        except Exception:
            _logger.exception("Failed to record reprocess audit for document_id=%s", document_id)
        return self.document_response(refreshed)

    async def submit_document(self, document_id: str) -> DocumentResponse:
        record = self.get_store().get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found")
        js = self._job_store()
        if not js:
            raise HTTPException(status_code=503, detail="PostgreSQL document jobs are not configured")
        js.ensure_received_from_record(record, batch_id=None)
        try:
            await asyncio.to_thread(self.submit_to_queue, record)
        except Exception as exc:
            js.update_status_queue_failed(document_id, str(exc))
            self.get_store().update_document_status(document_id, "queue_failed", error_details=str(exc))
            raise_service_error(
                COMPONENT_DOCUMENT_RECEIVER,
                code="queue_submission_failed",
                http_status=503,
                user_message="Document processing could not be queued. Please try again shortly.",
                reason=f"Queue submission failed during submit for document_id={document_id}",
                cause=exc,
            )
        refreshed = self.get_store().get_document(document_id) or record
        try:
            from src.features.observability.hooks.processor_hooks import record_document_submitted

            record_document_submitted(
                document_id,
                repository_id=refreshed.get("repository_id"),
                document_name=refreshed.get("document_name") or refreshed.get("original_file_name"),
                repository_name=refreshed.get("repository_name"),
            )
        except Exception:
            _logger.exception("Failed to record submit audit for document_id=%s", document_id)
        return self.document_response(refreshed)

    def recover_received_unqueued_jobs(self, *, limit: int = 200) -> dict[str, Any]:
        """Recover jobs that were inserted but never submitted to RabbitMQ.

        This uses the normal submit_to_queue path (seed_processor_plan -> publish ->
        update_status_queued), so it is not a database workaround.
        """
        js = self._job_store()
        if not js or not hasattr(js, "list_received_unqueued"):
            return {"recovered": [], "skipped": [], "count": 0}

        recovered: list[str] = []
        skipped: list[dict[str, str]] = []
        for row in js.list_received_unqueued(limit=limit):
            document_id = str(row.get("document_id") or "").strip()
            job_id = str(row.get("job_id") or "").strip()
            if not document_id:
                continue
            record = self.get_store().get_document(document_id)
            if not record:
                skipped.append({"document_id": document_id, "reason": "intake_record_missing"})
                _pipeline_trace(
                    "received_orphan_recovery_skipped",
                    document_id=document_id,
                    job_id=job_id,
                    current_status="RECEIVED",
                    next_status="RECEIVED",
                    reason="intake_record_missing",
                )
                continue
            source = Path(str(record.get("repository_path") or row.get("source_location") or ""))
            if not source.is_file():
                skipped.append({"document_id": document_id, "reason": f"source_file_missing:{source}"})
                _pipeline_trace(
                    "received_orphan_recovery_skipped",
                    document_id=document_id,
                    job_id=job_id,
                    current_status="RECEIVED",
                    next_status="RECEIVED",
                    reason=f"source_file_missing:{source}",
                )
                continue
            if not self._processing_allowed_after_security_review(record):
                skipped.append({"document_id": document_id, "reason": "pending_human_review"})
                _pipeline_trace(
                    "received_orphan_recovery_skipped",
                    document_id=document_id,
                    job_id=job_id,
                    current_status="RECEIVED",
                    next_status="human_review",
                    reason="pending_human_review",
                )
                continue
            try:
                _pipeline_trace(
                    "received_orphan_recovery_submit",
                    document_id=document_id,
                    job_id=job_id,
                    current_status="RECEIVED",
                    next_status="QUEUED",
                    reason="startup_or_periodic_recovery",
                )
                self.submit_to_queue(record)
                recovered.append(document_id)
            except Exception as exc:
                skipped.append({"document_id": document_id, "reason": str(exc)})
                _logger.exception("Failed to recover RECEIVED job document_id=%s", document_id)
        return {"recovered": recovered, "skipped": skipped, "count": len(recovered)}

    def delete_document(self, document_id: str, *, delete_file: bool = True) -> dict[str, Any]:
        return self.delete_document_cascade(document_id, delete_file=delete_file)

    def purge_document_for_repository(
        self,
        document_id: str,
        repository_id: str,
        *,
        delete_file: bool = True,
    ) -> dict[str, Any]:
        """Delete a document during repository purge (intake, link, or job-only orphans)."""
        from src.features.documents.application.document_purge_service import (
            purge_document_job,
            purge_repository_link,
        )

        record = self.get_store().get_document(document_id)
        if record:
            bound_repository_id = record.get("repository_id") or (record.get("metadata") or {}).get(
                "repository_id"
            )
            if bound_repository_id and str(bound_repository_id) != str(repository_id):
                raise HTTPException(
                    status_code=404,
                    detail="Document belongs to another repository.",
                )
            return self.delete_document_cascade(document_id, delete_file=delete_file)

        if self._document_linked_to_repository(document_id, repository_id):
            purge_repository_link(document_id)
            purge_document_job(document_id)
            return {
                "document_id": document_id,
                "repository_id": repository_id,
                "deleted": True,
                "orphan_cleanup": True,
            }

        js = self._job_store()
        if js and js.get_by_document_id(document_id):
            purge_document_job(document_id)
            return {
                "document_id": document_id,
                "repository_id": repository_id,
                "deleted": True,
                "orphan_cleanup": True,
            }

        raise HTTPException(status_code=404, detail="Document not found")

    @staticmethod
    def _document_linked_to_repository(document_id: str, repository_id: str) -> bool:
        from src.features.repositories.application.repository_service import get_repository_service

        return get_repository_service().is_document_linked_to_repository(repository_id, document_id)

    def _cleanup_missing_document(
        self,
        document_id: str,
        *,
        repository_id: str | None = None,
    ) -> dict[str, Any]:
        """Remove orphan repository links and job rows when intake metadata is gone."""
        from src.features.documents.application.document_purge_service import (
            purge_document_job,
            purge_repository_link,
        )

        if repository_id and not self._document_linked_to_repository(document_id, repository_id):
            raise HTTPException(status_code=404, detail="Document not found")

        purge = {
            "repository_link": purge_repository_link(document_id),
            "document_job": purge_document_job(document_id),
        }
        return {
            "document_id": document_id,
            "repository_id": repository_id,
            "deleted": True,
            "orphan_cleanup": True,
            "purge": purge,
        }

    def delete_document_cascade(
        self,
        document_id: str,
        *,
        delete_file: bool = True,
        repository_id: str | None = None,
    ) -> dict[str, Any]:
        from src.application.consumer_api.context import get_current_user_from_context, get_request_id
        from src.features.documents.application.document_purge_service import purge_all_artifacts

        record = self.get_store().get_document(document_id)
        if not record:
            return self._cleanup_missing_document(document_id, repository_id=repository_id)

        bound_repository_id = record.get("repository_id") or (record.get("metadata") or {}).get("repository_id")
        if repository_id and bound_repository_id and str(bound_repository_id) != str(repository_id):
            raise HTTPException(
                status_code=404,
                detail="Document is not linked to this repository.",
            )
        if repository_id and not bound_repository_id:
            if self._document_linked_to_repository(document_id, repository_id):
                bound_repository_id = repository_id
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Document is not linked to this repository.",
                )

        actor = get_current_user_from_context()
        repo_role: str | None = None
        from src.features.authentication.domain.authentication_exceptions import AuthenticationError

        if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
            raise AuthenticationError("Authentication required")
        if bound_repository_id:
            from src.features.users.application.user_service import get_platform_security_service

            repo_role = get_platform_security_service().check_document_delete_access(
                actor,
                str(bound_repository_id),
            )
        elif not actor.is_platform_admin and not actor.is_admin_api_key:
            raise HTTPException(
                status_code=403,
                detail="Document delete requires repository context or platform administrator role.",
            )

        audit_snapshot = {
            "document_id": document_id,
            "original_file_name": record.get("original_file_name"),
            "document_name": record.get("document_name"),
            "repository_id": bound_repository_id,
            "collection_name": record.get("collection_name"),
        }

        purge = purge_all_artifacts(record, delete_file=delete_file)

        if not self.get_store().delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")

        if actor and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous":
            try:
                from src.features.audit.application.security_audit_service import AuditService
                from src.features.users.infrastructure.user_repository import get_platform_security_store

                sec_store = get_platform_security_store()
                if sec_store:
                    AuditService(sec_store).record(
                        event_category="document_intake",
                        event_type="document.deleted",
                        action="delete",
                        outcome="success",
                        actor=actor,
                        repository_role=repo_role if repo_role not in {"administrator", "api_key"} else None,
                        repository_id=str(bound_repository_id) if bound_repository_id else None,
                        document_id=document_id,
                        old_value=audit_snapshot,
                        new_value={"purge": purge},
                        request_id=get_request_id(),
                    )
            except Exception:
                _logger.exception("Failed to audit document deletion for %s", document_id)

        return {
            **audit_snapshot,
            "deleted": True,
            "purge": purge,
        }

    def list_documents(
        self,
        *,
        status: str | None = None,
        tenant_id: str | None = None,
        collection_name: str | None = None,
        repository_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentResponse]:
        js = self._job_store()
        store_status = None if (js and status) else status
        records = self.get_store().list_documents(
            status=store_status,
            tenant_id=tenant_id,
            collection_name=collection_name,
            repository_id=repository_id,
            limit=max(1, min(limit, 500)),
            offset=max(0, offset),
        )
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.authentication.domain.authentication_exceptions import AuthenticationError
        from src.features.users.application.user_service import get_platform_security_service

        actor = get_current_user_from_context()
        if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
            raise AuthenticationError("Authentication required")
        security = get_platform_security_service()
        if repository_id:
            security.check_retrieval_access(actor, repository_id)
        elif not getattr(actor, "is_platform_admin", False) and not getattr(actor, "is_admin_api_key", False):
            allowed = set(security.list_accessible_repository_ids(actor))
            records = [
                record
                for record in records
                if str(record.get("repository_id") or (record.get("metadata") or {}).get("repository_id") or "") in allowed
            ]
        prefetched = js.fetch_for_documents([r["document_id"] for r in records]) if js else None
        out = [self.document_response(record, prefetched=prefetched) for record in records]
        if js and status:
            out = [r for r in out if r.status == status]
        return out

    def get_document(self, document_id: str) -> DocumentResponse:
        record = self.get_store().get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found")
        from src.features.documents.application.document_metadata_service import check_document_access

        check_document_access(record)
        response = self.document_response(record)
        try:
            from src.features.documents.application.document_metadata_service import (
                resolve_key_field_metadata_bundle,
            )

            key_field_bundle = resolve_key_field_metadata_bundle(record)
            from src.features.documents.application.document_metadata_service import (
                resolve_validation_bundle,
            )

            validation = resolve_validation_bundle(record)
            enriched_metadata = dict(response.metadata or {})
            enriched_metadata["extracted_metadata"] = key_field_bundle.get("extracted_metadata") or {}
            enriched_metadata["key_field_metadata"] = key_field_bundle.get("key_field_metadata") or {}
            enriched_metadata["effective_fields"] = key_field_bundle.get("effective_fields") or []
            enriched_metadata["validation"] = {
                "status": validation.get("status"),
                "missing_required_fields": validation.get("missing_required_fields") or [],
                "invalid_fields": validation.get("invalid_fields") or [],
                "low_confidence_fields": validation.get("low_confidence_fields") or [],
            }
            if validation.get("document_status"):
                enriched_metadata["validation_status"] = validation.get("document_status")
            if key_field_bundle.get("document_type_id"):
                enriched_metadata["document_type_id"] = key_field_bundle.get("document_type_id")
            if key_field_bundle.get("document_type_name"):
                enriched_metadata["document_type_name"] = key_field_bundle.get("document_type_name")
            updates: dict[str, Any] = {"metadata": enriched_metadata}
            if key_field_bundle.get("document_type_id") and not response.document_type_id:
                updates["document_type_id"] = key_field_bundle.get("document_type_id")
            if key_field_bundle.get("document_type_name") and not response.document_type_name:
                updates["document_type_name"] = key_field_bundle.get("document_type_name")
            if validation.get("document_status") and not response.validation_status:
                updates["validation_status"] = validation.get("document_status")
            return response.model_copy(update=updates)
        except Exception:
            return response

    def update_document_metadata(
        self,
        document_id: str,
        metadata: dict[str, Any],
        *,
        merge: bool = True,
        change_reason: str | None = None,
    ) -> DocumentResponse:
        if not metadata:
            raise HTTPException(status_code=422, detail="metadata must contain at least one key.")

        record = self.get_store().get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found")

        from src.application.consumer_api.context import get_current_user_from_context, get_request_id

        document_type_id = self._document_type_id_from_record(record)
        allowed_fields = self._allowed_effective_field_names(document_type_id)
        if document_type_id:
            # When a document type is bound, only effective fields may be patched.
            self._validate_known_metadata_fields(metadata, allowed_fields)

        actor = get_current_user_from_context()
        repository_id = record.get("repository_id") or (record.get("metadata") or {}).get("repository_id")
        if actor is not None and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous":
            if repository_id:
                from src.features.users.application.user_service import get_platform_security_service

                get_platform_security_service().check_upload_access(actor, str(repository_id))

        old_meta = {
            k: v
            for k, v in (record.get("metadata") or {}).items()
            if k not in MetadataStore.SYSTEM_METADATA_KEYS
        }
        updated = self.get_store().update_document_metadata(document_id, metadata, merge=merge)
        if not updated:
            raise HTTPException(status_code=404, detail="Document not found")

        new_meta = {
            k: v
            for k, v in (updated.get("metadata") or {}).items()
            if k not in MetadataStore.SYSTEM_METADATA_KEYS
        }

        if actor and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous":
            try:
                from src.features.audit.application.security_audit_service import AuditService
                from src.features.users.infrastructure.user_repository import get_platform_security_store

                sec_store = get_platform_security_store()
                if sec_store:
                    AuditService(sec_store).record(
                        event_category="document_intake",
                        event_type="document.metadata_updated",
                        action="update",
                        outcome="success",
                        actor=actor,
                        repository_id=str(repository_id) if repository_id else None,
                        document_id=document_id,
                        old_value=old_meta,
                        new_value=new_meta,
                        change_reason=change_reason,
                        request_id=get_request_id(),
                    )
            except Exception:
                pass

        return self.document_response(updated)

    def get_document_status(self, document_id: str) -> DocumentJobStatusResponse:
        record = self.get_store().get_document(document_id)
        if not record:
            raise HTTPException(status_code=404, detail="Document not found")
        from src.features.documents.application.document_metadata_service import check_document_access

        check_document_access(record)
        js = self._job_store()
        if not js:
            raise HTTPException(
                status_code=503,
                detail="PostgreSQL document jobs are not configured; DOCUMENT_JOBS_POSTGRES_HOST or POSTGRES_HOST is required.",
            )
        job = js.get_by_document_id(document_id)
        if not job:
            raise HTTPException(status_code=404, detail="Document job not found in MySQL")
        from src.infrastructure.database.document_jobs import effective_job_status

        job_view = dict(job)
        raw_status = str(job_view.get("status") or "").strip().upper()
        effective = effective_job_status(job_view)
        job_view["raw_status"] = raw_status
        job_view["status"] = effective
        merged_record = self.merge_job_into_record(dict(record), job)
        return DocumentJobStatusResponse(
            document_id=document_id,
            job=job_view,
            document=self.document_response(merged_record),
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        js = self._job_store()
        if not js:
            raise HTTPException(status_code=503, detail="PostgreSQL document jobs are not configured")
        row = js.get_by_job_id(job_id)
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        return row

    def list_jobs_for_batch(self, batch_id: str) -> list[dict[str, Any]]:
        js = self._job_store()
        if not js:
            raise HTTPException(status_code=503, detail="PostgreSQL document jobs are not configured")
        return js.list_by_batch(batch_id)

    def mark_kill_requested(self, document_id: str) -> None:
        self.get_store().update_document_status(document_id, "kill_requested")
