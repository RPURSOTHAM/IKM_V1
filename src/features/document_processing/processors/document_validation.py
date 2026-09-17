"""Phase 4 document validation processor — runs after key_field_extraction."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.infrastructure.document_databases.neo4j_store import store_document_graph
from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_validation.application.validation_pipeline import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    run_document_validation,
)
from src.features.document_processing.shared_processor.types import ProcessorType

logger = logging.getLogger(__name__)


def _fetch_key_field_artifact(document_id: str) -> dict[str, Any] | None:
    try:
        from src.features.documents.application.document_metadata_service import (
            fetch_key_field_artifact_from_neo4j,
        )

        return fetch_key_field_artifact_from_neo4j(document_id)
    except Exception:
        logger.debug("DMS Neo4j key-field fetch unavailable; using local query", exc_info=True)

    try:
        import json

        from neo4j import GraphDatabase

        from src.infrastructure.document_databases.neo4j_store import _neo4j_credentials

        uri, user, password = _neo4j_credentials()
        cypher = """
        MATCH (a:DocumentArtifact {document_id: $document_id, processor_type: 'key_field_extraction'})
        RETURN a.payload_json AS payload_json
        LIMIT 1
        """
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as session:
                record = session.run(cypher, document_id=document_id).single()
            if not record:
                return None
            raw = record.get("payload_json")
            if isinstance(raw, str):
                payload = json.loads(raw)
            elif isinstance(raw, dict):
                payload = raw
            else:
                return None
            return {"artifact_payload": payload}
        finally:
            driver.close()
    except Exception:
        logger.debug("Local Neo4j key-field artifact fetch failed", exc_info=True)
        return None


def _wait_for_key_field_artifact(document_id: str, *, timeout_sec: float = 90.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout_sec
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        last = _fetch_key_field_artifact(document_id)
        if last and isinstance(last.get("artifact_payload"), dict):
            return last
        time.sleep(2.0)
    return last


def _resolve_effective_fields(request: ProcessRequest) -> list[dict[str, Any]]:
    extra = getattr(request, "effective_fields", None)
    if isinstance(extra, list) and extra:
        return [item for item in extra if isinstance(item, dict)]

    if request.key_fields:
        normalized: list[dict[str, Any]] = []
        for item in request.key_fields:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "field_name": item.get("field_name") or item.get("name"),
                    "type": item.get("type") or item.get("field_type") or item.get("data_type") or "string",
                    "field_type": item.get("field_type") or item.get("data_type") or item.get("type") or "string",
                    "required": bool(item.get("required", False)),
                    "description": item.get("description"),
                }
            )
        if normalized:
            return normalized

    document_type_id = request.document_type_id
    if not document_type_id:
        return []
    try:
        from src.features.document_types.application.document_type_service import get_document_type_service

        bundle = get_document_type_service().resolve_effective_fields(str(document_type_id))
        fields = bundle.get("fields") if isinstance(bundle, dict) else []
        return [item for item in fields if isinstance(item, dict)] if isinstance(fields, list) else []
    except Exception:
        logger.debug("Could not resolve effective fields for %s", document_type_id, exc_info=True)
        return []


class DocumentValidationProcessor(BaseProcessor):
    processor_type = ProcessorType.DOCUMENT_VALIDATION

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        set_status("loading_extraction_results", 10.0)
        artifact = _wait_for_key_field_artifact(request.document_id)
        payload = (artifact or {}).get("artifact_payload") if isinstance(artifact, dict) else None
        if not isinstance(payload, dict):
            set_status("skipped_no_extraction_artifact", 100.0)
            result_payload = {
                "status": "FAIL",
                "document_status": "INVALID",
                "missing_required_fields": [],
                "invalid_fields": [],
                "low_confidence_fields": [],
                "skip_reason": "key_field_extraction_artifact_missing",
                "message": "Validation requires a completed key_field_extraction artifact.",
            }
            location = store_document_graph(
                document_id=request.document_id,
                repository_id=request.repository_id,
                processor_type=self.processor_type.value,
                payload=result_payload,
                document_type_id=request.document_type_id,
                document_type_name=request.document_type_name,
            )
            return ProcessorResult(
                processor_type=self.processor_type.value,
                document_id=request.document_id,
                storage_backend=self.storage_backend,
                result_location=location,
                document_metadata=result_payload,
                artifacts=result_payload,
            )

        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        set_status("resolving_effective_fields", 35.0)
        effective_fields = _resolve_effective_fields(request)
        extracted_fields = payload.get("fields") or payload.get("extracted_fields") or []

        threshold = DEFAULT_CONFIDENCE_THRESHOLD
        raw_threshold = getattr(request, "validation_confidence_threshold", None)
        if raw_threshold is None and request.repository_settings is not None:
            settings_payload = request.repository_settings.model_dump()
            raw_threshold = settings_payload.get("validation_confidence_threshold")
        if raw_threshold is not None:
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError):
                threshold = DEFAULT_CONFIDENCE_THRESHOLD

        set_status("validating", 60.0)
        validation = run_document_validation(
            effective_fields=effective_fields,
            extracted_fields=extracted_fields,
            confidence_threshold=threshold,
        )
        validation.update(
            {
                "document_type_id": request.document_type_id or payload.get("document_type_id"),
                "document_type_name": request.document_type_name,
                "source": "document_validation",
                "source_pages": payload.get("source_pages"),
            }
        )

        set_status("storing", 85.0)
        location = store_document_graph(
            document_id=request.document_id,
            repository_id=request.repository_id,
            processor_type=self.processor_type.value,
            payload=validation,
            document_type_id=request.document_type_id,
            document_type_name=request.document_type_name,
        )
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata=validation,
            artifacts=validation,
        )
