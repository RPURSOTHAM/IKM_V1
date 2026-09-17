"""Resolve document-info and document-type metadata for consumer APIs."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_UUID_FILENAME_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?:\.[a-z0-9]+)?$",
    re.IGNORECASE,
)

DOCUMENT_INFO_KEYS = (
    "document_id",
    "document_name",
    "original_file_name",
    "document_type",
    "file_extension",
    "file_size_bytes",
    "page_count",
    "character_count",
    "line_count",
    "chunk_count",
    "embedded_chunk_count",
    "collection_name",
    "tenant_id",
    "repository_id",
    "citation_retainment",
    "indexing_status",
    "processor_status",
    "completed_at",
    "result_location",
)


from src.features.documents.infrastructure.content.page_count import page_count_from_file


def is_internal_storage_filename(name: str | None, document_id: str | None = None) -> bool:
    normalized = str(name or "").strip()
    if not normalized:
        return True
    doc_id = str(document_id or "").strip()
    if doc_id and normalized == doc_id:
        return True
    if doc_id and Path(normalized).stem == doc_id:
        return True
    return bool(_UUID_FILENAME_RE.match(normalized))


def resolve_display_filename(record: dict[str, Any], *, job: dict[str, Any] | None = None) -> str | None:
    """Return the user-facing upload filename when intake stored an internal uuid-based name."""
    document_id = str(record.get("document_id") or "").strip()
    metadata = dict(record.get("metadata") or {})
    candidates: list[Any] = [
        record.get("original_file_name"),
        metadata.get("original_file_name"),
        metadata.get("original_filename"),
        metadata.get("filename"),
    ]
    doc_info = metadata.get("document_info")
    if isinstance(doc_info, dict):
        candidates.extend([doc_info.get("original_file_name"), doc_info.get("document_name")])
    if job:
        chunk_outcome = _chunking_outcome(job)
        chunk_meta = dict(chunk_outcome.get("document_metadata") or {})
        candidates.append(chunk_meta.get("original_file_name"))
    candidates.append(record.get("document_name"))
    for candidate in candidates:
        name = str(candidate or "").strip()
        if name and not is_internal_storage_filename(name, document_id):
            return name
    return None


def resolve_display_filename_from_job(job: dict[str, Any], *, document_id: str = "") -> str | None:
    """Resolve a human-readable filename from a MySQL document job when intake JSON is missing."""
    doc_id = document_id or str(job.get("document_id") or "").strip()
    meta = job.get("scheduling_metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    chunk_meta = dict(_chunking_outcome(job).get("document_metadata") or {})
    candidates: list[Any] = [
        meta.get("original_file_name"),
        meta.get("original_filename"),
        meta.get("filename"),
        chunk_meta.get("original_file_name"),
        chunk_meta.get("original_filename"),
        job.get("document_name"),
    ]
    for candidate in candidates:
        name = str(candidate or "").strip()
        if name and not is_internal_storage_filename(name, doc_id):
            return name
    return None


def _page_count_from_repository_file(record: dict[str, Any]) -> int | None:
    repository_path = str(record.get("repository_path") or "").strip()
    if not repository_path:
        return None
    try:
        return page_count_from_file(Path(repository_path))
    except Exception:
        _logger.debug("Page count refresh failed for %s", record.get("document_id"), exc_info=True)
        return None


def _resolved_page_count(record: dict[str, Any], *candidates: Any) -> int | None:
    values: list[int] = []
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    from_file = _page_count_from_repository_file(record)
    if from_file is not None and from_file > 0:
        values.append(from_file)
    return max(values) if values else None


def _file_size_from_path(repository_path: str | None) -> int | None:
    if not repository_path:
        return None
    try:
        path = Path(repository_path)
        if path.is_file():
            return int(path.stat().st_size)
    except OSError:
        return None
    return None


def _chunking_outcome(job: dict[str, Any] | None) -> dict[str, Any]:
    if not job:
        return {}
    meta = job.get("scheduling_metadata") or {}
    if not isinstance(meta, dict):
        return {}
    results = meta.get("processor_results") or {}
    if not isinstance(results, dict):
        return {}
    outcome = results.get("chunking_vectorizing") or {}
    return outcome if isinstance(outcome, dict) else {}


def _metadata_extraction_outcome(job: dict[str, Any] | None) -> dict[str, Any]:
    if not job:
        return {}
    meta = job.get("scheduling_metadata") or {}
    if not isinstance(meta, dict):
        return {}
    results = meta.get("processor_results") or {}
    if not isinstance(results, dict):
        return {}
    outcome = results.get("metadata_extraction") or {}
    return outcome if isinstance(outcome, dict) else {}


def _weaviate_document_stats(record: dict[str, Any]) -> dict[str, Any]:
    collection_name = str(record.get("collection_name") or "").strip()
    document_id = str(record.get("document_id") or "").strip()
    if collection_name and document_id:
        by_id = weaviate_stats_for_document_id(
            collection_name,
            document_id,
            tenant=str(record.get("tenant_id")) if record.get("tenant_id") else None,
            file_size_bytes=record.get("file_size_bytes"),
        )
        if by_id.get("chunk_count"):
            return by_id
    document_name = str(record.get("original_file_name") or record.get("document_name") or "").strip()
    if not collection_name or not document_name:
        return {}
    return _weaviate_stats_for_doc_name(
        collection_name,
        document_name,
        tenant=str(record.get("tenant_id")) if record.get("tenant_id") else None,
    )


def _weaviate_stats_for_doc_name(
    collection_name: str,
    document_name: str,
    *,
    tenant: str | None = None,
) -> dict[str, Any]:
    if not collection_name or not document_name:
        return {}
    try:
        from src.features.repositories.infrastructure import weaviate_admin as repository_admin

        payload = repository_admin.list_chunks_metadata(
            collection_name,
            tenant=tenant,
            document_name=document_name,
            limit=500,
            offset=0,
            include_text=False,
            include_vector=False,
        )
        returned = int(payload.get("returned") or 0)
        if returned <= 0:
            return {"chunk_count": 0, "indexing_status": "not_indexed"}
        chunks = payload.get("chunks") or []
        props = (chunks[0].get("properties") or {}) if chunks else {}
        file_size = props.get("file_size_bytes")
        line_count = props.get("line_count")
        return {
            "chunk_count": returned,
            "file_size_bytes": int(file_size) if file_size is not None else None,
            "line_count": int(line_count) if line_count is not None else None,
            "indexing_status": "indexed",
            "doc_name": document_name,
        }
    except Exception:
        _logger.debug("Weaviate document stats unavailable for %s", document_name, exc_info=True)
        return {}


def _weaviate_stats_by_file_size(
    collection_name: str,
    file_size_bytes: int,
    *,
    tenant: str | None = None,
) -> dict[str, Any]:
    if not collection_name or file_size_bytes <= 0:
        return {}
    try:
        from weaviate.classes.query import Filter
        from weaviate.connect import executor as wexec

        from src.features.repositories.infrastructure.weaviate_backend import backend
        from src.features.repositories.infrastructure.weaviate_admin import _collection_handle

        client = backend.require_client()
        col = _collection_handle(client, collection_name, tenant)
        result = wexec.result(
            col.query.fetch_objects(
                filters=Filter.by_property("file_size_bytes").equal(int(file_size_bytes)),
                limit=1,
            )
        )
        if not result.objects:
            return {}
        props = dict(result.objects[0].properties or {})
        doc_name = str(props.get("doc_name") or "").strip()
        if not doc_name:
            return {}
        stats = _weaviate_stats_for_doc_name(collection_name, doc_name, tenant=tenant)
        return stats if stats.get("chunk_count") else {}
    except Exception:
        _logger.debug(
            "Weaviate file-size lookup failed for collection=%s size=%s",
            collection_name,
            file_size_bytes,
            exc_info=True,
        )
        return {}


def weaviate_stats_for_document_id(
    collection_name: str,
    document_id: str,
    *,
    tenant: str | None = None,
    file_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Locate indexed chunks for a portal document id when intake metadata is gone."""
    doc_id = str(document_id or "").strip()
    if not collection_name or not doc_id:
        return {}

    # Prefer the processor-written document_id property (authoritative).
    try:
        from src.features.repositories.infrastructure import weaviate_admin as repository_admin

        payload = repository_admin.list_chunks_metadata(
            collection_name,
            tenant=tenant,
            document_id=doc_id,
            limit=500,
            offset=0,
            include_text=False,
            include_vector=False,
        )
        returned = int(payload.get("returned") or 0)
        if returned > 0:
            chunks = payload.get("chunks") or []
            props = (chunks[0].get("properties") or {}) if chunks else {}
            file_size = props.get("file_size_bytes")
            line_count = props.get("line_count")
            return {
                "chunk_count": returned,
                "file_size_bytes": int(file_size) if file_size is not None else None,
                "line_count": int(line_count) if line_count is not None else None,
                "indexing_status": "indexed",
                "doc_name": str(props.get("doc_name") or props.get("document_name") or doc_id).strip(),
            }
    except Exception:
        _logger.debug("Weaviate document_id stats unavailable for %s", doc_id, exc_info=True)

    # Legacy fallback: some older objects only had doc_name == "{uuid}" / "{uuid}.ext".
    for doc_name in [doc_id, *[f"{doc_id}.{ext}" for ext in ("pdf", "docx", "doc", "txt")]]:
        stats = _weaviate_stats_for_doc_name(collection_name, doc_name, tenant=tenant)
        if stats.get("chunk_count"):
            return stats

    if file_size_bytes is not None and file_size_bytes > 0:
        matched = _weaviate_stats_by_file_size(collection_name, int(file_size_bytes), tenant=tenant)
        if matched.get("chunk_count"):
            return matched
    return {"chunk_count": 0, "indexing_status": "not_indexed"}


def resolve_document_info(record: dict[str, Any], *, job: dict[str, Any] | None = None) -> dict[str, Any]:
    """Document facts produced by chunking / vectorization and indexing."""
    intake_meta = dict(record.get("metadata") or {})
    persisted = intake_meta.get("document_info")
    info = dict(persisted) if isinstance(persisted, dict) else {}

    chunk_outcome = _chunking_outcome(job)
    chunk_meta = dict(chunk_outcome.get("document_metadata") or {})
    weaviate_stats = _weaviate_document_stats(record)

    file_size = (
        info.get("file_size_bytes")
        or chunk_meta.get("file_size_bytes")
        or weaviate_stats.get("file_size_bytes")
        or _file_size_from_path(str(record.get("repository_path") or ""))
    )
    chunk_count = info.get("chunk_count") or chunk_meta.get("chunk_count") or weaviate_stats.get("chunk_count")

    resolved: dict[str, Any] = {
        "document_id": record.get("document_id"),
        "document_name": record.get("document_name"),
        "original_file_name": resolve_display_filename(record, job=job) or record.get("original_file_name"),
        "document_type": record.get("document_type"),
        "file_extension": chunk_meta.get("file_extension") or record.get("document_type"),
        "file_size_bytes": int(file_size) if file_size is not None else None,
        "page_count": _resolved_page_count(record, chunk_meta.get("page_count"), info.get("page_count")),
        "character_count": chunk_meta.get("character_count"),
        "line_count": chunk_meta.get("line_count"),
        "chunk_count": int(chunk_count) if chunk_count is not None else None,
        "embedded_chunk_count": chunk_meta.get("embedded_chunk_count") or chunk_meta.get("chunk_count"),
        "collection_name": chunk_meta.get("collection_name") or record.get("collection_name"),
        "tenant_id": chunk_meta.get("tenant_id") or record.get("tenant_id"),
        "repository_id": record.get("repository_id") or intake_meta.get("repository_id"),
        "citation_retainment": chunk_meta.get("citation_retainment"),
        "indexing_status": weaviate_stats.get("indexing_status")
        or ("indexed" if chunk_count else "pending"),
        "processor_status": chunk_outcome.get("status"),
        "completed_at": chunk_outcome.get("completed_at"),
        "result_location": chunk_outcome.get("result_location"),
        "source": info.get("source") or ("chunking_vectorizing" if chunk_meta else "intake"),
    }
    out = {key: resolved.get(key) for key in DOCUMENT_INFO_KEYS if resolved.get(key) is not None}
    out["document_id"] = record.get("document_id")
    return out


def _normalize_extracted_fields(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        fields: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            field_name = str(item.get("field_name") or item.get("name") or "").strip()
            if not field_name:
                continue
            fields.append(
                {
                    "field_name": field_name,
                    "value": item.get("value"),
                    "confidence": item.get("confidence"),
                    "extraction_model": item.get("extraction_model"),
                }
            )
        return fields
    if isinstance(raw, dict):
        return [
            {"field_name": str(key), "value": value, "confidence": None, "extraction_model": "intake"}
            for key, value in raw.items()
            if key not in {"document_type_id", "status", "completed_at", "source", "fields"}
        ]
    return []


def fetch_type_metadata_from_neo4j(document_id: str) -> dict[str, Any] | None:
    uri = (os.getenv("NEO4J_URI") or os.getenv("NEO4J_BOLT_URI") or "").strip()
    if not uri:
        return None
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None

    user = (os.getenv("NEO4J_USER") or "neo4j").strip()
    password_raw = os.getenv("NEO4J_PASSWORD")
    if password_raw:
        password = password_raw.strip()
    else:
        password = (os.getenv("NEO4J_AUTH") or "neo4j/password").split("/")[-1].strip()

    cypher = """
    MATCH (a:DocumentArtifact {document_id: $document_id, processor_type: 'metadata_extraction'})
    RETURN a.payload_json AS payload_json, a.updated_at AS updated_at
    LIMIT 1
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            row = session.run(cypher, document_id=document_id).single()
            if not row:
                return None
            payload_raw = row.get("payload_json")
            if not payload_raw:
                return None
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            if not isinstance(payload, dict):
                return None
            return {
                "document_type_id": payload.get("document_type_id"),
                "fields": _normalize_extracted_fields(payload.get("fields")),
                "extracted_field_count": len(payload.get("fields") or []),
                "source": "neo4j",
                "completed_at": str(row.get("updated_at") or ""),
            }
    except Exception:
        _logger.debug("Neo4j type metadata lookup failed for %s", document_id, exc_info=True)
        return None
    finally:
        driver.close()


def _normalize_key_field_map(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, Any] = {}
    for field_name, value in raw.items():
        key = str(field_name).strip()
        if not key:
            continue
        if isinstance(value, dict):
            normalized_value = {
                "value": value.get("value"),
                "confidence": value.get("confidence"),
            }
            for extra_key in ("source", "updated_at", "updated_by", "change_reason"):
                if value.get(extra_key) is not None:
                    normalized_value[extra_key] = value.get(extra_key)
            normalized[key] = normalized_value
        else:
            normalized[key] = {"value": value, "confidence": None}
    return normalized


def fetch_key_field_artifact_from_neo4j(document_id: str) -> dict[str, Any] | None:
    uri = (os.getenv("NEO4J_URI") or os.getenv("NEO4J_BOLT_URI") or "").strip()
    if not uri:
        return None
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None

    user = (os.getenv("NEO4J_USER") or "neo4j").strip()
    password_raw = os.getenv("NEO4J_PASSWORD")
    if password_raw:
        password = password_raw.strip()
    else:
        password = (os.getenv("NEO4J_AUTH") or "neo4j/password").split("/")[-1].strip()

    cypher = """
    MATCH (a:DocumentArtifact {document_id: $document_id, processor_type: 'key_field_extraction'})
    RETURN a.payload_json AS payload_json, a.updated_at AS updated_at
    LIMIT 1
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            row = session.run(cypher, document_id=document_id).single()
            if not row:
                return None
            payload_raw = row.get("payload_json")
            if not payload_raw:
                return None
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            if not isinstance(payload, dict):
                return None
            return {
                "artifact_payload": payload,
                "document_type_id": payload.get("document_type_id"),
                "fields": _normalize_key_field_map(payload.get("fields")),
                "source": "neo4j",
                "updated_at": str(row.get("updated_at") or ""),
            }
    except Exception:
        _logger.debug("Neo4j key field lookup failed for %s", document_id, exc_info=True)
        return None
    finally:
        driver.close()


def fetch_key_fields_from_neo4j(document_id: str) -> dict[str, Any] | None:
    payload = fetch_key_field_artifact_from_neo4j(document_id)
    if not payload:
        return None
    return {
        "document_type_id": payload.get("document_type_id"),
        "fields": payload.get("fields") or {},
        "source": payload.get("source"),
        "updated_at": payload.get("updated_at"),
    }


def resolve_key_fields(record: dict[str, Any]) -> dict[str, Any]:
    document_id = str(record.get("document_id") or "")
    document_type_id = (record.get("metadata") or {}).get("document_type_id")
    if not document_type_id:
        try:
            from src.features.document_types.infrastructure.document_type_repository import get_document_type_store

            store = get_document_type_store()
            if store:
                instance = store.get_document_instance(document_id)
                if instance:
                    document_type_id = instance.get("document_type_id")
        except Exception:
            _logger.debug("Could not load document_instance for document_id=%s", document_id, exc_info=True)

    neo = fetch_key_fields_from_neo4j(document_id)
    if neo:
        return {
            "document_id": document_id,
            "document_type_id": neo.get("document_type_id") or document_type_id,
            "fields": neo.get("fields") or {},
            "updated_at": neo.get("updated_at"),
            "source": neo.get("source") or "neo4j",
        }

    # Fall back to intake cache written by record_processor_outcome.
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    cached_fields = metadata.get("key_field_metadata")
    cached_extracted = metadata.get("extracted_metadata")
    fields: dict[str, Any] = {}
    if isinstance(cached_fields, dict) and isinstance(cached_fields.get("fields"), list):
        for item in cached_fields.get("fields") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("field_name") or item.get("name")
            if name:
                fields[str(name)] = item
    elif isinstance(cached_extracted, dict) and cached_extracted:
        fields = {
            str(name): {"field_name": str(name), "value": value}
            for name, value in cached_extracted.items()
        }
    if fields:
        return {
            "document_id": document_id,
            "document_type_id": document_type_id or metadata.get("document_type_id"),
            "fields": fields,
            "updated_at": (cached_fields or {}).get("completed_at") if isinstance(cached_fields, dict) else None,
            "source": (
                (cached_fields or {}).get("source")
                if isinstance(cached_fields, dict) and cached_fields.get("source")
                else "intake"
            ),
        }

    return {
        "document_id": document_id,
        "document_type_id": document_type_id,
        "fields": {},
        "updated_at": None,
        "source": "none",
    }


def _derived_source_pages(page_count: int | None) -> list[int]:
    if not page_count or page_count <= 0:
        return []
    try:
        from src.features.document_processing.key_fields.page_context import select_extraction_page_numbers

        return select_extraction_page_numbers(int(page_count))
    except Exception:
        return []


def resolve_key_field_metadata_bundle(record: dict[str, Any], *, job: dict[str, Any] | None = None) -> dict[str, Any]:
    key_fields = resolve_key_fields(record)
    fields_map = key_fields.get("fields") if isinstance(key_fields.get("fields"), dict) else {}
    extracted_metadata: dict[str, Any] = {}
    for field_name, value in (fields_map or {}).items():
        if isinstance(value, dict):
            scalar = value.get("value")
        else:
            scalar = value
        if scalar is not None:
            extracted_metadata[str(field_name)] = scalar

    source_pages: list[int] = []
    artifact_payload: dict[str, Any] = {}
    artifact = fetch_key_field_artifact_from_neo4j(str(record.get("document_id") or ""))
    if artifact and isinstance(artifact.get("artifact_payload"), dict):
        artifact_payload = artifact.get("artifact_payload") or {}
        raw_pages = artifact_payload.get("source_pages")
        if isinstance(raw_pages, list):
            source_pages = [int(p) for p in raw_pages if str(p).isdigit() and int(p) > 0]

    if not source_pages:
        page_count = resolve_document_info(record, job=job).get("page_count")
        source_pages = _derived_source_pages(int(page_count)) if page_count is not None else []

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    processing = record.get("processing") if isinstance(record.get("processing"), dict) else {}
    document_type_id = (
        key_fields.get("document_type_id")
        or record.get("document_type_id")
        or metadata.get("document_type_id")
        or processing.get("document_type_id")
    )
    document_type_name = (
        record.get("document_type_name")
        or metadata.get("document_type_name")
        or processing.get("document_type_name")
    )
    effective_fields = metadata.get("effective_fields") or processing.get("effective_fields") or []
    if not isinstance(effective_fields, list):
        effective_fields = []
    if not effective_fields and document_type_id:
        try:
            from src.features.document_types.application.document_type_service import get_document_type_service

            bundle = get_document_type_service().resolve_effective_fields(str(document_type_id))
            fields = bundle.get("fields") if isinstance(bundle, dict) else []
            if isinstance(fields, list):
                effective_fields = fields
        except Exception:
            effective_fields = []
    if not document_type_name and document_type_id:
        try:
            from src.features.document_types.application.document_type_service import get_document_type_service

            doc_type = get_document_type_service().get_type(str(document_type_id))
            document_type_name = str(doc_type.get("name") or "").strip() or None
        except Exception:
            document_type_name = None

    kfe_outcome: dict[str, Any] = {}
    if isinstance(job, dict):
        sched = job.get("scheduling_metadata") or {}
        if isinstance(sched, dict):
            results = sched.get("processor_results") or {}
            if isinstance(results, dict):
                outcome = results.get("key_field_extraction") or {}
                if isinstance(outcome, dict):
                    kfe_outcome = outcome

    key_field_metadata = {
        "processor": "key_field_extraction",
        "source": key_fields.get("source"),
        "source_pages": source_pages,
        "updated_at": key_fields.get("updated_at") or kfe_outcome.get("completed_at"),
        "document_type_id": document_type_id,
        "document_type_name": document_type_name,
        "processor_status": kfe_outcome.get("status"),
        "completed_at": kfe_outcome.get("completed_at"),
        "result_location": kfe_outcome.get("result_location"),
    }
    if isinstance(artifact_payload.get("source"), str):
        key_field_metadata["artifact_source"] = artifact_payload.get("source")
    return {
        "document_type_id": document_type_id,
        "document_type_name": document_type_name,
        "effective_fields": effective_fields,
        "extracted_metadata": extracted_metadata,
        "key_field_metadata": key_field_metadata,
    }


def resolve_type_metadata(record: dict[str, Any], *, job: dict[str, Any] | None = None) -> dict[str, Any]:
    """Document-type field metadata from metadata_extraction processor."""
    intake_meta = dict(record.get("metadata") or {})
    persisted = intake_meta.get("type_metadata")
    base = dict(persisted) if isinstance(persisted, dict) else {}

    extraction_outcome = _metadata_extraction_outcome(job)
    extraction_meta = dict(extraction_outcome.get("document_metadata") or {})
    fields = (
        base.get("fields")
        or _normalize_extracted_fields(extraction_meta.get("extracted_fields"))
        or _normalize_extracted_fields(intake_meta.get("metadata_extraction"))
    )

    if not fields:
        neo = fetch_type_metadata_from_neo4j(str(record.get("document_id") or ""))
        if neo:
            fields = neo.get("fields") or []
            base.setdefault("document_type_id", neo.get("document_type_id"))
            base.setdefault("completed_at", neo.get("completed_at"))
            base.setdefault("source", neo.get("source"))

    document_type_id = (
        base.get("document_type_id")
        or extraction_meta.get("document_type_id")
        or (record.get("processing") or {}).get("document_type_id")
    )

    processor_status = extraction_outcome.get("status") or base.get("status")
    # Intake may persist source="pending" before the processor finishes; do not
    # treat that as a final source once the processor has a terminal status.
    persisted_source = str(base.get("source") or "").strip().lower()
    if persisted_source and persisted_source not in {"pending", "none"}:
        source = base.get("source")
    elif fields:
        source = "metadata_extraction"
    elif processor_status and str(processor_status).upper() in {"COMPLETED", "SKIPPED"}:
        source = "metadata_extraction"
    else:
        source = "pending"

    return {
        "document_id": record.get("document_id"),
        "document_type_id": document_type_id,
        "status": processor_status,
        "fields": fields,
        "extracted_field_count": len(fields),
        "processor_status": extraction_outcome.get("status"),
        "completed_at": extraction_outcome.get("completed_at") or base.get("completed_at"),
        "result_location": extraction_outcome.get("result_location") or base.get("result_location"),
        "error_details": extraction_outcome.get("error_details"),
        "source": source,
    }


def fetch_validation_artifact_from_neo4j(document_id: str) -> dict[str, Any] | None:
    uri = (os.getenv("NEO4J_URI") or os.getenv("NEO4J_BOLT_URI") or "").strip()
    if not uri:
        return None
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return None

    user = (os.getenv("NEO4J_USER") or "neo4j").strip()
    password_raw = os.getenv("NEO4J_PASSWORD")
    if password_raw:
        password = password_raw.strip()
    else:
        password = (os.getenv("NEO4J_AUTH") or "neo4j/password").split("/")[-1].strip()

    cypher = """
    MATCH (a:DocumentArtifact {document_id: $document_id, processor_type: 'document_validation'})
    RETURN a.payload_json AS payload_json, a.updated_at AS updated_at
    LIMIT 1
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            row = session.run(cypher, document_id=document_id).single()
            if not row:
                return None
            payload_raw = row.get("payload_json")
            if not payload_raw:
                return None
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            if not isinstance(payload, dict):
                return None
            return {
                "artifact_payload": payload,
                "source": "neo4j",
                "updated_at": str(row.get("updated_at") or ""),
            }
    except Exception:
        _logger.debug("Neo4j validation lookup failed for %s", document_id, exc_info=True)
        return None
    finally:
        driver.close()


def resolve_validation_bundle(record: dict[str, Any], *, job: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    persisted = metadata.get("validation") if isinstance(metadata.get("validation"), dict) else {}
    validation = dict(persisted) if persisted else {}

    if job:
        meta = job.get("scheduling_metadata") or {}
        if isinstance(meta, dict):
            outcome = (meta.get("processor_results") or {}).get("document_validation") or {}
            doc_meta = outcome.get("document_metadata") if isinstance(outcome, dict) else None
            if isinstance(doc_meta, dict) and doc_meta.get("status"):
                validation = {
                    "status": doc_meta.get("status"),
                    "document_status": doc_meta.get("document_status"),
                    "missing_required_fields": doc_meta.get("missing_required_fields") or [],
                    "invalid_fields": doc_meta.get("invalid_fields") or [],
                    "low_confidence_fields": doc_meta.get("low_confidence_fields") or [],
                    "confidence_threshold": doc_meta.get("confidence_threshold"),
                    "source": "document_job",
                    "completed_at": outcome.get("completed_at"),
                }

    if not validation.get("status"):
        neo = fetch_validation_artifact_from_neo4j(str(record.get("document_id") or ""))
        payload = (neo or {}).get("artifact_payload") if isinstance(neo, dict) else None
        if isinstance(payload, dict):
            validation = {
                "status": payload.get("status"),
                "document_status": payload.get("document_status"),
                "missing_required_fields": payload.get("missing_required_fields") or [],
                "invalid_fields": payload.get("invalid_fields") or [],
                "low_confidence_fields": payload.get("low_confidence_fields") or [],
                "confidence_threshold": payload.get("confidence_threshold"),
                "source": "neo4j",
                "updated_at": (neo or {}).get("updated_at"),
            }

    if not validation:
        return {
            "status": None,
            "document_status": None,
            "missing_required_fields": [],
            "invalid_fields": [],
            "low_confidence_fields": [],
        }

    status = validation.get("status")
    document_status = validation.get("document_status")
    if not document_status and status:
        try:
            from src.features.document_validation.application.validation_pipeline import map_validation_status_to_document_status

            document_status = map_validation_status_to_document_status(str(status))
        except Exception:
            mapping = {"PASS": "VALID", "WARNING": "WARNING", "FAIL": "INVALID"}
            document_status = mapping.get(str(status).upper())

    return {
        "status": status,
        "document_status": document_status,
        "missing_required_fields": validation.get("missing_required_fields") or [],
        "invalid_fields": validation.get("invalid_fields") or [],
        "low_confidence_fields": validation.get("low_confidence_fields") or [],
        "confidence_threshold": validation.get("confidence_threshold"),
        "source": validation.get("source"),
        "completed_at": validation.get("completed_at") or validation.get("updated_at"),
    }


def resolve_document_metadata_bundle(record: dict[str, Any], *, job: dict[str, Any] | None = None) -> dict[str, Any]:
    document_info = resolve_document_info(record, job=job)
    type_metadata = resolve_type_metadata(record, job=job)
    key_field_bundle = resolve_key_field_metadata_bundle(record, job=job)
    validation = resolve_validation_bundle(record, job=job)
    processing = record.get("processing") or {}
    if not isinstance(processing, dict):
        processing = {}
    enabled = processing.get("enabled_processor_types")
    if not enabled and job:
        meta = job.get("scheduling_metadata") or {}
        if isinstance(meta, dict):
            enabled = meta.get("enabled_processor_types")
    document_type_id = (
        key_field_bundle.get("document_type_id")
        or processing.get("document_type_id")
        or record.get("document_type_id")
    )
    document_type_name = (
        key_field_bundle.get("document_type_name")
        or processing.get("document_type_name")
        or record.get("document_type_name")
    )
    return {
        "document_id": record.get("document_id"),
        "document_info": document_info,
        "type_metadata": type_metadata,
        "document_type_id": document_type_id,
        "document_type_name": document_type_name,
        "effective_fields": key_field_bundle.get("effective_fields") or [],
        "extracted_metadata": key_field_bundle.get("extracted_metadata") or {},
        "key_field_metadata": key_field_bundle.get("key_field_metadata") or {},
        "validation": {
            "status": validation.get("status"),
            "missing_required_fields": validation.get("missing_required_fields") or [],
            "invalid_fields": validation.get("invalid_fields") or [],
            "low_confidence_fields": validation.get("low_confidence_fields") or [],
        },
        "validation_status": validation.get("document_status"),
        "review": _resolve_review_section(record),
        "processing": {
            "enabled_processor_types": enabled,
            "metadata_extraction_enabled": bool(processing.get("metadata_extraction")),
            "document_type_id": processing.get("document_type_id"),
            "document_type_name": processing.get("document_type_name"),
            "validation_enabled": bool(processing.get("validation_enabled")),
        },
    }


def _resolve_review_section(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    persisted = metadata.get("review") if isinstance(metadata.get("review"), dict) else None
    document_id = str(record.get("document_id") or "").strip()
    if document_id:
        try:
            from src.features.human_review.application.review_service import get_document_review_service

            live = get_document_review_service().get_review_for_metadata(document_id)
            if live:
                return live
        except Exception:
            pass
    if persisted:
        return {
            "review_id": persisted.get("review_id"),
            "status": persisted.get("status") or persisted.get("review_status"),
            "assigned_to": persisted.get("assigned_to"),
            "last_updated": persisted.get("last_updated") or persisted.get("updated_at"),
            "validation_status": persisted.get("validation_status"),
        }
    return None


def check_document_access(record: dict[str, Any]) -> None:
    from src.application.consumer_api.context import get_current_user_from_context
    from src.features.authentication.domain.authentication_exceptions import AuthenticationError
    from src.features.users.application.user_service import get_platform_security_service

    actor = get_current_user_from_context()
    if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
        raise AuthenticationError("Authentication required")
    repository_id = record.get("repository_id") or (record.get("metadata") or {}).get("repository_id")
    if repository_id:
        get_platform_security_service().check_retrieval_access(actor, str(repository_id))
