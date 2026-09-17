"""Sync terminal job status into the DMS JSON intake store.

Processors mount ``shared`` but not ``dms_service``. Job-row updates must still
advance the intake ``status`` / completion timestamps so clients that read the
JSON store (or upload responses cached from it) do not remain stuck at
``queued`` after ``document_job`` reaches COMPLETED/FAILED.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_DEFAULT_INTAKE_DB = "/app/src/dms_service/consumer_api_service/data/rag_api.db"


def resolve_intake_json_path() -> Path:
    raw = (os.getenv("RAG_API_DB_PATH") or _DEFAULT_INTAKE_DB).strip()
    path = Path(raw)
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        path = path.with_suffix(".json")
    return path


def sync_intake_document_status(
    document_id: str,
    status: str,
    *,
    queued_at: float | None = None,
    completed_at: float | None = None,
    error_details: str | None = None,
) -> bool:
    """Update intake JSON status fields. Returns True when a record was patched."""
    if not document_id or not status:
        return False
    path = resolve_intake_json_path()
    if not path.parent.exists():
        logger.debug(
            "Intake sync skipped for %s: parent missing (%s)",
            document_id,
            path.parent,
        )
        return False

    with _lock:
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8") or "{}") or {}
            else:
                payload = {"documents": {}, "repository_config": None}
            documents = payload.setdefault("documents", {})
            record = documents.get(document_id)
            if not isinstance(record, dict):
                logger.debug("Intake sync skipped for %s: document not in store", document_id)
                return False
            record["status"] = status
            if queued_at is not None:
                record["queue_submission_timestamp"] = queued_at
            if completed_at is not None:
                record["processing_completion_timestamp"] = completed_at
            if error_details is not None:
                record["error_details"] = error_details
            documents[document_id] = record
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return True
        except Exception:
            logger.debug("Intake sync failed for document_id=%s", document_id, exc_info=True)
            return False


def sync_intake_document_metadata(
    document_id: str,
    metadata: dict[str, Any],
    *,
    merge: bool = True,
) -> bool:
    """Merge metadata onto an intake record when the JSON store is reachable."""
    if not document_id or not metadata:
        return False
    path = resolve_intake_json_path()
    if not path.parent.exists():
        return False
    system_keys = frozenset({"job_id", "batch_id", "repository_id"})
    with _lock:
        try:
            if not path.exists():
                return False
            payload = json.loads(path.read_text(encoding="utf-8") or "{}") or {}
            documents = payload.setdefault("documents", {})
            record = documents.get(document_id)
            if not isinstance(record, dict):
                return False
            old_meta = dict(record.get("metadata") or {})
            system_values = {key: old_meta[key] for key in system_keys if key in old_meta}
            if merge:
                updated_meta = {**old_meta, **metadata}
            else:
                updated_meta = {**system_values, **metadata}
            for key, value in system_values.items():
                updated_meta[key] = value
            record["metadata"] = updated_meta
            record["metadata_updated_at"] = time.time()
            documents[document_id] = record
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return True
        except Exception:
            logger.debug(
                "Intake metadata sync failed for document_id=%s",
                document_id,
                exc_info=True,
            )
            return False
