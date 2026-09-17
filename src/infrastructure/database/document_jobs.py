"""PostgreSQL-backed ``document_job`` rows — authoritative processing status across API, scheduler, processor."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

from src.infrastructure.database.postgres import (
    MySQLConnectionParams,
    PostgresConnectionParams,
    connect,
    connection,
    mysql_params_from_env,
    postgres_params_from_env,
)

# Re-export connection helpers so existing imports from this module keep working.
__all__ = [
    "DocumentJobStore",
    "MySQLConnectionParams",
    "PostgresConnectionParams",
    "TERMINAL_JOB_STATUSES",
    "aggregate_processor_plan_status",
    "all_enabled_processors_terminal",
    "connect",
    "connection",
    "effective_job_status",
    "get_document_job_store",
    "mysql_params_from_env",
    "postgres_params_from_env",
    "reset_document_job_store_for_tests",
]

logger = logging.getLogger(__name__)

TERMINAL_JOB_STATUSES = frozenset({"COMPLETED", "FAILED", "TIMED_OUT", "QUEUE_FAILED", "STOPPED"})
# Hard-closed statuses that must not be reopened by a sibling processor assignment.
_HARD_CLOSED_JOB_STATUSES = frozenset({"COMPLETED", "TIMED_OUT", "QUEUE_FAILED", "STOPPED"})
_TERMINAL_JOB_STATUS_LIST = ", ".join(f"'{status}'" for status in sorted(TERMINAL_JOB_STATUSES))
_HARD_CLOSED_JOB_STATUS_LIST = ", ".join(f"'{status}'" for status in sorted(_HARD_CLOSED_JOB_STATUSES))
_PROCESSOR_OUTCOME_TERMINAL = frozenset({"COMPLETED", "FAILED"})


def _normalize_processor_results(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    results = meta.get("processor_results") or {}
    return results if isinstance(results, dict) else {}


def _enabled_processor_types(meta: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(meta, dict):
        return ["chunking_vectorizing"]
    enabled = list(meta.get("enabled_processor_types") or ["chunking_vectorizing"])
    return [str(pt) for pt in enabled if str(pt or "").strip()]


def _processor_outcome_status(results: Mapping[str, Any], processor_type: str) -> str:
    outcome = results.get(processor_type) or {}
    if not isinstance(outcome, dict):
        return ""
    return str(outcome.get("status") or "").strip().upper()


def all_enabled_processors_terminal(meta: Mapping[str, Any] | None) -> bool:
    """True when every enabled processor has a COMPLETED or FAILED outcome."""
    enabled = _enabled_processor_types(meta)
    if not enabled:
        return False
    results = _normalize_processor_results(meta)
    return all(_processor_outcome_status(results, pt) in _PROCESSOR_OUTCOME_TERMINAL for pt in enabled)


def aggregate_processor_plan_status(meta: Mapping[str, Any] | None) -> str | None:
    """Derive plan-level status from per-processor outcomes.

    Returns:
      - COMPLETED when every enabled processor completed
      - FAILED when every enabled processor finished and at least one failed
      - None when the plan is still in progress (some processors pending)
    """
    enabled = _enabled_processor_types(meta)
    if not enabled:
        return None
    results = _normalize_processor_results(meta)
    statuses = [_processor_outcome_status(results, pt) for pt in enabled]
    if not all(st in _PROCESSOR_OUTCOME_TERMINAL for st in statuses):
        return None
    if all(st == "COMPLETED" for st in statuses):
        return "COMPLETED"
    return "FAILED"


def effective_job_status(job: Mapping[str, Any] | None) -> str:
    """Resolve job status when parallel processors share one row.

    Per-processor outcomes in ``scheduling_metadata.processor_results`` are the
    source of truth once the enabled plan has finished. A single processor
    failure no longer forces FAILED while sibling processors are still pending.
    """
    if not job:
        return ""
    status = str(job.get("status") or "").strip().upper()
    meta = job.get("scheduling_metadata") or {}
    if not isinstance(meta, dict):
        return status

    aggregated = aggregate_processor_plan_status(meta)
    if aggregated:
        return aggregated

    enabled = _enabled_processor_types(meta)
    if enabled:
        results = _normalize_processor_results(meta)
        pending = [
            pt
            for pt in enabled
            if _processor_outcome_status(results, pt) not in _PROCESSOR_OUTCOME_TERMINAL
        ]
        if pending and status in {"ASSIGNED", "DISPATCHING", "STARTED"}:
            return "IN_PROGRESS"

    # Plan still running: never surface premature FAILED from a sibling failure.
    if status == "FAILED" and _enabled_processor_types(meta):
        results = _normalize_processor_results(meta)
        pending = [
            pt
            for pt in _enabled_processor_types(meta)
            if _processor_outcome_status(results, pt) not in _PROCESSOR_OUTCOME_TERMINAL
        ]
        if pending:
            return "IN_PROGRESS"
    return status


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DocumentJobStore:
    """CRUD for ``document_job`` (one row per document intake; ``job_id`` is distinct from ``document_id``)."""

    CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS document_job (
      job_id CHAR(36) NOT NULL,
      document_id CHAR(36) NOT NULL,
      batch_id CHAR(36) NULL,
      document_name VARCHAR(512) NOT NULL,
      source_location VARCHAR(2048) NOT NULL,
      collection_name VARCHAR(256) NULL,
      tenant_id VARCHAR(256) NULL,
      status VARCHAR(32) NOT NULL,
      retry_count INT NOT NULL DEFAULT 0,
      processor_id VARCHAR(256) NULL,
      assigned_at TIMESTAMPTZ NULL,
      scheduling_metadata JSONB NULL,
      progress_percent DECIMAL(6,2) NULL,
      heartbeat_at TIMESTAMPTZ NULL,
      processing_notes TEXT NULL,
      started_at TIMESTAMPTZ NULL,
      completed_at TIMESTAMPTZ NULL,
      error_details TEXT NULL,
      result_location VARCHAR(2048) NULL,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (job_id),
      UNIQUE (document_id)
    );
    """

    INDEX_SQL = (
        "CREATE INDEX IF NOT EXISTS idx_document_job_batch ON document_job (batch_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_job_status ON document_job (status)",
    )

    def __init__(self, params: PostgresConnectionParams) -> None:
        self._params = params

    @contextmanager
    def _conn(self, *, autocommit: bool = True) -> Iterator[Any]:
        with connection(self._params, autocommit=autocommit) as conn:
            yield conn

    def _parse_scheduling_metadata_value(self, raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                loaded = json.loads(raw)
                return dict(loaded) if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _locked_scheduling_metadata_update(
        self,
        document_id: str,
        mutator: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Atomically read-modify-write ``scheduling_metadata`` under a row lock.

        Concurrent processor outcomes previously used read-then-full-write and could
        drop sibling ``processor_results`` entries (lost update), leaving intake stuck
        in ``processing`` forever.
        """
        now = _utc_naive()
        with self._conn(autocommit=False) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT scheduling_metadata FROM document_job WHERE document_id=%s FOR UPDATE",
                        (document_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        conn.rollback()
                        return {}
                    raw = row["scheduling_metadata"] if isinstance(row, Mapping) else row[0]
                    meta = self._parse_scheduling_metadata_value(raw)
                    mutator(meta)
                    cur.execute(
                        "UPDATE document_job SET scheduling_metadata=%s, updated_at=%s WHERE document_id=%s",
                        (json.dumps(meta, default=str), now, document_id),
                    )
                conn.commit()
                return meta
            except Exception:
                conn.rollback()
                raise

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_SQL)
                for index_sql in self.INDEX_SQL:
                    cur.execute(index_sql)

    def ping(self) -> bool:
        """Return True when a trivial query succeeds (connectivity check)."""
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            logger.debug("PostgreSQL ping failed", exc_info=True)
            return False

    def update_for_reprocess(self, document_id: str, *, notes: str | None = None) -> None:
        """Reset a document job to ``QUEUED`` so it can be published to RabbitMQ again."""
        now = _utc_naive()
        msg = (notes or "reprocess_requested")[:65000]
        meta = self._scheduling_metadata(document_id)
        meta["processor_results"] = {}
        meta["processor_assignments"] = []
        meta.pop("current_dispatch", None)
        meta_json = json.dumps(meta, default=str)
        sql = """
        UPDATE document_job SET
          status='QUEUED',
          processor_id=NULL,
          assigned_at=NULL,
          started_at=NULL,
          completed_at=NULL,
          error_details=NULL,
          progress_percent=NULL,
          heartbeat_at=NULL,
          processing_notes=%s,
          scheduling_metadata=%s,
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (msg, meta_json, now, document_id))

    def ensure_received_from_record(self, record: Mapping[str, Any] | MutableMapping[str, Any], *, batch_id: str | None) -> str:
        """Create RECEIVED row if missing; return ``job_id``."""
        existing = self.get_by_document_id(str(record["document_id"]))
        if existing:
            return str(existing["job_id"])
        return self.insert_received(
            document_id=str(record["document_id"]),
            batch_id=batch_id,
            document_name=str(record.get("document_name") or ""),
            source_location=str(record.get("repository_path") or ""),
            collection_name=record.get("collection_name"),
            tenant_id=record.get("tenant_id"),
        )

    def insert_received(
        self,
        *,
        document_id: str,
        batch_id: str | None,
        document_name: str,
        source_location: str,
        collection_name: str | None,
        tenant_id: str | None,
    ) -> str:
        job_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO document_job (
          job_id, document_id, batch_id, document_name, source_location,
          collection_name, tenant_id, status, retry_count,
          created_at, updated_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,'RECEIVED',0,%s,%s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        job_id,
                        document_id,
                        batch_id,
                        document_name,
                        source_location,
                        collection_name,
                        tenant_id,
                        now,
                        now,
                    ),
                )
        return job_id

    def update_status_queued(self, document_id: str) -> None:
        now = _utc_naive()
        sql = "UPDATE document_job SET status='QUEUED', updated_at=%s WHERE document_id=%s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (now, document_id))

    def update_requeued_after_dispatch_failure(
        self,
        document_id: str,
        *,
        notes: str | None = None,
        error_details: str | None = None,
    ) -> None:
        """Return a dequeued job to ``QUEUED`` when dispatch never reached a processor."""
        now = _utc_naive()
        meta = self._scheduling_metadata(document_id)
        meta.pop("current_dispatch", None)
        meta_json = json.dumps(meta, default=str)
        sql = """
        UPDATE document_job SET
          status='QUEUED',
          processor_id=NULL,
          assigned_at=NULL,
          processing_notes=COALESCE(%s, processing_notes),
          error_details=COALESCE(%s, error_details),
          scheduling_metadata=%s,
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (notes, error_details, meta_json, now, document_id))

    def update_human_review(self, document_id: str, *, notes: str | None = None) -> None:
        """Hold a job that the processor routed to human review.

        This is a terminal state for the current RabbitMQ delivery.  A reviewer
        decision explicitly publishes a new delivery when processing may resume.
        """
        now = _utc_naive()
        meta = self._scheduling_metadata(document_id)
        meta.pop("current_dispatch", None)
        meta_json = json.dumps(meta, default=str)
        sql = """
        UPDATE document_job SET
          status='HUMAN_REVIEW',
          processor_id=NULL,
          assigned_at=NULL,
          processing_notes=COALESCE(%s, processing_notes),
          scheduling_metadata=%s,
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (notes, meta_json, now, document_id))

    def _scheduling_metadata(self, document_id: str) -> dict[str, Any]:
        row = self.get_by_document_id(document_id)
        if not row:
            return {}
        return self._parse_scheduling_metadata_value(row.get("scheduling_metadata") or {})

    def _write_scheduling_metadata(self, document_id: str, metadata: Mapping[str, Any]) -> None:
        now = _utc_naive()
        meta_json = json.dumps(dict(metadata), default=str)
        sql = "UPDATE document_job SET scheduling_metadata=%s, updated_at=%s WHERE document_id=%s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (meta_json, now, document_id))

    def _merge_scheduling_metadata(
        self,
        document_id: str,
        patch: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        meta = self._scheduling_metadata(document_id)
        if patch:
            meta.update(dict(patch))
        return meta

    def seed_processor_plan(
        self,
        document_id: str,
        enabled_types: Sequence[str],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        meta = self._scheduling_metadata(document_id)
        meta["enabled_processor_types"] = list(enabled_types)
        meta["processor_results"] = {}
        meta["processor_assignments"] = []
        meta.pop("current_dispatch", None)
        if context:
            for key, value in dict(context).items():
                if value is not None:
                    meta[key] = value
        self._write_scheduling_metadata(document_id, meta)

    def mark_processor_republished(self, document_id: str, processor_types: Sequence[str]) -> None:
        meta = self._scheduling_metadata(document_id)
        republished = meta.setdefault("republished_types", {})
        stamp = _utc_naive().isoformat()
        results = meta.get("processor_results") or {}
        if not isinstance(results, dict):
            results = {}
        for processor_type in processor_types:
            republished[processor_type] = stamp
            # Clear prior terminal outcome so a manual/automatic retry can succeed.
            if processor_type in results:
                results.pop(processor_type, None)
        if results != meta.get("processor_results"):
            meta["processor_results"] = results
        self._write_scheduling_metadata(document_id, meta)
        self.update_status_queued(document_id)

    def list_pending_processor_types(
        self,
        document_id: str,
        *,
        dispatch_stale_after_seconds: float = 300.0,
    ) -> list[str]:
        """Return enabled processor types that have not yet reached a terminal outcome."""
        meta = self._scheduling_metadata(document_id)
        enabled = list(meta.get("enabled_processor_types") or ["chunking_vectorizing"])
        results = meta.get("processor_results") or {}
        if not isinstance(results, dict):
            results = {}
        dispatch = meta.get("current_dispatch") or {}
        active_type = dispatch.get("processor_type")
        active_phase = dispatch.get("phase")
        dispatch_is_fresh = True
        stamp = dispatch.get("assigned_at") or dispatch.get("dispatching_at")
        if stamp:
            try:
                parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                age = (_utc_naive() - parsed).total_seconds()
                dispatch_is_fresh = age <= float(dispatch_stale_after_seconds)
            except Exception:
                dispatch_is_fresh = True
        pending: list[str] = []
        for processor_type in enabled:
            outcome = results.get(processor_type) or {}
            outcome_status = str(outcome.get("status") or "").upper()
            # COMPLETED and FAILED are both done — do not republish failed types forever.
            if outcome_status in _PROCESSOR_OUTCOME_TERMINAL:
                continue
            if (
                dispatch_is_fresh
                and active_type == processor_type
                and active_phase in {"dispatching", "assigned"}
            ):
                continue
            pending.append(processor_type)
        return pending

    def record_processor_outcome(
        self,
        document_id: str,
        *,
        processor_type: str,
        status: str,
        result_location: str | None = None,
        document_metadata: Mapping[str, Any] | None = None,
        error_details: str | None = None,
    ) -> None:
        completed_at = _utc_naive().isoformat()

        def _merge_outcome(meta: dict[str, Any]) -> None:
            results = meta.setdefault("processor_results", {})
            if not isinstance(results, dict):
                results = {}
                meta["processor_results"] = results
            results[processor_type] = {
                "status": status,
                "result_location": result_location,
                "document_metadata": dict(document_metadata or {}),
                "error_details": error_details,
                "completed_at": completed_at,
            }
            dispatch = meta.get("current_dispatch") or {}
            if isinstance(dispatch, dict) and dispatch.get("processor_type") == processor_type:
                meta.pop("current_dispatch", None)
            republished = meta.get("republished_types")
            if isinstance(republished, dict):
                republished.pop(processor_type, None)

        meta = self._locked_scheduling_metadata_update(document_id, _merge_outcome)
        if not meta:
            return
        results = meta.setdefault("processor_results", {})
        if not isinstance(results, dict):
            results = {}
            meta["processor_results"] = results

        if processor_type == "chunking_vectorizing" and status == "COMPLETED" and document_metadata:
            try:
                from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
                from src.features.configuration.platform_settings import get_settings

                document_info = {
                    "document_id": document_id,
                    "source": "chunking_vectorizing",
                    **dict(document_metadata),
                }
                MetadataStore(get_settings().db_path).update_document_metadata(
                    document_id,
                    {"document_info": document_info},
                    merge=True,
                )
            except Exception:
                logger.debug("Could not persist document_info for %s", document_id, exc_info=True)

        if processor_type == "metadata_extraction" and document_metadata:
            extracted = document_metadata.get("extracted_fields")
            fields = extracted if isinstance(extracted, list) else []
            type_metadata = {
                "document_type_id": document_metadata.get("document_type_id"),
                "fields": fields,
                "extracted_field_count": document_metadata.get("extracted_field_count") or len(fields),
                "status": status.lower() if status else None,
                "completed_at": _utc_naive().isoformat(),
                "source": "metadata_extraction",
            }
            try:
                from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
                from src.features.configuration.platform_settings import get_settings

                patch = {"type_metadata": type_metadata}
                if fields:
                    patch["metadata_extraction"] = {
                        str(item.get("field_name")): item.get("value")
                        for item in fields
                        if isinstance(item, dict) and item.get("field_name")
                    }
                MetadataStore(get_settings().db_path).update_document_metadata(document_id, patch, merge=True)
            except Exception:
                logger.debug("Could not merge type_metadata into intake record", exc_info=True)

        if processor_type == "key_field_extraction" and document_metadata:
            extracted = document_metadata.get("extracted_fields")
            fields = extracted if isinstance(extracted, list) else []
            extracted_metadata = {
                str(item.get("field_name")): item.get("value")
                for item in fields
                if isinstance(item, dict) and item.get("field_name")
            }
            type_metadata = {
                "document_type_id": document_metadata.get("document_type_id"),
                "fields": fields,
                "extracted_field_count": document_metadata.get("extracted_field_count") or len(fields),
                "status": status.lower() if status else None,
                "completed_at": _utc_naive().isoformat(),
                "source": "key_field_extraction",
            }
            try:
                from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
                from src.features.configuration.platform_settings import get_settings

                patch = {"key_field_metadata": type_metadata}
                if extracted_metadata:
                    patch["extracted_metadata"] = extracted_metadata
                MetadataStore(get_settings().db_path).update_document_metadata(document_id, patch, merge=True)
            except Exception:
                logger.debug("Could not merge key_field_metadata into intake record", exc_info=True)

            # Unlock deferred document_validation for scheduler reconcile / immediate follow-up.
            if status == "COMPLETED" and "document_validation" in list(
                meta.get("enabled_processor_types") or []
            ):

                def _unlock_validation(locked_meta: dict[str, Any]) -> None:
                    republished = locked_meta.setdefault("republished_types", {})
                    if isinstance(republished, dict):
                        republished.pop("document_validation", None)
                    locked_meta["follow_up_processors"] = ["document_validation"]

                meta = self._locked_scheduling_metadata_update(document_id, _unlock_validation) or meta
                results = meta.get("processor_results") or results

        # Logical folders are an optional, repository-scoped view of extracted
        # metadata.  Run only after successful extraction; when key-field
        # extraction is scheduled, wait for it and use its richer result.
        if processor_type in {"metadata_extraction", "key_field_extraction"} and status == "COMPLETED":
            try:
                metadata_result = results.get("metadata_extraction") or {}
                key_field_result = results.get("key_field_extraction") or {}
                enabled_processors = set(meta.get("enabled_processor_types") or [])
                key_fields_required = "key_field_extraction" in enabled_processors
                if (
                    str(metadata_result.get("status") or "").upper() == "COMPLETED"
                    and (not key_fields_required or str(key_field_result.get("status") or "").upper() == "COMPLETED")
                ):
                    selected_metadata = (
                        key_field_result.get("document_metadata") if key_fields_required else metadata_result.get("document_metadata")
                    )
                    repository_id = None
                    try:
                        job = self.get_by_document_id(document_id)
                        repository_id = job.get("repository_id") if isinstance(job, dict) else None
                    except Exception:
                        pass
                    from src.features.logical_folders.application.folder_service import (
                        assign_document_folders_after_extraction,
                    )

                    assign_document_folders_after_extraction(
                        document_id=document_id,
                        repository_id=str(repository_id) if repository_id else None,
                        document_metadata=selected_metadata if isinstance(selected_metadata, dict) else {},
                    )
            except Exception:
                logger.debug("Could not assign logical folder for document_id=%s", document_id, exc_info=True)

        if processor_type == "document_validation" and document_metadata:
            validation = {
                "status": document_metadata.get("status"),
                "document_status": document_metadata.get("document_status"),
                "missing_required_fields": document_metadata.get("missing_required_fields") or [],
                "invalid_fields": document_metadata.get("invalid_fields") or [],
                "low_confidence_fields": document_metadata.get("low_confidence_fields") or [],
                "confidence_threshold": document_metadata.get("confidence_threshold"),
                "completed_at": _utc_naive().isoformat(),
                "source": "document_validation",
            }
            try:
                from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
                from src.features.configuration.platform_settings import get_settings

                patch = {
                    "validation": validation,
                    "validation_status": document_metadata.get("document_status")
                    or document_metadata.get("status"),
                }
                MetadataStore(get_settings().db_path).update_document_metadata(document_id, patch, merge=True)
            except Exception:
                logger.debug("Could not merge validation into intake record", exc_info=True)

            # Phase 5: auto-create / refresh human review for WARNING/INVALID outcomes.
            if status == "COMPLETED":
                try:
                    from src.features.human_review.application.review_service import (
                        get_document_review_service,
                    )

                    repository_id = None
                    document_type_id = document_metadata.get("document_type_id")
                    try:
                        job_row = self.get_by_document_id(document_id)
                        if isinstance(job_row, dict):
                            repository_id = job_row.get("repository_id")
                            sched = job_row.get("scheduling_metadata") or {}
                            if isinstance(sched, dict) and not document_type_id:
                                document_type_id = sched.get("document_type_id")
                    except Exception:
                        pass
                    get_document_review_service().maybe_create_review_from_validation(
                        document_id=document_id,
                        validation_status=document_metadata.get("document_status")
                        or document_metadata.get("status"),
                        repository_id=str(repository_id) if repository_id else None,
                        document_type_id=str(document_type_id) if document_type_id else None,
                        validation_payload=dict(document_metadata),
                    )
                except Exception:
                    logger.debug(
                        "Could not create/update human review for document_id=%s",
                        document_id,
                        exc_info=True,
                    )

        enabled = list(meta.get("enabled_processor_types") or ["chunking_vectorizing"])
        self._sync_processor_results_to_intake(document_id, meta)

        # Independent processors: never terminal-fail the shared job row until
        # every enabled processor has recorded COMPLETED or FAILED.
        if status == "FAILED":
            if all_enabled_processors_terminal(meta):
                failed_parts = []
                for pt in enabled:
                    outcome = results.get(pt) or {}
                    if str(outcome.get("status") or "").upper() != "FAILED":
                        continue
                    detail = str(outcome.get("error_details") or "").strip() or f"{pt} failed"
                    failed_parts.append(f"{pt}: {detail}")
                self.update_failed(
                    document_id,
                    error_details="; ".join(failed_parts) or (error_details or f"{processor_type} failed"),
                    notes="enabled_processors_finished_with_failures",
                )
            else:
                self.heartbeat(document_id, notes=f"{processor_type}:failed;waiting_siblings")
            return

        if status != "COMPLETED":
            return

        aggregated = aggregate_processor_plan_status(meta)
        if aggregated == "COMPLETED":
            primary = results.get("chunking_vectorizing", {}).get("result_location") or result_location
            # Prefer any successful result_location if chunking did not run/succeed.
            if not primary:
                for pt in enabled:
                    loc = (results.get(pt) or {}).get("result_location")
                    if loc:
                        primary = loc
                        break
            self.update_completed(
                document_id,
                result_location=primary,
                notes="all_enabled_processors_complete",
            )
            return

        if aggregated == "FAILED":
            failed_parts = []
            for pt in enabled:
                outcome = results.get(pt) or {}
                if str(outcome.get("status") or "").upper() != "FAILED":
                    continue
                detail = str(outcome.get("error_details") or "").strip() or f"{pt} failed"
                failed_parts.append(f"{pt}: {detail}")
            self.update_failed(
                document_id,
                error_details="; ".join(failed_parts) or "one or more processors failed",
                notes="enabled_processors_finished_with_failures",
            )
            return

        self.heartbeat(document_id, notes=f"{processor_type}:completed")
        try:
            from src.features.documents.application.job_republish import enqueue_ready_processors

            enqueue_ready_processors(document_id)
        except Exception:
            logger.debug(
                "Could not enqueue ready follow-on processors for %s",
                document_id,
                exc_info=True,
            )

    def _sync_processor_results_to_intake(self, document_id: str, meta: Mapping[str, Any]) -> None:
        """Mirror processor_results onto the intake document so APIs can show siblings."""
        try:
            from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
            from src.features.configuration.platform_settings import get_settings

            results = _normalize_processor_results(meta)
            patch: dict[str, Any] = {
                "processor_results": results,
                "enabled_processor_types": _enabled_processor_types(meta),
            }
            template = results.get("template_extraction") if isinstance(results.get("template_extraction"), dict) else {}
            if str(template.get("status") or "").upper() == "COMPLETED" and template.get("result_location"):
                patch["template_extraction"] = {
                    "status": "COMPLETED",
                    "result_location": template.get("result_location"),
                    "completed_at": template.get("completed_at"),
                    "source": "template_extraction",
                }
            MetadataStore(get_settings().db_path).update_document_metadata(
                document_id,
                patch,
                merge=True,
            )
        except Exception:
            logger.debug("Could not sync processor_results into intake for %s", document_id, exc_info=True)

    def update_dispatching(
        self,
        document_id: str,
        *,
        scheduling_metadata: Mapping[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        """Scheduler dequeued the job from RabbitMQ; not yet accepted by ``POST /process``."""
        now = _utc_naive()
        meta = self._merge_scheduling_metadata(document_id, scheduling_metadata)
        processor_type = (scheduling_metadata or {}).get("processor_type")
        if processor_type:
            meta["current_dispatch"] = {
                "processor_type": processor_type,
                "phase": "dispatching",
                "dequeued_at": now.isoformat(),
            }
        meta_json = json.dumps(meta, default=str)
        sql = """
        UPDATE document_job SET
          status=CASE
            WHEN status IN ('QUEUED', 'queued') THEN 'DISPATCHING'
            ELSE status
          END,
          scheduling_metadata=%s,
          processing_notes=COALESCE(%s, processing_notes),
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (meta_json, notes, now, document_id))

    def list_active_processing(self) -> list[dict[str, Any]]:
        """Rows the scheduler considers in-flight (consumed from queue or on a processor)."""
        sql = """
        SELECT * FROM document_job
        WHERE status IN ('DISPATCHING','ASSIGNED','STARTED','IN_PROGRESS')
        ORDER BY updated_at DESC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for r in rows:
            n = self._normalize_row(r)
            if n:
                out.append(n)
        return out

    def update_status_queue_failed(self, document_id: str, error_details: str) -> None:
        now = _utc_naive()
        sql = """
        UPDATE document_job SET status='QUEUE_FAILED', error_details=%s, updated_at=%s, completed_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (error_details[:65000], now, now, document_id))

    def update_assigned(
        self,
        document_id: str,
        *,
        processor_id: str,
        scheduling_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = _utc_naive()
        meta = self._merge_scheduling_metadata(document_id, scheduling_metadata)
        processor_type = (scheduling_metadata or {}).get("processor_type")
        processor_port = (scheduling_metadata or {}).get("processor_port")
        assignment = {
            "processor_id": processor_id,
            "processor_type": processor_type,
            "processor_port": processor_port,
            "assigned_at": now.isoformat(),
        }
        meta.setdefault("processor_assignments", []).append(assignment)
        meta["current_dispatch"] = {
            **assignment,
            "phase": "assigned",
        }
        meta_json = json.dumps(meta, default=str)
        # Only advance from pre-accept states. A late ASSIGNED write must not
        # overwrite STARTED/IN_PROGRESS/FAILED after the processor already began
        # (or failed) and must not clear error_details.
        sql = """
        UPDATE document_job SET
          status='ASSIGNED',
          processor_id=%s,
          assigned_at=%s,
          scheduling_metadata=%s,
          retry_count=retry_count+1,
          updated_at=%s,
          completed_at=NULL,
          error_details=NULL
        WHERE document_id=%s AND status IN ('QUEUED', 'DISPATCHING')
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (processor_id, now, meta_json, now, document_id))

    def record_processor_assignment(
        self,
        document_id: str,
        *,
        processor_id: str,
        scheduling_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a follow-up processor dispatch without resetting status to ASSIGNED."""
        now = _utc_naive()
        meta = self._merge_scheduling_metadata(document_id, scheduling_metadata)
        processor_type = (scheduling_metadata or {}).get("processor_type")
        processor_port = (scheduling_metadata or {}).get("processor_port")
        assignment = {
            "processor_id": processor_id,
            "processor_type": processor_type,
            "processor_port": processor_port,
            "assigned_at": now.isoformat(),
        }
        meta.setdefault("processor_assignments", []).append(assignment)
        meta["current_dispatch"] = {**assignment, "phase": "assigned"}
        meta_json = json.dumps(meta, default=str)
        sql = f"""
        UPDATE document_job SET
          processor_id=%s,
          scheduling_metadata=%s,
          updated_at=%s
        WHERE document_id=%s
          AND status NOT IN ({_HARD_CLOSED_JOB_STATUS_LIST})
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (processor_id, meta_json, now, document_id))

    def update_started(self, document_id: str, *, processor_identity: str, notes: str | None = None) -> None:
        now = _utc_naive()
        sql = f"""
        UPDATE document_job SET
          status='STARTED',
          processor_id=COALESCE(processor_id,%s),
          started_at=COALESCE(started_at,%s),
          heartbeat_at=%s,
          processing_notes=%s,
          updated_at=%s,
          completed_at=NULL
        WHERE document_id=%s AND status NOT IN ({_HARD_CLOSED_JOB_STATUS_LIST})
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (processor_identity, now, now, notes, now, document_id))

    def update_in_progress(
        self,
        document_id: str,
        *,
        progress_percent: float | None,
        notes: str | None = None,
    ) -> None:
        now = _utc_naive()
        sql = f"""
        UPDATE document_job SET
          status='IN_PROGRESS',
          progress_percent=%s,
          heartbeat_at=%s,
          processing_notes=COALESCE(%s, processing_notes),
          updated_at=%s,
          completed_at=NULL
        WHERE document_id=%s AND status NOT IN ({_HARD_CLOSED_JOB_STATUS_LIST})
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (progress_percent, now, notes, now, document_id))

    def heartbeat(
        self,
        document_id: str,
        *,
        progress_percent: float | None = None,
        notes: str | None = None,
    ) -> None:
        now = _utc_naive()
        sets = ["heartbeat_at=%s", "updated_at=%s"]
        args: list[Any] = [now, now]
        if progress_percent is not None:
            sets.append("progress_percent=%s")
            args.append(progress_percent)
        if notes is not None:
            sets.append("processing_notes=%s")
            args.append(notes)
        args.append(document_id)
        # Allow heartbeat while a sibling is still running even if a prior
        # outcome prematurely marked the shared row FAILED.
        sql = (
            f"UPDATE document_job SET {', '.join(sets)} "
            f"WHERE document_id=%s AND status NOT IN ({_HARD_CLOSED_JOB_STATUS_LIST})"
        )
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
    def update_completed(
        self,
        document_id: str,
        *,
        result_location: str | None = None,
        notes: str | None = None,
    ) -> None:
        now = _utc_naive()
        sql = """
        UPDATE document_job SET
          status='COMPLETED',
          progress_percent=100,
          completed_at=%s,
          result_location=%s,
          processing_notes=COALESCE(%s, processing_notes),
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (now, result_location, notes, now, document_id))
        try:
            from src.infrastructure.database.intake_status_sync import sync_intake_document_status

            synced = sync_intake_document_status(
                document_id,
                "completed",
                completed_at=now.timestamp(),
                error_details=None,
            )
            if not synced:
                # Fallback when running inside DMS (shared helper path may differ).
                from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
                from src.features.configuration.platform_settings import get_settings

                MetadataStore(get_settings().db_path).update_document_status(
                    document_id,
                    "completed",
                    completed_at=now.timestamp(),
                    error_details=None,
                )
        except Exception:
            logger.debug("Could not sync intake status for completed document %s", document_id, exc_info=True)

    def update_failed(self, document_id: str, *, error_details: str, notes: str | None = None) -> None:
        now = _utc_naive()
        sql = """
        UPDATE document_job SET
          status='FAILED',
          completed_at=%s,
          error_details=%s,
          processing_notes=COALESCE(%s, processing_notes),
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (now, error_details[:65000], notes, now, document_id))
        try:
            from src.infrastructure.database.intake_status_sync import sync_intake_document_status

            synced = sync_intake_document_status(
                document_id,
                "failed",
                completed_at=now.timestamp(),
                error_details=error_details[:65000],
            )
            if not synced:
                from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
                from src.features.configuration.platform_settings import get_settings

                MetadataStore(get_settings().db_path).update_document_status(
                    document_id,
                    "failed",
                    completed_at=now.timestamp(),
                    error_details=error_details[:65000],
                )
        except Exception:
            logger.debug("Could not sync intake status for failed document %s", document_id, exc_info=True)

    def update_operator_stopped(self, document_id: str, *, details: str) -> None:
        now = _utc_naive()
        sql = """
        UPDATE document_job SET
          status='STOPPED',
          completed_at=%s,
          error_details=%s,
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (now, details[:65000], now, document_id))

    def update_timed_out(self, document_id: str, *, error_details: str | None = None) -> None:
        now = _utc_naive()
        sql = """
        UPDATE document_job SET
          status='TIMED_OUT',
          completed_at=%s,
          error_details=%s,
          updated_at=%s
        WHERE document_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (now, (error_details or "timed out")[:65000], now, document_id))

    def list_non_terminal_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Jobs still open for processing, including FAILED rows with pending siblings."""
        sql = """
        SELECT * FROM document_job
        WHERE status NOT IN ('COMPLETED','TIMED_OUT','QUEUE_FAILED','STOPPED','HUMAN_REVIEW')
        ORDER BY updated_at DESC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (max(1, min(limit, 500)),))
                rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            normalized = self._normalize_row(row)
            if not normalized:
                continue
            status = str(normalized.get("status") or "").upper()
            if status == "FAILED":
                # Only keep FAILED rows that still have unfinished sibling processors.
                if not self.list_pending_processor_types(str(normalized.get("document_id") or "")):
                    continue
            out.append(normalized)
        return out

    def list_received_unqueued(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Jobs stuck in RECEIVED without a processor plan (never submitted to RabbitMQ)."""
        sql = """
        SELECT * FROM document_job
        WHERE status = 'RECEIVED'
          AND (
            scheduling_metadata IS NULL
            OR scheduling_metadata = 'null'::jsonb
            OR scheduling_metadata = '{}'::jsonb
            OR scheduling_metadata = '[]'::jsonb
          )
        ORDER BY created_at ASC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (max(1, min(limit, 200)),))
                rows = cur.fetchall() or []
        out: list[dict[str, Any]] = []
        for row in rows:
            normalized = self._normalize_row(row)
            if normalized:
                out.append(normalized)
        return out

    def get_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        sql = "SELECT * FROM document_job WHERE job_id=%s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (job_id,))
                row = cur.fetchone()
        return self._normalize_row(row)

    def get_by_document_id(self, document_id: str) -> dict[str, Any] | None:
        sql = "SELECT * FROM document_job WHERE document_id=%s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                row = cur.fetchone()
        return self._normalize_row(row)

    def list_by_batch(self, batch_id: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM document_job WHERE batch_id=%s ORDER BY created_at ASC"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (batch_id,))
                rows = cur.fetchall() or []
        return [self._normalize_row(r) for r in rows]

    def fetch_for_documents(self, document_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not document_ids:
            return {}
        placeholders = ",".join(["%s"] * len(document_ids))
        sql = f"SELECT * FROM document_job WHERE document_id IN ({placeholders})"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(document_ids))
                rows = cur.fetchall() or []
        return {str(r["document_id"]): self._normalize_row(r) for r in rows}

    def delete_by_document_id(self, document_id: str) -> None:
        sql = "DELETE FROM document_job WHERE document_id=%s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))

    @staticmethod
    def _normalize_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        out = dict(row)
        meta = out.get("scheduling_metadata")
        if isinstance(meta, str):
            try:
                out["scheduling_metadata"] = json.loads(meta)
            except json.JSONDecodeError:
                out["scheduling_metadata"] = {}
        for k in ("created_at", "updated_at", "assigned_at", "heartbeat_at", "started_at", "completed_at"):
            v = out.get(k)
            if hasattr(v, "isoformat"):
                out[k] = v.isoformat(sep=" ", timespec="milliseconds")
        return out


_store: DocumentJobStore | None = None


def get_document_job_store() -> DocumentJobStore | None:
    """Singleton store; returns None when PostgreSQL is not configured."""
    global _store
    params = mysql_params_from_env()
    if params is None:
        return None
    if _store is None:
        _store = DocumentJobStore(params)
        try:
            _store.ensure_schema()
        except Exception:
            logger.exception("document_job schema ensure failed")
            raise
    return _store


def reset_document_job_store_for_tests() -> None:
    global _store
    _store = None
