from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.infrastructure.database.document_jobs import get_document_job_store, DocumentJobStore
from src.features.scheduler_server.domain.job_state import JobState

logger = logging.getLogger(__name__)

_ACTIVE_JOB_STATUSES = frozenset(
    {"DISPATCHING", "ASSIGNED", "IN_PROGRESS", "PROCESSING", "STARTED"}
)
_TERMINAL_SKIP_STATUSES = frozenset(
    {"COMPLETED", "FAILED", "TIMED_OUT", "STOPPED", "QUEUE_FAILED", "HUMAN_REVIEW", "RECEIVED"}
)
_PROCESSING_PAYLOAD_KEYS = (
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
    "processor_type",
    "collection_name",
    "tenant_id",
    "repository_id",
    "key_fields",
    "original_file_name",
    "strict_key_field_page_scope",
)


class SchedulerJobRepository:
    """Relational DB Persistence interface for document job state tracking."""

    def __init__(self):
        self._store: DocumentJobStore = get_document_job_store()

    def update_job_status(
        self,
        job_id: str,
        status: str | JobState,
        processor_id: Optional[str] = None,
        processor_container_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update job lifecycle status in relational DB (document_job)."""
        status_val = status.value if isinstance(status, JobState) else str(status)
        try:
            now = datetime.utcnow()
            if status_val == JobState.QUEUED.value:
                self._store.update_status_queued(job_id)
                return True

            # Use generic update_job or direct SQL via store helper
            sql = """
            UPDATE document_job SET
                status = %s,
                processor_id = COALESCE(%s, processor_id),
                error_details = COALESCE(%s, error_details),
                updated_at = %s
            WHERE job_id = %s OR document_id = %s
            """
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (status_val, processor_id, error_message, now, job_id, job_id))
            return True
        except Exception as exc:
            logger.error("Failed to update DB state for job %s -> %s: %s", job_id, status_val, exc)
            return False

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch job record dictionary by job_id."""
        try:
            res = self._store.get_by_job_id(job_id)
            return dict(res) if res else None
        except Exception as exc:
            logger.error("Error retrieving job %s from DB: %s", job_id, exc)
            return None

    def get_job_by_document_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch job record dictionary by document_id."""
        try:
            res = self._store.get_by_document_id(document_id)
            return dict(res) if res else None
        except Exception as exc:
            logger.error("Error retrieving job for document %s from DB: %s", document_id, exc)
            return None

    def count_queued_jobs(self) -> int:
        """Count jobs waiting for first processor assignment."""
        try:
            sql = "SELECT COUNT(*) AS count FROM document_job WHERE status IN ('QUEUED', 'queued')"
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    row = cur.fetchone() or {}
                    return int(row.get("count") or 0)
        except Exception as exc:
            logger.error("Error counting queued jobs: %s", exc)
            return 0

    def is_processor_slot_active(self, processor_id: str) -> bool:
        """True when this processor slot has a live active job in the database."""
        slot_id = str(processor_id or "").strip()
        if not slot_id:
            return False
        try:
            sql = """
            SELECT 1 FROM document_job
            WHERE processor_id = %s
              AND status IN ('DISPATCHING', 'ASSIGNED', 'IN_PROGRESS', 'PROCESSING', 'STARTED')
            LIMIT 1
            """
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (slot_id,))
                    return cur.fetchone() is not None
        except Exception as exc:
            logger.error("Error checking active slot assignment for %s: %s", slot_id, exc)
            return True

    def message_dispatch_priority(
        self,
        payload: Dict[str, Any],
        record: Optional[Dict[str, Any]],
        *,
        sla_seconds: int = 30,
    ) -> int:
        """Higher score = dispatch sooner. Fresh QUEUED uploads beat follow-up republishes."""
        score = 0
        status = self.normalized_status(record)
        if status == JobState.QUEUED.value:
            score += 1000
        if not payload.get("requeued"):
            score += 500
        updated = (record or {}).get("updated_at")
        if updated is not None:
            try:
                if hasattr(updated, "timestamp"):
                    age = max(0.0, datetime.utcnow().timestamp() - updated.timestamp())
                else:
                    parsed = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                    if parsed.tzinfo is not None:
                        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                    age = max(0.0, (datetime.utcnow() - parsed).total_seconds())
                if age >= max(0, sla_seconds - 10):
                    score += 300
                score += min(int(age), 120)
            except Exception:
                pass
        processor_type = str(payload.get("processor_type") or "chunking_vectorizing").strip().lower()
        if processor_type == "chunking_vectorizing":
            score += 50
        return score

    def list_running_jobs(self) -> List[Dict[str, Any]]:
        """List active jobs with status DISPATCHING, ASSIGNED, or IN_PROGRESS."""
        try:
            sql = """
            SELECT job_id, document_id, processor_id, status, document_name, created_at, updated_at
            FROM document_job
            WHERE status IN ('DISPATCHING', 'ASSIGNED', 'IN_PROGRESS', 'processing')
            LIMIT 100
            """
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("Error listing running jobs from DB: %s", exc)
            return []

    def get_job_retry_count(self, job_id: str) -> int:
        """Get current retry count for a job."""
        try:
            record = self.get_job_by_id(job_id) or self.get_job_by_document_id(job_id)
            if record:
                return int(record.get("retry_count") or 0)
            return 0
        except Exception as exc:
            logger.error("Error fetching retry count for job %s: %s", job_id, exc)
            return 0

    def get_job_record(self, job_id: str, document_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Resolve a job row by job_id and/or document_id."""
        record = self.get_job_by_id(job_id)
        if not record and document_id:
            record = self.get_job_by_document_id(document_id)
        if not record and document_id != job_id:
            record = self.get_job_by_document_id(job_id)
        return record

    @staticmethod
    def normalized_status(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return ""
        return str(record.get("status") or "").strip().upper()

    def is_job_completed(self, job_id_or_doc_id: str) -> bool:
        """Check if job is already in COMPLETED status (idempotency guard)."""
        try:
            record = self.get_job_record(job_id_or_doc_id, job_id_or_doc_id)
            if record:
                return self.normalized_status(record) == JobState.COMPLETED.value
            return False
        except Exception as exc:
            logger.error("Error checking job completion for %s: %s", job_id_or_doc_id, exc)
            return False

    def is_job_active(self, job_id: str, document_id: Optional[str] = None) -> bool:
        """True when the job is actively being scheduled or processed."""
        try:
            record = self.get_job_record(job_id, document_id)
            return self.normalized_status(record) in _ACTIVE_JOB_STATUSES
        except Exception as exc:
            logger.error("Error checking active status for job %s: %s", job_id, exc)
            return False

    def should_skip_queue_message(self, job_id: str, document_id: str) -> bool:
        """Skip broker messages for terminal or intake-hold statuses."""
        record = self.get_job_record(job_id, document_id)
        return self.normalized_status(record) in _TERMINAL_SKIP_STATUSES

    @staticmethod
    def _scheduling_meta(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not record:
            return {}
        meta = record.get("scheduling_metadata") or {}
        return meta if isinstance(meta, dict) else {}

    def processor_outcome_status(self, document_id: str, processor_type: str) -> str:
        record = self.get_job_by_document_id(document_id)
        outcome = (self._scheduling_meta(record).get("processor_results") or {}).get(processor_type) or {}
        if not isinstance(outcome, dict):
            return ""
        return str(outcome.get("status") or "").strip().upper()

    def is_processor_terminal(self, document_id: str, processor_type: str) -> bool:
        return self.processor_outcome_status(document_id, processor_type) in {"COMPLETED", "FAILED"}

    def is_processor_pending(self, document_id: str, processor_type: str) -> bool:
        try:
            return processor_type in self._store.list_pending_processor_types(document_id)
        except Exception as exc:
            logger.error("Failed pending check for %s/%s: %s", document_id, processor_type, exc)
            return not self.is_processor_terminal(document_id, processor_type)

    def list_ready_processor_types(self, document_id: str) -> List[str]:
        """Processor types whose dependencies are satisfied and outcome is not terminal."""
        record = self.get_job_by_document_id(document_id)
        meta = self._scheduling_meta(record)
        enabled = list(meta.get("enabled_processor_types") or ["chunking_vectorizing"])
        results = meta.get("processor_results") or {}
        if not isinstance(results, dict):
            results = {}
        try:
            from src.features.document_processing.shared_processor.types import processor_dependencies_met
        except Exception as exc:
            logger.error("Failed to import processor_dependencies_met: %s", exc)
            pending = self._store.list_pending_processor_types(document_id)
            return pending[:1] if pending else ["chunking_vectorizing"]

        ready: List[str] = []
        for processor_type in enabled:
            pt = str(processor_type or "").strip().lower()
            if not pt or self.is_processor_terminal(document_id, pt):
                continue
            if processor_dependencies_met(pt, results, enabled_types=enabled):
                ready.append(pt)
        return ready

    def is_processor_ready(self, document_id: str, processor_type: str) -> bool:
        pt = str(processor_type or "").strip().lower()
        if not pt or self.is_processor_terminal(document_id, pt):
            return False
        return pt in self.list_ready_processor_types(document_id)

    def republish_cooled_down(self, document_id: str, cooldown_seconds: int) -> bool:
        """Return True when scheduler may republish (outside cooldown window)."""
        if cooldown_seconds <= 0:
            return True
        record = self.get_job_by_document_id(document_id)
        meta = self._scheduling_meta(record)
        stamp = meta.get("last_scheduler_republish_at")
        if not stamp:
            return True
        try:
            parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            age = (datetime.utcnow() - parsed).total_seconds()
            return age >= float(cooldown_seconds)
        except Exception:
            return True

    def record_scheduler_republish(self, document_id: str) -> None:
        """Track last scheduler-driven RabbitMQ republish to throttle recovery."""
        try:
            record = self.get_job_by_document_id(document_id)
            meta = dict(self._scheduling_meta(record))
            meta["last_scheduler_republish_at"] = datetime.utcnow().isoformat()
            self._store._write_scheduling_metadata(document_id, meta)
        except Exception as exc:
            logger.debug("Could not record scheduler republish for %s: %s", document_id, exc)

    def is_processor_actively_dispatching(self, document_id: str, processor_type: str) -> bool:
        record = self.get_job_by_document_id(document_id)
        dispatch = self._scheduling_meta(record).get("current_dispatch") or {}
        if not isinstance(dispatch, dict):
            return False
        if str(dispatch.get("processor_type") or "").strip().lower() != processor_type.strip().lower():
            return False
        phase = str(dispatch.get("phase") or "").strip().lower()
        return phase in {"dispatching", "assigned"}

    def prepare_processor_dispatch(
        self,
        document_id: str,
        processor_type: str,
        processor_id: str,
    ) -> bool:
        """Record scheduler dequeue for a specific processor type (multi-processor plan)."""
        try:
            self._store.update_dispatching(
                document_id,
                scheduling_metadata={"processor_type": processor_type, "processor_id": processor_id},
                notes=f"scheduler:dequeued:{processor_type}",
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to prepare dispatch for document %s processor %s: %s",
                document_id,
                processor_type,
                exc,
            )
            return False

    def requeue_after_dispatch_failure(
        self,
        document_id: str,
        *,
        error_message: str | None = None,
    ) -> None:
        """Reset a failed dispatch back to QUEUED and clear slot assignment."""
        try:
            self._store.update_requeued_after_dispatch_failure(
                document_id,
                notes="scheduler:dispatch_failed",
                error_details=error_message,
            )
        except Exception as exc:
            logger.error("Failed to requeue document %s after dispatch failure: %s", document_id, exc)

    def reset_failed_for_republish(self, document_id: str) -> bool:
        """Allow intentional recovery republish of a FAILED document job row."""
        record = self.get_job_by_document_id(document_id)
        if not record or self.normalized_status(record) != JobState.FAILED.value:
            return False
        try:
            self._store.update_status_queued(document_id)
            return True
        except Exception as exc:
            logger.error("Failed to reset FAILED job %s for republish: %s", document_id, exc)
            return False

    def _payload_from_job_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Build a minimal processor payload from a document_job row."""
        doc_id = str(record.get("document_id") or "").strip()
        payload: Dict[str, Any] = {
            "document_id": doc_id,
            "document_name": record.get("document_name") or f"{doc_id}.pdf",
        }
        source = record.get("source_location")
        if source:
            payload["document_path"] = str(source)
        meta = record.get("scheduling_metadata") or {}
        if isinstance(meta, dict):
            for key in _PROCESSING_PAYLOAD_KEYS:
                if meta.get(key) is not None:
                    payload[key] = meta[key]
        enabled = payload.get("enabled_processor_types") or (
            meta.get("enabled_processor_types") if isinstance(meta, dict) else None
        )
        if enabled and not payload.get("processor_type"):
            payload["processor_type"] = enabled[0]
        elif not payload.get("processor_type"):
            payload["processor_type"] = "chunking_vectorizing"
        return payload

    def build_dispatch_payload(
        self,
        job_id: str,
        document_id: str,
        partial_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a full processor dispatch payload from intake metadata and DB."""
        partial = dict(partial_payload or {})
        doc_id = str(document_id or partial.get("document_id") or job_id).strip()
        jid = str(job_id or partial.get("job_id") or doc_id).strip()
        record = self.get_job_record(jid, doc_id)

        enriched: Dict[str, Any]
        try:
            from src.features.documents.application.job_republish import (
                get_intake_record,
                queue_payload_from_record,
            )

            intake = get_intake_record(doc_id)
            if intake:
                enriched = queue_payload_from_record(intake)
            elif record:
                enriched = self._payload_from_job_record(record)
            else:
                enriched = {
                    "document_id": doc_id,
                    "document_name": partial.get("document_name") or f"{doc_id}.pdf",
                    "processor_type": partial.get("processor_type") or "chunking_vectorizing",
                }
        except Exception as exc:
            logger.warning("Failed to enrich dispatch payload for %s: %s", doc_id, exc)
            enriched = self._payload_from_job_record(record) if record else {
                "document_id": doc_id,
                "document_name": partial.get("document_name") or f"{doc_id}.pdf",
            }

        enriched["job_id"] = jid
        enriched.setdefault("document_id", doc_id)

        for key, value in partial.items():
            if value is None:
                continue
            if key in {"document_name", "collection_name", "document_path"} and not value:
                continue
            enriched[key] = value

        if not enriched.get("document_name"):
            enriched["document_name"] = (
                (record or {}).get("document_name")
                or partial.get("document_name")
                or f"{doc_id}.pdf"
            )
        if not enriched.get("processor_type"):
            enabled = enriched.get("enabled_processor_types") or []
            enriched["processor_type"] = enabled[0] if enabled else "chunking_vectorizing"
        if "security_prevalidated" not in enriched:
            enriched["security_prevalidated"] = partial.get("security_prevalidated", True)
        return enriched

    def list_queued_job_ids(self) -> List[str]:
        """Return document/job ids currently in QUEUED status."""
        try:
            sql = """
            SELECT job_id, document_id
            FROM document_job
            WHERE status IN ('QUEUED', 'queued')
            ORDER BY updated_at ASC
            LIMIT 200
            """
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            ids: List[str] = []
            for row in rows:
                jid = row.get("job_id") or row.get("document_id")
                if jid:
                    ids.append(str(jid))
            return ids
        except Exception as exc:
            logger.error("Failed to list queued jobs: %s", exc)
            return []

    def requeue_job(self, job_id: str, error_message: Optional[str] = None) -> bool:
        """Atomically increment retry_count, reset status to QUEUED, and clear processor_id."""
        try:
            now = datetime.utcnow()
            sql = """
            UPDATE document_job SET
                status = %s,
                processor_id = NULL,
                retry_count = COALESCE(retry_count, 0) + 1,
                error_details = COALESCE(%s, error_details),
                updated_at = %s
            WHERE job_id = %s OR document_id = %s
            """
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (JobState.QUEUED.value, error_message, now, job_id, job_id))
            logger.info("Requeued job %s (retried with reason: %s)", job_id, error_message)
            return True
        except Exception as exc:
            logger.error("Failed to requeue job %s: %s", job_id, exc)
            return False

    def mark_job_failed(self, job_id: str, error_message: str) -> bool:
        """Transition job status to FAILED with error details."""
        return self.update_job_status(
            job_id=job_id,
            status=JobState.FAILED,
            error_message=error_message,
        )

    def mark_dispatch_success(
        self,
        *,
        job_id: str,
        document_id: str,
        processor_id: str,
        processor_type: str,
        processor_port: int | None = None,
    ) -> bool:
        """Persist successful HTTP dispatch without downgrading IN_PROGRESS follow-ups."""
        did = str(document_id or job_id).strip()
        if not did:
            return False
        meta = {
            "processor_type": processor_type,
            "processor_id": processor_id,
            "processor_port": processor_port,
        }
        record = self.get_job_record(job_id, did)
        status = self.normalized_status(record)
        try:
            if status in {JobState.QUEUED.value, JobState.DISPATCHING.value}:
                self._store.update_assigned(
                    did,
                    processor_id=processor_id,
                    scheduling_metadata=meta,
                )
            else:
                self._store.record_processor_assignment(
                    did,
                    processor_id=processor_id,
                    scheduling_metadata=meta,
                )
            return True
        except Exception as exc:
            logger.error(
                "Failed to mark dispatch success for %s processor %s: %s",
                did,
                processor_type,
                exc,
            )
            if status in {JobState.QUEUED.value, JobState.DISPATCHING.value}:
                return self.update_job_status(
                    job_id=job_id,
                    status=JobState.ASSIGNED,
                    processor_id=processor_id,
                )
            return False

    def claim_job_atomically(self, job_id: str, processor_id: str) -> bool:
        """Atomically claim a QUEUED job by setting status to DISPATCHING.
        Returns True if claimed (rowcount == 1), False if already claimed.
        """
        try:
            now = datetime.utcnow()
            sql = """
            UPDATE document_job SET
                status = %s,
                processor_id = %s,
                updated_at = %s
            WHERE (job_id = %s OR document_id = %s)
              AND status IN ('QUEUED', 'queued')
            """
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (JobState.DISPATCHING.value, processor_id, now, job_id, job_id))
                    claimed = cur.rowcount > 0
                    if claimed:
                        logger.info("Atomically claimed job %s for processor %s", job_id, processor_id)
                    return claimed
        except Exception as exc:
            logger.error("Failed atomic claim for job %s: %s", job_id, exc)
            return False

    def recover_stale_dispatching_jobs(self, timeout_seconds: int = 60) -> List[str]:
        """Recover jobs left in DISPATCHING from previous scheduler crash/restart or dispatch drop."""
        try:
            sql = f"""
            SELECT job_id, document_id FROM document_job
            WHERE status = 'DISPATCHING'
              AND updated_at < (NOW() - INTERVAL '{int(timeout_seconds)} seconds')
            LIMIT 50
            """
            recovered = []
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    for r in rows:
                        jid = r.get("job_id") or r.get("document_id")
                        if jid:
                            recovered.append(jid)

            if recovered:
                logger.warning("Found %d stale DISPATCHING jobs (>%ds). Requeuing: %s", len(recovered), timeout_seconds, recovered)
                for jid in recovered:
                    self.requeue_job(jid, error_message=f"Recovered stale DISPATCHING (> {timeout_seconds}s)")
            return recovered
        except Exception as exc:
            logger.error("Failed to recover stale dispatching jobs: %s", exc)
            return []

    def recover_stale_assigned_jobs(
        self,
        timeout_seconds: int = 300,
        exclude_job_ids: Optional[set[str]] = None,
        exclude_document_ids: Optional[set[str]] = None,
    ) -> List[str]:
        """Recover jobs left in ASSIGNED that workers never started."""
        exclude_job_ids = exclude_job_ids or set()
        exclude_document_ids = exclude_document_ids or set()
        try:
            sql = f"""
            SELECT job_id, document_id FROM document_job
            WHERE status = 'ASSIGNED'
              AND updated_at < (NOW() - INTERVAL '{int(timeout_seconds)} seconds')
            LIMIT 50
            """
            recovered = []
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    for r in rows:
                        jid = str(r.get("job_id") or r.get("document_id") or "").strip()
                        did = str(r.get("document_id") or jid).strip()
                        if not jid:
                            continue
                        if (
                            jid in exclude_job_ids
                            or did in exclude_job_ids
                            or did in exclude_document_ids
                        ):
                            continue
                        recovered.append(jid)

            if recovered:
                logger.warning("Found %d stale ASSIGNED jobs (>%ds). Requeuing: %s", len(recovered), timeout_seconds, recovered)
                for jid in recovered:
                    self.requeue_job(jid, error_message=f"Recovered stale ASSIGNED (> {timeout_seconds}s)")
            return recovered
        except Exception as exc:
            logger.error("Failed to recover stale assigned jobs: %s", exc)
            return []

    def recover_orphaned_active_jobs(
        self,
        live_job_ids: set[str],
        *,
        heartbeat_grace_seconds: int = 120,
        recently_released_document_ids: Optional[set[str]] = None,
    ) -> List[str]:
        """Requeue active DB rows that are not held on any live processor slot.

        IN_PROGRESS rows with a fresh heartbeat are left alone (processor may be
        between slot release and sibling enqueue). IN_PROGRESS orphans enqueue
        pending processor types instead of resetting the whole row to QUEUED.
        """
        recently_released_document_ids = recently_released_document_ids or set()
        try:
            sql = f"""
            SELECT job_id, document_id, status, updated_at, heartbeat_at
            FROM document_job
            WHERE status IN ('DISPATCHING', 'ASSIGNED', 'IN_PROGRESS', 'processing', 'STARTED')
              AND updated_at < (NOW() - INTERVAL '{int(heartbeat_grace_seconds)} seconds')
            LIMIT 100
            """
            orphaned: List[str] = []
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            for row in rows:
                jid = str(row.get("job_id") or row.get("document_id") or "").strip()
                did = str(row.get("document_id") or jid).strip()
                if not jid:
                    continue
                if jid in live_job_ids or did in live_job_ids:
                    continue
                if did in recently_released_document_ids:
                    continue
                status = str(row.get("status") or "").strip().upper()
                if status in {"IN_PROGRESS", "PROCESSING", "STARTED"}:
                    pending = self._store.list_pending_processor_types(did)
                    if pending:
                        try:
                            from src.features.documents.application.job_republish import (
                                republish_processor_types,
                            )

                            republish_processor_types(did, pending[:1])
                            logger.info(
                                "Republished pending processor %s for orphaned IN_PROGRESS doc %s",
                                pending[0],
                                did,
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed to republish pending processors for %s: %s",
                                did,
                                exc,
                            )
                        continue
                orphaned.append(jid)
            if orphaned:
                logger.warning(
                    "Found %d orphaned active jobs with no live slot. Requeuing: %s",
                    len(orphaned),
                    orphaned,
                )
                for key in orphaned:
                    self.requeue_job(key, error_message="Orphaned active job with no live processor slot")
            return orphaned
        except Exception as exc:
            logger.error("Failed to recover orphaned active jobs: %s", exc)
            return []

    def recover_stale_in_progress_jobs(self, timeout_seconds: int = 1800) -> List[str]:
        """Timeout orphaned IN_PROGRESS jobs exceeding max job execution time."""
        try:
            sql = f"""
            SELECT job_id, document_id FROM document_job
            WHERE status IN ('IN_PROGRESS', 'processing')
              AND updated_at < (NOW() - INTERVAL '{int(timeout_seconds)} seconds')
            LIMIT 50
            """
            timed_out = []
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    for r in rows:
                        jid = r.get("job_id") or r.get("document_id")
                        if jid:
                            timed_out.append(jid)

            if timed_out:
                logger.warning("Found %d timed-out IN_PROGRESS jobs (>%ds). Marking TIMED_OUT: %s", len(timed_out), timeout_seconds, timed_out)
                for jid in timed_out:
                    self.update_job_status(
                        job_id=jid,
                        status=JobState.TIMED_OUT,
                        error_message=f"Job exceeded execution timeout ({timeout_seconds}s)",
                    )
            return timed_out
        except Exception as exc:
            logger.error("Failed to recover stale in-progress jobs: %s", exc)
            return []

    def recover_stale_queued_jobs(self, timeout_seconds: int = 300) -> List[Dict[str, Any]]:
        """Query PostgreSQL for jobs stuck in QUEUED for longer than timeout_seconds."""
        try:
            sql = f"""
            SELECT job_id, document_id, document_name, created_at, updated_at
            FROM document_job
            WHERE status IN ('QUEUED', 'queued')
              AND updated_at < (NOW() - INTERVAL '{int(timeout_seconds)} seconds')
            ORDER BY updated_at ASC
            LIMIT 50
            """
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    result = [dict(r) for r in rows]
                    if result:
                        job_ids = [r["job_id"] for r in result]
                        cur.execute(
                            "UPDATE document_job SET updated_at = NOW() WHERE job_id = ANY(%s)",
                            (job_ids,),
                        )
                        conn.commit()
                    return result
        except Exception as exc:
            logger.error("Failed to query stale queued jobs: %s", exc)
            return []

    def recover_pending_follow_up_processors(self) -> List[Tuple[str, List[str]]]:
        """Find IN_PROGRESS jobs with uncompleted follow-up processors."""
        try:
            sql = """
            SELECT document_id, scheduling_metadata
            FROM document_job
            WHERE status IN ('IN_PROGRESS', 'in_progress')
              AND scheduling_metadata IS NOT NULL
              AND scheduling_metadata->'follow_up_processors' IS NOT NULL
              AND jsonb_array_length(scheduling_metadata->'follow_up_processors') > 0
              AND updated_at < (NOW() - INTERVAL '2 seconds')
            LIMIT 20
            """
            pending_republishes: List[Tuple[str, List[str]]] = []
            with self._store._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
                    for r in rows:
                        doc_id = str(r["document_id"]).strip()
                        meta = r["scheduling_metadata"] or {}
                        if not isinstance(meta, dict):
                            continue
                        follow_ups = meta.get("follow_up_processors") or []
                        results = meta.get("processor_results") or {}
                        uncompleted = [
                            str(pt).strip()
                            for pt in follow_ups
                            if str(pt).strip()
                            and (pt not in results or (results.get(pt) or {}).get("status") not in ("COMPLETED", "FAILED"))
                        ]
                        if uncompleted:
                            pending_republishes.append((doc_id, uncompleted))
            return pending_republishes
        except Exception as exc:
            logger.error("Failed to query pending follow-up processors: %s", exc)
            return []

