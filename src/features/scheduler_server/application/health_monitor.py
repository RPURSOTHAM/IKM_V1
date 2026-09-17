from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional

from src.features.scheduler_server.application.pool_manager import ProcessorPoolManager
from src.features.scheduler_server.infrastructure.job_repository import SchedulerJobRepository
from src.features.scheduler_server.domain.job_state import JobState
from src.features.scheduler_server.domain.slot_models import SlotStatus
from src.features.scheduler_server.scheduler_config import SchedulerRuntimeConfig

logger = logging.getLogger(__name__)


class ContainerHealthMonitor:
    """Container reconciliation loop & job timeout detection engine."""

    def __init__(
        self,
        pool_mgr: ProcessorPoolManager,
        job_repo: SchedulerJobRepository,
        runtime_config: SchedulerRuntimeConfig | None = None,
        poll_frequency: int | None = None,
        job_timeout_seconds: int | None = None,
        config_refresh_callback: Optional[Callable[[], Awaitable[None]]] = None,
        queue_consumer: Optional[Any] = None,
    ):
        self.pool_mgr = pool_mgr
        self.job_repo = job_repo
        self.runtime = runtime_config or SchedulerRuntimeConfig.from_env()
        self.poll_frequency = poll_frequency if poll_frequency is not None else self.runtime.container_poll_frequency
        self.job_timeout_seconds = (
            job_timeout_seconds if job_timeout_seconds is not None else self.runtime.job_timeout_seconds
        )
        self.config_refresh_callback = config_refresh_callback
        self.queue_consumer = queue_consumer
        self._running = False
        self.orphan_grace_seconds = self.runtime.orphan_grace_seconds
        self.stale_queued_seconds = self.runtime.stale_queued_seconds
        self.stale_dispatching_seconds = self.runtime.stale_dispatching_seconds
        self.stale_assigned_seconds = self.runtime.stale_assigned_seconds
        self.republish_cooldown_seconds = self.runtime.republish_cooldown_seconds
        self.stale_queued_republish_limit = self.runtime.stale_queued_republish_limit

    async def start_monitoring(self) -> None:
        """Asynchronous monitoring loop executing every `poll_frequency` seconds."""
        self._running = True
        logger.info("Starting Container Health Monitor loop (interval: %ds)...", self.poll_frequency)
        while self._running:
            try:
                await asyncio.to_thread(self.pool_mgr.reconcile_pool)
                await asyncio.to_thread(self.check_job_timeouts)
                await asyncio.to_thread(self.reconcile_database_jobs)
                if self.config_refresh_callback:
                    await self.config_refresh_callback()
            except Exception as exc:
                logger.error("Error in Container Health Monitor iteration: %s", exc)

            await asyncio.sleep(self.poll_frequency)

    def _live_slot_job_ids(self) -> set[str]:
        live_ids: set[str] = set()
        for slot in self.pool_mgr.slots.values():
            if slot.current_job_id and slot.status in {
                SlotStatus.BUSY,
                SlotStatus.RESERVED,
            }:
                live_ids.add(str(slot.current_job_id))
            if slot.current_document_id and slot.status in {
                SlotStatus.BUSY,
                SlotStatus.RESERVED,
            }:
                live_ids.add(str(slot.current_document_id))
        return live_ids

    def _live_slot_document_ids(self) -> set[str]:
        """Document IDs actively running on a processor slot (avoid false stale recovery)."""
        live_docs: set[str] = set()
        for slot in self.pool_mgr.slots.values():
            if not slot.current_document_id:
                continue
            if slot.status in {SlotStatus.BUSY, SlotStatus.RESERVED} or slot.state == "busy":
                live_docs.add(str(slot.current_document_id))
        return live_docs

    def _recently_released_document_ids(self) -> set[str]:
        cutoff = datetime.utcnow() - timedelta(seconds=self.orphan_grace_seconds)
        released: set[str] = set()
        for slot in self.pool_mgr.slots.values():
            if not slot.released_at or slot.released_at < cutoff:
                continue
            if slot.last_released_document_id:
                released.add(str(slot.last_released_document_id))
        return released

    def reconcile_database_jobs(self) -> None:
        """Continuously reconcile orphaned/stale database jobs."""
        try:
            queued_count = self.job_repo.count_queued_jobs()
            if queued_count:
                self.pool_mgr.recover_capacity_for_queued(queued_count)

            live_job_ids = self._live_slot_job_ids()
            live_document_ids = self._live_slot_document_ids()

            self.job_repo.recover_stale_dispatching_jobs(
                timeout_seconds=self.stale_dispatching_seconds,
            )
            self.job_repo.recover_stale_assigned_jobs(
                timeout_seconds=self.stale_assigned_seconds,
                exclude_job_ids=live_job_ids,
                exclude_document_ids=live_document_ids,
            )
            self.job_repo.recover_stale_in_progress_jobs(timeout_seconds=self.job_timeout_seconds)

            orphaned = self.job_repo.recover_orphaned_active_jobs(
                live_job_ids,
                heartbeat_grace_seconds=self.orphan_grace_seconds,
                recently_released_document_ids=self._recently_released_document_ids(),
            )
            if self.queue_consumer:
                for jid in orphaned:
                    record = self.job_repo.get_job_record(jid)
                    did = str((record or {}).get("document_id") or jid)
                    if self.job_repo.republish_cooled_down(did, self.republish_cooldown_seconds):
                        self.queue_consumer.republish_job_by_id(jid, did)

            if self.queue_consumer and self.stale_queued_republish_limit > 0:
                stale_queued = self.job_repo.recover_stale_queued_jobs(
                    timeout_seconds=self.stale_queued_seconds,
                )
                republished = 0
                for qjob in stale_queued:
                    if republished >= self.stale_queued_republish_limit:
                        break
                    jid = qjob.get("job_id") or qjob.get("document_id")
                    did = str(qjob.get("document_id") or jid or "").strip()
                    if not jid or not did:
                        continue
                    if not self.job_repo.republish_cooled_down(did, self.republish_cooldown_seconds):
                        continue
                    ready = self.job_repo.list_ready_processor_types(did)
                    if not ready:
                        continue
                    logger.info(
                        "Re-publishing stale QUEUED job %s processors %s to RabbitMQ",
                        jid,
                        ready,
                    )
                    self.queue_consumer.republish_job_by_id(
                        str(jid),
                        did,
                        processor_types=ready,
                    )
                    republished += 1

            try:
                from src.features.documents.application.job_republish import enqueue_ready_processors

                pending_followups = self.job_repo.recover_pending_follow_up_processors()
                for doc_id, _pts in pending_followups:
                    enqueue_ready_processors(doc_id)
            except Exception as ex:
                logger.debug("Could not auto-republish follow-up processors: %s", ex)
        except Exception as exc:
            logger.error("Error in reconcile_database_jobs: %s", exc)

    def check_job_timeouts(self) -> None:
        """Check for busy slots executing beyond `job_timeout_seconds`."""
        now = datetime.utcnow()
        for slot in self.pool_mgr.slots.values():
            if slot.current_job_id and slot.assigned_at:
                elapsed = (now - slot.assigned_at).total_seconds()
                if elapsed > self.job_timeout_seconds:
                    logger.error(
                        "Job %s on slot %s exceeded timeout (%.0fs > %ds). Marking TIMED_OUT...",
                        slot.current_job_id,
                        slot.slot_id,
                        elapsed,
                        self.job_timeout_seconds,
                    )
                    self.job_repo.update_job_status(
                        job_id=slot.current_job_id,
                        status=JobState.TIMED_OUT,
                        error_message=f"Job processing timed out after {elapsed:.0f} seconds",
                    )
                    if slot.current_document_id:
                        self.pool_mgr.docker_mgr.stop_processor_job_http(
                            slot_port=slot.host_port,
                            container_name=slot.container_name,
                            document_id=slot.current_document_id,
                        )
                    slot.mark_idle()

    def stop(self) -> None:
        """Stop monitoring loop."""
        self._running = False
        logger.info("Stopped Container Health Monitor.")
