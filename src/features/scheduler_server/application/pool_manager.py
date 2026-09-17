from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from src.features.scheduler_server.domain.slot_models import ProcessorSlot, SlotStatus
from src.features.scheduler_server.infrastructure.docker_manager import DockerServiceManager
from src.features.scheduler_server.infrastructure.job_repository import SchedulerJobRepository

logger = logging.getLogger(__name__)


class ProcessorPoolManager:
    """Manages the pool of processor slots, allocation, dynamic auto-scaling (MIN=5, MAX=10), and auto scale-down."""

    def __init__(
        self,
        docker_mgr: DockerServiceManager,
        job_repo: SchedulerJobRepository,
        port_range: Tuple[int, int] = (3100, 3120),
        min_parallel_jobs: int = 5,
        max_parallel_jobs: int = 10,
        document_host_path: str = "./data/documents",
        document_container_path: str = "/app/documents",
        models_host_path: Optional[str] = None,
        models_container_path: str = "/app/src/models",
        processor_image: str = "rag-processor:latest",
        idle_scale_down_seconds: int = 300,
        max_retries: int = 3,
        startup_failure_threshold: int = 24,
        runtime_failure_threshold: int = 12,
        unhealthy_restart_threshold: int = 3,
    ):
        self.docker_mgr = docker_mgr
        self.job_repo = job_repo
        self.port_range_start, self.port_range_end = port_range
        self.min_parallel_jobs = min_parallel_jobs
        self.max_parallel_jobs = max_parallel_jobs
        self.document_host_path = document_host_path
        self.document_container_path = document_container_path
        self.models_host_path = models_host_path
        self.models_container_path = models_container_path
        self.processor_image = processor_image
        self.idle_scale_down_seconds = idle_scale_down_seconds
        self.max_retries = max_retries
        self.startup_failure_threshold = startup_failure_threshold
        self.runtime_failure_threshold = runtime_failure_threshold
        self.unhealthy_restart_threshold = unhealthy_restart_threshold
        self.requeue_callback = None

        self.slots: Dict[str, ProcessorSlot] = {}
        self._initialize_slots_inventory()

    def set_requeue_callback(self, callback) -> None:
        """Register callback to republish recovered jobs to queue broker."""
        self.requeue_callback = callback

    def recover_failed_worker_job(self, slot: ProcessorSlot, reason: str = "Worker failure") -> None:
        """Recover an in-flight job from a failed or crashed worker slot."""
        job_id = slot.current_job_id
        if not job_id:
            return
        doc_id = slot.current_document_id or job_id
        logger.warning("Recovering in-flight job %s from failed slot %s (reason: %s)", job_id, slot.slot_id, reason)
        retry_count = self.job_repo.get_job_retry_count(job_id)
        if retry_count >= self.max_retries:
            err = f"Job {job_id} exceeded maximum retries ({self.max_retries}). Last failure: {reason}"
            logger.error(err)
            self.job_repo.mark_job_failed(job_id, error_message=err)
        else:
            self.job_repo.requeue_job(job_id, error_message=reason)
            if self.requeue_callback:
                try:
                    self.requeue_callback(job_id, doc_id)
                except Exception as exc:
                    logger.error("Error in requeue_callback for job %s: %s", job_id, exc)
        slot.mark_idle()

    def _initialize_slots_inventory(self) -> None:
        """Create baseline inventory for MIN_PARALLEL_JOBS."""
        for i in range(self.min_parallel_jobs):
            slot_name = f"doc_processor_{i+1}"
            port = self.port_range_start + i
            self.slots[slot_name] = ProcessorSlot(
                slot_id=slot_name,
                host_port=port,
                container_name=slot_name,
                status=SlotStatus.STARTING,
                is_dynamic=False,
            )

    def reconcile_pool(self) -> Dict[str, ProcessorSlot]:
        """Reconcile all container slots against Docker and HTTP /health."""
        existing_containers = {c.name: c for c in self.docker_mgr.find_slot_containers()}

        # Reconcile baseline slots
        for slot_name, slot in list(self.slots.items()):
            container = existing_containers.get(slot_name)
            if not container:
                logger.warning("Slot container %s missing. Spawning replacement...", slot_name)
                if slot.current_job_id:
                    self.recover_failed_worker_job(slot, "Slot container missing from Docker host")
                c = self.docker_mgr.start_slot_container(
                    slot_name=slot_name,
                    image_name=self.processor_image,
                    host_port=slot.host_port,
                    document_host_path=self.document_host_path,
                    document_container_path=self.document_container_path,
                    models_host_path=self.models_host_path,
                    models_container_path=self.models_container_path,
                )
                if c:
                    slot.container_id = c.id
                    slot.status = SlotStatus.STARTING
                continue

            slot.container_id = container.id
            if container.status != "running":
                logger.warning("Container %s is in status '%s'. Restarting...", slot_name, container.status)
                if slot.current_job_id:
                    self.recover_failed_worker_job(slot, f"Container status transitioned to '{container.status}'")
                c = self.docker_mgr.start_slot_container(
                    slot_name=slot_name,
                    image_name=self.processor_image,
                    host_port=slot.host_port,
                    document_host_path=self.document_host_path,
                    document_container_path=self.document_container_path,
                    models_host_path=self.models_host_path,
                    models_container_path=self.models_container_path,
                )
                if c:
                    slot.container_id = c.id
                    slot.status = SlotStatus.STARTING
                continue

            # HTTP Health Check
            health_info = self.docker_mgr.check_processor_health(
                slot.host_port,
                slot_name,
                timeout=self.docker_mgr.processor_health_timeout,
            )
            slot.last_health_check = datetime.utcnow()
            slot.health = health_info.get("health", "unhealthy")
            slot.state = health_info.get("state", "idle")
            details = health_info.get("details") if isinstance(health_info.get("details"), dict) else {}

            if slot.health == "healthy":
                slot.failure_count = 0
                processor_busy = details.get("processor_busy")
                pipeline_stage = str(details.get("pipeline_stage") or "").lower()
                if processor_busy is False and slot.current_job_id is not None:
                    logger.info(
                        "Slot %s worker finished (processor_busy=false); releasing doc %s",
                        slot_name,
                        slot.current_document_id or slot.current_job_id,
                    )
                    slot.mark_idle()
                elif pipeline_stage in {"completed", "failed", "stopped"} and slot.current_job_id is not None:
                    logger.info(
                        "Slot %s pipeline terminal (%s); releasing doc %s",
                        slot_name,
                        pipeline_stage,
                        slot.current_document_id or slot.current_job_id,
                    )
                    slot.mark_idle()
                elif slot.state == "idle":
                    if slot.current_job_id is not None:
                        logger.info(
                            "Slot %s completed work for doc %s processor %s (worker idle). Releasing slot.",
                            slot_name,
                            slot.current_document_id or slot.current_job_id,
                            slot.current_processor_type,
                        )
                        # Terminal job status is owned by the processor via
                        # record_processor_outcome(); only release the slot here.
                        slot.mark_idle()
                    else:
                        slot.status = SlotStatus.HEALTHY_IDLE
                elif slot.state == "busy":
                    slot.status = SlotStatus.BUSY
            else:
                slot.failure_count += 1
                threshold = (
                    self.startup_failure_threshold
                    if slot.status == SlotStatus.STARTING
                    else self.runtime_failure_threshold
                )
                if container.status == "running":
                    idle_slot = slot.current_job_id is None and slot.status != SlotStatus.BUSY
                    if idle_slot and slot.failure_count >= self.unhealthy_restart_threshold:
                        logger.warning(
                            "Slot %s unresponsive while idle (%d failures); restarting container",
                            slot_name,
                            slot.failure_count,
                        )
                        self._restart_slot_container(slot_name, slot)
                        continue
                    logger.warning(
                        "Slot %s health check failed (%d) while Docker status=running; leaving container running",
                        slot_name,
                        slot.failure_count,
                    )
                    continue
                logger.warning("Slot %s health check failed (%d/%d)", slot_name, slot.failure_count, threshold)
                if slot.failure_count >= threshold:
                    logger.error("Slot %s exceeded failure threshold (%d/%d). Replacing container...", slot_name, slot.failure_count, threshold)
                    slot.status = SlotStatus.UNHEALTHY
                    if slot.current_job_id:
                        self.recover_failed_worker_job(slot, f"Exceeded health check failure threshold ({threshold}/{threshold})")
                    self.docker_mgr.stop_and_remove_container(slot_name)
                    c = self.docker_mgr.start_slot_container(
                        slot_name=slot_name,
                        image_name=self.processor_image,
                        host_port=slot.host_port,
                        document_host_path=self.document_host_path,
                        document_container_path=self.document_container_path,
                        models_host_path=self.models_host_path,
                        models_container_path=self.models_container_path,
                    )
                    if c:
                        slot.container_id = c.id
                        slot.status = SlotStatus.STARTING
                    slot.failure_count = 0

            self._reconcile_stale_slot_assignment(slot_name, slot)

        # Auto Scale-Down check for dynamic slots
        self._check_auto_scale_down()
        return self.slots

    def _restart_slot_container(self, slot_name: str, slot: ProcessorSlot) -> None:
        """Replace a running but unresponsive processor container."""
        if slot.current_job_id:
            self.recover_failed_worker_job(slot, "Processor container restarted after health failures")
        self.docker_mgr.stop_and_remove_container(slot_name)
        c = self.docker_mgr.start_slot_container(
            slot_name=slot_name,
            image_name=self.processor_image,
            host_port=slot.host_port,
            document_host_path=self.document_host_path,
            document_container_path=self.document_container_path,
            models_host_path=self.models_host_path,
            models_container_path=self.models_container_path,
        )
        if c:
            slot.container_id = c.id
            slot.status = SlotStatus.STARTING
        slot.failure_count = 0
        slot.mark_idle()

    def _reconcile_stale_slot_assignment(self, slot_name: str, slot: ProcessorSlot) -> None:
        """Release slots that are busy in memory but no longer active in the database."""
        if slot.status not in {SlotStatus.BUSY, SlotStatus.RESERVED}:
            return
        if slot.current_job_id and not self.job_repo.is_processor_slot_active(slot.slot_id):
            logger.info(
                "Slot %s has stale in-memory assignment for %s; releasing for new work",
                slot_name,
                slot.current_document_id or slot.current_job_id,
            )
            slot.mark_idle()
            return
        if (
            slot.status == SlotStatus.BUSY
            and slot.state == "idle"
            and slot.health == "healthy"
            and not self.job_repo.is_processor_slot_active(slot.slot_id)
        ):
            logger.info("Slot %s worker idle with no active DB assignment; releasing", slot_name)
            slot.mark_idle()

    def recover_capacity_for_queued(self, queued_count: int = 0) -> int:
        """Free or restart slots so fresh QUEUED uploads can dispatch within SLA."""
        if queued_count <= 0:
            return 0
        freed = 0
        for slot_name, slot in list(self.slots.items()):
            self._reconcile_stale_slot_assignment(slot_name, slot)
            if slot.is_available_for_job:
                freed += 1
        if freed:
            return freed

        for slot_name, slot in list(self.slots.items()):
            if slot.current_job_id is None and slot.health != "healthy":
                if slot.failure_count >= self.unhealthy_restart_threshold:
                    logger.info("Recovering queued capacity by restarting unhealthy idle slot %s", slot_name)
                    self._restart_slot_container(slot_name, slot)
                    freed += 1
        return freed

    def get_available_slot(self) -> Optional[ProcessorSlot]:
        """Find an idle healthy processor slot."""
        for slot in self.slots.values():
            if slot.is_available_for_job:
                return slot

        # Check if dynamic scale-up can be triggered
        return self._maybe_scale_up()

    def _maybe_scale_up(self) -> Optional[ProcessorSlot]:
        """Dynamically create a new slot if capacity allows (up to MAX_PARALLEL_JOBS)."""
        current_count = len(self.slots)
        if current_count >= self.max_parallel_jobs:
            logger.debug("Pool reached max parallel capacity (%d). Cannot scale up.", self.max_parallel_jobs)
            return None

        new_index = current_count + 1
        slot_name = f"doc_processor_{new_index}"
        port = self.port_range_start + (new_index - 1)

        logger.info("Dynamically scaling up pool: Creating slot %s on port %d", slot_name, port)
        c = self.docker_mgr.start_slot_container(
            slot_name=slot_name,
            image_name=self.processor_image,
            host_port=port,
            document_host_path=self.document_host_path,
            document_container_path=self.document_container_path,
            models_host_path=self.models_host_path,
            models_container_path=self.models_container_path,
        )

        new_slot = ProcessorSlot(
            slot_id=slot_name,
            host_port=port,
            container_id=c.id if c else None,
            container_name=slot_name,
            status=SlotStatus.STARTING,
            is_dynamic=True,
        )
        if c:
            health_info = self.docker_mgr.check_processor_health(port, slot_name, timeout=1.0)
            if health_info.get("health") == "healthy":
                new_slot.health = "healthy"
                new_slot.state = health_info.get("state", "idle")
                new_slot.status = SlotStatus.HEALTHY_IDLE

        self.slots[slot_name] = new_slot
        return new_slot if (c and new_slot.is_available_for_job) else None

    def _check_auto_scale_down(self) -> None:
        """Scale down extra dynamic slots if idle beyond idle_scale_down_seconds."""
        now = datetime.utcnow()
        for slot_name, slot in list(self.slots.items()):
            if not slot.is_dynamic:
                continue

            if slot.status == SlotStatus.HEALTHY_IDLE and slot.last_health_check:
                idle_duration = (now - slot.last_health_check).total_seconds()
                if idle_duration > self.idle_scale_down_seconds and len(self.slots) > self.min_parallel_jobs:
                    logger.info("Scaling down idle dynamic slot %s (idle for %.0fs)", slot_name, idle_duration)
                    self.docker_mgr.stop_and_remove_container(slot_name)
                    del self.slots[slot_name]

    def stop_all_slots(self) -> None:
        """Gracefully stop and remove all managed container slots."""
        logger.info("Stopping all managed processor slots...")
        for slot_name in list(self.slots.keys()):
            self.docker_mgr.stop_and_remove_container(slot_name)
        self.slots.clear()
