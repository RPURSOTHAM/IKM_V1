from __future__ import annotations

import asyncio
import json
import logging
import pika
from typing import Any, Dict, Literal, Optional

from src.features.scheduler_server.application.pool_manager import ProcessorPoolManager
from src.features.scheduler_server.infrastructure.job_repository import SchedulerJobRepository
from src.features.scheduler_server.domain.job_state import JobState
from src.features.scheduler_server.domain.slot_models import ProcessorSlot, SlotStatus

logger = logging.getLogger(__name__)

_ACTIVE_DB_STATUSES = frozenset(
    {"DISPATCHING", "ASSIGNED", "IN_PROGRESS", "PROCESSING", "STARTED"}
)


def _normalize_processor_type(payload: Dict[str, Any]) -> str:
    return str(payload.get("processor_type") or "chunking_vectorizing").strip().lower()


class RabbitMQQueueConsumer:
    """Consumes document processing jobs from RabbitMQ and dispatches to idle processor slots."""

    def __init__(
        self,
        rabbitmq_config: Dict[str, Any],
        pool_mgr: ProcessorPoolManager,
        job_repo: SchedulerJobRepository,
        check_frequency: int = 5,
        queue_scan_limit: int = 100,
        dispatch_sla_seconds: int = 30,
        queue_priority_batch: int = 25,
    ):
        self.rabbitmq_config = rabbitmq_config
        self.pool_mgr = pool_mgr
        self.job_repo = job_repo
        self.check_frequency = check_frequency
        self.queue_scan_limit = max(10, int(queue_scan_limit))
        self.dispatch_sla_seconds = max(10, int(dispatch_sla_seconds))
        self.queue_priority_batch = max(5, int(queue_priority_batch))
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self._running = False

    def _connect(self) -> bool:
        """Establish connection to RabbitMQ broker."""
        try:
            rmq_host = self.rabbitmq_config.get("host", "localhost")
            if rmq_host in {"rabbitmq", "rag-rabbitmq"}:
                try:
                    import socket
                    socket.gethostbyname(rmq_host)
                except socket.gaierror:
                    rmq_host = "localhost"

            credentials = pika.PlainCredentials(
                self.rabbitmq_config.get("user", "rabbitmq_user"),
                self.rabbitmq_config.get("pass", "rabbitmq_password"),
            )
            parameters = pika.ConnectionParameters(
                host=rmq_host,
                port=int(self.rabbitmq_config.get("port", 5672)),
                credentials=credentials,
                connection_attempts=3,
                retry_delay=2,
            )
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            self._channel.queue_declare(
                queue=self.rabbitmq_config.get("queue_name", "document_processing_queue"),
                durable=True,
            )
            logger.info("Connected to RabbitMQ queue: %s", self.rabbitmq_config.get("queue_name"))
            return True
        except Exception as exc:
            logger.error("Failed to connect to RabbitMQ: %s", exc)
            self._connection = None
            self._channel = None
            return False

    async def start_polling(self) -> None:
        """Asynchronous polling loop checking RabbitMQ at `check_frequency` intervals."""
        self._running = True
        logger.info("Starting RabbitMQ polling loop (check interval: %ds)...", self.check_frequency)
        while self._running:
            try:
                if not self._connection or self._connection.is_closed or not self._channel or self._channel.is_closed:
                    if not self._connect():
                        await asyncio.sleep(self.check_frequency)
                        continue

                await asyncio.to_thread(self.compact_queue_backlog)
                await asyncio.to_thread(self.drain_available_jobs)
            except Exception as exc:
                logger.error("Error in queue polling iteration: %s", exc)
                try:
                    if self._connection and not self._connection.is_closed:
                        self._connection.close()
                except Exception:
                    pass
                self._connection = None
                self._channel = None

            await asyncio.sleep(self.check_frequency)

    def _slot_holding_document(self, document_id: str) -> Optional[ProcessorSlot]:
        """Return a live slot currently working on this document."""
        for slot in self.pool_mgr.slots.values():
            if slot.current_document_id == document_id:
                if slot.status in {SlotStatus.BUSY, SlotStatus.RESERVED} or slot.state == "busy":
                    return slot
        return None

    def _slot_holding_processor_work(
        self,
        document_id: str,
        processor_type: str,
    ) -> Optional[ProcessorSlot]:
        """Return a live slot running this document + processor_type pair."""
        for slot in self.pool_mgr.slots.values():
            if slot.current_document_id != document_id:
                continue
            slot_type = str(slot.current_processor_type or "").strip().lower()
            if slot_type and slot_type != processor_type:
                continue
            if slot.status in {SlotStatus.BUSY, SlotStatus.RESERVED} or slot.state == "busy":
                return slot
        return None

    def _release_slot_reservation(self, slot: ProcessorSlot) -> None:
        if slot.status == SlotStatus.RESERVED and slot.state != "busy":
            slot.mark_idle()
        elif slot.status == SlotStatus.RESERVED:
            slot.current_job_id = None
            slot.current_document_id = None
            slot.current_processor_type = None
            slot.assigned_at = None
            slot.status = SlotStatus.HEALTHY_IDLE

    def _classify_queue_message(
        self,
        payload: Dict[str, Any],
    ) -> tuple[Literal["ack", "hold", "dispatch"], str, str, str, Dict[str, Any] | None, str]:
        document_id = str(payload.get("document_id") or "").strip()
        job_id = str(payload.get("job_id") or document_id).strip()
        processor_type = _normalize_processor_type(payload)
        if not document_id:
            return "ack", job_id, document_id, processor_type, None, ""

        if self.job_repo.is_processor_terminal(document_id, processor_type):
            return "ack", job_id, document_id, processor_type, None, ""
        if not self.job_repo.is_processor_ready(document_id, processor_type):
            return "ack", job_id, document_id, processor_type, None, ""

        record = self.job_repo.get_job_record(job_id, document_id)
        db_status = self.job_repo.normalized_status(record)

        if db_status == JobState.COMPLETED.value:
            return "ack", job_id, document_id, processor_type, record, db_status

        if db_status in {"FAILED", "TIMED_OUT", "STOPPED", "QUEUE_FAILED"}:
            if payload.get("requeued") or self.job_repo.is_processor_pending(document_id, processor_type):
                self.job_repo.reset_failed_for_republish(document_id)
                record = self.job_repo.get_job_record(job_id, document_id)
                db_status = self.job_repo.normalized_status(record)
            elif self.job_repo.should_skip_queue_message(job_id, document_id):
                return "ack", job_id, document_id, processor_type, record, db_status

        if self.job_repo.should_skip_queue_message(job_id, document_id):
            if not self.job_repo.is_processor_pending(document_id, processor_type):
                return "ack", job_id, document_id, processor_type, record, db_status

        if self.job_repo.is_processor_actively_dispatching(document_id, processor_type):
            return "ack", job_id, document_id, processor_type, record, db_status

        if self._slot_holding_processor_work(document_id, processor_type):
            return "ack", job_id, document_id, processor_type, record, db_status

        if not self.job_repo.is_processor_pending(document_id, processor_type):
            return "ack", job_id, document_id, processor_type, record, db_status

        return "dispatch", job_id, document_id, processor_type, record, db_status

    def compact_queue_backlog(self) -> int:
        """Remove stale/non-actionable messages without needing a free processor slot."""
        if not self._channel:
            return 0

        queue_name = self.rabbitmq_config.get("queue_name", "document_processing_queue")
        removed = 0
        for _ in range(self.queue_scan_limit):
            method_frame, _header, body = self._channel.basic_get(queue=queue_name, auto_ack=False)
            if not method_frame:
                break
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as exc:
                logger.error("Failed to decode queue message payload: %s", exc)
                self._channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
                removed += 1
                continue

            disposition, _job_id, document_id, processor_type, _record, _db_status = self._classify_queue_message(payload)
            if disposition == "ack":
                self._channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                removed += 1
                if document_id and processor_type:
                    logger.debug(
                        "Compacted queue message for doc %s processor %s",
                        document_id,
                        processor_type,
                    )
                continue

            self._channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
            break

        if removed:
            logger.info("Compacted %d stale queue message(s) from %s", removed, queue_name)
        return removed

    def _requeue_message(self, delivery_tag: int) -> None:
        if self._channel:
            self._channel.basic_nack(delivery_tag=delivery_tag, requeue=True)

    def _fetch_priority_dispatch_message(
        self,
        queue_name: str,
    ) -> tuple[Any, Dict[str, Any], str, str, str, str, Dict[str, Any] | None, str] | tuple[None, ...]:
        """Peek a batch and prefer fresh QUEUED uploads over follow-up republishes."""
        batch_limit = min(self.queue_priority_batch, self.queue_scan_limit)
        batch: list[tuple[Any, Dict[str, Any], str, str, str, str, Dict[str, Any] | None, str]] = []

        for _ in range(batch_limit):
            method_frame, _header, body = self._channel.basic_get(queue=queue_name, auto_ack=False)
            if not method_frame:
                break
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as exc:
                logger.error("Failed to decode queue message payload: %s", exc)
                self._channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
                continue

            disposition, job_id, document_id, processor_type, record, db_status = self._classify_queue_message(payload)
            batch.append(
                (method_frame, payload, disposition, job_id, document_id, processor_type, record, db_status)
            )

        if not batch:
            return (None, None, "", "", "", "", None, "")

        best_index = 0
        best_score = -1
        for index, item in enumerate(batch):
            _mf, payload, disposition, _jid, _did, _ptype, record, _status = item
            if disposition != "dispatch":
                continue
            score = self.job_repo.message_dispatch_priority(
                payload,
                record,
                sla_seconds=self.dispatch_sla_seconds,
            )
            if score > best_score:
                best_score = score
                best_index = index

        for index, item in enumerate(batch):
            if index == best_index:
                continue
            self._requeue_message(item[0].delivery_tag)

        return batch[best_index]

    def drain_available_jobs(self) -> None:
        """Drain messages from RabbitMQ while idle healthy slots exist."""
        if not self._channel:
            return

        queue_name = self.rabbitmq_config.get("queue_name", "document_processing_queue")
        scans = 0
        while self._running and scans < self.queue_scan_limit:
            scans += 1
            slot = self.pool_mgr.get_available_slot()
            if not slot:
                queued_count = self.job_repo.count_queued_jobs()
                if queued_count:
                    self.pool_mgr.recover_capacity_for_queued(queued_count)
                    slot = self.pool_mgr.get_available_slot()
            if not slot:
                break

            method_frame, payload, disposition, job_id, document_id, processor_type, record, db_status = (
                self._fetch_priority_dispatch_message(queue_name)
            )
            if not method_frame or payload is None:
                break
            if not document_id:
                logger.error("Invalid job payload missing document_id: %s", payload)
                self._channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
                continue

            if disposition == "ack":
                self._channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                continue

            if disposition == "hold":
                blocking_slot = self._slot_holding_document(document_id)
                logger.info(
                    "Doc %s busy on slot %s (%s); requeueing %s message (continuing drain)",
                    document_id,
                    blocking_slot.slot_id if blocking_slot else "?",
                    blocking_slot.current_processor_type if blocking_slot else "?",
                    processor_type,
                )
                self._channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
                continue

            slot.mark_reserved(job_id=job_id, document_id=document_id, processor_type=processor_type)

            claimed = False
            if db_status in {JobState.QUEUED.value, "QUEUED"}:
                claimed = self.job_repo.claim_job_atomically(job_id=job_id, processor_id=slot.slot_id)
            elif db_status in _ACTIVE_DB_STATUSES:
                claimed = self.job_repo.prepare_processor_dispatch(
                    document_id,
                    processor_type,
                    slot.slot_id,
                )
            else:
                claimed = self.job_repo.prepare_processor_dispatch(
                    document_id,
                    processor_type,
                    slot.slot_id,
                )

            if not claimed:
                record = self.job_repo.get_job_record(job_id, document_id)
                db_status = self.job_repo.normalized_status(record)
                if self._slot_holding_processor_work(document_id, processor_type):
                    self._release_slot_reservation(slot)
                    self._channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                    continue
                if db_status in {JobState.COMPLETED.value, "FAILED", "TIMED_OUT", "STOPPED", "QUEUE_FAILED"}:
                    self._release_slot_reservation(slot)
                    self._channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                    continue
                if not self.job_repo.is_processor_pending(document_id, processor_type):
                    self._release_slot_reservation(slot)
                    self._channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                    continue
                if db_status not in _ACTIVE_DB_STATUSES and db_status not in {JobState.QUEUED.value, "QUEUED"}:
                    self._release_slot_reservation(slot)
                    self._channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                    continue
                self.job_repo.prepare_processor_dispatch(document_id, processor_type, slot.slot_id)

            payload["processor_type"] = processor_type
            dispatch_payload = self.job_repo.build_dispatch_payload(job_id, document_id, payload)

            success, status_code, err_msg = self.pool_mgr.docker_mgr.dispatch_job_http(
                slot_port=slot.host_port,
                container_name=slot.container_name,
                job_payload=dispatch_payload,
            )

            if success:
                self.job_repo.mark_dispatch_success(
                    job_id=job_id,
                    document_id=document_id,
                    processor_id=slot.slot_id,
                    processor_type=processor_type,
                    processor_port=slot.host_port,
                )
                slot.mark_busy(job_id=job_id, document_id=document_id, processor_type=processor_type)
                self._channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                logger.info(
                    "Successfully assigned doc %s processor %s to slot %s",
                    document_id,
                    processor_type,
                    slot.slot_id,
                )
            else:
                self._release_slot_reservation(slot)
                if status_code in {400, 404, 422}:
                    logger.error(
                        "Doc %s processor %s permanently rejected (HTTP %s): %s. Dropping message.",
                        document_id,
                        processor_type,
                        status_code,
                        err_msg,
                    )
                    self.job_repo.update_job_status(
                        job_id=job_id,
                        status=JobState.FAILED,
                        error_message=err_msg,
                    )
                    self._channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=False)
                else:
                    if db_status in {JobState.QUEUED.value, "QUEUED"}:
                        self.job_repo.requeue_after_dispatch_failure(
                            document_id,
                            error_message=err_msg,
                        )
                    self._channel.basic_nack(delivery_tag=method_frame.delivery_tag, requeue=True)
                    logger.warning(
                        "Dispatch failed for doc %s processor %s. Requeued message.",
                        document_id,
                        processor_type,
                    )
                    break

    def republish_job_by_id(
        self,
        job_id: str,
        document_id: Optional[str] = None,
        processor_types: Optional[list[str]] = None,
    ) -> bool:
        """Publish recovered job payload back to document_processing_queue for retry."""
        try:
            if not self._connection or self._connection.is_closed or not self._channel or self._channel.is_closed:
                if not self._connect():
                    logger.error("Cannot republish job %s: RabbitMQ connection unavailable", job_id)
                    return False
            queue_name = self.rabbitmq_config.get("queue_name", "document_processing_queue")
            did = str(document_id or job_id).strip()
            self.job_repo.reset_failed_for_republish(did)

            types = processor_types
            if not types:
                ready = self.job_repo.list_ready_processor_types(did)
                types = ready[:1] if ready else ["chunking_vectorizing"]

            published = False
            for processor_type in types:
                partial = {
                    "requeued": True,
                    "security_prevalidated": True,
                    "processor_type": processor_type,
                }
                payload = self.job_repo.build_dispatch_payload(job_id, did, partial)
                body = json.dumps(payload).encode("utf-8")
                self._channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2),
                )
                self.job_repo.record_scheduler_republish(did)
                published = True
                logger.info(
                    "Republished doc %s processor %s to queue %s",
                    did,
                    processor_type,
                    queue_name,
                )
            return published
        except Exception as exc:
            logger.error("Failed to republish job %s to RabbitMQ: %s", job_id, exc)
            return False

    def stop(self) -> None:
        """Stop polling and close RabbitMQ connection."""
        self._running = False
        if self._connection and not self._connection.is_closed:
            try:
                self._connection.close()
            except Exception:
                pass
        logger.info("Stopped RabbitMQ consumer.")
