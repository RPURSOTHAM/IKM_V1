"""Unit and component integration tests for Scheduler Server Feature."""

import asyncio
import json
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from src.features.scheduler_server.api.protocol import decode_frame, encode_frame
from src.features.scheduler_server.api.tcp_server import TCPServerHandler
from src.features.scheduler_server.application.health_monitor import ContainerHealthMonitor
from src.features.scheduler_server.application.pool_manager import ProcessorPoolManager
from src.features.scheduler_server.application.queue_consumer import RabbitMQQueueConsumer
from src.features.scheduler_server.client.scheduler_client import SchedulerClient
from src.features.scheduler_server.domain.job_state import JobState, is_valid_state_transition
from src.features.scheduler_server.domain.slot_models import ProcessorSlot, SlotStatus
from src.features.scheduler_server.infrastructure.docker_manager import DockerServiceManager
from src.features.scheduler_server.infrastructure.job_repository import SchedulerJobRepository


# ============================================================================
# 1. DOMAIN LAYER TESTS
# ============================================================================

def test_job_state_properties():
    assert JobState.QUEUED.is_active is False
    assert JobState.DISPATCHING.is_active is True
    assert JobState.ASSIGNED.is_active is True
    assert JobState.IN_PROGRESS.is_active is True
    assert JobState.COMPLETED.is_terminal is True
    assert JobState.FAILED.is_terminal is True
    assert JobState.TIMED_OUT.is_terminal is True
    assert JobState.STOPPED.is_terminal is True


def test_job_state_transitions():
    # Valid transitions
    assert is_valid_state_transition(JobState.QUEUED, JobState.DISPATCHING) is True
    assert is_valid_state_transition(JobState.DISPATCHING, JobState.ASSIGNED) is True
    assert is_valid_state_transition(JobState.ASSIGNED, JobState.IN_PROGRESS) is True
    assert is_valid_state_transition(JobState.IN_PROGRESS, JobState.COMPLETED) is True
    assert is_valid_state_transition(JobState.IN_PROGRESS, JobState.TIMED_OUT) is True

    # Same state is valid (idempotent)
    assert is_valid_state_transition(JobState.IN_PROGRESS, JobState.IN_PROGRESS) is True

    # Terminal state cannot transition to anything new
    assert is_valid_state_transition(JobState.COMPLETED, JobState.IN_PROGRESS) is False
    assert is_valid_state_transition(JobState.FAILED, JobState.ASSIGNED) is False

    # Invalid jump
    assert is_valid_state_transition(JobState.QUEUED, JobState.COMPLETED) is False


def test_slot_model_lifecycle():
    slot = ProcessorSlot(
        slot_id="doc_processor_1",
        host_port=3100,
        container_name="doc_processor_1",
        status=SlotStatus.STARTING,
    )
    assert slot.is_available_for_job is False

    # Becomes healthy idle
    slot.health = "healthy"
    slot.state = "idle"
    slot.status = SlotStatus.HEALTHY_IDLE
    assert slot.is_available_for_job is True

    # Mark busy
    slot.mark_busy(job_id="job-123", document_id="doc-456")
    assert slot.is_available_for_job is False
    assert slot.status == SlotStatus.BUSY
    assert slot.current_job_id == "job-123"
    assert slot.current_document_id == "doc-456"
    assert slot.assigned_at is not None

    # Serialization
    data = slot.to_dict()
    assert data["slot_id"] == "doc_processor_1"
    assert data["status"] == "BUSY"
    assert data["current_job_id"] == "job-123"

    # Mark idle
    slot.mark_idle()
    assert slot.status == SlotStatus.HEALTHY_IDLE
    assert slot.current_job_id is None
    assert slot.is_available_for_job is True


# ============================================================================
# 2. PROTOCOL LAYER TESTS
# ============================================================================

def test_protocol_encode_decode():
    payload = {"signal_type": "query", "job_id": "test-123", "nested": {"key": "val"}}
    frame = encode_frame(payload)

    assert len(frame) > 4
    decoded = decode_frame(frame)
    assert decoded == payload


def test_protocol_invalid_frame():
    with pytest.raises(ValueError, match="Invalid frame"):
        decode_frame(b"12")


# ============================================================================
# 3. POOL MANAGER TESTS
# ============================================================================

def test_pool_manager_initialization():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_docker.find_slot_containers.return_value = []
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3110),
        min_parallel_jobs=3,
        max_parallel_jobs=5,
    )

    assert len(pool.slots) == 3
    assert "doc_processor_1" in pool.slots
    assert "doc_processor_3" in pool.slots
    assert pool.slots["doc_processor_1"].host_port == 3100
    assert pool.slots["doc_processor_3"].host_port == 3102


def test_pool_manager_get_available_slot_and_scale_up():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_docker.find_slot_containers.return_value = []
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3110),
        min_parallel_jobs=2,
        max_parallel_jobs=3,
    )

    # All slots busy
    for slot in pool.slots.values():
        slot.health = "healthy"
        slot.mark_busy("j1", "d1")

    # Mock docker starting container for dynamic scale up
    mock_container = MagicMock()
    mock_container.id = "mock-dyn-id-123"
    mock_docker.start_slot_container.return_value = mock_container
    mock_docker.check_processor_health.return_value = {"health": "healthy", "state": "idle"}

    # Available slot should attempt scale up to 3 slots
    new_slot = pool._maybe_scale_up()
    assert new_slot is not None
    assert new_slot.slot_id == "doc_processor_3"
    assert new_slot.is_dynamic is True
    assert len(pool.slots) == 3

    # Attempting scale up beyond max (3) returns None
    assert pool._maybe_scale_up() is None


def test_queue_consumer_acks_duplicate_for_live_slot():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)
    mock_channel = MagicMock()
    method_frame = MagicMock()
    method_frame.delivery_tag = 1
    payload = json.dumps({"job_id": "job-dup", "document_id": "doc-dup"}).encode("utf-8")
    mock_channel.basic_get.side_effect = [
        (method_frame, None, payload),
        (None, None, None),
    ] + [(None, None, None)] * 10

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3105),
        min_parallel_jobs=2,
    )
    busy_slot = pool.slots["doc_processor_1"]
    busy_slot.health = "healthy"
    busy_slot.mark_busy("job-dup", "doc-dup", processor_type="chunking_vectorizing")
    idle_slot = pool.slots["doc_processor_2"]
    idle_slot.health = "healthy"
    idle_slot.status = SlotStatus.HEALTHY_IDLE

    mock_job_repo.is_processor_terminal.return_value = False
    mock_job_repo.is_processor_ready.return_value = True
    mock_job_repo.is_processor_actively_dispatching.return_value = False
    mock_job_repo.get_job_record.return_value = {"status": "IN_PROGRESS"}
    mock_job_repo.normalized_status.return_value = "IN_PROGRESS"
    mock_job_repo.should_skip_queue_message.return_value = False
    mock_job_repo.count_queued_jobs.return_value = 0
    mock_job_repo.message_dispatch_priority.return_value = 0

    consumer = RabbitMQQueueConsumer(
        rabbitmq_config={"queue_name": "document_processing_queue"},
        pool_mgr=pool,
        job_repo=mock_job_repo,
    )
    consumer._channel = mock_channel
    consumer._running = True
    consumer.drain_available_jobs()

    mock_channel.basic_ack.assert_called_once_with(delivery_tag=1)
    mock_job_repo.claim_job_atomically.assert_not_called()
    mock_docker.dispatch_job_http.assert_not_called()


def test_build_dispatch_payload_enriches_from_intake_record():
    mock_store = MagicMock()
    intake = {
        "document_id": "doc-abc",
        "document_name": "invoice.pdf",
        "collection_name": "default",
        "tenant_id": "tenant-1",
        "processing": {"enabled_processor_types": ["chunking_vectorizing"]},
    }
    with patch(
        "src.features.scheduler_server.infrastructure.job_repository.get_document_job_store",
        return_value=mock_store,
    ), patch(
        "src.features.documents.application.job_republish.get_intake_record",
        return_value=intake,
    ):
        repo = SchedulerJobRepository()
        payload = repo.build_dispatch_payload("job-abc", "doc-abc", {"requeued": True})

    assert payload["job_id"] == "job-abc"
    assert payload["document_name"] == "invoice.pdf"
    assert payload["collection_name"] == "default"
    assert payload["requeued"] is True
    assert payload.get("security_prevalidated") is True


def test_queue_consumer_acks_premature_processor_message():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)
    mock_channel = MagicMock()
    method_frame = MagicMock()
    method_frame.delivery_tag = 9
    payload = json.dumps(
        {
            "job_id": "job-early",
            "document_id": "doc-early",
            "processor_type": "metadata_extraction",
        }
    ).encode("utf-8")
    mock_channel.basic_get.side_effect = [
        (method_frame, None, payload),
        (None, None, None),
    ] + [(None, None, None)] * 10

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3105),
        min_parallel_jobs=1,
    )
    pool.slots["doc_processor_1"].health = "healthy"
    pool.slots["doc_processor_1"].status = SlotStatus.HEALTHY_IDLE

    mock_job_repo.is_processor_terminal.return_value = False
    mock_job_repo.is_processor_ready.return_value = False
    mock_job_repo.count_queued_jobs.return_value = 0
    mock_job_repo.message_dispatch_priority.return_value = 0

    consumer = RabbitMQQueueConsumer(
        rabbitmq_config={"queue_name": "document_processing_queue"},
        pool_mgr=pool,
        job_repo=mock_job_repo,
    )
    consumer._channel = mock_channel
    consumer._running = True
    consumer.drain_available_jobs()

    mock_channel.basic_ack.assert_called_once_with(delivery_tag=9)
    mock_docker.dispatch_job_http.assert_not_called()


def test_republish_job_by_id_includes_document_name():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)
    mock_job_repo.build_dispatch_payload.return_value = {
        "job_id": "job-xyz",
        "document_id": "doc-xyz",
        "document_name": "receipt.pdf",
        "collection_name": "default",
        "processor_type": "chunking_vectorizing",
        "requeued": True,
        "security_prevalidated": True,
    }
    mock_job_repo.reset_failed_for_republish.return_value = False
    mock_job_repo.get_job_record.return_value = {"scheduling_metadata": {}}
    mock_job_repo._scheduling_meta.return_value = {}
    mock_job_repo.list_ready_processor_types.return_value = ["chunking_vectorizing"]
    mock_channel = MagicMock()
    mock_connection = MagicMock()
    mock_connection.is_closed = False
    mock_channel.is_closed = False

    consumer = RabbitMQQueueConsumer(
        rabbitmq_config={"queue_name": "document_processing_queue"},
        pool_mgr=MagicMock(),
        job_repo=mock_job_repo,
    )
    consumer._connection = mock_connection
    consumer._channel = mock_channel

    assert consumer.republish_job_by_id("job-xyz", "doc-xyz") is True
    mock_job_repo.build_dispatch_payload.assert_called_once()
    published = json.loads(mock_channel.basic_publish.call_args.kwargs["body"].decode("utf-8"))
    assert published["document_name"] == "receipt.pdf"


def test_queue_consumer_does_not_dispatch_when_claim_fails_for_non_active_job():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)
    mock_channel = MagicMock()
    method_frame = MagicMock()
    method_frame.delivery_tag = 2
    mock_channel.basic_get.return_value = (
        method_frame,
        None,
        json.dumps({"job_id": "job-x", "document_id": "doc-x"}).encode("utf-8"),
    )
    mock_channel.basic_get.side_effect = [
        (
            method_frame,
            None,
            json.dumps({"job_id": "job-x", "document_id": "doc-x"}).encode("utf-8"),
        ),
        (None, None, None),
    ] + [(None, None, None)] * 10

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3105),
        min_parallel_jobs=1,
    )
    pool.slots["doc_processor_1"].health = "healthy"
    pool.slots["doc_processor_1"].status = SlotStatus.HEALTHY_IDLE

    mock_job_repo.is_processor_terminal.return_value = False
    mock_job_repo.is_processor_ready.return_value = True
    mock_job_repo.is_processor_actively_dispatching.return_value = False
    mock_job_repo.is_processor_pending.side_effect = [True, False]
    mock_job_repo.should_skip_queue_message.return_value = False
    mock_job_repo.claim_job_atomically.return_value = False
    mock_job_repo.count_queued_jobs.return_value = 0
    mock_job_repo.message_dispatch_priority.return_value = 100
    mock_job_repo.get_job_record.side_effect = [
        {"status": "QUEUED"},
        {"status": "COMPLETED"},
    ]
    mock_job_repo.normalized_status.side_effect = ["QUEUED", "COMPLETED"]

    consumer = RabbitMQQueueConsumer(
        rabbitmq_config={"queue_name": "document_processing_queue"},
        pool_mgr=pool,
        job_repo=mock_job_repo,
    )
    consumer._channel = mock_channel
    consumer._running = True
    consumer.drain_available_jobs()

    mock_channel.basic_ack.assert_called_once_with(delivery_tag=2)
    mock_docker.dispatch_job_http.assert_not_called()


# ============================================================================
# 4. HEALTH MONITOR & TIMEOUT TESTS
# ============================================================================

def test_health_monitor_timeout():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3105),
        min_parallel_jobs=1,
        max_parallel_jobs=2,
    )

    slot = pool.slots["doc_processor_1"]
    slot.health = "healthy"
    slot.mark_busy(job_id="timed-out-job", document_id="doc-999")
    # Simulate job started 2000 seconds ago
    slot.assigned_at = datetime.utcnow() - timedelta(seconds=2000)

    monitor = ContainerHealthMonitor(
        pool_mgr=pool,
        job_repo=mock_job_repo,
        poll_frequency=10,
        job_timeout_seconds=1800,
    )

    monitor.check_job_timeouts()

    # Verify job marked TIMED_OUT in repository
    mock_job_repo.update_job_status.assert_called_once()
    args, kwargs = mock_job_repo.update_job_status.call_args
    assert kwargs.get("job_id") == "timed-out-job"
    assert kwargs.get("status") == JobState.TIMED_OUT

    # Verify stop request made to processor
    mock_docker.stop_processor_job_http.assert_called_once_with(
        slot_port=3100,
        container_name="doc_processor_1",
        document_id="doc-999",
    )

    # Slot reset to idle
    assert slot.current_job_id is None
    assert slot.status == SlotStatus.HEALTHY_IDLE


# ============================================================================
# 5. TCP CONTROL SERVER & CLIENT END-TO-END TESTS
# ============================================================================

def test_tcp_server_and_client_communication():
    async def _run_test():
        mock_docker = MagicMock(spec=DockerServiceManager)
        mock_job_repo = MagicMock(spec=SchedulerJobRepository)
        mock_job_repo.get_job_by_id.return_value = {
            "job_id": "job-abc",
            "document_id": "doc-abc",
            "status": "COMPLETED",
            "processor_id": "doc_processor_1",
            "progress_percent": 100,
            "error_details": None,
            "updated_at": "2026-09-10T12:00:00",
        }
        mock_job_repo.list_running_jobs.return_value = [
            {"job_id": "job-1", "document_id": "doc-1", "processor_id": "slot-1", "status": "IN_PROGRESS"}
        ]

        pool = ProcessorPoolManager(
            docker_mgr=mock_docker,
            job_repo=mock_job_repo,
            port_range=(3100, 3105),
            min_parallel_jobs=1,
        )

        shutdown_called = False

        def on_shutdown():
            nonlocal shutdown_called
            shutdown_called = True

        test_port = 3299
        server = TCPServerHandler(
            host="127.0.0.1",
            port=test_port,
            pool_mgr=pool,
            job_repo=mock_job_repo,
            shutdown_trigger=on_shutdown,
        )

        await server.start()

        try:
            client = SchedulerClient(host="127.0.0.1", port=test_port, timeout=5.0)

            # 1. Healthcheck
            res = await asyncio.to_thread(client.healthcheck)
            assert res.get("status") == "Healthy"
            assert res.get("service") == "SchedulerServer"

            # 2. Query job
            query_res = await asyncio.to_thread(client.query_job_status, document_id="doc-abc")
            assert query_res.get("document_processing_status") == "COMPLETED"
            assert query_res.get("job_id") == "job-abc"

            # 3. Running jobs
            running = await asyncio.to_thread(client.get_running_jobs)
            assert isinstance(running, list)
            assert len(running) == 1
            assert running[0]["job_id"] == "job-1"

            # 4. Shutdown trigger
            shut_res = await asyncio.to_thread(client.shutdown_scheduler)
            assert shut_res.get("status") == "shutdown_initiated"
            assert shutdown_called is True

        finally:
            await server.stop()

    asyncio.run(_run_test())


# ============================================================================
# 6. FASTAPI ROUTER TESTS
# ============================================================================

def test_fastapi_router_endpoints(monkeypatch):
    from fastapi import FastAPI
    from src.features.scheduler_server.api.fastapi_router import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    with patch("src.features.scheduler_server.api.fastapi_router.SchedulerClient") as MockClient:
        instance = MockClient.return_value
        instance.healthcheck.return_value = {"status": "Healthy", "service": "SchedulerServer"}
        instance.get_running_jobs.return_value = [{"job_id": "test-1"}]
        instance.query_job_status.return_value = {"document_processing_status": "COMPLETED"}
        instance.kill_job.return_value = {"status": "killed"}
        instance.shutdown_scheduler.return_value = {"status": "shutdown_initiated"}

        # Health
        resp = client.get("/scheduler/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "Healthy"

        # Running jobs
        resp = client.get("/scheduler/running-jobs")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # Query job
        resp = client.get("/scheduler/jobs/doc-123")
        assert resp.status_code == 200
        assert resp.json()["document_processing_status"] == "COMPLETED"

        # Kill job
        resp = client.post("/scheduler/jobs/doc-123/kill")
        assert resp.status_code == 200
        assert resp.json()["status"] == "killed"

        # Shutdown
        resp = client.post("/scheduler/shutdown")
        assert resp.status_code == 200
        assert resp.json()["status"] == "shutdown_initiated"


def test_message_dispatch_priority_prefers_fresh_queued_upload():
    mock_store = MagicMock()
    with patch(
        "src.features.scheduler_server.infrastructure.job_repository.get_document_job_store",
        return_value=mock_store,
    ):
        repo = SchedulerJobRepository()

    fresh = {
        "status": "QUEUED",
        "updated_at": datetime.utcnow() - timedelta(seconds=5),
    }
    follow_up = {
        "status": "IN_PROGRESS",
        "updated_at": datetime.utcnow() - timedelta(seconds=120),
    }
    fresh_score = repo.message_dispatch_priority(
        {"processor_type": "chunking_vectorizing"},
        fresh,
        sla_seconds=30,
    )
    follow_up_score = repo.message_dispatch_priority(
        {"processor_type": "template_extraction", "requeued": True},
        follow_up,
        sla_seconds=30,
    )
    assert fresh_score > follow_up_score


def test_pool_manager_releases_stale_busy_slot_without_db_assignment():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)
    mock_job_repo.is_processor_slot_active.return_value = False

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3105),
        min_parallel_jobs=1,
    )
    slot = pool.slots["doc_processor_1"]
    slot.health = "healthy"
    slot.mark_busy(job_id="job-stale", document_id="doc-stale", processor_type="chunking_vectorizing")

    pool._reconcile_stale_slot_assignment("doc_processor_1", slot)

    assert slot.current_job_id is None
    assert slot.status == SlotStatus.HEALTHY_IDLE


def test_queue_consumer_prioritizes_fresh_queued_message():
    mock_docker = MagicMock(spec=DockerServiceManager)
    mock_docker.dispatch_job_http.return_value = (True, 202, "")
    mock_job_repo = MagicMock(spec=SchedulerJobRepository)
    mock_channel = MagicMock()

    stale_frame = MagicMock()
    stale_frame.delivery_tag = 1
    fresh_frame = MagicMock()
    fresh_frame.delivery_tag = 2

    stale_payload = json.dumps(
        {
            "job_id": "job-stale",
            "document_id": "doc-stale",
            "processor_type": "template_extraction",
            "requeued": True,
        }
    ).encode("utf-8")
    fresh_payload = json.dumps(
        {
            "job_id": "job-fresh",
            "document_id": "doc-fresh",
            "processor_type": "chunking_vectorizing",
        }
    ).encode("utf-8")

    mock_channel.basic_get.side_effect = [
        (stale_frame, None, stale_payload),
        (fresh_frame, None, fresh_payload),
        (None, None, None),
    ]

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker,
        job_repo=mock_job_repo,
        port_range=(3100, 3105),
        min_parallel_jobs=1,
    )
    pool.slots["doc_processor_1"].health = "healthy"
    pool.slots["doc_processor_1"].status = SlotStatus.HEALTHY_IDLE

    mock_job_repo.is_processor_terminal.return_value = False
    mock_job_repo.is_processor_ready.return_value = True
    mock_job_repo.is_processor_actively_dispatching.return_value = False
    mock_job_repo.is_processor_pending.return_value = True
    mock_job_repo.should_skip_queue_message.return_value = False
    mock_job_repo.claim_job_atomically.return_value = True
    mock_job_repo.build_dispatch_payload.return_value = {"document_id": "doc-fresh"}
    mock_job_repo.count_queued_jobs.return_value = 0
    mock_job_repo.get_job_record.side_effect = [
        {"status": "IN_PROGRESS"},
        {"status": "QUEUED"},
    ]
    mock_job_repo.normalized_status.side_effect = ["IN_PROGRESS", "QUEUED"]
    mock_job_repo.message_dispatch_priority.side_effect = [100, 2000]

    consumer = RabbitMQQueueConsumer(
        rabbitmq_config={"queue_name": "document_processing_queue"},
        pool_mgr=pool,
        job_repo=mock_job_repo,
        queue_priority_batch=5,
    )
    consumer._channel = mock_channel
    consumer._running = True
    consumer.drain_available_jobs()

    mock_channel.basic_nack.assert_any_call(delivery_tag=1, requeue=True)
    mock_channel.basic_ack.assert_called_once_with(delivery_tag=2)
    mock_docker.dispatch_job_http.assert_called_once()
