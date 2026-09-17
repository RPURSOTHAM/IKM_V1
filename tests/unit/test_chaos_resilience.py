"""Unit tests for Chaos Monkey resilience experiments and scheduler fault-tolerance logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.features.scheduler_server.domain.job_state import JobState
from src.features.scheduler_server.domain.slot_models import ProcessorSlot, SlotStatus
from src.features.scheduler_server.application.pool_manager import ProcessorPoolManager
from src.features.scheduler_server.application.queue_consumer import RabbitMQQueueConsumer
from src.features.scheduler_server.infrastructure.job_repository import SchedulerJobRepository

from chaos.worker_failure import (
    run_ch01_worker_stopped,
    run_ch02_worker_crash_in_flight,
    run_ch06_slot_exhaustion,
    run_ch07_slow_worker_timeout,
)
from chaos.rabbitmq_failure import run_ch04_rabbitmq_failure
from chaos.scheduler_failure import run_ch03_scheduler_restart
from chaos.db_failure import run_ch05_database_failure, run_ch08_duplicate_message_idempotency


@pytest.fixture
def mock_job_repo():
    repo = MagicMock(spec=SchedulerJobRepository)
    repo.get_job_retry_count.return_value = 0
    repo.is_job_completed.return_value = False
    repo.requeue_job.return_value = True
    repo.mark_job_failed.return_value = True
    repo.recover_stale_dispatching_jobs.return_value = ["stale-job-1"]
    return repo


@pytest.fixture
def mock_docker_mgr():
    mgr = MagicMock()
    mgr.find_slot_containers.return_value = []
    mgr.start_slot_container.return_value = MagicMock(id="c-new-123")
    mgr.check_processor_health.return_value = {"health": "healthy", "state": "idle"}
    return mgr


# ============================================================================
# 1. SCHEDULER FAULT-TOLERANCE CORE TESTS
# ============================================================================

def test_worker_crash_in_flight_recovery(mock_docker_mgr, mock_job_repo):
    """Verify that when an in-flight worker slot crashes, the scheduler recovers the job."""
    pool = ProcessorPoolManager(
        docker_mgr=mock_docker_mgr,
        job_repo=mock_job_repo,
        min_parallel_jobs=2,
        max_parallel_jobs=4,
        max_retries=3,
    )
    requeue_cb = MagicMock()
    pool.set_requeue_callback(requeue_cb)

    slot = pool.slots["doc_processor_1"]
    slot.mark_busy(job_id="job-crash-1", document_id="doc-123")
    assert slot.current_job_id == "job-crash-1"

    # Trigger recovery
    pool.recover_failed_worker_job(slot, reason="Worker killed by Chaos Monkey")

    # Verify job requeued and retry count checked
    mock_job_repo.get_job_retry_count.assert_called_with("job-crash-1")
    mock_job_repo.requeue_job.assert_called_with("job-crash-1", error_message="Worker killed by Chaos Monkey")
    requeue_cb.assert_called_with("job-crash-1", "doc-123")

    # Slot should now be marked idle
    assert slot.current_job_id is None
    assert slot.status == SlotStatus.HEALTHY_IDLE


def test_worker_crash_exceeding_max_retries(mock_docker_mgr, mock_job_repo):
    """Verify that after 3 retries, the job transitions to FAILED instead of looping indefinitely."""
    mock_job_repo.get_job_retry_count.return_value = 3  # Already hit max_retries

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker_mgr,
        job_repo=mock_job_repo,
        min_parallel_jobs=2,
        max_retries=3,
    )
    requeue_cb = MagicMock()
    pool.set_requeue_callback(requeue_cb)

    slot = pool.slots["doc_processor_1"]
    slot.mark_busy(job_id="job-max-retry", document_id="doc-999")

    pool.recover_failed_worker_job(slot, reason="Continuous crash")

    # Should mark FAILED and NOT requeue
    mock_job_repo.mark_job_failed.assert_called_once()
    mock_job_repo.requeue_job.assert_not_called()
    requeue_cb.assert_not_called()
    assert slot.current_job_id is None


def test_idempotency_duplicate_message_prevention(mock_docker_mgr, mock_job_repo):
    """Verify that when a duplicate message arrives for an already COMPLETED job, it is skipped."""
    mock_job_repo.is_job_completed.return_value = True

    pool = ProcessorPoolManager(
        docker_mgr=mock_docker_mgr,
        job_repo=mock_job_repo,
        min_parallel_jobs=2,
    )
    consumer = RabbitMQQueueConsumer(
        rabbitmq_config={"queue_name": "test_queue"},
        pool_mgr=pool,
        job_repo=mock_job_repo,
    )

    mock_channel = MagicMock()
    mock_method = MagicMock(delivery_tag=42)
    payload_bytes = json.dumps({"job_id": "job-completed-dup", "document_id": "doc-dup"}).encode()
    # First call returns message, second call returns None to cleanly break the loop
    mock_channel.basic_get.side_effect = [(mock_method, None, payload_bytes), (None, None, None)]
    consumer._channel = mock_channel
    consumer._running = True

    # Make slot available
    slot = pool.slots["doc_processor_1"]
    slot.health = "healthy"
    slot.state = "idle"
    slot.status = SlotStatus.HEALTHY_IDLE

    consumer.drain_available_jobs()

    # The consumer should have inspected is_job_completed
    # and acknowledged the message without dispatching
    mock_channel.basic_ack.assert_called_with(delivery_tag=42)
    mock_docker_mgr.dispatch_job_http.assert_not_called()


def test_republish_job_by_id(mock_docker_mgr, mock_job_repo):
    """Verify republish_job_by_id publishes back to RabbitMQ queue with persistent delivery."""
    pool = ProcessorPoolManager(
        docker_mgr=mock_docker_mgr,
        job_repo=mock_job_repo,
        min_parallel_jobs=2,
    )
    consumer = RabbitMQQueueConsumer(
        rabbitmq_config={"queue_name": "document_processing_queue"},
        pool_mgr=pool,
        job_repo=mock_job_repo,
    )
    mock_channel = MagicMock()
    mock_channel.is_closed = False
    consumer._channel = mock_channel
    consumer._connection = MagicMock(is_closed=False)

    success = consumer.republish_job_by_id(job_id="job-recovered-123", document_id="doc-456")
    assert success is True
    mock_channel.basic_publish.assert_called_once()
    args, kwargs = mock_channel.basic_publish.call_args
    assert kwargs.get("routing_key") == "document_processing_queue"
    published_body = json.loads(kwargs.get("body").decode())
    assert published_body["job_id"] == "job-recovered-123"
    assert published_body["requeued"] is True


# ============================================================================
# 2. CHAOS EXPERIMENT DRY-RUN TESTS (CH-01 TO CH-08)
# ============================================================================

def test_chaos_ch01_worker_stopped_dry_run():
    res = run_ch01_worker_stopped(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-01"


def test_chaos_ch02_worker_crash_dry_run():
    res = run_ch02_worker_crash_in_flight(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-02"


def test_chaos_ch03_scheduler_restart_dry_run():
    res = run_ch03_scheduler_restart(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-03"


def test_chaos_ch04_rabbitmq_failure_dry_run():
    res = run_ch04_rabbitmq_failure(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-04"


def test_chaos_ch05_database_failure_dry_run():
    res = run_ch05_database_failure(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-05"


def test_chaos_ch06_slot_exhaustion_dry_run():
    res = run_ch06_slot_exhaustion(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-06"


def test_chaos_ch07_slow_worker_timeout_dry_run():
    res = run_ch07_slow_worker_timeout(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-07"


def test_chaos_ch08_idempotency_dry_run():
    res = run_ch08_duplicate_message_idempotency(docker_client=None, config={}, dry_run=True)
    assert res["status"] == "PASSED"
    assert res["test_id"] == "CH-08"
    assert res["metrics"]["duplicates_prevented"] == 1
