# RAG Builder Resilience Scheduler — End-to-End Specification & Operations Guide

The **Resilience Scheduler Server** (`src/features/scheduler_server`) is an asynchronous, fault-tolerant Python daemon and orchestration engine. It manages document processing jobs from RabbitMQ, schedules work across a dynamic pool of Docker processor containers, enforces a state machine with automated recovery, and integrates with the **Chaos Monkey** resilience validation framework.

---

## 📌 1. Architectural Overview

```text
                                  ┌────────────────────────┐
                                  │   FastAPI Parent API   │
                                  │   (DMS / CIH Service)  │
                                  └───────────┬────────────┘
                                              │
                                              │ 1. POST /upload (Job submitted)
                                              ▼
                                  ┌────────────────────────┐
                                  │       RabbitMQ         │
                                  │ document_processing_q  │
                                  └───────────┬────────────┘
                                              │
                                              │ 2. basic_get / consume
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SCHEDULER SERVER DAEMON                                  │
│                                                                                             │
│  ┌─────────────────────────┐   ┌─────────────────────────┐   ┌───────────────────────────┐  │
│  │   RabbitMQ Consumer     │   │   Pool & Slot Manager   │   │   Health & Healing Loop   │  │
│  │   - Polling / Event     │   │   - Min/Max Inventory   │   │   - 10s Reconcile Loop    │  │
│  │   - Idempotency Guard   │   │   - Port Range (3100+)  │   │   - Worker Recovery       │  │
│  │   - Requeue / Nack DLQ  │   │   - Auto Scale-Down     │   │   - Job Timeout Watchdog  │  │
│  └────────────┬────────────┘   └────────────┬────────────┘   └─────────────┬─────────────┘  │
│               │                             │                              │                │
│               └─────────────────────────────┼──────────────────────────────┘                │
│                                             │                                               │
│                                3. POST /process (HTTP Dispatch)                             │
│                                             ▼                                               │
│                                ┌─────────────────────────┐                                  │
│                                │   TCP Control Server    │ (Port 3200)                      │
│                                │   Status, Kill, Config  │                                  │
│                                └─────────────────────────┘                                  │
└─────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │  doc_processor_1 (3100) │                       │  doc_processor_n (310n) │
        │  - validation           │                       │  - validation           │
        │  - metadata_extraction  │                       │  - metadata_extraction  │
        │  - chunking_vectorizing │                       │  - chunking_vectorizing │
        │  - key_field_extraction │                       │  - key_field_extraction │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │ 4. Outcome & Status Persistence
                                              ▼
                                 ┌─────────────────────────┐
                                 │     PostgreSQL 15       │
                                 │     `document_job`      │
                                 │  `document_instance`    │
                                 └─────────────────────────┘
```

---

## 🔄 2. End-to-End Document Job Lifecycle

### A. The State Machine

```text
                    ┌─────────────┐
                    │   UPLOAD    │
                    └──────┬──────┘
                           │ Document registered & enqueued
                           ▼
                    ┌─────────────┐
                    │   QUEUED    │ ◄────────────────────────┐
                    └──────┬──────┘                          │
                           │ Slot available; message claimed │ Retry on in-flight crash
                           ▼                                 │ (retry_count < max_retries)
                    ┌─────────────┐                          │
                    │ DISPATCHING │                          │
                    └──────┬──────┘                          │
                           │ HTTP POST /process accepted     │
                           ▼                                 │
                    ┌─────────────┐                          │
                    │  ASSIGNED   │                          │
                    └──────┬──────┘                          │
                           │ Processor starts work           │
                           ▼                                 │
                    ┌─────────────┐                          │
                    │ IN_PROGRESS │ ─────────────────────────┘
                    └──────┬──────┘
                           │
            ┌──────────────┴──────────────┬─────────────────────────────┐
            ▼                             ▼                             ▼
     ┌─────────────┐               ┌─────────────┐               ┌─────────────┐
     │  COMPLETED  │               │   FAILED    │               │  TIMED_OUT  │
     └─────────────┘               └─────────────┘               └─────────────┘
```

### B. Lifecycle Stage Details

1. **Upload & Enqueue (`QUEUED`)**:
   - The user or client calls `POST /api/v1/documents/upload` or `POST /api/v1/cih/upload`.
   - The document is saved to `/app/documents/`, a row is created in `document_job` with status `QUEUED`, and a message containing `{document_id, job_id, repository_id}` is published to RabbitMQ.
   - **Immediate HTTP response is 200 OK with `status: "queued"`.** This is by design for non-blocking asynchronous processing.

2. **Scheduling & Pre-Assignment (`DISPATCHING`)**:
   - The `RabbitMQQueueConsumer` reads the message while a healthy, idle worker slot is available.
   - **Idempotency Guard**: Before dispatching, the scheduler checks if the job is already `COMPLETED`. If so, it acknowledges the message and drops it to avoid duplicate runs.
   - The database status is updated to `DISPATCHING` to track in-flight ownership.

3. **HTTP Dispatch (`ASSIGNED`)**:
   - The scheduler issues `POST /process` with the job payload to `http://doc_processor_n:3100/process`.
   - The slot is marked `BUSY`, and the database status updates to `ASSIGNED`.
   - If the processor returns an unrecoverable client error (HTTP 400, 404, 422 - e.g. file missing), the job is marked `FAILED` and permanently dropped (`basic_nack(requeue=False)`), preventing infinite retry loops.

4. **Execution (`IN_PROGRESS`)**:
   - The processor container transitions internal status:
     - `loading_document`
     - `extracting_metadata`
     - `chunking_vectorizing` (Weaviate)
     - `extracting_key_fields` (Neo4j)
     - `storing`
   - Worker writes heartbeat timestamps periodically to `document_job.heartbeat_at`.

5. **Terminal Resolution (`COMPLETED` / `FAILED`)**:
   - The processor records outcome in PostgreSQL `document_job` (`COMPLETED` or `FAILED`), updates `completed_at`, and frees its processing thread.
   - Its `/health` endpoint transitions `state` from `busy` back to `idle`.
   - The Scheduler's 10-second reconciliation loop detects `state == "idle"` on the worker, invokes `slot.mark_idle()`, and immediately releases the slot to pick up the next queued job.

---

## 🛡️ 3. Resilience & Self-Healing Mechanics

The Scheduler incorporates automated fault-tolerance mechanisms:

| Failure Scenario | Automatic Resilience Action | Verification Code / Matrix |
| :--- | :--- | :--- |
| **Worker Crash During Job** | In-flight worker crash is detected during pool reconciliation. The job is recovered from PostgreSQL, `retry_count` is incremented, the crashed container is replaced, and the job is republished to RabbitMQ for an alternate worker. | `CH-02`, `test_worker_crash_in_flight_recovery` |
| **Worker Offline / Stopped** | If a worker container disappears or stops, the scheduler marks the slot `UNHEALTHY` and automatically launches a replacement container pointing to the host document mount. | `CH-01`, `test_chaos_ch01_worker_stopped_dry_run` |
| **Scheduler Restart** | On daemon startup, `job_repo.recover_stale_dispatching_jobs()` queries PostgreSQL for any orphaned `DISPATCHING` jobs and republishes them to RabbitMQ, ensuring zero job loss. | `CH-03`, `test_chaos_ch03_scheduler_restart_dry_run` |
| **RabbitMQ Outage** | If the AMQP connection drops, the queue consumer initiates an exponential-backoff reconnection loop (every 5 seconds) until RabbitMQ is restored. | `CH-04`, `test_chaos_ch04_rabbitmq_failure_dry_run` |
| **PostgreSQL Downtime** | Database queries are wrapped in connection retry blocks. If PostgreSQL is temporarily unreachable, the scheduler logs the warning and reconnects once available. | `CH-05`, `test_chaos_ch05_database_failure_dry_run` |
| **Pool Saturation (High Load)** | If all slots are busy, incoming jobs remain buffered safely in RabbitMQ without dropping messages until worker slots are freed or auto-scaled up. | `CH-06`, `test_pool_manager_get_available_slot_and_scale_up` |
| **Slow Worker / Hung Job** | The `ContainerHealthMonitor` enforces `job_timeout_seconds`. If an active job exceeds the limit, an HTTP `POST /stop` is sent, the job is marked `TIMED_OUT`, and the slot is reset to idle. | `CH-07`, `test_health_monitor_timeout` |
| **Duplicate Delivery** | Idempotency guard checks `job_repo.is_job_completed(job_id)`. Duplicates are acknowledged and discarded without re-triggering embedding or extraction. | `CH-08`, `test_idempotency_duplicate_message_prevention` |

---

## 🐒 4. Chaos Monkey Resilience Framework (`chaos/`)

The repository includes a standalone automated test harness for validating scheduler resilience:

```text
chaos/
├── chaos_controller.py      # Main CLI test runner & tabular reporting
├── config.json              # Configurable thresholds & container targets
├── worker_failure.py        # CH-01 (Stop), CH-02 (Crash in-flight), CH-06 (Capacity), CH-07 (Timeout)
├── scheduler_failure.py     # CH-03 (Scheduler daemon restart)
├── rabbitmq_failure.py      # CH-04 (Broker outage & reconnection)
└── db_failure.py             # CH-05 (Postgres outage), CH-08 (Idempotency)
```

### Running Chaos Tests

```powershell
# 1. Run full suite in dry-run simulation mode (safe for production)
python -m chaos.chaos_controller --dry-run

# 2. Run a specific chaos experiment against running containers
python -m chaos.chaos_controller --test CH-01
python -m chaos.chaos_controller --test CH-02
python -m chaos.chaos_controller --test CH-08

# 3. Run all experiments live
python -m chaos.chaos_controller --all
```

---

## ⚙️ 5. Configuration Reference

Configuration is sourced from `configs/processors.yaml` and environment variables:

| Setting | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| **Min Parallel Jobs** | `SCHEDULER_MIN_PARALLEL_JOBS` | `5` | Minimum baseline processor containers maintained active. |
| **Max Parallel Jobs** | `SCHEDULER_MAX_PARALLEL_JOBS` | `5` | Maximum dynamic auto-scaled container slots. |
| **Port Range** | `SCHEDULER_PORT_RANGE_START` | `3100` | Base host port for processor slots (3100–3120). |
| **Queue Frequency** | `SCHEDULER_QUEUE_CHECK_FREQUENCY`| `5` | RabbitMQ poll interval in seconds. |
| **Health Frequency** | `SCHEDULER_CONTAINER_HEALTH_INTERVAL` | `10` | Container reconciliation loop interval in seconds. |
| **Job Timeout** | `SCHEDULER_JOB_TIMEOUT_SECONDS` | `1800` | Maximum job processing time before triggering `TIMED_OUT`. |
| **Scale-Down Cooldown**| `SCHEDULER_AUTO_SCALE_DOWN_COOLDOWN`| `300` | Seconds an idle dynamic container stays alive before removal. |
| **Max Retries** | `SCHEDULER_MAX_RETRIES` | `3` | Maximum automatic retries for an in-flight crashed job. |
| **Control Port** | `SCHEDULER_TCP_PORT` | `3200` | TCP port for IPC commands and health checks. |

---

## 📡 6. API & CLI Interface

### A. REST Endpoints (FastAPI DMS / Swagger UI)
* `GET /api/v1/scheduler/health`: Returns scheduler status, slot counts, and uptime.
* `GET /api/v1/scheduler/running-jobs`: Returns all active jobs currently assigned to processor slots.
* `GET /api/v1/scheduler/jobs/{id}`: Returns status, processor assignment, and error details of a specific job.
* `POST /api/v1/scheduler/jobs/{id}/kill`: Sends graceful stop signal to the worker and marks job `STOPPED`.
* `GET /api/v1/scheduler/config`: Returns current pool configuration.
* `PATCH /api/v1/scheduler/config`: Dynamically updates min/max parallel jobs without restarting the daemon.

### B. CLI Control Commands
```powershell
# Check scheduler health via TCP port 3200
python -m src.features.scheduler_server.cli.control healthcheck

# List currently running jobs
python -m src.features.scheduler_server.cli.control running-jobs

# Query details for a specific job
python -m src.features.scheduler_server.cli.control query-job <job_id>

# Terminate an active job
python -m src.features.scheduler_server.cli.control kill-job <job_id>

# Graceful daemon shutdown
python -m src.features.scheduler_server.cli.control shutdown
```

---

## 🧪 7. Verification & Automated Test Suites

All 30 unit and resilience tests pass cleanly:

```powershell
# Run the complete test suite
pytest tests/unit/test_scheduler_server_feature.py tests/unit/test_scheduler_probe_resilience.py tests/unit/test_chaos_resilience.py -v
```

```text
============================= test session starts =============================
tests/unit/test_scheduler_server_feature.py::test_job_state_properties PASSED
tests/unit/test_scheduler_server_feature.py::test_job_state_transitions PASSED
tests/unit/test_scheduler_server_feature.py::test_slot_model_lifecycle PASSED
tests/unit/test_scheduler_server_feature.py::test_protocol_encode_decode PASSED
tests/unit/test_scheduler_server_feature.py::test_protocol_invalid_frame PASSED
tests/unit/test_scheduler_server_feature.py::test_pool_manager_initialization PASSED
tests/unit/test_scheduler_server_feature.py::test_pool_manager_get_available_slot_and_scale_up PASSED
tests/unit/test_scheduler_server_feature.py::test_health_monitor_timeout PASSED
tests/unit/test_scheduler_server_feature.py::test_tcp_server_and_client_communication PASSED
tests/unit/test_scheduler_server_feature.py::test_fastapi_router_endpoints PASSED
tests/unit/test_scheduler_probe_resilience.py::test_single_transient_probe_failure_keeps_active_job PASSED
tests/unit/test_scheduler_probe_resilience.py::test_threshold_probe_failures_clear_ownership PASSED
tests/unit/test_scheduler_probe_resilience.py::test_healthy_sync_resets_probe_failure_counter PASSED
tests/unit/test_scheduler_probe_resilience.py::test_non_running_docker_clears_ownership_immediately PASSED
tests/unit/test_scheduler_probe_resilience.py::test_get_idle_processor_uses_cached_status_without_docker_probe PASSED
tests/unit/test_scheduler_probe_resilience.py::test_failed_assign_releases_idle_reservation PASSED
tests/unit/test_scheduler_probe_resilience.py::test_ensure_processor_pool_does_not_restart_running_slot PASSED
tests/unit/test_scheduler_probe_resilience.py::test_should_discard_stale_queue_deliveries PASSED
tests/unit/test_chaos_resilience.py::test_worker_crash_in_flight_recovery PASSED
tests/unit/test_chaos_resilience.py::test_worker_crash_exceeding_max_retries PASSED
tests/unit/test_chaos_resilience.py::test_idempotency_duplicate_message_prevention PASSED
tests/unit/test_chaos_resilience.py::test_republish_job_by_id PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch01_worker_stopped_dry_run PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch02_worker_crash_dry_run PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch03_scheduler_restart_dry_run PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch04_rabbitmq_failure_dry_run PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch05_database_failure_dry_run PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch06_slot_exhaustion_dry_run PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch07_slow_worker_timeout_dry_run PASSED
tests/unit/test_chaos_resilience.py::test_chaos_ch08_idempotency_dry_run PASSED

============================= 30 passed in 0.44s ==============================
```
