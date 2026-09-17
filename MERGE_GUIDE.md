# Scheduler Server Feature - Integration & Merge Guide

## 📦 What is inside this zip?

This package contains the complete **Scheduler Server Feature** and its accompanying **Chaos Monkey Resilience Test Harness**:

`	ext
.
├── MERGE_GUIDE.md                              # This setup and integration guide
├── src/
│   └── features/
│       └── scheduler_server/                   # Main Scheduler Server Feature
│           ├── domain/                         # JobState, ProcessorSlot, SlotStatus
│           ├── application/                    # PoolManager, QueueConsumer, HealthMonitor
│           ├── infrastructure/                 # DockerServiceManager, JobRepository
│           ├── api/                            # TCP Control Server (port 3200) & FastAPI router
│           ├── client/                         # SchedulerClient (IPC client)
│           ├── cli/                            # CLI commands: start, control, kill
│           ├── main.py                         # Daemon entrypoint
│           └── README.md                       # Architectural documentation
├── chaos/                                      # Chaos Monkey Test Framework
│   ├── config.json                             # Chaos test configuration
│   ├── chaos_controller.py                     # Main CLI test runner
│   ├── worker_failure.py                       # CH-01, CH-02, CH-06, CH-07 experiments
│   ├── rabbitmq_failure.py                     # CH-04 RabbitMQ disconnection test
│   ├── scheduler_failure.py                    # CH-03 Scheduler restart test
│   ├── db_failure.py                           # CH-05 DB failure & CH-08 Idempotency test
│   └── README.md                               # Chaos framework documentation
└── tests/
    └── unit/
        ├── test_scheduler_server_feature.py    # Unit tests for core scheduler
        └── test_chaos_resilience.py            # Unit tests for chaos & resilience
`

---

## 🔧 Prerequisites & Dependencies

### Python packages
Ensure the following packages are installed:
`ash
pip install docker pika requests fastapi uvicorn pytest python-dotenv
`

### Infrastructure Services
- **RabbitMQ**: AMQP message broker (port 5672)
- **Docker Engine**: For dynamically spinning up worker containers
- **PostgreSQL / MySQL**: document_job table for job state persistence

---

## 🚀 How to Merge into Your Repository

1. **Extract Files**:
   Unzip the archive into your repository root. The directory paths src/features/scheduler_server, chaos/, and 	ests/unit/ will map cleanly into standard Python project layouts.

2. **Environment Configuration**:
   Set or export the following environment variables (or add to your .env file):
   `ini
   SCHEDULER_TCP_PORT=3200
   SCHEDULER_MIN_PARALLEL_JOBS=5
   SCHEDULER_MAX_PARALLEL_JOBS=10
   SCHEDULER_PORT_RANGE_START=3100
   SCHEDULER_PORT_RANGE_END=3120
   SCHEDULER_JOB_TIMEOUT_SECONDS=1800
   RABBITMQ_HOST=localhost
   RABBITMQ_PORT=5672
   RABBITMQ_QUEUE_NAME=document_processing_queue
   PROCESSOR_IMAGE_NAME=rag-processor:latest
   `

3. **FastAPI Route Registration (Optional)**:
   If integrating with an existing parent FastAPI application:
   `python
   from src.features.scheduler_server.api.fastapi_router import router as scheduler_router
   app.include_router(scheduler_router)
   `

---

## 🛠️ CLI Operations

### Start Scheduler Server
`ash
python -m src.features.scheduler_server.cli.start
`

### Check Scheduler Status & Running Jobs
`ash
python -m src.features.scheduler_server.cli.control healthcheck
python -m src.features.scheduler_server.cli.control running-jobs
python -m src.features.scheduler_server.cli.control query-job <job_id>
`

### Stop / Kill Scheduler
`ash
python -m src.features.scheduler_server.cli.kill
`

---

## 🧪 Running Automated & Chaos Tests

### Run Unit Tests
`ash
pytest tests/unit/test_scheduler_server_feature.py tests/unit/test_chaos_resilience.py -v
`

### Run Chaos Monkey Resilience Verification
`ash
python -m chaos.chaos_controller --dry-run
python -m chaos.chaos_controller --test CH-01
`
