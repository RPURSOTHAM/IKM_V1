# Deployment Guide

This guide explains how to deploy the **DMS Platform Engine** locally or on a single host. The platform consists of:

1. **Infrastructure dependencies** (PostgreSQL, RabbitMQ, Weaviate, Neo4j, Redis)
2. **Tier 1 — DMS Service** (`src/dms_service`)
3. **Tier 2 — Scheduler daemon** (`src/scheduler_service`)
4. **Tier 3 — Processor image** (`src/processor_service`) — runtime managed by the Scheduler

For production topology, scaling, and security, see [`_documents/deployment_strategy.md`](_documents/deployment_strategy.md).

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Deployment modes (local vs Docker)](#deployment-modes-local-vs-docker)
- [1. Infrastructure dependencies](#1-infrastructure-dependencies)
- [1.5 Application tier — Docker Compose (DMS + Scheduler)](#15-application-tier--docker-compose-dms--scheduler)
- [1.6 Local development — DMS + Scheduler (without Docker)](#16-local-development--dms--scheduler-without-docker)
- [2. Tier 3 — Build the processor image](#2-tier-3--build-the-processor-image)
- [3. Tier 2 — Scheduler daemon](#3-tier-2--scheduler-daemon)
- [4. Tier 1 — DMS Service](#4-tier-1--dms-service)
- [5. Startup order](#5-startup-order)
- [6. Verify deployment](#6-verify-deployment)
- [7. Environment alignment checklist](#7-environment-alignment-checklist)
- [8. Optional processor types](#8-optional-processor-types)
- [9. Troubleshooting](#9-troubleshooting)

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11+** | All three application tiers |
| **Docker Engine** | Infrastructure Compose stack + Scheduler processor pool |
| **Docker Compose v2** | Run infrastructure and build processor image |
| **Git clone** | Repository root is the working directory for commands below |

Recommended host resources for a minimal dev stack:

- 8 GB RAM (processors load embedding models)
- 20 GB free disk (Weaviate volume + model weights)
- Ports available: 8088 (API), 3200 (scheduler TCP), 3100–3120 (processors), 5432, 5672, 6379, 7474, 7687, 8086, 50051

---

## Deployment modes (local vs Docker)

**DMS Service** and **Scheduler** can run in two ways:

| Mode | How | Infrastructure connectivity | Processor connectivity (Scheduler → pool) |
|------|-----|----------------------------|----------------------------------------|
| **Local (manual)** | `python -m …` on the host | `localhost` or any remote host/port in `.env` | `SCHEDULER_PROCESSOR_HTTP_MODE=localhost` (default) |
| **Docker Compose (application tier)** | [`deploy/application/docker-compose.application.yml`](deploy/application/docker-compose.application.yml) | Docker Compose **service names** on network `docnet` | `SCHEDULER_PROCESSOR_HTTP_MODE=docker_network` |

Infrastructure (PostgreSQL, RabbitMQ, Weaviate, Neo4j, Redis) always runs via [`deploy/infrastructure/docker-compose.yml`](deploy/infrastructure/docker-compose.yml) on the shared **`docnet`** network. Application containers join that network as an **external** network — they must **not** use `localhost` to reach infrastructure or each other.

### Hostname reference

| Logical service | Local / manual `.env` | Docker Compose (`DEPLOYMENT_MODE=docker`) |
|-----------------|----------------------|-------------------------------------------|
| PostgreSQL | `localhost:5432` | `postgres:5432` |
| RabbitMQ | `localhost:5672` | `rabbitmq:5672` |
| Weaviate HTTP | `http://localhost:8086` (host-mapped port) | `http://weaviate:8080` (container port) |
| Weaviate gRPC | `50051` | `50051` |
| Neo4j Bolt | `bolt://localhost:7687` | `bolt://neo4j:7687` |
| Redis | `localhost:6379` | `redis:6379` |
| Scheduler (DMS → TCP) | `localhost:3200` | `scheduler-service:3200` |
| Processor HTTP (scheduler) | `http://localhost:{port}` | `http://processor_{type}_{slot}:{port}` |

Set `DEPLOYMENT_MODE=local` or `DEPLOYMENT_MODE=docker` (or rely on auto-detection: `RUNNING_IN_DOCKER=1` / `/.dockerenv` → docker). Shared defaults live in [`src/shared/networking/hosts.py`](src/shared/networking/hosts.py).

### Local (manual) — quick path

See **[§1.6 Local development — DMS + Scheduler (without Docker)](#16-local-development--dms--scheduler-without-docker)** for the full guide (dependencies, configuration, and startup).

Summary:

1. Start infrastructure (§1).
2. Build processor image (§2).
3. Configure `src/scheduler_service/.env` and `src/dms_service/consumer_api_service/.env` with **`localhost`** hosts.
4. Run Scheduler and DMS Service as Python modules in separate terminals.

### Docker Compose — application tier

See **[§1.5 Application tier — Docker Compose](#15-application-tier--docker-compose-dms--scheduler)** for the full step-by-step guide to start **DMS Service** and **Scheduler** together in one Compose stack.

**Hybrid:** Scheduler in Docker with processors publishing ports on the host — set `SCHEDULER_PROCESSOR_HTTP_MODE=host_gateway` and `SCHEDULER_PROCESSOR_HTTP_HOST=host.docker.internal` (Windows/macOS) instead of `docker_network`.

---

## 1. Infrastructure dependencies

Infrastructure is defined in [`deploy/infrastructure/docker-compose.yml`](deploy/infrastructure/docker-compose.yml). Application tiers connect to these services over the network — they are **not** embedded in application images.

### 1.1 Configure environment

Create `deploy/infrastructure/.env` with at least:

```env
# Required secrets (change for non-dev)
POSTGRES_PASSWORD=rag_root_password
DOCUMENT_JOBS_POSTGRES_PASSWORD=rag_password
REDIS_PASSWORD=redis_dev_password
RABBITMQ_DEFAULT_USER=rabbitmq_user
RABBITMQ_DEFAULT_PASS=rabbitmq_password

# Weaviate (align with DMS + processor env)
WEAVIATE_PORT=8086
WEAVIATE_GRPC_PORT=50051
WEAVIATE_AUTH_APIKEY_ALLOWED_KEYS=weaviate_secret_key

# Neo4j (metadata / template / reference processors)
NEO4J_AUTH=neo4j/neo4j_password
```

### 1.2 Start the stack

From the repository root:

```powershell
cd deploy/infrastructure
docker compose up -d
```

Wait until health checks pass:

```powershell
docker compose ps
```

### 1.3 Services and ports

| Service | Host port (default) | Used by |
|---------|---------------------|---------|
| PostgreSQL | 5432 | DMS, Scheduler, Processors — `document_job` |
| RabbitMQ AMQP | 5672 | DMS (publish), Scheduler (consume) |
| RabbitMQ Management | 15672 | Ops / queue admin routes |
| Weaviate HTTP | 8086 | DMS retrieval, chunking processor |
| Weaviate gRPC | 50051 | DMS + processor client v4 |
| Neo4j HTTP / Bolt | 7474 / 7687 | Metadata, template, reference processors |
| Redis | 6379 | Conversion-for-rendering processor |
| PostgreSQL | 5432 | DMS, Scheduler, Processors, Observability |

### 1.4 Connection variables (all tiers)

**Local / manual** — use host-mapped ports (`localhost` when everything runs on one machine):

```env
DEPLOYMENT_MODE=local

DOCUMENT_JOBS_POSTGRES_HOST=localhost
DOCUMENT_JOBS_POSTGRES_PORT=5432
DOCUMENT_JOBS_POSTGRES_USER=rag
DOCUMENT_JOBS_POSTGRES_PASSWORD=rag_password
DOCUMENT_JOBS_POSTGRES_DATABASE=rag_builder

RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=rabbitmq_user
RABBITMQ_PASS=rabbitmq_password
RABBITMQ_QUEUE_NAME=document_processing_queue

WEAVIATE_URL=http://localhost:8086
WEAVIATE_GRPC_PORT=50051
WEAVIATE_API_KEY=weaviate_secret_key
WEAVIATE_COLLECTION=DocumentChunk

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password
```

**Docker Compose application tier** — use Compose **service names** (see [`deploy/application/.env.docker.sample`](deploy/application/.env.docker.sample); Weaviate uses internal port **8080**, not the host-mapped 8086):

```env
DEPLOYMENT_MODE=docker
RUNNING_IN_DOCKER=1

DOCUMENT_JOBS_POSTGRES_HOST=postgres
RABBITMQ_HOST=rabbitmq
WEAVIATE_URL=http://weaviate:8080
NEO4J_URI=bolt://neo4j:7687
REDIS_HOST=redis
SCHEDULER_HOST=scheduler-service
```

PostgreSQL schema for `document_job` is created automatically on first connection via `DocumentJobStore.ensure_schema()`.

---

## 1.5 Application tier — Docker Compose (DMS + Scheduler)

Use this path when **both** the DMS Service and Scheduler should run as containers on the shared infrastructure network **`docnet`**. They are defined together in [`deploy/application/docker-compose.application.yml`](deploy/application/docker-compose.application.yml).

| Compose service | Container name | Host port (default) | Role |
|-----------------|----------------|---------------------|------|
| `dms-service` | `rag-dms-service` | 8088 | Public HTTP API |
| `scheduler-service` | `rag-scheduler-service` | 3200 | RabbitMQ drain + processor pool (Docker socket) |

The Scheduler container spawns **processor** containers (`rag-processor:latest`) dynamically — they are not services in this Compose file.

### Prerequisites (complete before starting)

| # | Requirement | How to verify |
|---|-------------|---------------|
| 1 | Infrastructure running on `docnet` | §1.2 — `docker compose ps` in `deploy/infrastructure` |
| 2 | Processor image built | `docker image ls rag-processor` shows `latest` (§2) |
| 3 | Embedding models on host | `src/models/<model_dir>/` exists (e.g. `bge-base-en/`) |
| 4 | Document upload folder | `data/documents/` exists at repo root (created automatically if missing) |
| 5 | Application env file | `deploy/application/.env.docker` configured (step 1 below) |

### Step 1 — Configure application env

From the repository root:

```powershell
cd deploy/application
copy .env.docker.sample .env.docker
```

Edit `deploy/application/.env.docker`. **Required** settings:

```env
# Absolute host path to this repository (forward slashes work on Windows)
REPO_ROOT=C:/Users/you/path/to/rag-builder

# Must match deploy/infrastructure/.env secrets
DOCUMENT_JOBS_POSTGRES_PASSWORD=rag_password
RABBITMQ_PASS=rabbitmq_password
REDIS_PASSWORD=redis_password
WEAVIATE_API_KEY=weaviate_secret_key
NEO4J_PASSWORD=neo4j_password

# Bootstrap admin (first DMS startup only)
PLATFORM_BOOTSTRAP_ADMIN_USER=admin
PLATFORM_BOOTSTRAP_ADMIN_PASSWORD=YourSecureInitialPassword
JWT_SIGNING_KEY=change-me-to-a-long-random-secret
```

All infrastructure hostnames in `.env.docker.sample` already use Docker service names (`postgres`, `rabbitmq`, `weaviate:8080`, `scheduler-service`, etc.) — do **not** change them to `localhost`.

### Step 2 — Build the processor image

The Scheduler container creates processor workers from **`rag-processor:latest`**. Build it before starting the application stack (§2):

```powershell
cd ../processor_service
powershell -File build-image.ps1
cd ../deployment
```

### Step 3 — Start infrastructure (if not already running)

```powershell
cd ../dependencies
docker compose up -d
docker compose ps
cd ../deployment
```

This creates the external network **`docnet`** that the application Compose file joins.

### Step 4 — Build DMS + Scheduler images

On **Windows / OneDrive**, `docker compose ... --build` fails with `invalid file request` on cloud-only files. Build images with the staging script:

```powershell
powershell -File build-application.ps1
```

On Linux or when the repo is fully local (not OneDrive placeholders), you may skip the script and use `docker compose ... up -d --build` in step 5 instead.

The script produces **`rag-dms-service:latest`** and **`rag-scheduler-service:latest`**.

### Step 5 — Start DMS + Scheduler together

From `deploy/application`:

```powershell
docker compose -f docker-compose.application.yml --env-file .env.docker up -d
```

Add `--build` only if you did **not** run `build-application.ps1` and your host supports direct Compose builds.

This command:

1. Starts **`scheduler-service`**, then **`dms-service`** (`depends_on`)
2. Joins both containers to **`docnet`**
3. Bind-mounts `${REPO_ROOT}/_documents` and `${REPO_ROOT}/src/models` into DMS
4. Mounts **`/var/run/docker.sock`** into the Scheduler so it can spawn processor containers

### Step 6 — Verify containers

```powershell
docker compose -f docker-compose.application.yml --env-file .env.docker ps
docker logs rag-scheduler-service --tail 50
docker logs rag-dms-service --tail 50
```

| Check | Command |
|-------|---------|
| API health | `curl http://localhost:8088/health` |
| Full dependency snapshot | `curl http://localhost:8088/api/v1/health` |
| Processor pool | `curl http://localhost:8088/api/v1/scheduler/processors` |

Expect processor slots with `healthy: true`, `document_mount_ok: true`, and `models_mount_ok: true`.

### Stop, restart, and rebuild

```powershell
# Stop both services (keep containers)
docker compose -f docker-compose.application.yml --env-file .env.docker stop

# Start again
docker compose -f docker-compose.application.yml --env-file .env.docker start

# Rebuild images after code changes, then recreate containers
powershell -File build-application.ps1
docker compose -f docker-compose.application.yml --env-file .env.docker up -d

# Stop and remove containers (images kept)
docker compose -f docker-compose.application.yml --env-file .env.docker down
```

Infrastructure (`deploy/infrastructure`) is **separate** — stopping the application stack does not stop PostgreSQL, RabbitMQ, or Weaviate.

### What not to do in Docker mode

- Do **not** run `python -m src.workers.scheduler.service.main` or `python -m src.application.consumer_api.main` on the host at the same time — you would duplicate services and port bindings.
- Do **not** set `RAG_API_AUTO_START_SCHEDULER=true` in `.env.docker` — the Scheduler is already a Compose service.
- Do **not** use `localhost` for PostgreSQL, RabbitMQ, or Weaviate inside `.env.docker` — containers reach infrastructure via **`docnet`** DNS names.

---

## 1.6 Local development — DMS + Scheduler (without Docker)

Use this path when **DMS Service** and **Scheduler** run directly on your machine as Python processes (`python -m …`), while infrastructure (PostgreSQL, RabbitMQ, Weaviate, Neo4j, Redis) still runs in Docker via [`deploy/infrastructure/docker-compose.yml`](deploy/infrastructure/docker-compose.yml).

| Component | Local (this section) | Docker Compose (§1.5) |
|-----------|----------------------|------------------------|
| DMS Service | `python -m src.application.consumer_api.main` on host | `rag-dms-service` container |
| Scheduler | `python -m src.workers.scheduler.service.main` on host | `rag-scheduler-service` container |
| PostgreSQL, RabbitMQ, Weaviate, … | Docker Compose in `deploy/infrastructure` | Same |
| Processor workers | Docker containers spawned by Scheduler on host | Docker containers on `docnet` |

The Scheduler **still requires Docker Engine** — it creates and manages `rag-processor` containers. Only the DMS and Scheduler daemons themselves run outside Docker.

### Scope and ports

| Process | Default port | Purpose |
|---------|--------------|---------|
| DMS Service (HTTP) | **8088** | Public REST API |
| Scheduler (TCP) | **3200** | Control plane (health, pool snapshot) |
| Processor slots | **3100–3120** | One HTTP port per processor container |

Run all commands from the **repository root** unless noted otherwise.

### Prerequisites

Complete these before starting the application tiers.

| # | Requirement | Notes |
|---|-------------|-------|
| 1 | **Python 3.11+** | Same version as production images |
| 2 | **Docker Engine + Compose v2** | Infrastructure stack + processor pool |
| 3 | **Git clone / repo on disk** | Avoid OneDrive cloud-only placeholders when building processor images (§2, §9) |
| 4 | **Infrastructure running** | §1.2 — `docker compose up -d` in `deploy/infrastructure` |
| 5 | **Processor image** | `rag-processor:latest` built (§2) |
| 6 | **Embedding models** | At least one model under `src/models/<model_dir>/` (e.g. `bge-base-en/`) |
| 7 | **Document folder** | `data/documents/` at repo root (created automatically if missing) |
| 8 | **Free ports** | 8088, 3200, 3100–3120 (adjust range in scheduler `.env` if needed) |

Recommended host resources: **8 GB RAM**, **20 GB disk** (Weaviate volume + model weights).

### Step 1 — Python environment and dependencies

Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
```

Install both application tiers in editable mode from the repository root:

```powershell
pip install -e src/scheduler_service
pip install -e "src/dms_service[retrieval,preview]"
```

| Package | Install target | Why |
|---------|----------------|-----|
| `scheduler-service` | `src/scheduler_service` | RabbitMQ drain, Docker pool, TCP listener |
| `dms-service[retrieval]` | `src/dms_service` | Semantic/hybrid search (`RAG_API_ENABLE_RETRIEVAL=true`) |
| `dms-service[preview]` | optional extra | DOCX/PDF preview and Redis-backed render cache |

Core runtime dependencies (installed automatically):

- **DMS:** FastAPI, Uvicorn, Pydantic, psycopg2, Pika, Weaviate client, PyJWT, bcrypt; retrieval adds `sentence-transformers` + `torch`.
- **Scheduler:** Docker SDK, Pika, psycopg2, Requests, python-dotenv.

Shared code under `src/shared/` is included via the editable installs — no separate package step.

### Step 2 — Infrastructure and processor image

1. **Configure** `deploy/infrastructure/.env` (§1.1) and start the stack:

   ```powershell
   cd deploy/infrastructure
   docker compose up -d
   docker compose ps
   cd ../..
   ```

2. **Build** the processor image (Scheduler will not start healthy slots without it):

   ```powershell
   cd src/processor_service
   powershell -File build-image.ps1
   cd ../..
   ```

3. **Verify** Weaviate is ready:

   ```powershell
   curl http://localhost:8086/v1/.well-known/ready
   ```

### Step 3 — Configuration before first run

Configuration is the most common source of local failures. Set **`DEPLOYMENT_MODE=local`** and use **`localhost`** (or host-mapped ports from §1.3) for all infrastructure hostnames.

#### 3.1 Environment file layering (DMS only)

The DMS Service loads env files in this order (later overrides earlier):

1. Repository root `.env` (optional)
2. `deploy/infrastructure/.env`
3. `deploy/application/.env.docker` — **omit or rename when running locally**; if present, it overrides with Docker service names (`postgres`, `rabbitmq`, …) and breaks host-side connectivity
4. `src/processor_service/.env` (optional)
5. `src/dms_service/consumer_api_service/.env` — **primary local config** (highest priority)

The Scheduler reads **`src/scheduler_service/.env`** only (via `python-dotenv` in its entrypoint).

#### 3.2 Scheduler — `src/scheduler_service/.env`

Copy from [`.env_sample`](src/scheduler_service/.env_sample):

```powershell
copy src\scheduler_service\.env_sample src\scheduler_service\.env
```

Edit required values (use **absolute paths** on Windows):

```env
DEPLOYMENT_MODE=local
SCHEDULER_PROCESSOR_HTTP_MODE=localhost

SCHEDULER_TCP_PORT=3200
SCHEDULER_PORT_RANGE_START=3100
SCHEDULER_PORT_RANGE_END=3120
SCHEDULER_MAX_PARALLEL_JOBS=5
PROCESSOR_IMAGE_NAME=rag-processor:latest

# Must match DMS document upload folder (absolute path recommended)
SCHEDULER_DOCUMENT_HOST_PATH=C:/path/to/rag-builder/_documents
SCHEDULER_DOCUMENT_CONTAINER_PATH=/app/_documents

# Folder containing bge-base-en/, etc. (absolute path recommended)
SCHEDULER_MODELS_HOST_PATH=C:/path/to/rag-builder/src/models
SCHEDULER_MODELS_CONTAINER_PATH=/app/src/models

# Infrastructure — localhost / §1.4
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=rabbitmq_user
RABBITMQ_PASS=rabbitmq_password
RABBITMQ_QUEUE_NAME=document_processing_queue

DOCUMENT_JOBS_POSTGRES_HOST=localhost
DOCUMENT_JOBS_POSTGRES_PORT=5432
DOCUMENT_JOBS_POSTGRES_USER=rag
DOCUMENT_JOBS_POSTGRES_PASSWORD=rag_password
DOCUMENT_JOBS_POSTGRES_DATABASE=rag_builder

WEAVIATE_URL=http://localhost:8086
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password
```

| Setting | Must align with |
|---------|-----------------|
| `SCHEDULER_DOCUMENT_HOST_PATH` | DMS `DOCUMENT_ROOT` (same physical folder) |
| `SCHEDULER_MODELS_HOST_PATH` | DMS `MODELS_ROOT` / parent of `MODEL_DIR` |
| `DOCUMENT_JOBS_POSTGRES_*`, `RABBITMQ_*` | DMS `.env` and `deploy/infrastructure/.env` secrets |
| `WEAVIATE_*`, `NEO4J_*`, `REDIS_*` | Passed through to processor containers for optional processors |

Optional typed processor pools (§8):

```env
SCHEDULER_PROCESSOR_POOL=chunking_vectorizing:3,metadata_extraction:1,conversion_for_rendering:1
```

Ensure `SCHEDULER_PORT_RANGE_END - SCHEDULER_PORT_RANGE_START + 1` is **≥ total slot count**.

#### 3.3 DMS Service — `src/dms_service/consumer_api_service/.env`

Copy from [`.env_sample`](src/dms_service/consumer_api_service/.env_sample):

```powershell
copy src\dms_service\consumer_api_service\.env_sample src\dms_service\consumer_api_service\.env
```

Minimum local configuration:

```env
DEPLOYMENT_MODE=local

RAG_API_HOST=0.0.0.0
RAG_API_PORT=8088
RAG_API_ENABLE_RETRIEVAL=true
RAG_API_AUTO_START_SCHEDULER=false

DOCUMENT_ROOT=_documents

SCHEDULER_HOST=localhost
SCHEDULER_TCP_PORT=3200

RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=rabbitmq_user
RABBITMQ_PASS=rabbitmq_password
RABBITMQ_QUEUE_NAME=document_processing_queue

DOCUMENT_JOBS_POSTGRES_HOST=localhost
DOCUMENT_JOBS_POSTGRES_PORT=5432
DOCUMENT_JOBS_POSTGRES_USER=rag
DOCUMENT_JOBS_POSTGRES_PASSWORD=rag_password
DOCUMENT_JOBS_POSTGRES_DATABASE=rag_builder

WEAVIATE_URL=http://localhost:8086
WEAVIATE_GRPC_PORT=50051
WEAVIATE_API_KEY=weaviate_secret_key
WEAVIATE_COLLECTION=DocumentChunk

EMBEDDING_MODEL=BAAI/bge-base-en
MODEL_DIR=bge-base-en
MODELS_ROOT=src/models

PLATFORM_BOOTSTRAP_ADMIN_USER=admin
PLATFORM_BOOTSTRAP_ADMIN_PASSWORD=YourSecureInitialPassword
JWT_SIGNING_KEY=change-me-to-a-long-random-secret
```

| Setting | Purpose |
|---------|---------|
| `DOCUMENT_ROOT` | Relative to repo root unless absolute; must resolve to the same folder as `SCHEDULER_DOCUMENT_HOST_PATH` |
| `RAG_API_AUTO_START_SCHEDULER` | Keep **`false`** when running Scheduler manually in a second terminal |
| `PLATFORM_BOOTSTRAP_*` / `JWT_SIGNING_KEY` | First-start platform admin and API auth (see `_documents/platform-security-management.md`) |
| `WEAVIATE_API_KEY` | Must match `WEAVIATE_AUTH_APIKEY_ALLOWED_KEYS` in `deploy/infrastructure/.env` |

Create upload and model directories if missing:

```powershell
mkdir _documents
# Ensure src/models/bge-base-en/ (or your MODEL_DIR) exists
```

#### 3.4 Pre-flight configuration checklist

| Check | Action |
|-------|--------|
| PostgreSQL credentials | Match `deploy/infrastructure/.env` (`DOCUMENT_JOBS_POSTGRES_PASSWORD`, user `rag`) |
| No Docker hostname leakage | Remove or rename `deploy/application/.env.docker` for local runs |
| Document path parity | `DOCUMENT_ROOT` ↔ `SCHEDULER_DOCUMENT_HOST_PATH` |
| Model path parity | `MODELS_ROOT` + `MODEL_DIR` ↔ models visible under `SCHEDULER_MODELS_HOST_PATH` |
| Weaviate port | Host API uses **8086** (`WEAVIATE_PORT` in dependencies `.env`), not container port 8080 |
| Scheduler not duplicated | Do not run §1.5 Compose **and** local Scheduler on port 3200 simultaneously |

### Step 4 — Start services (two terminals)

**Terminal 1 — Scheduler (start first):**

```powershell
cd C:\path\to\rag-builder
.\.venv\Scripts\Activate.ps1
python -m src.workers.scheduler.service.main
```

Alternative helper:

```powershell
python src/scheduler_service/start_scheduler.py
```

On success the Scheduler connects to RabbitMQ, starts processor containers (`processor_*` in `docker ps`), and listens on TCP **3200**.

**Terminal 2 — DMS Service:**

```powershell
cd C:\path\to\rag-builder
.\.venv\Scripts\Activate.ps1
python -m src.application.consumer_api.main
```

The API binds to **http://localhost:8088** by default. Startup verifies PostgreSQL (`document_job` schema), repository/document-type schemas, and optional Weaviate connectivity.

> **Optional:** Set `RAG_API_AUTO_START_SCHEDULER=true` in DMS `.env` to spawn the Scheduler as a subprocess on API startup. Prefer two terminals during development so logs stay separated.

### Step 5 — Verify local deployment

| Check | Command |
|-------|---------|
| DMS liveness | `curl http://localhost:8088/health` |
| Full dependency snapshot | `curl http://localhost:8088/api/v1/health` |
| Processor pool | `curl http://localhost:8088/api/v1/scheduler/processors` |
| Processor containers | `docker ps --filter "name=processor"` |
| RabbitMQ UI | http://localhost:15672 (`rabbitmq_user` / password from dependencies `.env`) |

Expect processor slots with `healthy: true`, `document_mount_ok: true`, and `models_mount_ok: true`.

End-to-end upload test (§6):

```powershell
curl -X POST "http://localhost:8088/api/v1/documents/upload" -F "file=@sample.pdf" -F "submit_for_processing=true"
```

### Local vs Docker — common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `deploy/application/.env.docker` present while running on host | PostgreSQL/RabbitMQ connection refused to `postgres` / `rabbitmq` | Rename or remove `.env.docker` for local runs |
| Relative `SCHEDULER_DOCUMENT_HOST_PATH` on Windows | `document_mount_ok: false` | Use absolute path to `_documents` |
| Scheduler started after DMS only | Pool empty until Scheduler runs | Start Scheduler first (or enable auto-start) |
| Port 3200 or 8088 already in use | Bind error on startup | Stop Compose application tier (§1.5) or change ports |
| Missing `rag-processor:latest` | Scheduler logs container create failures | Run `build-image.ps1` (§2) |
| Embedding model folder missing | Chunking jobs fail at runtime | Add model under `src/models/<MODEL_DIR>/` |

### Stop local services

- **DMS / Scheduler:** `Ctrl+C` in each terminal.
- **Processor containers:** Stopped when Scheduler shuts down gracefully; or `docker stop` on `processor_*` containers.
- **Infrastructure:** `docker compose down` in `deploy/infrastructure` (only when you want to tear down PostgreSQL, RabbitMQ, etc.).

Further detail: §3 (Scheduler reference), §4 (DMS reference), §7 (alignment checklist), §9 (troubleshooting).

### What not to do in local mode

- Do **not** run the application Compose stack (§1.5) at the same time — it duplicates DMS and Scheduler on the same ports.
- Do **not** set infrastructure hostnames to Docker service names (`postgres`, `weaviate:8080`) in local `.env` files — use **`localhost`** and host-mapped ports.
- Do **not** set `SCHEDULER_PROCESSOR_HTTP_MODE=docker_network` when the Scheduler runs on the host — use **`localhost`** (default).

---

## 2. Tier 3 — Build the processor image

Processors run as Docker containers created by the Scheduler. Build the image **before** starting the Scheduler.

### 2.1 Install (optional, for local debugging only)

```powershell
pip install -e src/processor_service
```

### 2.2 Build Docker image

The image contains **Python dependencies and processor code only**. Embedding models are **not** copied into the image; they are loaded from a **shared host volume** mounted by the Scheduler (see §2.3 and §3.2).

From `src/processor_service`:

```powershell
powershell -File build-image.ps1
```

On Windows with the repo under **OneDrive**, use this script — `docker compose build` fails on cloud-only (reparse point) files. On other hosts you may use `docker compose -f docker-compose.processor-build.yml build` instead.

### 2.3 Shared models volume

Place all embedding model directories on the host (or NFS/EFS/Azure Files) under one folder, for example:

```text
src/models/
├── bge-base-en/
├── bge-large-en/
└── models--BAAI--bge-base-en/   # Hugging Face hub-style layout also supported
```

The Scheduler bind-mounts this folder **read-only** into every processor container at `/app/src/models`. Each repository/job selects a subfolder via `model_dir` in the queue payload (from repository `embedding_model.local_model_dir`).

| Variable | Default | Role |
|----------|---------|------|
| `SCHEDULER_MODELS_HOST_PATH` | `{repo}/src/models` | Host path containing all model directories |
| `SCHEDULER_MODELS_CONTAINER_PATH` | `/app/src/models` | In-container mount point (`MODELS_ROOT`) |

All processor slots share the **same disk**; each container loads the chosen model into its **own RAM** when a chunking job runs.

The DMS Service (retrieval) should use the **same** `src/models/` tree on the host when running locally (`MODEL_DIR` / `EMBEDDING_MODEL` in API env).

### 2.4 Processor runtime notes

- The Scheduler starts containers with `PROCESSOR_TYPE` set per slot (see [§8 Optional processor types](#8-optional-processor-types)).
- Each container exposes HTTP on a port in `SCHEDULER_PORT_RANGE_START` … `SCHEDULER_PORT_RANGE_END`.
- Documents are read from a bind mount: host `SCHEDULER_DOCUMENT_HOST_PATH` → container `/app/_documents` (default).
- Models are read from a bind mount: host `SCHEDULER_MODELS_HOST_PATH` → container `/app/src/models` (read-only).
- **Do not** run `uvicorn` manually for production flow unless debugging a single processor type.

---

## 3. Tier 2 — Scheduler daemon

The Scheduler consumes RabbitMQ, manages the processor Docker pool, and listens on TCP port 3200 for control signals from the DMS Service.

### 3.1 Install

From the repository root:

```powershell
pip install -e src/scheduler_service
```

### 3.2 Configure

**Local:** Create or edit `src/scheduler_service/.env` (copy from [`.env_sample`](src/scheduler_service/.env_sample)):

```env
DEPLOYMENT_MODE=local
SCHEDULER_PROCESSOR_HTTP_MODE=localhost

SCHEDULER_TCP_PORT=3200
SCHEDULER_PORT_RANGE_START=3100
SCHEDULER_PORT_RANGE_END=3120
SCHEDULER_MAX_PARALLEL_JOBS=5
PROCESSOR_IMAGE_NAME=rag-processor:latest

# Typed processor pools (optional; default = chunking only)
# SCHEDULER_PROCESSOR_POOL=chunking_vectorizing:3,metadata_extraction:1,conversion_for_rendering:1

QUEUE_CHECK_FREQUENCY=5
CONTAINER_POLLING_FREQUENCY=5

# Same document folder the DMS Service writes uploads to (absolute path)
SCHEDULER_DOCUMENT_HOST_PATH=C:/path/to/rag-builder/_documents
SCHEDULER_DOCUMENT_CONTAINER_PATH=/app/_documents

# Shared embedding models (absolute path to folder containing bge-base-en/, etc.)
SCHEDULER_MODELS_HOST_PATH=C:/path/to/rag-builder/src/models
SCHEDULER_MODELS_CONTAINER_PATH=/app/src/models

# RabbitMQ + PostgreSQL + optional processor backends — localhost (§1.4)
RABBITMQ_HOST=localhost
DOCUMENT_JOBS_POSTGRES_HOST=localhost
WEAVIATE_URL=http://localhost:8086
NEO4J_URI=bolt://localhost:7687
REDIS_HOST=localhost
```

**Docker Compose:** Use [`deploy/application/.env.docker`](deploy/application/.env.docker) (from [`.env.docker.sample`](deploy/application/.env.docker.sample)). The scheduler service sets `SCHEDULER_PROCESSOR_HTTP_MODE=docker_network`, `SCHEDULER_DOCKER_NETWORK=docnet`, and passes Docker service hostnames to spawned processor containers.

**Critical:** `SCHEDULER_DOCUMENT_HOST_PATH` must match DMS `DOCUMENT_ROOT`. `SCHEDULER_MODELS_HOST_PATH` must contain every embedding model referenced by repository settings.

### 3.3 Start

**Local:** See **[§1.6](#16-local-development--dms--scheduler-without-docker)** — preferred entry point for host-side deployment.

From the repository root:

```powershell
python -m src.workers.scheduler.service.main
```

Or use the helper script:

```powershell
python src/scheduler_service/start_scheduler.py
```

**Docker Compose:** Started together with DMS in **[§1.5](#15-application-tier--docker-compose-dms--scheduler)** — do not start the Scheduler manually when using the application Compose stack.

On startup the Scheduler:

1. Connects to RabbitMQ
2. Ensures processor containers exist for each entry in `SCHEDULER_PROCESSOR_POOL` (or chunking-only default)
3. Bind-mounts documents (`SCHEDULER_DOCUMENT_HOST_PATH`) and models (`SCHEDULER_MODELS_HOST_PATH`, read-only) into each processor
4. Listens on `SCHEDULER_TCP_PORT` (default 3200)

### 3.4 Scheduler requirements

| Requirement | Reason |
|-------------|--------|
| Docker socket access | Creates/manages processor containers |
| Port range free | One host port per processor slot |
| RabbitMQ reachable | Job consumption |
| Processor image present | `docker images rag-processor:latest` |

---

## 4. Tier 1 — DMS Service

The DMS Service is the only tier exposed to external HTTP clients.

### 4.1 Install

From the repository root:

```powershell
pip install -e "src/dms_service[retrieval]"
```

Use `[retrieval]` when semantic/hybrid search routes are required (`RAG_API_ENABLE_RETRIEVAL=true`).

### 4.2 Configure

Environment is layered (later files override earlier):

1. Repository root `.env` (optional)
2. `deploy/infrastructure/.env`
3. `deploy/application/.env.docker` (when present — Docker application tier)
4. `src/processor_service/.env`
5. `src/dms_service/consumer_api_service/.env`

**Local:** Copy [`src/dms_service/consumer_api_service/.env_sample`](src/dms_service/consumer_api_service/.env_sample) to `.env`:

```env
DEPLOYMENT_MODE=local

RAG_API_HOST=0.0.0.0
RAG_API_PORT=8088
RAG_API_ENABLE_RETRIEVAL=true

DOCUMENT_ROOT=_documents

# Scheduler control plane (not used for document status reads)
SCHEDULER_HOST=localhost
SCHEDULER_TCP_PORT=3200
RAG_API_AUTO_START_SCHEDULER=false

# RabbitMQ + PostgreSQL + Weaviate — localhost (§1.4)
RABBITMQ_HOST=localhost
DOCUMENT_JOBS_POSTGRES_HOST=localhost
WEAVIATE_URL=http://localhost:8086
WEAVIATE_GRPC_PORT=50051
WEAVIATE_API_KEY=weaviate_secret_key
WEAVIATE_COLLECTION=DocumentChunk

# Retrieval query embeddings (align with processor MODEL_DIR)
EMBEDDING_MODEL=BAAI/bge-base-en
MODEL_DIR=bge-base-en
```

**Docker Compose:** Configure via `deploy/application/.env.docker` (`SCHEDULER_HOST=scheduler-service`, `WEAVIATE_URL=http://weaviate:8080`, etc.). DMS loads that file automatically when present.

Create the document upload directory:

```powershell
mkdir _documents
```

If using platform security (JWT/API keys), configure variables documented in `_documents/platform-security-management.md`.

### 4.3 Start

**Local:** See **[§1.6](#16-local-development--dms--scheduler-without-docker)** — preferred entry point for host-side deployment.

From the repository root:

```powershell
python -m src.application.consumer_api.main
```

**Docker Compose:** Started together with the Scheduler in **[§1.5](#15-application-tier--docker-compose-dms--scheduler)**. The API is at **http://localhost:8088** (`RAG_API_PORT` in `.env.docker`).

- Liveness: `GET /health`
- Full dependency snapshot: `GET /api/v1/health`
- OpenAPI (when enabled): `GET /docs`

Set `RAG_API_AUTO_START_SCHEDULER=true` only if you want the API process to spawn the Scheduler subprocess on startup (otherwise start Tier 2 separately as in §3.3).

---

## 5. Startup order

### Local (manual)

Full guide: **[§1.6 Local development — DMS + Scheduler (without Docker)](#16-local-development--dms--scheduler-without-docker)**.

| Step | Component | Command / action |
|------|-----------|------------------|
| 1 | PostgreSQL, RabbitMQ, Weaviate | `docker compose up -d` in `deploy/infrastructure` |
| 2 | Neo4j, Redis | Started with the same Compose stack (required when optional processors are enabled) |
| 3 | Python venv + pip install | §1.6 Step 1 |
| 4 | Processor image | `powershell -File build-image.ps1` in `src/processor_service` |
| 5 | Embedding models on host | Ensure `src/models/<model_dir>/` exists for each repository embedding model |
| 6 | Configure `.env` files | §1.6 Step 3 |
| 7 | Scheduler (Tier 2) | `python -m src.workers.scheduler.service.main` |
| 8 | DMS Service (Tier 1) | `python -m src.application.consumer_api.main` |

### Docker Compose (application tier)

Full guide: **[§1.5 Application tier — Docker Compose](#15-application-tier--docker-compose-dms--scheduler)**.

| Step | Component | Command / action |
|------|-----------|------------------|
| 1 | Infrastructure | `docker compose up -d` in `deploy/infrastructure` |
| 2 | Application env | Copy `deploy/application/.env.docker.sample` → `.env.docker`; set `REPO_ROOT` and secrets |
| 3 | Processor image | `powershell -File build-image.ps1` in `src/processor_service` |
| 4 | DMS + Scheduler images | `powershell -File build-application.ps1` in `deploy/application` |
| 5 | DMS + Scheduler containers | `docker compose -f docker-compose.application.yml --env-file .env.docker up -d` in `deploy/application` |

Neo4j and Redis can remain idle until repository settings enable processors that write to them.

---

## 6. Verify deployment

### Infrastructure

```powershell
# PostgreSQL
docker exec rag_PostgreSQL PostgreSQLadmin ping -h 127.0.0.1 -uroot -p<root_password>

# Weaviate
curl http://localhost:8086/v1/.well-known/ready

# RabbitMQ management (browser)
# http://localhost:15672  (rabbitmq_user / rabbitmq_password)
```

### Scheduler processor pool

```powershell
curl http://localhost:8088/api/v1/scheduler/processors
```

Expect typed slots with `processor_type`, `healthy: true`, `document_mount_ok: true`, and `models_mount_ok: true`.

### End-to-end ingest

```powershell
curl -X POST "http://localhost:8088/api/v1/documents/upload" ^
  -F "file=@sample.pdf" ^
  -F "submit_for_processing=true"
```

Poll status (PostgreSQL-backed):

```powershell
curl http://localhost:8088/api/v1/documents/{document_id}/status
```

When `status` is `COMPLETED`, test retrieval (if enabled):

```powershell
curl -X POST "http://localhost:8088/api/v1/retrieve" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"your search text\", \"collection_name\": \"DocumentChunk\"}"
```

---

## 7. Environment alignment checklist

These values **must match** across tiers or jobs will fail silently (wrong paths, auth errors, embedding mismatch).

| Setting | Tiers |
|---------|-------|
| `DOCUMENT_JOBS_POSTGRES_*` | DMS, Scheduler, Processors |
| `RABBITMQ_*`, `RABBITMQ_QUEUE_NAME` | DMS, Scheduler |
| `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION` | DMS, chunking processors |
| `DOCUMENT_ROOT` (DMS) = `SCHEDULER_DOCUMENT_HOST_PATH` (Scheduler) | DMS, Scheduler, Processors |
| `SCHEDULER_MODELS_HOST_PATH` → same tree as DMS `MODEL_DIR` parent (`src/models`) | Scheduler, Processors, DMS retrieval |
| `EMBEDDING_MODEL` / `MODEL_NAME` + `MODEL_DIR` (per repository/job) | DMS retrieval, chunking processors |
| `SCHEDULER_HOST`, `SCHEDULER_TCP_PORT` | DMS only (use `scheduler-service` in Docker) |
| `DEPLOYMENT_MODE` / infra hostnames | All tiers — must match deployment mode (see hostname table) |
| `SCHEDULER_PROCESSOR_HTTP_MODE` | Scheduler — `localhost` (local), `docker_network` (Compose) |
| `NEO4J_*` | Processors when metadata/template/reference enabled |
| `REDIS_*` | Processors when conversion-for-rendering enabled |

---

## 8. Optional processor types

By default only **`chunking_vectorizing`** slots are created (`SCHEDULER_MAX_PARALLEL_JOBS`).

To enable additional processor containers, set on the **Scheduler**:

```env
SCHEDULER_PROCESSOR_POOL=chunking_vectorizing:3,metadata_extraction:1,template_extraction:1,reference_document_extraction:1,conversion_for_rendering:1
```

Ensure `SCHEDULER_PORT_RANGE_END - SCHEDULER_PORT_RANGE_START + 1` is **≥ total slot count**.

Enable matching processors on the **repository** before first document upload (settings lock after first upload):

| Repository setting | Processor type |
|--------------------|----------------|
| *(always)* | `chunking_vectorizing` |
| `metadata_extraction: true` | `metadata_extraction` |
| `template_extraction: true` | `template_extraction` |
| `reference_document_extraction: true` | `reference_document_extraction` |
| `conversion_for_rendering: true` | `conversion_for_rendering` |

The DMS Service publishes **one RabbitMQ message per enabled type** on upload.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Processor image build: `invalid file request …/.env` or `…/__init__.py` | OneDrive cloud-only (reparse point) files in build context | Run `powershell -File src/processor_service/build-image.ps1`, or set repo folder to **Always keep on this device** in OneDrive |
| DMS/Scheduler build: `invalid file request …/README.md` | Same OneDrive issue on application tier | Run `powershell -File deploy/application/build-application.ps1`, then `docker compose … up -d` **without** `--build` |
| RabbitMQ / Weaviate warnings at DMS startup but services are on `docnet` | PostgreSQL platform config had quoted strings (e.g. `"rabbitmq"`, `"http://weaviate:8080"`) overriding `.env.docker` | Rebuild DMS after config fix; restart container. Ensure `WEAVIATE_API_KEY` matches `deploy/infrastructure/.env` |
| Connection refused from DMS/Scheduler container to PostgreSQL/RabbitMQ/Weaviate | Using `localhost` inside a container | Set Docker service hostnames (`postgres`, `rabbitmq`, `weaviate:8080`) and `DEPLOYMENT_MODE=docker` |
| Scheduler cannot reach processors | Wrong `SCHEDULER_PROCESSOR_HTTP_MODE` | Use `docker_network` when scheduler and processors are on `docnet`; `localhost` when scheduler runs on host |
| API fails at startup — PostgreSQL | Infrastructure not up | Start Compose stack; verify `DOCUMENT_JOBS_POSTGRES_*` |
| Queue depth grows, no processing | Scheduler not running or no healthy processors | Start Scheduler; check `docker ps` for `processor_*` containers |
| Processor `document_mount_ok: false` | Path mismatch | Align `DOCUMENT_ROOT` and `SCHEDULER_DOCUMENT_HOST_PATH` |
| Jobs stuck in `DISPATCHING` | No idle slot for `processor_type` | Add slot to `SCHEDULER_PROCESSOR_POOL` or enable fewer repository processors |
| Processor `models_mount_ok: false` | Models path mismatch or missing host folder | Set `SCHEDULER_MODELS_HOST_PATH`; verify `src/models` exists |
| Embedding / chunking fails at runtime | Model subfolder missing on shared volume | Add model under `SCHEDULER_MODELS_HOST_PATH/<model_dir>` |
| Retrieval returns poor/no results | Embedding mismatch | Use same `MODEL_DIR` on DMS and in repository settings; model must exist on shared volume |
| Neo4j/Redis processor failures | Service down or wrong password | Verify Compose health; pass env into processor containers via Scheduler host env |
| `409` on `/process` | Normal — slot busy | Scheduler will retry when slot returns idle |

Detailed runbooks:

- `src/dms_service/consumer_api_service/README.md`
- `src/scheduler_service/README.md`
- `src/processor_service/README.md`
- `_documents/document-processor-service.md`

---

*Last updated: June 6, 2026*
