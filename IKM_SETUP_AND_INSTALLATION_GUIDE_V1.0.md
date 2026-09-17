# IKM

## Information & Knowledge Management — Setup & Installation Guide

**Version:** 1.0  
**Audience:** Deployment and support teams

This guide contains the commands and configuration needed to install and run IKM on a new machine.

## 1. Before you start

- Use a computer with internet access, a modern multi-core CPU, at least **16 GB RAM**, and several GB of free disk space for Docker images and models.
- Obtain access to the IKM GitHub repository.
- Use an account with permission to install software and run Docker.
- Ensure the required ports are not in use: `8088`, `3200`, `5432`, `5672`, `15672`, `6379`, `7474`, `7687`, `8086`, `50051`, and `27017`.

## 2. Get IKM from GitHub

Install Git from [git-scm.com/downloads](https://git-scm.com/downloads), then verify it:

```powershell
git --version
```

Set the repository URL provided by your team and clone the release:

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd rag-builder
git checkout v1.0.0
```

Use the `main` branch only when your team instructs you to do so. For a versioned deployment, use the release tag `v1.0.0`.

## 3. Software prerequisites

| Software | Required version | Verification command |
| --- | --- | --- |
| Git | 2.30+ | `git --version` |
| Python | 3.10 or 3.11 | `python --version` |
| pip | 22.0+ | `pip --version` |
| Docker Desktop / Engine | 20.10+ | `docker --version` |
| Docker Compose | v2.20+ | `docker compose version` |
| PowerShell | 5.1+ or 7.x | `$PSVersionTable.PSVersion` |

## 4. Install IKM from scratch

### Step 1 — Clone the repository

```powershell
git clone <GITHUB_REPOSITORY_URL>
cd rag-builder
git checkout v1.0.0
```

### Step 2 — Create and activate a Python virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Create required data folders

```powershell
New-Item -ItemType Directory -Force data/documents
New-Item -ItemType Directory -Force models
```

### Step 5 — Download required models

Download the base embedding model:

```powershell
python scripts/download-embedding-model.py --model-id BAAI/bge-base-en --local-dir bge-base-en --models-root models
```

Download the reranker model:

```powershell
python scripts/download_reranker_model.py --model-id BAAI/bge-reranker-base --dest models/bge-reranker-base
```

Download Faster-Whisper Base:

```powershell
python scripts/_download_faster_whisper_base.py
```

Download the required Content Intelligence Hub models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download-cih-models.ps1 -Profile Required
```

To download the recommended MiniLM model as well, use:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download-cih-models.ps1 -Profile Recommended
```

Expected model folders include:

```text
models/
├── bge-base-en/
├── bge-reranker-base/
├── faster-whisper-base/
└── cih/
    └── whisper/
```

### Step 6 — Configure the environment file

Edit this file:

```text
deploy/application/.env
```

Set `REPO_ROOT` to the absolute path of the cloned repository. Use forward slashes on Windows:

```env
REPO_ROOT=C:/Users/<USERNAME>/Downloads/rag-builder
```

Set secure, deployment-specific values for at least the following variables:

```env
PLATFORM_BOOTSTRAP_ADMIN_USER=admin
PLATFORM_BOOTSTRAP_ADMIN_PASSWORD=<STRONG_ADMIN_PASSWORD>
JWT_SIGNING_KEY=<LONG_RANDOM_SECRET>

POSTGRES_PASSWORD=<STRONG_POSTGRES_PASSWORD>
DOCUMENT_JOBS_POSTGRES_PASSWORD=<STRONG_JOB_DATABASE_PASSWORD>

RABBITMQ_DEFAULT_USER=<RABBITMQ_USER>
RABBITMQ_DEFAULT_PASS=<STRONG_RABBITMQ_PASSWORD>
RABBITMQ_PASS=<STRONG_RABBITMQ_PASSWORD>

WEAVIATE_API_KEY=<STRONG_WEAVIATE_KEY>
NEO4J_PASSWORD=<STRONG_NEO4J_PASSWORD>
REDIS_PASSWORD=<STRONG_REDIS_PASSWORD>
```

Keep actual secrets only in the local `.env` file. Do not commit secrets to Git.

### Optional — temporary uploads without a selected repository

When files are uploaded without a selected repository, IKM creates one temporary repository for that upload request. A single-file request receives one temporary repository; all files in one multi-file request share one temporary repository; later upload requests receive different temporary repositories. Names use the `_temp_<batch-uuid>` pattern. Uploads with a selected repository continue through the existing repository flow unchanged. The standard document-processing and chunking pipeline is used. Once every document in a temporary batch completes successfully, IKM retains that batch repository for the configured period and then purges it.

Add or adjust these settings in `deploy/application/.env`:

```env
# Set false to disable automatic temporary-repository routing.
TEMP_REPOSITORY_ENABLED=true
TEMP_REPOSITORY_NAME=_temp
TEMP_REPOSITORY_RETENTION_HOURS=24
TEMP_REPOSITORY_CLEANUP_INTERVAL_SECONDS=300
```

The cleanup service evaluates every temporary batch separately. It waits for successful processing, starts the retention timer from the most recently completed document in that batch, then purges the documents and artifacts before deleting that batch repository and its Weaviate collection. Set `TEMP_REPOSITORY_ENABLED=false` to disable automatic routing and cleanup.

For a one-minute test, set the following values and restart `dms-service`:

```env
TEMP_REPOSITORY_ENABLED=true
TEMP_REPOSITORY_TEST_MODE=true
TEMP_REPOSITORY_TEST_RETENTION_SECONDS=60
TEMP_REPOSITORY_CLEANUP_INTERVAL_SECONDS=30
```

`TEMP_REPOSITORY_TEST_RETENTION_SECONDS` applies only when `TEMP_REPOSITORY_TEST_MODE=true`. Keep test mode `false` in production, where `TEMP_REPOSITORY_RETENTION_HOURS=24` retains each temporary batch for 24 hours. All temporary-repository settings are kept in the single `deploy/application/.env` file.

Swagger endpoints for temporary-repository operations:

```text
POST /api/v1/repositories/temporary
GET  /api/v1/repositories/temporary
GET  /api/v1/repositories/temporary/documents
POST /api/v1/repositories/temporary/cleanup
```

### Step 7 — Start infrastructure containers

The infrastructure Compose project automatically creates the external Docker network named `docnet`. Do **not** create a separate network manually.

```powershell
cd deploy/infrastructure
docker compose --env-file ../application/.env up -d
docker compose --env-file ../application/.env ps
```

Wait for PostgreSQL, RabbitMQ, Weaviate, Neo4j, and Redis to become healthy.

### Step 8 — Build the processor image

From the repository root:

```powershell
cd ../..
powershell -ExecutionPolicy Bypass -File deploy/application/build-processor-image.ps1
docker image ls rag-processor
```

### Step 9 — Build application images

```powershell
cd deploy/application
docker compose -f docker-compose.application.yml --env-file .env build
```

### Step 10 — Start application services

```powershell
docker compose -f docker-compose.application.yml --env-file .env up -d
```

### Step 11 — Verify services

```powershell
docker compose -f docker-compose.application.yml --env-file .env ps
docker ps
curl http://localhost:8088/health
curl http://localhost:8088/api/v1/health
curl http://localhost:8088/api/v1/scheduler/processors
```

When OpenAPI is enabled, open:

```text
http://localhost:8088/docs
```

## 5. Day-to-day startup

Start infrastructure:

```powershell
cd deploy/infrastructure
docker compose --env-file ../application/.env up -d
```

Start the application:

```powershell
cd ../application
docker compose -f docker-compose.application.yml --env-file .env up -d
```

Check status:

```powershell
docker ps
```

## 6. Logs and shutdown

View service logs:

```powershell
docker logs rag-dms-service --tail 100
docker logs rag-scheduler-service --tail 100
```

Stop application services without deleting containers:

```powershell
cd deploy/application
docker compose -f docker-compose.application.yml --env-file .env stop
```

Stop and remove application containers while retaining persistent data volumes:

```powershell
docker compose -f docker-compose.application.yml --env-file .env down
```

## 7. Quick command reference

```powershell
# Clone and enter the release
git clone <GITHUB_REPOSITORY_URL>
cd rag-builder
git checkout v1.0.0

# Python environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Models
python scripts/download-embedding-model.py --model-id BAAI/bge-base-en --local-dir bge-base-en --models-root models
python scripts/download_reranker_model.py --model-id BAAI/bge-reranker-base --dest models/bge-reranker-base
python scripts/_download_faster_whisper_base.py
powershell -ExecutionPolicy Bypass -File scripts/download-cih-models.ps1 -Profile Required

# Configure deploy/application/.env before continuing

# Infrastructure
cd deploy/infrastructure
docker compose --env-file ../application/.env up -d

# Build processor, build app, and start app
cd ../..
powershell -ExecutionPolicy Bypass -File deploy/application/build-processor-image.ps1
cd deploy/application
docker compose -f docker-compose.application.yml --env-file .env build
docker compose -f docker-compose.application.yml --env-file .env up -d

# Verify
curl http://localhost:8088/health
curl http://localhost:8088/api/v1/scheduler/processors
```
