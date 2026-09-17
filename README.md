# DMS Platform Engine (RAG Builder)

Modular document-ingestion and retrieval platform. Clients talk to a FastAPI **Consumer API**;
heavy processing runs in **scheduler-managed processor containers** driven by **RabbitMQ**.

**Layout:** see **[docs/STRUCTURE.md](docs/STRUCTURE.md)** for the production folder map.  
**Runbook:** **[DEPLOYMENT.md](DEPLOYMENT.md)**

---

## Production layout (summary)

```text
rag-builder/
├── src/                  # application, features, infrastructure, workers, shared
├── deploy/
│   ├── infrastructure/   # Postgres, RabbitMQ, Weaviate, Neo4j, Redis
│   └── application/      # DMS + Scheduler Compose
├── data/documents/       # Uploaded originals
├── models/               # Offline embedding / reranker weights
├── configs/              # processors.yaml, security policies
├── docs/                 # Design & ops docs
├── tools/simulators/     # Local/dev helpers (not production runtime)
├── scripts/              # Maintenance utilities
└── tests/                # Automated tests
```

Python imports stay under `src.*` (e.g. `python -m src.application.consumer_api.main`).

---

## What is in `src/`

| Package | Role |
|---------|------|
| `application/` | FastAPI app factory + Consumer API |
| `features/` | Business features (auth, repos, documents, retrieval, PubMed, …) |
| `infrastructure/` | Postgres, Weaviate, Neo4j, Redis, support libs |
| `workers/` | Scheduler, document processor, reference processor |
| `shared/` | Cross-cutting errors, networking, semantic types |

Legacy `dms_service/` / `processor_service/` are **import shims only** — prefer `src.features.*`.

Deployable Compose lives under **`deploy/`**, not under `src/`.

---

## Quick start

```powershell
# Single env: deploy/application/.env

# Infrastructure
cd deploy/infrastructure
docker compose --env-file ../application/.env up -d

# Application
cd ../application
powershell -File build-application.ps1
docker compose -f docker-compose.application.yml --env-file .env up -d
```

Default API: `http://localhost:8088` — health at `/health`, OpenAPI when enabled for the environment.

---

## Feature areas

- Document types & key fields  
- Chunking strategies & embedding model catalogs  
- Content intelligence (`processor_service/extractors`)  
- Citations (anchors + retrieval resolver)  
- Metrics & audit trail (`shared/observability`)  
- Platform security audit (separate from observability audit)

All relational persistence uses **PostgreSQL** (see `DOCUMENT_JOBS_POSTGRES_*` / `POSTGRES_*`).

---

## Compatibility

Legacy folders `_documents`, `_templates`, `_reports` are junctions to `data/*` so older scripts keep working. Prefer `data/documents` in new configuration.
