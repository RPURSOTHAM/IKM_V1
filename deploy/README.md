# Deploy

Production deployment assets live here (not under `src/`).

| Path | Purpose |
|------|---------|
| `infrastructure/` | Shared dependencies Compose (Postgres, RabbitMQ, Weaviate, Neo4j, Redis) |
| `application/` | DMS + Scheduler Compose, build & recovery scripts |

Quick start: see [../docs/STRUCTURE.md](../docs/STRUCTURE.md) and [../DEPLOYMENT.md](../DEPLOYMENT.md).
