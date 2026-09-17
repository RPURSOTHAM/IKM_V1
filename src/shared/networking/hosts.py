"""Resolve infrastructure hostnames for local vs container deployment."""

from __future__ import annotations

import os
from typing import Literal

DeploymentMode = Literal["local", "docker"]

_LOCAL_DEFAULTS: dict[str, str] = {
    "postgres": "localhost",
    "mysql": "localhost",  # legacy alias; app stores use postgres
    "rabbitmq": "localhost",
    "weaviate": "localhost",
    "weaviate_url": "http://localhost:8086",
    "weaviate_grpc_port": "50051",
    "neo4j": "localhost",
    "neo4j_uri": "bolt://localhost:7687",
    "redis": "localhost",
    "scheduler": "localhost",
    "rabbitmq_management_url": "http://localhost:15672",
}

_DOCKER_DEFAULTS: dict[str, str] = {
    "postgres": "postgres",
    "mysql": "postgres",  # legacy alias redirected to postgres service
    "rabbitmq": "rabbitmq",
    "weaviate": "weaviate",
    "weaviate_url": "http://weaviate:8080",
    "weaviate_grpc_port": "50051",
    "neo4j": "neo4j",
    "neo4j_uri": "bolt://neo4j:7687",
    "redis": "redis",
    "scheduler": "scheduler-service",
    "rabbitmq_management_url": "http://rabbitmq:15672",
}


def resolve_deployment_mode(raw: str | None = None) -> DeploymentMode:
    value = (raw or os.getenv("DEPLOYMENT_MODE") or "local").strip().lower()
    if value in {"local", "host", "native"}:
        return "local"
    if value in {"docker", "container", "compose"}:
        return "docker"
    raise ValueError(f"Unsupported DEPLOYMENT_MODE '{raw or value}'. Use 'local' or 'docker'.")


def detect_deployment_mode() -> DeploymentMode:
    """Best-effort mode detection when DEPLOYMENT_MODE is unset."""
    explicit = (os.getenv("DEPLOYMENT_MODE") or "").strip()
    if explicit:
        return resolve_deployment_mode(explicit)
    if os.path.exists("/.dockerenv") or os.getenv("RUNNING_IN_DOCKER", "").lower() in {"1", "true", "yes"}:
        return "docker"
    return "local"


def infra_hosts_for_mode(mode: DeploymentMode | None = None) -> dict[str, str]:
    return dict(_DOCKER_DEFAULTS if (mode or detect_deployment_mode()) == "docker" else _LOCAL_DEFAULTS)


def env_default(key: str, *, mode: DeploymentMode | None = None) -> str:
    """Return the default host/URL for a logical service when an env var is unset."""
    hosts = infra_hosts_for_mode(mode)
    return hosts[key]
