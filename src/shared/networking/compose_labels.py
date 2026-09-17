"""Docker Compose project labels for scheduler-managed containers."""

from __future__ import annotations

import os


def compose_stack_settings() -> dict[str, str | None]:
    return {
        "project": (os.getenv("SCHEDULER_COMPOSE_PROJECT") or "rag-builder-application").strip(),
        "service": (os.getenv("SCHEDULER_COMPOSE_SERVICE") or "doc-processor").strip(),
        "config_files": (os.getenv("SCHEDULER_COMPOSE_CONFIG_FILES") or "").strip() or None,
        "working_dir": (os.getenv("SCHEDULER_COMPOSE_WORKING_DIR") or "").strip() or None,
    }


def compose_stack_labels(
    *,
    project: str,
    service: str,
    container_number: int,
    config_files: str | None = None,
    working_dir: str | None = None,
) -> dict[str, str]:
    """Labels that attach dynamically created containers to a Compose project stack."""
    labels = {
        "com.docker.compose.project": project,
        "com.docker.compose.service": service,
        "com.docker.compose.container-number": str(container_number),
        "com.docker.compose.version": "2",
        "com.docker.compose.oneoff": "False",
        "com.docker.compose.depends_on": "",
    }
    if config_files:
        labels["com.docker.compose.project.config_files"] = config_files.replace("\\", "/")
    if working_dir:
        labels["com.docker.compose.project.working_dir"] = working_dir.replace("\\", "/")
    return labels
