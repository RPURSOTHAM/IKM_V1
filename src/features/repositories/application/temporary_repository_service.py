"""Opt-in routing and retention cleanup for uploads without a repository."""

from __future__ import annotations

import os
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    return str(os.getenv(name, str(default))).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(str(os.getenv(name, default)).strip()))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TemporaryRepositoryConfig:
    enabled: bool
    test_mode: bool
    name: str
    retention_hours: int
    retention_seconds: int
    cleanup_interval_seconds: int


def _lifecycle_status_file() -> Path:
    """Return the persistent status file used by the temporary-repository API."""
    document_root = str(os.getenv("DOCUMENT_ROOT", "_documents")).strip() or "_documents"
    return Path(document_root) / ".temp_repository_lifecycle.json"


def _record_successful_cleanup(result: dict[str, Any]) -> None:
    """Persist a successful deletion so it remains visible after the next poll."""
    try:
        status_file = _lifecycle_status_file()
        status_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_successful_cleanup": {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "repository_name": result.get("repository_name"),
                "repository_id": result.get("repository_id"),
                "expires_at": result.get("expires_at"),
                "retention_seconds": result.get("retention_seconds"),
                "test_mode": result.get("test_mode"),
                "deleted": True,
            }
        }
        temporary_file = status_file.with_suffix(".tmp")
        temporary_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        temporary_file.replace(status_file)
    except Exception:
        # Cleanup must not fail merely because the diagnostic status cannot be saved.
        return


def get_temporary_repository_cleanup_status() -> dict[str, Any] | None:
    """Return the last successfully deleted `_temp` repository, if recorded."""
    try:
        payload = json.loads(_lifecycle_status_file().read_text(encoding="utf-8"))
        status = payload.get("last_successful_cleanup")
        return status if isinstance(status, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def get_temporary_repository_config() -> TemporaryRepositoryConfig:
    """Read temporary-repository settings from the deployment environment."""
    retention_hours = _env_int("TEMP_REPOSITORY_RETENTION_HOURS", 24, minimum=1)
    # Test retention is isolated behind an explicit switch. Production remains
    # hour-based even when test values are present in an environment file.
    test_mode = _env_bool("TEMP_REPOSITORY_TEST_MODE", False)
    retention_seconds = (
        _env_int("TEMP_REPOSITORY_TEST_RETENTION_SECONDS", 60, minimum=60)
        if test_mode
        else retention_hours * 3600
    )
    return TemporaryRepositoryConfig(
        enabled=_env_bool("TEMP_REPOSITORY_ENABLED", True),
        test_mode=test_mode,
        name=str(os.getenv("TEMP_REPOSITORY_NAME", "_temp")).strip() or "_temp",
        retention_hours=retention_hours,
        retention_seconds=retention_seconds,
        cleanup_interval_seconds=_env_int("TEMP_REPOSITORY_CLEANUP_INTERVAL_SECONDS", 300, minimum=30),
    )


def should_route_to_temporary_repository(repository_id: str | None) -> bool:
    """Use a temporary batch repository only when no repository was selected."""
    return not bool(str(repository_id or "").strip())


def temporary_repository_name(batch_id: str, config: TemporaryRepositoryConfig | None = None) -> str:
    """Return the deterministic, system-managed repository name for an upload batch."""
    try:
        normalized_batch_id = str(uuid.UUID(str(batch_id))).lower()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("batch_id must be a UUID.") from exc
    prefix = (config or get_temporary_repository_config()).name.rstrip("_") or "_temp"
    return f"{prefix}_{normalized_batch_id}"


def is_temporary_repository_name(name: str | None, config: TemporaryRepositoryConfig | None = None) -> bool:
    """Identify batch temporary repositories, including a legacy shared `_temp`."""
    configured = (config or get_temporary_repository_config()).name.rstrip("_") or "_temp"
    value = str(name or "").strip()
    return value == configured or value.startswith(f"{configured}_")


def list_temporary_repositories(*, service: Any | None = None) -> list[dict[str, Any]]:
    """List every active temporary upload-batch repository."""
    config = get_temporary_repository_config()
    repository_service = service
    if repository_service is None:
        from src.features.repositories.application.repository_service import get_repository_service

        repository_service = get_repository_service()
    payload = repository_service.list_repositories()
    return [
        repository
        for repository in list(payload.get("repositories") or [])
        if is_temporary_repository_name(repository.get("name"), config)
    ]


def ensure_temporary_repository(*, batch_id: str | None = None) -> dict[str, Any]:
    """Create or reuse the distinct temporary repository belonging to one upload batch."""
    from src.application.consumer_api.context import get_current_user_from_context
    from src.features.repositories.application.repository_service import get_repository_service
    from src.features.repositories.domain.repository_exceptions import RepositoryAlreadyExists

    config = get_temporary_repository_config()
    if not config.enabled:
        raise RuntimeError("Temporary repository routing is disabled.")

    service = get_repository_service()
    effective_batch_id = str(uuid.UUID(batch_id)) if batch_id else str(uuid.uuid4())
    repository_name = temporary_repository_name(effective_batch_id, config)
    repository_id = service.get_repository_id_by_name(repository_name)
    if repository_id is None:
        actor = get_current_user_from_context()
        owner_user_id = str(
            getattr(actor, "user_id", "") or os.getenv("TEMP_REPOSITORY_OWNER_USER", "temporary-system")
        ).strip()
        try:
            created = service.create_repository(
                {
                    "name": repository_name,
                    "owner_user_id": owner_user_id,
                    "owner_user_name": "Temporary Repository",
                    # The repository must be usable immediately by the existing
                    # upload and processor pipeline.
                    "status": "active",
                }
            )
            repository_id = str(created["repository_id"])
        except RepositoryAlreadyExists:
            # Another request may have created it between the lookup and insert.
            repository_id = service.get_repository_id_by_name(repository_name)
            if repository_id is None:
                raise

    repository = service.validate_repository_active(repository_id)
    _grant_current_uploader_access(repository_id)
    return {
        "repository_id": repository_id,
        "repository_name": repository_name,
        "batch_id": effective_batch_id,
        "weaviate_collection": repository.get("weaviate_collection"),
        "retention_hours": config.retention_hours,
        "retention_seconds": config.retention_seconds,
        "test_mode": config.test_mode,
    }


def _grant_current_uploader_access(repository_id: str) -> None:
    """Give the current authenticated uploader contributor access to `_temp`."""
    try:
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.users.infrastructure.user_repository import get_platform_security_store

        actor = get_current_user_from_context()
        user_id = str(getattr(actor, "user_id", "") or "").strip()
        if not user_id:
            return
        store = get_platform_security_store()
        if store is not None:
            store.grant_repository_role(
                repository_id=repository_id,
                user_id=user_id,
                role="contributor",
                granted_by=user_id,
            )
    except Exception:
        # The normal authorization check remains authoritative.  This best-effort
        # grant supports user-owned JWT deployments without blocking API-key flows.
        return


def _cleanup_repository(
    service: Any,
    repository: dict[str, Any],
    *,
    config: TemporaryRepositoryConfig,
    now: datetime | None,
) -> dict[str, Any]:
    """Clean one temporary upload-batch repository when its retention has expired."""
    repository_id = str(repository.get("repository_id") or "")
    repository_name = str(repository.get("name") or "")
    result: dict[str, Any] = {
        "repository_id": repository_id,
        "repository_name": repository_name,
        "deleted": False,
    }
    documents_payload = service.list_documents(repository_id, limit=5000)
    documents = list(documents_payload.get("documents") or [])
    if not documents:
        # A batch repository is created only by an upload request. If every file
        # in that request was rejected, it has no useful data to retain.
        result["repository"] = service.delete_repository(repository_id)
        result["deleted"] = True
        _record_successful_cleanup({**result, "retention_seconds": config.retention_seconds, "test_mode": config.test_mode})
        return result
    incomplete = [
        str(document.get("document_id") or "")
        for document in documents
        if str(document.get("status") or "").strip().lower() != "completed"
    ]
    if incomplete:
        result.update({"reason": "processing_not_completed", "incomplete_document_ids": incomplete[:20]})
        return result
    completion_times: list[float] = []
    for document in documents:
        try:
            completion_times.append(float(document.get("processing_completion_timestamp")))
        except (TypeError, ValueError):
            result["reason"] = "completion_timestamp_unavailable"
            return result
    current = now or datetime.now(timezone.utc)
    completed_at = datetime.fromtimestamp(max(completion_times), tz=timezone.utc)
    expires_at = completed_at + timedelta(seconds=config.retention_seconds)
    result["expires_at"] = expires_at.isoformat()
    if current < expires_at:
        result["reason"] = "retention_not_expired"
        return result
    purge = service.purge_all_documents(repository_id, delete_file=True)
    result["purge"] = purge
    if int(purge.get("remaining") or 0) or int(purge.get("failed") or 0):
        result["reason"] = "document_purge_incomplete"
        return result
    result["repository"] = service.delete_repository(repository_id)
    result["deleted"] = True
    _record_successful_cleanup({**result, "retention_seconds": config.retention_seconds, "test_mode": config.test_mode})
    return result


def cleanup_expired_temporary_repository(*, now: datetime | None = None) -> dict[str, Any]:
    """Clean every expired temporary upload-batch repository.

    Each upload receives its own repository. A batch is deleted only after all
    of its documents complete and the most recently completed one reaches expiry.
    """
    config = get_temporary_repository_config()
    result: dict[str, Any] = {
        "enabled": config.enabled,
        "repository_name_prefix": config.name,
        "retention_hours": config.retention_hours,
        "retention_seconds": config.retention_seconds,
        "test_mode": config.test_mode,
        "deleted": False,
    }
    if not config.enabled:
        return result

    from src.features.repositories.application.repository_service import get_repository_service

    service = get_repository_service()
    repositories = list_temporary_repositories(service=service)
    result["repositories_checked"] = len(repositories)
    result["results"] = [
        _cleanup_repository(service, repository, config=config, now=now)
        for repository in repositories
    ]
    result["deleted_count"] = sum(1 for item in result["results"] if item.get("deleted"))
    result["deleted"] = bool(result["deleted_count"])
    if not repositories:
        result["reason"] = "temporary_repositories_not_found"
    return result
