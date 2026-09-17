"""Phase 3.8 — ACTIVE repository gate for retrieval (SettingsResolver via RepositoryService).

Retrieval must not call RepositoryStore or validators directly.
"""

from __future__ import annotations

from typing import Any

from src.shared.errors import COMPONENT_RETRIEVAL, raise_client_error, raise_service_error
from src.features.repositories.domain.repository_exceptions import NotFoundError, RepositoryInactiveError


def load_active_repository_context(repository_id: str) -> dict[str, Any]:
    """Validate repository is ACTIVE and return SettingsResolver wire context.

    Returns the same dict shape as ``RepositoryService.resolve_settings`` /
    ``validate_repository_active`` (settings + processing_hints + collection binding).
    Maps domain errors to retrieval client/service errors.
    """
    from src.features.repositories.application.repository_service import get_repository_service

    try:
        return get_repository_service().validate_repository_active(repository_id)
    except NotFoundError as exc:
        raise_client_error(
            COMPONENT_RETRIEVAL,
            code="repository_not_found",
            http_status=404,
            user_message=exc.message,
            reason=f"Repository not found during retrieval: {repository_id}",
            cause=exc,
        )
    except RepositoryInactiveError as exc:
        status = str((exc.details or {}).get("status") or "inactive")
        raise_client_error(
            COMPONENT_RETRIEVAL,
            code="repository_inactive",
            http_status=409,
            user_message=exc.message,
            reason=f"Repository status '{status}' cannot be used for retrieval: {repository_id}",
            cause=exc,
            details={"repository_id": repository_id, "status": status},
        )
    except Exception as exc:
        raise_service_error(
            COMPONENT_RETRIEVAL,
            code="repository_resolution_failed",
            http_status=422,
            user_message="The selected repository could not be resolved for search.",
            reason=f"Repository resolution failed for repository_id={repository_id}",
            cause=exc,
        )


def retrieval_settings_from_context(repo_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract effective settings dict used by search-mode / rerank / embedding helpers."""
    if not repo_context:
        return None
    settings = repo_context.get("settings")
    return dict(settings) if isinstance(settings, dict) else None
