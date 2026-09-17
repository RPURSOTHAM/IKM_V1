from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.features.logical_folders.application.folder_service import get_logical_folder_service
from src.features.logical_folders.domain.exceptions import LogicalFolderError
from src.application.consumer_api.context import get_current_user_from_context
from src.features.authentication.domain.authentication_exceptions import PlatformSecurityError
from src.features.users.application.user_service import get_platform_security_service

# Mounted onto the Repositories router (prefix /repositories, tag Repositories).
router = APIRouter()


def _folder_error(exc: LogicalFolderError) -> HTTPException:
    return HTTPException(
        status_code=exc.http_status,
        detail={"code": exc.code, "message": exc.message, "details": exc.details},
    )


def _authorize_read(repository_id: str) -> None:
    actor = get_current_user_from_context()
    get_platform_security_service().check_retrieval_access(actor, repository_id)


@router.get(
    "/{repository_id}/folders",
    summary="List logical folders for a repository",
    description=(
        "Return repository-scoped logical folders created from extracted key-field values. "
        "Folders are not created by clients; they appear after document processing."
    ),
    operation_id="listRepositoryLogicalFolders",
)
def list_repository_folders(repository_id: str) -> dict[str, Any]:
    try:
        _authorize_read(repository_id)
        return get_logical_folder_service().list_folders(repository_id)
    except PlatformSecurityError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": exc.message, "details": exc.details},
        ) from exc
    except LogicalFolderError as exc:
        raise _folder_error(exc) from exc


@router.get(
    "/{repository_id}/folders/{folder_id}/documents",
    summary="List documents assigned to a logical folder",
    operation_id="listRepositoryLogicalFolderDocuments",
)
def list_repository_folder_documents(
    repository_id: str,
    folder_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    try:
        _authorize_read(repository_id)
        payload = get_logical_folder_service().list_folder_documents(repository_id, folder_id)
        payload["documents"] = payload.get("documents") or []
        if len(payload["documents"]) > limit:
            payload["documents"] = payload["documents"][:limit]
            payload["returned"] = limit
        else:
            payload["returned"] = len(payload["documents"])
        return payload
    except PlatformSecurityError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": exc.message, "details": exc.details},
        ) from exc
    except LogicalFolderError as exc:
        raise _folder_error(exc) from exc
