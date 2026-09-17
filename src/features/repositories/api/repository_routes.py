"""Tracked FastAPI routes for repository domain (mounted before parameterized paths).

Consumer API mounts this router under ``settings.api_prefix`` (default ``/api/v1``).
HTTP layer only: parse → call RepositoryService → map RepositoryError to HTTP.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status

from src.application.consumer_api.security import security_scheme
from src.features.repositories.domain.repository_exceptions import RepositoryError
from src.features.repositories.schemas.repository_schemas import (
    CreateRepositoryKeyFieldRequest,
    CreateRepositoryRequest,
    PatchRepositoryKeyFieldsRequest,
    UpdateRepositoryRequest,
    UpdateRepositorySettingsRequest,
    UpsertRepositorySettingsRequest,
)
from src.features.users.schemas.user_schemas import GrantRepositoryRoleRequest
from src.features.repositories.application.repository_service import RepositoryService, get_repository_service

# Mounted on the authenticated /api/v1 router. Register BEFORE /repositories/{repository_id}.
# security_scheme documents Bearer JWT for Swagger; JwtAuthMiddleware enforces auth.
router = APIRouter(
    prefix="/repositories",
    tags=["Repositories"],
    dependencies=[Depends(security_scheme)],
)

# Read-only logical-folder endpoints are registered before parameterized
# repository routes so their paths are never interpreted as repository IDs.
from src.features.logical_folders.api.folder_routes import router as logical_folder_router

router.include_router(logical_folder_router)


def _svc() -> RepositoryService:
    return get_repository_service()


def _repo_error(exc: RepositoryError) -> HTTPException:
    return HTTPException(
        status_code=exc.http_status,
        detail={
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@router.get(
    "/embedding-models",
    summary="List selectable embedding models for repository settings",
    description="Catalog of allowed embedding_model values for create/settings flows.",
    operation_id="listEmbeddingModels",
    responses={200: {"description": "Embedding model catalog"}},
)
def list_embedding_models(
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    """Return all allowed embedding_model values for repository create/settings flows."""
    return service.get_embedding_model_catalog()


@router.get(
    "/chunking-strategies",
    summary="List supported repository chunking strategies",
    description="Catalog of chunking strategies and per-strategy configuration fields.",
    operation_id="listChunkingStrategies",
    responses={200: {"description": "Chunking strategy catalog"}},
)
def list_chunking_strategies(
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    """Return chunking strategies and the configuration fields required for each."""
    return service.get_chunking_strategy_catalog()


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create repository",
    description="Create a repository in configuring status with optional initial settings.",
    operation_id="createRepository",
    responses={
        201: {"description": "Repository created"},
        400: {"description": "Invalid request / repository name"},
        409: {"description": "Duplicate repository name"},
        503: {"description": "Persistence unavailable"},
    },
)
def create_repository(
    body: CreateRepositoryRequest,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.create_repository(body.model_dump(exclude_none=True))
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.get(
    "",
    summary="List repositories",
    description="List repositories with optional owner and status filters.",
    operation_id="listRepositories",
    responses={200: {"description": "Repository list"}, 503: {"description": "Persistence unavailable"}},
)
def list_repositories(
    owner_user_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status", description="Lifecycle status filter"),
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.list_repositories(owner_user_id=owner_user_id, status=status_filter)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post(
    "/temporary",
    status_code=status.HTTP_201_CREATED,
    summary="Create a temporary upload-batch repository",
    description=(
        "Creates one uniquely named temporary repository for a batch. Documents uploaded without repository_id "
        "create one automatically for that upload request."
    ),
    operation_id="ensureTemporaryRepository",
)

def ensure_temporary_repository_endpoint() -> dict[str, Any]:
    from src.features.repositories.application.temporary_repository_service import ensure_temporary_repository

    try:
        temporary = ensure_temporary_repository()
        return {"created": True, **temporary}
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "temporary_repository_unavailable", "message": str(exc)}) from exc


@router.get(
    "/temporary",
    summary="List temporary upload-batch repositories",
    description="Returns all current temporary batch repositories and their retention configuration.",
    operation_id="getTemporaryRepository",
)
def get_temporary_repository_endpoint(
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    from src.features.repositories.application.temporary_repository_service import (
        get_temporary_repository_config,
        list_temporary_repositories,
    )

    config = get_temporary_repository_config()
    repositories = list_temporary_repositories(service=service)
    return {
        "repository_name_prefix": config.name,
        "count": len(repositories),
        "repositories": repositories,
        "retention_hours": config.retention_hours,
        "retention_seconds": config.retention_seconds,
        "test_mode": config.test_mode,
        "cleanup_interval_seconds": config.cleanup_interval_seconds,
    }


@router.get(
    "/temporary/cleanup-status",
    summary="Check temporary repository deletion status",
    description=(
        "Reports active temporary upload batches and the last successfully completed automatic or manual cleanup. "
        "Use this endpoint to confirm that a temporary batch was deleted after its retention period."
    ),
    operation_id="getTemporaryRepositoryCleanupStatus",
)
def get_temporary_repository_cleanup_status_endpoint(
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    from src.features.repositories.application.temporary_repository_service import (
        get_temporary_repository_cleanup_status,
        get_temporary_repository_config,
        list_temporary_repositories,
    )

    config = get_temporary_repository_config()
    repositories = list_temporary_repositories(service=service)
    return {
        "repository_name_prefix": config.name,
        "active_batch_count": len(repositories),
        "active_batches": [
            {"repository_id": item.get("repository_id"), "repository_name": item.get("name")}
            for item in repositories
        ],
        "retention_hours": config.retention_hours,
        "retention_seconds": config.retention_seconds,
        "test_mode": config.test_mode,
        "cleanup_interval_seconds": config.cleanup_interval_seconds,
        "last_successful_cleanup": get_temporary_repository_cleanup_status(),
    }


@router.get(
    "/temporary/documents",
    summary="List temporary repository documents",
    description="Lists documents for all temporary batches, or one selected temporary batch.",
    operation_id="listTemporaryRepositoryDocuments",
)
def list_temporary_repository_documents_endpoint(
    limit: int = Query(default=500, ge=1, le=5000),
    repository_id: str | None = Query(default=None, description="Optional temporary batch repository ID"),
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    from src.features.repositories.application.temporary_repository_service import (
        get_temporary_repository_config,
        list_temporary_repositories,
    )

    config = get_temporary_repository_config()
    repositories = list_temporary_repositories(service=service)
    if repository_id:
        repositories = [item for item in repositories if str(item.get("repository_id")) == repository_id]
        if not repositories:
            raise HTTPException(status_code=404, detail={"code": "temporary_repository_not_found"})
    try:
        batches = [
            {
                "repository_id": item.get("repository_id"),
                "repository_name": item.get("name"),
                **service.list_documents(str(item.get("repository_id")), limit=limit),
            }
            for item in repositories
        ]
        return {"repository_name_prefix": config.name, "batch_count": len(batches), "batches": batches}
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post(
    "/temporary/cleanup",
    summary="Run temporary repository retention cleanup",
    description=(
        "Checks whether each temporary upload batch completed and its configured retention has expired. "
        "Eligible batches have their artifacts, Weaviate collection, and repository deleted."
    ),
    operation_id="cleanupTemporaryRepository",
)
def cleanup_temporary_repository_endpoint() -> dict[str, Any]:
    from src.features.repositories.application.temporary_repository_service import cleanup_expired_temporary_repository

    try:
        return cleanup_expired_temporary_repository()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "temporary_repository_cleanup_failed", "message": str(exc)}) from exc


@router.get(
    "/{repository_id}",
    summary="Get repository",
    description="Return a repository and its current (lenient) settings view.",
    operation_id="getRepository",
    responses={200: {"description": "Repository"}, 404: {"description": "Not found"}},
)
def get_repository(
    repository_id: str,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.get_repository(repository_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.patch(
    "/{repository_id}",
    summary="Update repository",
    description=(
        "Update editable repository metadata (owner_user_name, default_tenant_id). "
        "Immutable identity fields and lifecycle status are rejected; use activate/archive/"
        "reactivate or PATCH .../settings for those concerns."
    ),
    operation_id="updateRepository",
    responses={
        200: {"description": "Repository updated"},
        400: {"description": "Invalid or immutable field"},
        404: {"description": "Not found"},
        409: {"description": "Settings locked / conflict"},
    },
)
def update_repository(
    repository_id: str,
    body: UpdateRepositoryRequest,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.update_repository(repository_id, body.model_dump(exclude_none=True))
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.patch(
    "/{repository_id}/settings",
    summary="Patch repository settings",
    description="Merge processing/retrieval settings. Prefer dedicated lifecycle endpoints for status.",
    operation_id="patchRepositorySettings",
    responses={
        200: {"description": "Settings updated"},
        404: {"description": "Not found"},
        409: {"description": "Settings locked"},
        422: {"description": "Invalid settings"},
    },
)
def patch_repository_settings(
    repository_id: str,
    body: UpdateRepositorySettingsRequest,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.update_settings(repository_id, body.model_dump(exclude_none=True))
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.get(
    "/{repository_id}/settings",
    summary="Get repository settings",
    description="Return stored/effective settings for the repository.",
    operation_id="getRepositorySettings",
    responses={200: {"description": "Settings"}, 404: {"description": "Not found"}},
)
def get_repository_settings(
    repository_id: str,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.get_settings(repository_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.put(
    "/{repository_id}/settings",
    summary="Replace repository settings",
    description="Replace settings with a full upsert (defaults applied for omitted keys).",
    operation_id="putRepositorySettings",
    responses={
        200: {"description": "Settings replaced"},
        404: {"description": "Not found"},
        409: {"description": "Settings locked"},
        422: {"description": "Invalid settings"},
    },
)
def put_repository_settings(
    repository_id: str,
    body: UpsertRepositorySettingsRequest,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.replace_settings(repository_id, body.model_dump(exclude_none=True))
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.get("/{repository_id}/key-fields", summary="List repository key fields", operation_id="listRepositoryKeyFields")
def list_repository_key_fields(
    repository_id: str,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.list_repository_key_fields(repository_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post("/{repository_id}/key-fields", summary="Create repository key field", operation_id="createRepositoryKeyField")
def create_repository_key_field(
    repository_id: str,
    body: CreateRepositoryKeyFieldRequest,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.create_repository_key_field(repository_id, body.model_dump(exclude_none=True))
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.patch("/{repository_id}/key-fields", summary="Replace repository key fields", operation_id="patchRepositoryKeyFields")
def patch_repository_key_fields(
    repository_id: str,
    body: PatchRepositoryKeyFieldsRequest,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.patch_repository_key_fields(repository_id, body.model_dump()["key_fields"])
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.delete(
    "/{repository_id}/key-fields/{field_id}",
    summary="Delete repository key field",
    operation_id="deleteRepositoryKeyField",
)
def delete_repository_key_field(
    repository_id: str,
    field_id: str,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.delete_repository_key_field(repository_id, field_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post(
    "/{repository_id}/activate",
    summary="Activate repository",
    description=(
        "Transition repository to ACTIVE after activation validation. "
        "Locks settings. Does not provision Weaviate."
    ),
    operation_id="activateRepository",
    responses={
        200: {"description": "Repository active"},
        404: {"description": "Not found"},
        400: {"description": "Invalid status transition"},
        422: {"description": "Activation prerequisites failed"},
    },
)
def activate_repository(
    repository_id: str,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.activate_repository(repository_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post(
    "/{repository_id}/archive",
    summary="Archive repository",
    description="Soft-retire an ACTIVE repository (ACTIVE → ARCHIVED).",
    operation_id="archiveRepository",
    responses={
        200: {"description": "Repository archived"},
        404: {"description": "Not found"},
        400: {"description": "Invalid status transition"},
    },
)
def archive_repository(
    repository_id: str,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.archive_repository(repository_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post(
    "/{repository_id}/reactivate",
    summary="Reactivate repository",
    description="Reactivate a soft-retired repository (ARCHIVED → ACTIVE).",
    operation_id="reactivateRepository",
    responses={
        200: {"description": "Repository reactivated"},
        404: {"description": "Not found"},
        400: {"description": "Repository is not archived"},
        422: {"description": "Activation prerequisites failed"},
    },
)
def reactivate_repository(
    repository_id: str,
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.reactivate_repository(repository_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.delete(
    "/{repository_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete repository",
    description="Hard-delete a repository when document_count is zero. Archive when documents exist.",
    operation_id="deleteRepository",
    responses={
        204: {"description": "Deleted"},
        404: {"description": "Not found"},
        409: {"description": "Repository has linked documents"},
    },
)
def delete_repository(
    repository_id: str,
    service: RepositoryService = Depends(_svc),
) -> Response:
    try:
        service.delete_repository(repository_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{repository_id}/documents", summary="List repository documents", operation_id="listRepositoryDocuments")
def list_repository_documents(
    repository_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    from src.application.consumer_api.context import get_current_user_from_context
    from src.features.authentication.domain.authentication_exceptions import AuthenticationError
    from src.features.users.application.user_service import get_platform_security_service

    current = get_current_user_from_context()
    if current is None or current.auth_method not in {"jwt", "api_key"} or current.user_id in {"", "anonymous"}:
        raise AuthenticationError("Authentication required")
    security = get_platform_security_service()
    if getattr(current, "is_platform_admin", False) or getattr(current, "is_admin_api_key", False):
        security._ensure_repository_exists(repository_id)
    else:
        security.check_retrieval_access(current, repository_id)
    try:
        return service.list_documents(repository_id, limit=limit)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.delete(
    "/{repository_id}/documents",
    summary="Purge repository documents",
    operation_id="purgeRepositoryDocuments",
)
def purge_repository_documents(
    repository_id: str,
    delete_file: bool = Query(default=True),
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    from src.application.consumer_api.context import get_current_user_from_context
    from src.features.authorization.application.authorization_service import AuthenticatedUser
    from src.features.authentication.domain.authentication_exceptions import AuthenticationError
    from src.features.users.application.user_service import get_platform_security_service

    current = get_current_user_from_context()
    if current is None or current.auth_method not in {"jwt", "api_key"} or current.user_id in {"", "anonymous"}:
        raise AuthenticationError("Authentication required")
    actor = AuthenticatedUser(
        user_id=current.user_id,
        platform_role=current.platform_role,
        auth_method=current.auth_method,
        must_change_password=current.must_change_password,
        is_admin_api_key=current.is_admin_api_key,
        is_consumer_api_key=current.is_consumer_api_key,
    )
    get_platform_security_service().check_document_delete_access(actor, repository_id)
    try:
        return service.purge_all_documents(repository_id, delete_file=delete_file)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post(
    "/{repository_id}/access",
    status_code=status.HTTP_201_CREATED,
    summary="Grant repository role",
    operation_id="grantRepositoryRole",
)
def grant_repository_role(
    repository_id: str,
    body: GrantRepositoryRoleRequest,
    request: Request,
) -> dict[str, Any]:
    from src.application.consumer_api.context import get_current_user_from_context
    from src.features.authorization.application.authorization_service import AuthenticatedUser
    from src.features.authentication.domain.authentication_exceptions import AuthenticationError
    from src.features.users.application.user_service import get_platform_security_service

    current = get_current_user_from_context()
    if current is None or current.auth_method not in {"jwt", "api_key"} or current.user_id in {"", "anonymous"}:
        raise AuthenticationError("Authentication required")
    actor = AuthenticatedUser(
        user_id=current.user_id,
        platform_role=current.platform_role,
        auth_method=current.auth_method,
        must_change_password=current.must_change_password,
        is_admin_api_key=current.is_admin_api_key,
        is_consumer_api_key=current.is_consumer_api_key,
    )
    return get_platform_security_service().grant_repository_role(
        actor,
        repository_id,
        user_id=body.user_id,
        role=body.role,
        request_id=request.headers.get("x-request-id"),
    )


@router.delete(
    "/{repository_id}/access/{user_id}",
    summary="Revoke repository role",
    operation_id="revokeRepositoryRole",
)
def revoke_repository_role(
    repository_id: str,
    user_id: str,
    request: Request,
    role: str | None = Query(default=None),
) -> dict[str, Any]:
    from src.application.consumer_api.context import get_current_user_from_context
    from src.features.authorization.application.authorization_service import AuthenticatedUser
    from src.features.authentication.domain.authentication_exceptions import AuthenticationError
    from src.features.users.application.user_service import get_platform_security_service

    current = get_current_user_from_context()
    if current is None or current.auth_method not in {"jwt", "api_key"} or current.user_id in {"", "anonymous"}:
        raise AuthenticationError("Authentication required")
    actor = AuthenticatedUser(
        user_id=current.user_id,
        platform_role=current.platform_role,
        auth_method=current.auth_method,
        must_change_password=current.must_change_password,
        is_admin_api_key=current.is_admin_api_key,
        is_consumer_api_key=current.is_consumer_api_key,
    )
    return get_platform_security_service().revoke_repository_grant(
        actor,
        repository_id,
        user_id,
        role=role,
        request_id=request.headers.get("x-request-id"),
    )


@router.post(
    "/{repository_id}/grants",
    status_code=status.HTTP_201_CREATED,
    summary="Grant repository role",
    operation_id="grantRepositoryRoleLegacy",
    include_in_schema=False,
)
def grant_repository_role_legacy(
    repository_id: str,
    body: GrantRepositoryRoleRequest,
    request: Request,
) -> dict[str, Any]:
    return grant_repository_role(repository_id, body, request)


@router.delete(
    "/{repository_id}/grants/{user_id}",
    summary="Revoke repository role",
    operation_id="revokeRepositoryRoleLegacy",
    include_in_schema=False,
)
def revoke_repository_role_legacy(
    repository_id: str,
    user_id: str,
    request: Request,
    role: str | None = Query(default=None),
) -> dict[str, Any]:
    return revoke_repository_role(repository_id, user_id, request, role=role)


@router.post("/{repository_id}/documents", summary="Link document to repository", operation_id="linkRepositoryDocument")
def link_repository_document(
    repository_id: str,
    body: dict[str, Any] = Body(...),
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    document_id = str(body.get("document_id") or body.get("fileId") or body.get("file_id") or "").strip()
    if not document_id:
        raise HTTPException(status_code=422, detail="document_id is required")
    try:
        service.link_document(document_id, repository_id)
        return {"repository_id": repository_id, "document_id": document_id, "linked": True}
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.delete(
    "/{repository_id}/documents/{document_id}",
    summary="Delete repository document",
    operation_id="deleteRepositoryDocument",
)
def delete_repository_document(
    repository_id: str,
    document_id: str,
    delete_file: bool = Query(default=True),
    service: RepositoryService = Depends(_svc),
) -> dict[str, Any]:
    try:
        return service.delete_repository_document(repository_id, document_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc
