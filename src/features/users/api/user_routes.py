"""Tracked FastAPI routes for platform security catalogs and admin user management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status

from src.application.consumer_api.context import get_current_user_from_context
from src.features.configuration.platform_settings import settings
from src.features.authorization.application.authorization_service import AuthenticatedUser
from src.features.authentication.domain.authentication_exceptions import AuthenticationError
from src.features.users.schemas.user_schemas import CreatePlatformUserRequest
from src.features.users.application.user_service import PlatformSecurityService, get_platform_security_service

router = APIRouter(prefix="/platform", tags=["Platform Security"])


def _svc() -> PlatformSecurityService:
    return get_platform_security_service()


def _require_actor() -> AuthenticatedUser:
    current = get_current_user_from_context()
    if current is None or current.auth_method not in {"jwt", "api_key"} or current.user_id in {"", "anonymous"}:
        raise AuthenticationError("Authentication required")
    return AuthenticatedUser(
        user_id=current.user_id,
        platform_role=current.platform_role,
        auth_method=current.auth_method,
        must_change_password=current.must_change_password,
        is_admin_api_key=current.is_admin_api_key,
        is_consumer_api_key=current.is_consumer_api_key,
    )


@router.get(
    "/roles",
    summary="List platform roles",
    operation_id="listPlatformRoles",
)
def list_platform_roles(
    service: PlatformSecurityService = Depends(_svc),
) -> dict[str, Any]:
    """Return the catalog of platform roles for user administration and UI dropdowns."""
    return service.get_platform_role_catalog(api_prefix=settings.api_prefix)


@router.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    summary="Create platform user",
    operation_id="createPlatformUser",
)
def create_platform_user(
    body: CreatePlatformUserRequest,
    request: Request,
    service: PlatformSecurityService = Depends(_svc),
) -> dict[str, Any]:
    """Administrator-only user creation. Password hashes are never returned."""
    actor = _require_actor()
    return service.create_user(
        actor,
        user_id=body.user_id,
        password=body.password,
        display_name=body.display_name,
        platform_role=body.platform_role,
        request_id=request.headers.get("x-request-id"),
    )


@router.get(
    "/users",
    summary="List platform users",
    operation_id="listPlatformUsers",
)
def list_platform_users(
    service: PlatformSecurityService = Depends(_svc),
) -> dict[str, Any]:
    actor = _require_actor()
    return service.list_users(actor)
