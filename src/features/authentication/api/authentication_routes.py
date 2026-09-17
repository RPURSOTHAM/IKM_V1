"""Unauthenticated platform security routes (token issuance)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from src.features.authentication.domain.authentication_exceptions import AuthenticationError
from src.features.users.application.user_service import PlatformSecurityService, get_platform_security_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


class TokenRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="Platform user id (e.g. bootstrap admin).",
        examples=["admin"],
    )
    password: str = Field(
        ...,
        min_length=1,
        description=(
            "Platform user password. For the bootstrap admin, use "
            "PLATFORM_BOOTSTRAP_ADMIN_PASSWORD from deploy/application/.env "
            "(not a Swagger placeholder)."
        ),
        examples=["<PLATFORM_BOOTSTRAP_ADMIN_PASSWORD>"],
    )


def _svc() -> PlatformSecurityService:
    return get_platform_security_service()


def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip() or None
    if request.client:
        return request.client.host
    return None


@router.post(
    "/token",
    summary="Issue platform access token",
    operation_id="issuePlatformToken",
)
def issue_platform_token(
    body: TokenRequest,
    request: Request,
    service: PlatformSecurityService = Depends(_svc),
) -> dict[str, Any]:
    """Authenticate a platform user and return a JWT access token."""
    service.initialize()
    try:
        return service.authenticate(
            body.user_id,
            body.password,
            client_ip=_client_ip(request),
            request_id=request.headers.get("x-request-id"),
        )
    except AuthenticationError:
        try:
            service.record_login_failure(body.user_id.strip())
        except Exception:
            pass
        raise
