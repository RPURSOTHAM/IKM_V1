"""JWT authentication middleware for retrieval endpoints."""

from __future__ import annotations

import logging
import re
import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.application.consumer_api.context import (
    RequestActor,
    reset_current_user_in_context,
    set_current_user_in_context,
)
from src.features.authentication.domain.authentication_exceptions import AuthenticationError
from src.features.authentication.application.token_service import JwtService, looks_like_jwt

_logger = logging.getLogger(__name__)

_RETRIEVE_EXACT = re.compile(r"^/api/v1/retrieve/?$")
_RETRIEVE_SUB = re.compile(r"^/api/v1/retrieve/")
_BM25 = re.compile(r"^/api/v1/repositories/[^/]+/search/bm25/?$")
_REPO_CHUNKS = re.compile(r"^/api/v1/repositories/[^/]+/documents/[^/]+/chunks/?$")
_DOC_CHUNKS = re.compile(r"^/api/v1/documents/[^/]+/chunks/?$")
_PUBLIC_EXACT = frozenset({
    "/health",
    "/status",
    "/ready",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
})
_PUBLIC_PREFIXES = ("/docs/", "/redoc/", "/infra/")
_PUBLIC_API = re.compile(r"^/api/v1/auth/token/?$")


def requires_retrieval_auth(method: str, path: str) -> bool:
    """Return True for retrieval endpoints that require a Bearer JWT."""
    method_u = (method or "GET").upper()
    normalized = path or ""
    if _RETRIEVE_EXACT.match(normalized) and method_u == "POST":
        return True
    if _RETRIEVE_SUB.match(normalized):
        return True
    if _BM25.match(normalized) and method_u == "POST":
        return True
    if _REPO_CHUNKS.match(normalized) and method_u == "GET":
        return True
    if _DOC_CHUNKS.match(normalized) and method_u == "GET":
        return True
    return False


def requires_api_auth(method: str, path: str) -> bool:
    """Return True for authenticated platform API routes (JWT required)."""
    method_u = (method or "GET").upper()
    if method_u == "OPTIONS":
        return False
    normalized = path or ""
    if normalized in _PUBLIC_EXACT:
        return False
    if any(normalized.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
        return False
    if _PUBLIC_API.match(normalized):
        return False
    # Document Rendition (Swagger /api/v1/rendering/*) — public for page-image preview.
    if normalized.startswith("/api/v1/rendering"):
        return False
    if normalized.startswith("/api/v1/"):
        return True
    return requires_retrieval_auth(method, path)


def _unauthorized(message: str = "Authentication required", request_id: str | None = None) -> JSONResponse:
    error = {
        "code": "unauthorized",
        "message": message,
        "details": {},
    }
    if request_id:
        error["request_id"] = request_id
    return JSONResponse(status_code=401, content={"error": error})


def actor_from_jwt_payload(payload: dict) -> RequestActor:
    user_id = str(payload.get("sub") or "").strip()
    platform_role = payload.get("platform_role")
    platform_role_s = str(platform_role).strip() if platform_role else None
    is_admin = platform_role_s in {"administrator", "platform_owner"}
    return RequestActor(
        user_id=user_id or "anonymous",
        auth_method="jwt",
        platform_role=platform_role_s,
        is_platform_admin=is_admin,
        is_admin_api_key=False,
        is_consumer_api_key=False,
        must_change_password=bool(payload.get("must_change_password")),
    )


def authenticate_bearer_token(authorization_header: str | None) -> RequestActor:
    """Validate Authorization Bearer JWT and return a RequestActor."""
    if not authorization_header or not str(authorization_header).strip():
        raise AuthenticationError("Authentication required")
    parts = str(authorization_header).strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise AuthenticationError("Authentication required")
    token = parts[1].strip()
    if not looks_like_jwt(token):
        raise AuthenticationError("Invalid token")
    from src.features.users.infrastructure.user_repository import get_platform_security_store

    store = get_platform_security_store()
    payload = JwtService(store).decode_token(token)
    actor = actor_from_jwt_payload(payload)
    if not actor.user_id or actor.user_id == "anonymous":
        raise AuthenticationError("Invalid token")
    return actor


class JwtAuthMiddleware(BaseHTTPMiddleware):
    """Populate RequestActor for protected retrieval routes; reject missing/invalid JWTs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        method = request.method
        if not requires_api_auth(method, path):
            return await call_next(request)

        started = time.perf_counter()
        auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
        try:
            actor = authenticate_bearer_token(auth_header)
        except AuthenticationError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            _logger.info(
                "auth_result outcome=unauthorized endpoint=%s %s repository_hint=%s "
                "user_id=None latency_ms=%.1f reason=%s",
                method,
                path,
                request.path_params.get("repository_id") if hasattr(request, "path_params") else None,
                latency_ms,
                exc.message,
            )
            return _unauthorized(exc.message or "Authentication required", request.headers.get("x-request-id"))
        except Exception as exc:
            # Postgres/store outages previously surfaced as opaque ASGI 500s.
            _logger.error("auth backend unavailable for %s %s: %s", method, path, exc)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "service_unavailable",
                        "message": (
                            "Auth/database unavailable. Start Docker Desktop and infrastructure "
                            "(postgres on :5432), then retry."
                        ),
                        "details": {"reason": str(exc)[:300]},
                    }
                },
            )

        ctx_token = set_current_user_in_context(actor)
        try:
            response = await call_next(request)
            latency_ms = (time.perf_counter() - started) * 1000.0
            _logger.info(
                "auth_result outcome=authorized endpoint=%s %s user_id=%s "
                "platform_role=%s status=%s latency_ms=%.1f",
                method,
                path,
                actor.user_id,
                actor.platform_role,
                getattr(response, "status_code", None),
                latency_ms,
            )
            return response
        finally:
            reset_current_user_in_context(ctx_token)
