from __future__ import annotations

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from src.shared.errors import COMPONENT_CONSUMER_API, DmsServiceError, log_service_failure
from src.features.authentication.domain.authentication_exceptions import PlatformSecurityError
from src.features.repositories.domain.repository_exceptions import RepositoryError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PlatformSecurityError)
    async def platform_security_error_handler(request: Request, exc: PlatformSecurityError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": request.headers.get("x-request-id") or request.headers.get("X-Request-ID"),
                }
            },
        )

    @app.exception_handler(DmsServiceError)
    async def dms_service_error_handler(request: Request, exc: DmsServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.user_message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RepositoryError)
    async def repository_error_handler(request: Request, exc: RepositoryError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        log_service_failure(
            component=COMPONENT_CONSUMER_API,
            code="http_error",
            reason=str(detail),
            level=logging.WARNING if exc.status_code < 500 else logging.ERROR,
        )

        # Keep minimal validation contracts stable for clients that expect
        # {"error": "..."} instead of wrapped error envelopes.
        if (
            isinstance(detail, dict)
            and set(detail.keys()) == {"error"}
            and isinstance(detail.get("error"), str)
        ):
            return JSONResponse(status_code=exc.status_code, content=detail)

        # Preserve structured RepositoryError / domain detail payloads from routers.
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": detail.get("code"),
                        "message": detail.get("message"),
                        "details": detail.get("details") or {},
                        "request_id": request.headers.get("x-request-id") or request.headers.get("X-Request-ID"),
                    }
                },
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(detail),
                    "details": {},
                }
            },
        )
