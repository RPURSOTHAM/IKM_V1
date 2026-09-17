"""Shared DMS service error types and logging helpers."""

from __future__ import annotations

import logging
from typing import Any, NoReturn

from src.application.consumer_api.context import get_request_id

COMPONENT_CONSUMER_API = "consumer_api_service"
COMPONENT_DOCUMENT_RECEIVER = "document_receiver_service"
COMPONENT_DOCUMENT_PREVIEW = "document_preview"
COMPONENT_RETRIEVAL = "retrieval_service"
COMPONENT_REPOSITORY = "repository_service"
COMPONENT_QUEUE_ADMIN = "queue_admin_service"
COMPONENT_WEAVIATE_ADMIN = "weaviate_admin"

_logger = logging.getLogger("dms_service.errors")


class DmsServiceError(Exception):
    """Structured service error with separate internal and user-facing messages."""

    def __init__(
        self,
        reason: str,
        *,
        component: str,
        code: str = "service_error",
        http_status: int = 500,
        user_message: str | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.component = component
        self.code = code
        self.http_status = http_status
        self.user_message = user_message or "An unexpected error occurred. Please try again or contact support."
        self.details = details or {}
        self.cause = cause


def log_service_failure(
    *,
    component: str,
    code: str,
    reason: str,
    cause: BaseException | None = None,
    level: int = logging.ERROR,
) -> None:
    _logger.log(
        level,
        "component=%s code=%s request_id=%s reason=%s",
        component,
        code,
        get_request_id(),
        reason,
        exc_info=cause,
    )


def raise_service_error(
    component: str,
    *,
    code: str,
    http_status: int,
    user_message: str,
    reason: str,
    cause: BaseException | None = None,
    details: dict[str, Any] | None = None,
) -> NoReturn:
    log_service_failure(component=component, code=code, reason=reason, cause=cause)
    raise DmsServiceError(
        reason,
        component=component,
        code=code,
        http_status=http_status,
        user_message=user_message,
        details=details,
        cause=cause,
    )


def raise_client_error(
    component: str,
    *,
    code: str,
    http_status: int,
    user_message: str,
    reason: str,
    cause: BaseException | None = None,
    details: dict[str, Any] | None = None,
) -> NoReturn:
    log_service_failure(
        component=component,
        code=code,
        reason=reason,
        cause=cause,
        level=logging.WARNING,
    )
    raise DmsServiceError(
        reason,
        component=component,
        code=code,
        http_status=http_status,
        user_message=user_message,
        details=details,
        cause=cause,
    )
