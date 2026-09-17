from __future__ import annotations

from typing import Any


class PlatformSecurityError(Exception):
    code = "platform_security_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AuthenticationError(PlatformSecurityError):
    code = "unauthorized"
    http_status = 401


class AuthorizationError(PlatformSecurityError):
    code = "forbidden"
    http_status = 403


class AccountLockedError(PlatformSecurityError):
    code = "account_locked"
    http_status = 423


class NotFoundError(PlatformSecurityError):
    code = "not_found"
    http_status = 404


class ValidationError(PlatformSecurityError):
    code = "validation_error"
    http_status = 422
