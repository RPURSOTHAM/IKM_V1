from __future__ import annotations

from typing import Any


class PlatformConfigError(Exception):
    code = "platform_config_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(PlatformConfigError):
    code = "not_found"
    http_status = 404


class ValidationError(PlatformConfigError):
    code = "validation_error"
    http_status = 422


class ServiceUnavailableError(PlatformConfigError):
    code = "service_unavailable"
    http_status = 503
