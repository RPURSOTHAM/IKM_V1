"""Evidence service exceptions mapped to HTTP by the route layer."""

from __future__ import annotations

from typing import Any


class EvidenceError(Exception):
    code: str = "evidence_error"
    http_status: int = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class EvidenceValidationError(EvidenceError):
    code = "validation_error"
    http_status = 422


class EvidenceNotFoundError(EvidenceError):
    code = "not_found"
    http_status = 404


class EvidenceDuplicateError(EvidenceError):
    code = "duplicate_import"
    http_status = 409


class EvidenceExternalError(EvidenceError):
    code = "external_provider_error"
    http_status = 502


class EvidenceExternalTimeoutError(EvidenceExternalError):
    code = "external_provider_timeout"
    http_status = 504


class EvidenceExternalUnavailableError(EvidenceExternalError):
    code = "external_provider_unavailable"
    http_status = 503


class EvidenceRateLimitError(EvidenceExternalError):
    code = "external_provider_rate_limit"
    http_status = 429
