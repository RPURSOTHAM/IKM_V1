"""Phase 5 document review exceptions."""

from __future__ import annotations

from typing import Any


class DocumentReviewError(Exception):
    code = "document_review_error"
    http_status = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(DocumentReviewError):
    code = "review_not_found"
    http_status = 404


class ConflictError(DocumentReviewError):
    code = "review_conflict"
    http_status = 409


class InvalidTransitionError(DocumentReviewError):
    code = "invalid_review_transition"
    http_status = 409


class PermissionDeniedError(DocumentReviewError):
    code = "review_permission_denied"
    http_status = 403


class ValidationError(DocumentReviewError):
    code = "review_validation_error"
    http_status = 422
