from __future__ import annotations

from typing import Any


class DocumentTypeError(Exception):
    """Base error for document type domain failures."""

    code: str = "document_type_error"
    http_status: int = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(DocumentTypeError):
    code = "not_found"
    http_status = 404


class SystemTypeImmutableError(DocumentTypeError):
    code = "system_type_immutable"
    http_status = 403


class TypeInUseError(DocumentTypeError):
    code = "type_in_use"
    http_status = 409


class MetadataInUseError(DocumentTypeError):
    code = "metadata_in_use"
    http_status = 409


class DuplicateNameError(DocumentTypeError):
    code = "duplicate_name"
    http_status = 409


class MaxDepthExceededError(DocumentTypeError):
    code = "max_depth_exceeded"
    http_status = 422


class ParentInactiveError(DocumentTypeError):
    code = "parent_inactive"
    http_status = 422


class ServiceUnavailableError(DocumentTypeError):
    code = "service_unavailable"
    http_status = 503


class ValidationError(DocumentTypeError):
    code = "invalid_request"
    http_status = 400


class RepositoryMismatchError(DocumentTypeError):
    code = "repository_mismatch"
    http_status = 403


class CircularReferenceError(DocumentTypeError):
    code = "circular_reference"
    http_status = 422


class DuplicateFieldError(DocumentTypeError):
    code = "duplicate_field"
    http_status = 409
