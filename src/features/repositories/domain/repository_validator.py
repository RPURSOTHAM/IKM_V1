"""Model-level validation for Repository domain objects (Phase 3.1).

Structural checks only. Business rules (activate lock, delete-when-empty,
authorization, duplicate-name uniqueness against Store) belong to RepositoryService.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from src.features.repositories.domain.constants import (
    REPOSITORY_NAME_MAX_LENGTH,
    REPOSITORY_NAME_PATTERN,
    REPOSITORY_OWNER_ID_MAX_LENGTH,
    REPOSITORY_STATUS_VALUES,
    RESERVED_REPOSITORY_NAMES,
)
from src.features.repositories.domain.repository import (
    RepositoryRecord,
    RepositorySettings,
    RepositoryStatus,
    RepositoryValidationIssue,
    RepositoryValidationResult,
)


def _issue(
    code: str,
    message: str,
    *,
    field_name: str | None = None,
    details: dict[str, Any] | None = None,
) -> RepositoryValidationIssue:
    return RepositoryValidationIssue(
        code=code,
        message=message,
        field_name=field_name,
        details=details or {},
    )


def validate_repository_id(value: Any) -> RepositoryValidationResult:
    errors: list[RepositoryValidationIssue] = []
    text = str(value or "").strip()
    if not text:
        errors.append(
            _issue("invalid_repository_id", "repository_id is required.", field_name="repository_id")
        )
        return RepositoryValidationResult.failed(errors)
    try:
        UUID(text)
    except (TypeError, ValueError, AttributeError):
        errors.append(
            _issue(
                "invalid_repository_id",
                "repository_id must be a valid UUID.",
                field_name="repository_id",
                details={"value": text},
            )
        )
    return (
        RepositoryValidationResult.ok()
        if not errors
        else RepositoryValidationResult.failed(errors)
    )


def validate_repository_name(value: Any) -> RepositoryValidationResult:
    errors: list[RepositoryValidationIssue] = []
    if value is None:
        errors.append(_issue("invalid_name", "name is required.", field_name="name"))
        return RepositoryValidationResult.failed(errors)

    if not isinstance(value, str):
        errors.append(
            _issue(
                "invalid_name",
                "name must be a string.",
                field_name="name",
                details={"type": type(value).__name__},
            )
        )
        return RepositoryValidationResult.failed(errors)

    trimmed = value.strip()
    if not trimmed:
        errors.append(_issue("invalid_name", "name is required.", field_name="name"))
        return RepositoryValidationResult.failed(errors)

    if len(trimmed) > REPOSITORY_NAME_MAX_LENGTH:
        errors.append(
            _issue(
                "invalid_name",
                f"name must be at most {REPOSITORY_NAME_MAX_LENGTH} characters.",
                field_name="name",
                details={"length": len(trimmed), "max_length": REPOSITORY_NAME_MAX_LENGTH},
            )
        )

    if trimmed != value:
        # Leading/trailing whitespace is not part of the canonical name.
        errors.append(
            _issue(
                "invalid_name",
                "name must not have leading or trailing whitespace.",
                field_name="name",
            )
        )

    if not REPOSITORY_NAME_PATTERN.match(trimmed):
        errors.append(
            _issue(
                "invalid_name",
                "name must start with an alphanumeric character and contain only "
                "letters, digits, underscores, or hyphens.",
                field_name="name",
                details={"pattern": REPOSITORY_NAME_PATTERN.pattern},
            )
        )

    if trimmed.lower() in RESERVED_REPOSITORY_NAMES:
        errors.append(
            _issue(
                "reserved_name",
                f"name '{trimmed}' is reserved.",
                field_name="name",
                details={"name": trimmed},
            )
        )

    return (
        RepositoryValidationResult.ok()
        if not errors
        else RepositoryValidationResult.failed(errors)
    )


def validate_owner_user_id(value: Any) -> RepositoryValidationResult:
    errors: list[RepositoryValidationIssue] = []
    text = str(value or "").strip()
    if not text:
        errors.append(
            _issue("invalid_owner", "owner_user_id is required.", field_name="owner_user_id")
        )
        return RepositoryValidationResult.failed(errors)
    if len(text) > REPOSITORY_OWNER_ID_MAX_LENGTH:
        errors.append(
            _issue(
                "invalid_owner",
                f"owner_user_id must be at most {REPOSITORY_OWNER_ID_MAX_LENGTH} characters.",
                field_name="owner_user_id",
                details={"length": len(text), "max_length": REPOSITORY_OWNER_ID_MAX_LENGTH},
            )
        )
    return (
        RepositoryValidationResult.ok()
        if not errors
        else RepositoryValidationResult.failed(errors)
    )


def validate_repository_status(value: Any) -> RepositoryValidationResult:
    errors: list[RepositoryValidationIssue] = []
    if not RepositoryStatus.is_valid(value if not isinstance(value, str) else value.strip()):
        errors.append(
            _issue(
                "invalid_status",
                f"status must be one of {sorted(REPOSITORY_STATUS_VALUES)}.",
                field_name="status",
                details={"value": value, "allowed": sorted(REPOSITORY_STATUS_VALUES)},
            )
        )
    return (
        RepositoryValidationResult.ok()
        if not errors
        else RepositoryValidationResult.failed(errors)
    )


def validate_weaviate_collection(value: Any) -> RepositoryValidationResult:
    errors: list[RepositoryValidationIssue] = []
    text = str(value or "").strip()
    if not text:
        errors.append(
            _issue(
                "invalid_weaviate_collection",
                "weaviate_collection is required.",
                field_name="weaviate_collection",
            )
        )
    return (
        RepositoryValidationResult.ok()
        if not errors
        else RepositoryValidationResult.failed(errors)
    )


def validate_repository_settings_shape(settings: RepositorySettings | dict[str, Any] | None) -> RepositoryValidationResult:
    """Structural check only: settings must be a mapping. No processor/retrieval rules."""
    errors: list[RepositoryValidationIssue] = []
    if settings is None:
        return RepositoryValidationResult.ok()
    if isinstance(settings, RepositorySettings):
        payload = settings.settings
        repo_id = settings.repository_id
        id_result = validate_repository_id(repo_id)
        if not id_result.is_valid:
            errors.extend(id_result.errors)
    else:
        payload = settings
    if not isinstance(payload, dict):
        errors.append(
            _issue(
                "invalid_settings",
                "settings must be an object/mapping.",
                field_name="settings",
                details={"type": type(payload).__name__},
            )
        )
    return (
        RepositoryValidationResult.ok()
        if not errors
        else RepositoryValidationResult.failed(errors)
    )


def validate_repository_record(record: RepositoryRecord) -> RepositoryValidationResult:
    """Validate required fields and structural consistency on a RepositoryRecord."""
    errors: list[RepositoryValidationIssue] = []

    for result in (
        validate_repository_id(record.repository_id),
        validate_repository_name(record.name),
        validate_owner_user_id(record.owner_user_id),
        validate_repository_status(record.status),
        validate_weaviate_collection(record.weaviate_collection),
    ):
        if not result.is_valid:
            errors.extend(result.errors)

    if not isinstance(record.created_at, datetime):
        errors.append(
            _issue(
                "invalid_timestamp",
                "created_at must be a datetime.",
                field_name="created_at",
            )
        )
    if not isinstance(record.updated_at, datetime):
        errors.append(
            _issue(
                "invalid_timestamp",
                "updated_at must be a datetime.",
                field_name="updated_at",
            )
        )
    if record.settings_locked_at is not None and not isinstance(record.settings_locked_at, datetime):
        errors.append(
            _issue(
                "invalid_timestamp",
                "settings_locked_at must be a datetime or null.",
                field_name="settings_locked_at",
            )
        )

    if record.default_tenant_id is not None and not isinstance(record.default_tenant_id, str):
        errors.append(
            _issue(
                "invalid_tenant",
                "default_tenant_id must be a string or null.",
                field_name="default_tenant_id",
            )
        )

    return (
        RepositoryValidationResult.ok()
        if not errors
        else RepositoryValidationResult.failed(errors)
    )
