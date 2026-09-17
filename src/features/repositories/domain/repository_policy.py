"""Business validation rules for Repository Management (Phase 3.3).

Pure functions only — no Store, HTTP, Weaviate, Processing, or Retrieval imports.
RepositoryService loads persistence data and calls these validators before writes.
"""

from __future__ import annotations

import re
from typing import Any

from src.features.repositories.domain.constants import (
    ACTIVATION_SOURCE_STATUSES,
    ALLOWED_RETRIEVAL_SEARCH_MODES,
    ALLOWED_STATUS_TRANSITIONS,
    IMMUTABLE_REPOSITORY_FIELDS,
    LOCKED_SETTINGS_KEYS,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    REPOSITORY_NAME_MAX_LENGTH,
    REPOSITORY_NAME_MIN_LENGTH,
    REPOSITORY_NAME_PATTERN,
    RESERVED_REPOSITORY_NAMES,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
)
from src.features.repositories.domain.repository_exceptions import (
    ImmutableRepositorySetting,
    InvalidRepositoryName,
    InvalidRepositoryStatusTransition,
    RepositoryActivationFailed,
    RepositoryCannotBeDeleted,
)
from src.features.repositories.domain.repository import (
    RepositoryRecord,
    RepositoryStatus,
    RepositoryValidationIssue,
    RepositoryValidationResult,
)
from src.features.repositories.domain.repository_validator import (
    _issue,
    validate_repository_name,
    validate_repository_status,
)


def normalize_repository_name(value: Any) -> str:
    """Trim and return canonical repository name, or raise InvalidRepositoryName."""
    if value is None:
        raise InvalidRepositoryName("name is required.", details={"field": "name"})
    if not isinstance(value, str):
        raise InvalidRepositoryName(
            "name must be a string.",
            details={"field": "name", "type": type(value).__name__},
        )
    trimmed = value.strip()
    result = validate_repository_name_business(trimmed)
    if not result.is_valid:
        first = result.errors[0]
        raise InvalidRepositoryName(
            first.message,
            details={"field": first.field_name, "code": first.code, **first.details},
        )
    return trimmed


def validate_repository_name_business(value: Any) -> RepositoryValidationResult:
    """Name rules: required, trim already applied by caller, length, charset, reserved."""
    errors: list[RepositoryValidationIssue] = []

    # `_temp_<batch UUID>` is a system-managed repository created for an upload
    # request without a selected repository. It is intentionally the only
    # underscore-prefixed namespace, so this exception must precede generic
    # alphanumeric-first validation.
    if isinstance(value, str) and (
        value.strip() == "_temp"
        or re.fullmatch(r"_temp_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value.strip())
    ):
        return RepositoryValidationResult.ok(mode="structural")

    # Reuse structural validator when value has no surrounding whitespace concerns.
    if isinstance(value, str) and value != value.strip():
        errors.append(
            _issue(
                "invalid_name",
                "name must not have leading or trailing whitespace.",
                field_name="name",
            )
        )
        value = value.strip()

    base = validate_repository_name(value)
    if not base.is_valid:
        return base

    trimmed = str(value).strip()
    if len(trimmed) < REPOSITORY_NAME_MIN_LENGTH:
        errors.append(
            _issue(
                "invalid_name",
                f"name must be at least {REPOSITORY_NAME_MIN_LENGTH} character(s).",
                field_name="name",
                details={"min_length": REPOSITORY_NAME_MIN_LENGTH},
            )
        )
    if len(trimmed) > REPOSITORY_NAME_MAX_LENGTH:
        errors.append(
            _issue(
                "invalid_name",
                f"name must be at most {REPOSITORY_NAME_MAX_LENGTH} characters.",
                field_name="name",
                details={"max_length": REPOSITORY_NAME_MAX_LENGTH},
            )
        )
    if not REPOSITORY_NAME_PATTERN.match(trimmed):
        errors.append(
            _issue(
                "invalid_name",
                "name contains illegal characters.",
                field_name="name",
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

    return RepositoryValidationResult.ok() if not errors else RepositoryValidationResult.failed(errors)


def names_conflict_case_insensitive(left: str, right: str) -> bool:
    return str(left or "").strip().lower() == str(right or "").strip().lower()


def validate_status_transition(
    current_status: str,
    target_status: str,
) -> RepositoryValidationResult:
    """Enforce Phase 2 lifecycle transitions (pure; no persistence)."""
    errors: list[RepositoryValidationIssue] = []
    current = str(current_status or "").strip().lower()
    target = str(target_status or "").strip().lower()

    for label, value in (("current", current), ("target", target)):
        status_result = validate_repository_status(value)
        if not status_result.is_valid:
            errors.extend(status_result.errors)
            return RepositoryValidationResult.failed(errors)

    if current == target:
        return RepositoryValidationResult.ok()

    allowed = ALLOWED_STATUS_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        errors.append(
            _issue(
                "invalid_status_transition",
                f"Status transition '{current}' → '{target}' is not allowed.",
                field_name="status",
                details={
                    "from": current,
                    "to": target,
                    "allowed": sorted(allowed),
                },
            )
        )
        return RepositoryValidationResult.failed(errors)
    return RepositoryValidationResult.ok()


def assert_status_transition(current_status: str, target_status: str) -> None:
    result = validate_status_transition(current_status, target_status)
    if result.is_valid:
        return
    issue = result.errors[0]
    raise InvalidRepositoryStatusTransition(
        issue.message,
        details=issue.details,
    )


def validate_activation(
    record: RepositoryRecord,
    settings: dict[str, Any] | None,
) -> RepositoryValidationResult:
    """Validate whether a repository may become ACTIVE.

    First activation: ``configuring`` | ``approved`` → ``active`` (full settings checks).
    Reactivation (Phase 2): ``archived`` → ``active`` (settings already locked; no re-resolve).
    """
    errors: list[RepositoryValidationIssue] = []
    status = str(record.status or "").strip().lower()

    # Soft-retired repos may be reactivated without re-running first-activation settings gates.
    if status == STATUS_ARCHIVED:
        return RepositoryValidationResult.ok()

    if status not in ACTIVATION_SOURCE_STATUSES:
        errors.append(
            _issue(
                "repository_activation_failed",
                f"Repository status '{status}' cannot be activated.",
                field_name="status",
                details={
                    "status": status,
                    "allowed_source_statuses": sorted(
                        set(ACTIVATION_SOURCE_STATUSES) | {STATUS_ARCHIVED}
                    ),
                },
            )
        )

    settings_result = validate_settings_business(settings or {}, require_activation_fields=True)
    if not settings_result.is_valid:
        errors.extend(settings_result.errors)

    return RepositoryValidationResult.ok() if not errors else RepositoryValidationResult.failed(errors)


def assert_activation(record: RepositoryRecord, settings: dict[str, Any] | None) -> None:
    result = validate_activation(record, settings)
    if result.is_valid:
        return
    issue = result.errors[0]
    raise RepositoryActivationFailed(issue.message, details={"errors": [e.code for e in result.errors], **issue.details})


def validate_update_patch(
    record: RepositoryRecord,
    patch: dict[str, Any],
    *,
    allow_document_type_default: bool = False,
    allow_key_field_disable_recovery: bool = False,
) -> RepositoryValidationResult:
    """Validate update/patch rules: immutable fields + lock after ACTIVE."""
    errors: list[RepositoryValidationIssue] = []
    patch = dict(patch or {})

    for key in IMMUTABLE_REPOSITORY_FIELDS:
        if key in patch and patch[key] is not None:
            errors.append(
                _issue(
                    "immutable_repository_setting",
                    f"Field '{key}' is immutable.",
                    field_name=key,
                )
            )

    status_patch = patch.get("status")
    settings_keys = {k for k in patch.keys() if k != "status"}

    if status_patch is not None:
        transition = validate_status_transition(record.status, str(status_patch))
        if not transition.is_valid:
            errors.extend(transition.errors)

    locked = record.status == STATUS_ACTIVE or record.settings_locked_at is not None
    if locked and settings_keys:
        if allow_document_type_default and settings_keys == {"document_type_id"}:
            pass
        elif allow_key_field_disable_recovery and settings_keys.issubset(
            {"key_field_extraction", "key_field_extraction_enabled"}
        ):
            pass
        else:
            locked_hits = sorted(settings_keys & LOCKED_SETTINGS_KEYS) or sorted(settings_keys)
            errors.append(
                _issue(
                    "immutable_repository_setting",
                    "Repository settings are read-only after activation.",
                    field_name="settings",
                    details={
                        "status": record.status,
                        "locked_keys": locked_hits,
                    },
                )
            )

    if record.status == STATUS_ARCHIVED and status_patch is None and settings_keys:
        errors.append(
            _issue(
                "immutable_repository_setting",
                "Cannot update settings on an archived repository.",
                field_name="settings",
                details={"status": record.status},
            )
        )

    settings_only = {k: v for k, v in patch.items() if k != "status"}
    if settings_only and not (
        allow_document_type_default and set(settings_only.keys()) == {"document_type_id"}
    ):
        settings_result = validate_settings_business(settings_only, require_activation_fields=False)
        if not settings_result.is_valid:
            errors.extend(settings_result.errors)

    return RepositoryValidationResult.ok() if not errors else RepositoryValidationResult.failed(errors)


def assert_update_patch(
    record: RepositoryRecord,
    patch: dict[str, Any],
    *,
    allow_document_type_default: bool = False,
    allow_key_field_disable_recovery: bool = False,
) -> None:
    result = validate_update_patch(
        record,
        patch,
        allow_document_type_default=allow_document_type_default,
        allow_key_field_disable_recovery=allow_key_field_disable_recovery,
    )
    if result.is_valid:
        return
    issue = result.errors[0]
    if issue.code == "invalid_status_transition":
        raise InvalidRepositoryStatusTransition(issue.message, details=issue.details)
    raise ImmutableRepositorySetting(issue.message, details=issue.details)


def validate_delete(
    record: RepositoryRecord | None,
    *,
    document_count: int,
) -> RepositoryValidationResult:
    """Validate hard-delete eligibility. Does not perform delete."""
    errors: list[RepositoryValidationIssue] = []
    if record is None:
        errors.append(
            _issue(
                "not_found",
                "Repository not found.",
                field_name="repository_id",
            )
        )
        return RepositoryValidationResult.failed(errors)

    if int(document_count) > 0:
        errors.append(
            _issue(
                "repository_cannot_be_deleted",
                "Repository cannot be deleted while documents are linked. Archive it instead.",
                field_name="repository_id",
                details={
                    "repository_id": record.repository_id,
                    "document_count": int(document_count),
                    "suggested_action": "archive",
                    "status": record.status,
                },
            )
        )
    return RepositoryValidationResult.ok() if not errors else RepositoryValidationResult.failed(errors)


def assert_delete_allowed(record: RepositoryRecord | None, *, document_count: int) -> None:
    result = validate_delete(record, document_count=document_count)
    if result.is_valid:
        return
    issue = result.errors[0]
    if issue.code == "not_found":
        from src.features.repositories.domain.repository_exceptions import NotFoundError

        raise NotFoundError(issue.message, details=issue.details)
    raise RepositoryCannotBeDeleted(issue.message, details=issue.details)


def validate_settings_business(
    settings: dict[str, Any] | None,
    *,
    require_activation_fields: bool = False,
) -> RepositoryValidationResult:
    """Business rules for RepositorySettings values (no SettingsResolver / merge)."""
    errors: list[RepositoryValidationIssue] = []
    payload = dict(settings or {})

    if require_activation_fields:
        # Activation requires a usable embedding_model object when provided; missing is
        # allowed only if omitted entirely (platform defaults applied later by resolver).
        # When present, it must be complete.
        pass

    embedding = payload.get("embedding_model")
    if embedding is not None:
        if not isinstance(embedding, dict):
            errors.append(
                _issue(
                    "invalid_settings",
                    "embedding_model must be an object.",
                    field_name="embedding_model",
                )
            )
        else:
            provider = str(embedding.get("provider") or "").strip()
            model_id = str(embedding.get("model_id") or "").strip()
            if not provider:
                errors.append(
                    _issue(
                        "invalid_settings",
                        "embedding_model.provider is required.",
                        field_name="embedding_model",
                    )
                )
            if not model_id:
                errors.append(
                    _issue(
                        "invalid_settings",
                        "embedding_model.model_id is required.",
                        field_name="embedding_model",
                    )
                )

    if "chunk_size" in payload and payload["chunk_size"] is not None:
        try:
            chunk_size = int(payload["chunk_size"])
        except (TypeError, ValueError):
            errors.append(
                _issue("invalid_settings", "chunk_size must be an integer.", field_name="chunk_size")
            )
        else:
            if chunk_size < MIN_CHUNK_SIZE or chunk_size > MAX_CHUNK_SIZE:
                errors.append(
                    _issue(
                        "invalid_settings",
                        f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}.",
                        field_name="chunk_size",
                        details={"min": MIN_CHUNK_SIZE, "max": MAX_CHUNK_SIZE, "value": chunk_size},
                    )
                )

    if "chunk_overlap" in payload and payload["chunk_overlap"] is not None:
        try:
            overlap = int(payload["chunk_overlap"])
        except (TypeError, ValueError):
            errors.append(
                _issue(
                    "invalid_settings",
                    "chunk_overlap must be an integer.",
                    field_name="chunk_overlap",
                )
            )
        else:
            if overlap < 0:
                errors.append(
                    _issue(
                        "invalid_settings",
                        "chunk_overlap must be >= 0.",
                        field_name="chunk_overlap",
                    )
                )
            chunk_size_val = payload.get("chunk_size")
            try:
                if chunk_size_val is not None and overlap >= int(chunk_size_val):
                    errors.append(
                        _issue(
                            "invalid_settings",
                            "chunk_overlap must be less than chunk_size.",
                            field_name="chunk_overlap",
                        )
                    )
            except (TypeError, ValueError):
                pass

    mode = payload.get("retrieval_search_mode")
    if mode is not None and str(mode).strip():
        normalized = str(mode).strip().lower()
        if normalized not in ALLOWED_RETRIEVAL_SEARCH_MODES:
            errors.append(
                _issue(
                    "invalid_settings",
                    f"retrieval_search_mode must be one of {sorted(ALLOWED_RETRIEVAL_SEARCH_MODES)}.",
                    field_name="retrieval_search_mode",
                    details={"value": mode},
                )
            )

    if "reranking" in payload and payload["reranking"] is not None:
        if not isinstance(payload["reranking"], bool):
            errors.append(
                _issue(
                    "invalid_settings",
                    "reranking must be a boolean.",
                    field_name="reranking",
                )
            )

    # Illegal combination: key-field extraction enabled without fields
    extraction_enabled = payload.get("key_field_extraction_enabled")
    if extraction_enabled is None:
        extraction_enabled = payload.get("key_field_extraction")
    if extraction_enabled is True:
        key_fields = payload.get("key_fields")
        if not isinstance(key_fields, list) or len(key_fields) == 0:
            # Only enforce when key_fields key is present or activation requires completeness
            if "key_fields" in payload or require_activation_fields:
                errors.append(
                    _issue(
                        "invalid_settings",
                        "key_fields cannot be empty when key_field_extraction_enabled is true.",
                        field_name="key_fields",
                    )
                )

    if require_activation_fields and embedding is None and "embedding_model" in payload:
        errors.append(
            _issue(
                "invalid_settings",
                "embedding_model cannot be null when provided.",
                field_name="embedding_model",
            )
        )

    return RepositoryValidationResult.ok() if not errors else RepositoryValidationResult.failed(errors)


def assert_settings_business(
    settings: dict[str, Any] | None,
    *,
    require_activation_fields: bool = False,
) -> None:
    from src.features.repositories.domain.repository_exceptions import InvalidSettingsError

    result = validate_settings_business(
        settings,
        require_activation_fields=require_activation_fields,
    )
    if result.is_valid:
        return
    issue = result.errors[0]
    raise InvalidSettingsError(issue.message, details=issue.details)
