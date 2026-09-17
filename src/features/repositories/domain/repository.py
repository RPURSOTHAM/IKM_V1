"""Domain models for Repository Management (Phase 3.1).

Plain domain objects only. No HTTP, FastAPI, MySQL, Weaviate, Retrieval, or Processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from src.features.repositories.domain.constants import (
    DEFAULT_REPOSITORY_STATUS,
    REPOSITORY_STATUS_VALUES,
    STATUS_ACTIVE,
    STATUS_APPROVED,
    STATUS_ARCHIVED,
    STATUS_CONFIGURING,
)

# Backward-compatible string union used by type hints across the package.
RepositoryStatusLiteral = Literal["configuring", "approved", "active", "archived"]

# Historical alias (was Literal["active", "archived"] only). Prefer RepositoryStatus enum.
RepositoryStatusValue = RepositoryStatusLiteral

_UNSET = object()


class RepositoryStatus(str, Enum):
    """Lifecycle status for a Repository (Phase 2).

    DELETED is intentionally omitted: removal is a hard-delete operation when
    document_count == 0, not a persisted status value.
    """

    CONFIGURING = STATUS_CONFIGURING
    APPROVED = STATUS_APPROVED
    ACTIVE = STATUS_ACTIVE
    ARCHIVED = STATUS_ARCHIVED

    @classmethod
    def from_value(cls, value: str | RepositoryStatus | None) -> RepositoryStatus:
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Invalid repository status '{value}'. "
                f"Allowed: {sorted(REPOSITORY_STATUS_VALUES)}"
            ) from exc

    @classmethod
    def is_valid(cls, value: str | RepositoryStatus | None) -> bool:
        if isinstance(value, cls):
            return True
        return str(value or "").strip().lower() in REPOSITORY_STATUS_VALUES


@dataclass(frozen=True)
class RepositoryOwner:
    """Owner principal for a repository (user or system id)."""

    user_id: str
    user_name: str | None = None


@dataclass
class RepositorySettings:
    """One settings document per repository (Phase 2: upsert-in-place).

    Holds the raw settings map only. Effective merge / strict validation of
    processor and retrieval knobs belongs to SettingsResolver / Service.
    """

    repository_id: str
    settings: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime | None = None

    def copy_settings(self) -> dict[str, Any]:
        return dict(self.settings or {})


@dataclass
class RepositoryValidationIssue:
    """Single model-level validation finding."""

    code: str
    message: str
    field_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepositoryValidationResult:
    """Structured outcome of model-level (or later service-level) validation."""

    is_valid: bool
    errors: list[RepositoryValidationIssue] = field(default_factory=list)
    mode: Literal["structural", "lenient", "strict"] = "structural"

    @classmethod
    def ok(cls, *, mode: Literal["structural", "lenient", "strict"] = "structural") -> RepositoryValidationResult:
        return cls(is_valid=True, errors=[], mode=mode)

    @classmethod
    def failed(
        cls,
        errors: list[RepositoryValidationIssue],
        *,
        mode: Literal["structural", "lenient", "strict"] = "structural",
    ) -> RepositoryValidationResult:
        return cls(is_valid=False, errors=list(errors), mode=mode)

    def error_codes(self) -> list[str]:
        return [issue.code for issue in self.errors]


@dataclass
class RepositoryRecord:
    """Canonical repository entity (Phase 2 functional minimum).

    Also known as Repository in the domain blueprint. Persistable shape reused
    by Store / Service in later milestones; this module defines the domain only.
    """

    repository_id: str
    name: str
    owner_user_id: str
    weaviate_collection: str
    default_tenant_id: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    settings_locked_at: datetime | None = None
    owner_user_name: str | None = None

    @property
    def settings_locked(self) -> bool:
        return self.settings_locked_at is not None

    @property
    def status_enum(self) -> RepositoryStatus:
        return RepositoryStatus.from_value(self.status)

    def owner(self) -> RepositoryOwner:
        return RepositoryOwner(user_id=self.owner_user_id, user_name=self.owner_user_name)


# Phase 2 name alias — same object, no duplicate implementation.
Repository = RepositoryRecord


def parse_repository_id(value: str | UUID) -> str:
    """Normalize a repository_id to canonical UUID string form."""
    if isinstance(value, UUID):
        return str(value)
    text = str(value or "").strip()
    parsed = UUID(text)
    return str(parsed)


def default_status() -> str:
    return DEFAULT_REPOSITORY_STATUS


def repository_to_dict(
    record: RepositoryRecord,
    *,
    settings: dict[str, Any] | None = None,
    document_count: int | None = None,
    owner_user_name: str | None | object = _UNSET,
) -> dict[str, Any]:
    if owner_user_name is _UNSET:
        resolved_owner_name = str(record.owner_user_name or "").strip() or None
    else:
        resolved_owner_name = owner_user_name

    payload: dict[str, Any] = {
        "repository_id": record.repository_id,
        "name": record.name,
        "owner_user_id": record.owner_user_id,
        "owner_user_name": resolved_owner_name,
        "weaviate_collection": record.weaviate_collection,
        "default_tenant_id": record.default_tenant_id,
        "status": record.status,
        "created_at": record.created_at.isoformat() + "Z",
        "updated_at": record.updated_at.isoformat() + "Z",
        "settings_locked": record.settings_locked_at is not None,
        "settings_locked_at": (
            record.settings_locked_at.isoformat() + "Z"
            if record.settings_locked_at is not None
            else None
        ),
    }
    if document_count is not None:
        payload["document_count"] = document_count
    if settings is not None:
        payload["settings"] = settings
    return payload
