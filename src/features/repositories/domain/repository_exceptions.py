from __future__ import annotations

from typing import Any


class RepositoryError(Exception):
    """Base error for repository domain failures."""

    code: str = "repository_error"
    http_status: int = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(RepositoryError):
    code = "not_found"
    http_status = 404


class DuplicateNameError(RepositoryError):
    code = "duplicate_name"
    http_status = 409


class RepositoryInUseError(RepositoryError):
    code = "repository_in_use"
    http_status = 409


class ValidationError(RepositoryError):
    code = "invalid_request"
    http_status = 400


class ServiceUnavailableError(RepositoryError):
    code = "service_unavailable"
    http_status = 503


class InvalidSettingsError(RepositoryError):
    code = "invalid_settings"
    http_status = 422


class SettingsLockedError(RepositoryError):
    code = "settings_read_only"
    http_status = 409


class RepositoryInactiveError(RepositoryError):
    code = "repository_inactive"
    http_status = 409


# ---------------------------------------------------------------------------
# Business-validation exceptions (RepositoryService / business_validators).
# Persistencelayer exceptions remain separate below.
# ---------------------------------------------------------------------------


class InvalidRepositoryName(ValidationError):
    code = "invalid_repository_name"


class RepositoryAlreadyExists(DuplicateNameError):
    """Case-insensitive or exact name conflict detected by business validation."""

    code = "duplicate_name"


class InvalidRepositoryStatusTransition(ValidationError):
    code = "invalid_status_transition"


class RepositoryActivationFailed(ValidationError):
    code = "repository_activation_failed"
    http_status = 422


class ImmutableRepositorySetting(SettingsLockedError):
    """Update rejected because the field/setting is locked or immutable.

    Keeps HTTP code ``settings_read_only`` for API compatibility with
    SettingsLockedError while naming the business rule explicitly.
    """


class RepositoryCannotBeDeleted(RepositoryInUseError):
    code = "repository_cannot_be_deleted"


# ---------------------------------------------------------------------------
# Persistence-layer exceptions (Store only). Not business/HTTP errors.
# Service maps these to RepositoryError subtypes in later milestones if needed.
# ---------------------------------------------------------------------------


class PersistenceError(Exception):
    """Base error for MySQL / persistence failures raised by RepositoryStore."""

    code: str = "persistence_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RepositoryNotFound(PersistenceError):
    """Requested repository row does not exist."""

    code = "repository_not_found"


class DuplicateRepository(PersistenceError):
    """Unique constraint violated (name or weaviate_collection)."""

    code = "duplicate_repository"


class DatabaseUnavailable(PersistenceError):
    """MySQL connection or execution failed."""

    code = "database_unavailable"


class ConstraintViolation(PersistenceError):
    """Generic relational constraint failure."""

    code = "constraint_violation"
