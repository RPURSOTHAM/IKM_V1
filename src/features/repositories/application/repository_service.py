from __future__ import annotations

import logging
import uuid
from typing import Any

_logger = logging.getLogger(__name__)

from src.features.repositories.infrastructure.collection_naming import collection_name_from_repository_name
from src.features.repositories.configuration.embedding_model_catalog import build_embedding_model_catalog
from src.features.repositories.configuration.chunking_strategy_catalog import build_chunking_strategy_catalog
from src.features.repositories.domain.repository_exceptions import (
    ConstraintViolation,
    DatabaseUnavailable,
    DuplicateNameError,
    DuplicateRepository,
    ImmutableRepositorySetting,
    InvalidRepositoryStatusTransition,
    InvalidSettingsError,
    NotFoundError,
    PersistenceError,
    RepositoryAlreadyExists,
    RepositoryInactiveError,
    RepositoryError,
    RepositoryInUseError,
    RepositoryNotFound,
    ServiceUnavailableError,
    SettingsLockedError,
    ValidationError,
)
from src.features.repositories.domain.repository_policy import (
    assert_activation,
    assert_delete_allowed,
    assert_settings_business,
    assert_status_transition,
    assert_update_patch,
    names_conflict_case_insensitive,
    normalize_repository_name,
)
from src.features.repositories.domain.repository import RepositoryRecord, repository_to_dict
from src.features.repositories.application.owner_profiles import (
    is_raw_user_identifier,
    load_platform_display_names,
    resolve_owner_user_name,
)
from src.features.repositories.application.repository_settings_service import (
    merge_settings,
    merge_settings_lenient,
    patch_settings,
    resolve_for_repository,
    resolve_repository_context,
    validate_embedding_model,
)
from src.features.repositories.infrastructure.repository_repository import RepositoryStore, get_repository_store
from src.features.observability.audit.application.business_audit import audit_business_change


class RepositoryService:
    """Domain facade for FRD repository entities and settings (MySQL-backed)."""

    def __init__(self, store: RepositoryStore | None = None) -> None:
        self._store = store

    @staticmethod
    def _audit_repository_change(
        event_type: str,
        *,
        repository_id: str,
        old: dict[str, Any],
        new: dict[str, Any],
        action: str,
        repository_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        audit_business_change(
            event_type,
            entity_type="repository",
            entity_id=repository_id,
            old=old,
            new=new,
            category="repository",
            action=action,
            source="API",
            repository_name=repository_name,
            metadata={**(metadata or {}), "repository_id": repository_id},
        )

    def _require_store(self) -> RepositoryStore:
        try:
            store = self._store or get_repository_store()
        except Exception as exc:
            raise ServiceUnavailableError(
                "Repository Service persistence layer is unavailable.",
                details={
                    "hint": "Configure DOCUMENT_JOBS_POSTGRES_* / POSTGRES_* (or legacy DOCUMENT_JOBS_MYSQL_* / REPOSITORY_MYSQL_*) environment variables.",
                    "error": str(exc),
                },
            ) from exc
        if store is None:
            raise ServiceUnavailableError(
                "Repository Service persistence layer is unavailable.",
                details={"hint": "Configure DOCUMENT_JOBS_POSTGRES_* / POSTGRES_* (or legacy DOCUMENT_JOBS_MYSQL_* / REPOSITORY_MYSQL_*) environment variables."},
            )
        if not store.ping():
            raise ServiceUnavailableError(
                "Repository Service persistence layer is unavailable.",
                details={"hint": "PostgreSQL is not reachable; start the postgres service or fix DOCUMENT_JOBS_POSTGRES_* / POSTGRES_* settings."},
            )
        return store

    def initialize(self) -> None:
        self._require_store().ensure_schema()

    def _repository_payload(
        self,
        record: RepositoryRecord,
        *,
        settings: dict[str, Any] | None = None,
        include_document_count: bool = False,
        include_embedding_model_options: bool = False,
        platform_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        document_count = None
        if include_document_count:
            document_count = self._require_store().count_documents(record.repository_id)
        owner_user_name = resolve_owner_user_name(
            record.owner_user_id,
            stored_name=record.owner_user_name,
            platform_names=platform_names,
        )
        payload = repository_to_dict(
            record,
            settings=settings,
            document_count=document_count,
            owner_user_name=owner_user_name,
        )
        if include_embedding_model_options:
            payload["embedding_model_options"] = build_embedding_model_catalog()
        return payload

    @staticmethod
    def _resolve_owner_user_name_for_create(payload: dict[str, Any], owner_user_id: str) -> str | None:
        explicit = str(payload.get("owner_user_name") or "").strip()
        if explicit and explicit.lower() != "string":
            return explicit
        platform_names = load_platform_display_names([owner_user_id])
        return resolve_owner_user_name(owner_user_id, platform_names=platform_names)

    def _ensure_persisted_owner_user_names(
        self,
        store: RepositoryStore,
        records: list[RepositoryRecord],
    ) -> None:
        owner_ids: set[str] = set()
        for record in records:
            stored = str(record.owner_user_name or "").strip()
            if not stored or is_raw_user_identifier(stored, record.owner_user_id):
                owner_id = str(record.owner_user_id or "").strip()
                if owner_id:
                    owner_ids.add(owner_id)
        if not owner_ids:
            return

        platform_names = load_platform_display_names(owner_ids)
        if not platform_names:
            return

        for record in records:
            stored = str(record.owner_user_name or "").strip()
            if stored and not is_raw_user_identifier(stored, record.owner_user_id):
                continue
            name = str(platform_names.get(record.owner_user_id) or "").strip()
            if not name:
                continue
            store.update_owner_user_name(record.repository_id, name)
            record.owner_user_name = name

    def get_embedding_model_catalog(self) -> dict[str, Any]:
        return build_embedding_model_catalog()

    def get_chunking_strategy_catalog(self) -> dict[str, Any]:
        return build_chunking_strategy_catalog()

    @staticmethod
    def _non_status_patch_keys(patch: dict[str, Any]) -> set[str]:
        return {key for key in patch if key not in {"status"}}

    @staticmethod
    def _settings_are_mutable(record: RepositoryRecord) -> bool:
        return record.status != "active" and record.settings_locked_at is None

    @staticmethod
    def _is_key_field_disable_recovery_patch(settings_patch: dict[str, Any]) -> bool:
        if not settings_patch:
            return False
        allowed = {"key_field_extraction", "key_field_extraction_enabled"}
        keys = set(settings_patch.keys())
        if not keys.issubset(allowed):
            return False
        values = [
            settings_patch.get("key_field_extraction"),
            settings_patch.get("key_field_extraction_enabled"),
        ]
        present = [value for value in values if value is not None]
        if not present:
            return False
        return all(bool(value) is False for value in present)

    @staticmethod
    def _is_document_type_default_patch(settings_patch: dict[str, Any]) -> bool:
        if not settings_patch:
            return False
        if set(settings_patch.keys()) != {"document_type_id"}:
            return False
        return bool(str(settings_patch.get("document_type_id") or "").strip())

    def _assert_repository_document_type(self, repository_id: str, document_type_id: str) -> None:
        from src.features.document_types.domain.document_type_exceptions import DocumentTypeError
        from src.features.document_types.application.document_type_service import get_document_type_service

        try:
            dt_service = get_document_type_service()
            dt_service.initialize()
            dt_service.get_type(document_type_id, repository_id=repository_id)
        except DocumentTypeError as exc:
            raise ValidationError(
                exc.message,
                details={"repository_id": repository_id, "document_type_id": document_type_id, **(exc.details or {})},
            ) from exc

    @staticmethod
    def _is_swagger_placeholder(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            token = value.strip().lower()
            return token in {"", "string", "null"}
        return False

    def _sanitize_swagger_placeholders(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, raw in value.items():
                clean = self._sanitize_swagger_placeholders(raw)
                if clean is None:
                    continue
                if isinstance(clean, (dict, list)) and len(clean) == 0:
                    continue
                sanitized[key] = clean
            return sanitized
        if isinstance(value, list):
            out: list[Any] = []
            for item in value:
                clean = self._sanitize_swagger_placeholders(item)
                if clean is None:
                    continue
                if isinstance(clean, (dict, list)) and len(clean) == 0:
                    continue
                out.append(clean)
            return out
        if self._is_swagger_placeholder(value):
            return None
        return value

    def _resolve_owner_for_create(self, payload: dict[str, Any]) -> str:
        owner = str(payload.get("owner_user_id") or "").strip()
        if owner and owner.lower() != "string":
            return owner
        try:
            from src.application.consumer_api.context import get_current_user_from_context

            actor = get_current_user_from_context()
            if actor and actor.user_id and actor.user_id != "anonymous":
                return str(actor.user_id).strip()
        except Exception:
            pass
        return "system"

    def _normalize_create_settings(self, payload_settings: dict[str, Any] | None) -> dict[str, Any]:
        raw_settings = self._sanitize_swagger_placeholders(dict(payload_settings or {})) or {}
        embedding = raw_settings.get("embedding_model")
        if isinstance(embedding, dict):
            model_id = str(embedding.get("model_id") or "").strip()
            local_dir = str(embedding.get("local_model_dir") or "").strip()
            provider = str(embedding.get("provider") or "").strip().lower()
            if not model_id and local_dir:
                embedding["model_id"] = local_dir
                model_id = local_dir
            # Swagger examples can leave only provider=local after placeholders are stripped.
            # Treat this as unset so repository creation uses platform defaults.
            if not model_id and provider in {"", "local"}:
                raw_settings.pop("embedding_model", None)
            elif not model_id and not local_dir and not provider:
                raw_settings.pop("embedding_model", None)
            else:
                raw_settings["embedding_model"] = embedding
        elif embedding is not None:
            raw_settings.pop("embedding_model", None)
        return raw_settings

    def _assert_name_available(
        self,
        store: RepositoryStore,
        name: str,
        *,
        exclude_repository_id: str | None = None,
    ) -> None:
        """Case-insensitive uniqueness check (Service orchestration; Store stays dumb)."""
        for existing in store.list_repositories():
            if exclude_repository_id and existing.repository_id == exclude_repository_id:
                continue
            if names_conflict_case_insensitive(existing.name, name):
                raise RepositoryAlreadyExists(
                    "A repository with this name already exists.",
                    details={"name": name, "conflict_repository_id": existing.repository_id},
                )

    def _load_record(self, repository_id: str) -> RepositoryRecord:
        """Load repository or raise business NotFoundError (maps persistence gaps)."""
        store = self._require_store()
        try:
            record = store.get_by_id(repository_id)
        except DatabaseUnavailable as exc:
            raise ServiceUnavailableError(
                "Repository Service persistence layer is unavailable.",
                details={"error": str(exc), "repository_id": repository_id},
            ) from exc
        if record is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})
        return record

    def _map_persistence_error(self, exc: PersistenceError, *, repository_id: str | None = None) -> RepositoryError:
        """Translate Store persistence exceptions into business/facade exceptions."""
        if isinstance(exc, RepositoryNotFound):
            return NotFoundError(
                "Repository not found.",
                details={"repository_id": repository_id, **(exc.details or {})},
            )
        if isinstance(exc, DuplicateRepository):
            return RepositoryAlreadyExists(
                "A repository with this name already exists.",
                details=dict(exc.details or {}),
            )
        if isinstance(exc, DatabaseUnavailable):
            return ServiceUnavailableError(
                "Repository Service persistence layer is unavailable.",
                details={"error": str(exc), **(exc.details or {})},
            )
        if isinstance(exc, ConstraintViolation):
            return ValidationError(str(exc), details=dict(exc.details or {}))
        return ServiceUnavailableError(
            "Repository persistence operation failed.",
            details={"error": str(exc), **(getattr(exc, "details", None) or {})},
        )

    def _settings_view_for_record(self, record: RepositoryRecord, stored: dict[str, Any] | None) -> dict[str, Any]:
        stored = stored or {}
        if record.status == "active" or record.settings_locked_at is not None:
            try:
                return resolve_for_repository(stored)
            except Exception:
                return merge_settings_lenient(stored)
        return merge_settings_lenient(stored)

    def _persist_status(self, repository_id: str, target_status: str) -> RepositoryRecord:
        store = self._require_store()
        try:
            return store.update_repository_status(repository_id, target_status)
        except PersistenceError as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc

    def create_repository(self, payload: dict[str, Any]) -> dict[str, Any]:
        store = self._require_store()
        name_raw = payload.get("name")
        owner = self._resolve_owner_for_create(payload)
        if payload.get("weaviate_collection"):
            raise ValidationError(
                "weaviate_collection is derived from repository name and cannot be set explicitly.",
            )
        name = normalize_repository_name(str(name_raw or ""))
        self._assert_name_available(store, name)

        try:
            weaviate_collection = collection_name_from_repository_name(name)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        if store.get_by_collection(weaviate_collection):
            raise RepositoryAlreadyExists(
                "A repository with this name already exists.",
                details={"name": name, "weaviate_collection": weaviate_collection},
            )

        raw_settings = self._normalize_create_settings(payload.get("settings") or {})
        if payload.get("key_field_extraction_enabled") is not None:
            raw_settings["key_field_extraction_enabled"] = bool(payload.get("key_field_extraction_enabled"))
        if payload.get("key_fields") is not None:
            raw_settings["key_fields"] = payload.get("key_fields")
        assert_settings_business(raw_settings, require_activation_fields=False)
        try:
            patch_settings(
                merge_settings_lenient({}),
                {
                    "key_field_extraction": raw_settings.get("key_field_extraction"),
                    "key_field_extraction_enabled": raw_settings.get("key_field_extraction_enabled"),
                    "key_fields": raw_settings.get("key_fields"),
                },
            )
        except RepositoryError:
            raise
        except Exception as exc:
            raise InvalidSettingsError(str(exc)) from exc
        try:
            embedding = raw_settings.get("embedding_model")
            if embedding is not None:
                raw_settings["embedding_model"] = validate_embedding_model(embedding, strict_local=True)
        except RepositoryError:
            raise
        except Exception as exc:
            raise InvalidSettingsError(str(exc)) from exc

        default_tenant = payload.get("default_tenant_id")
        if default_tenant is not None:
            default_tenant = str(default_tenant).strip() or None

        repository_id = str(uuid.uuid4())
        basic_type_id: str | None = None
        try:
            from src.features.document_types.application.document_type_service import get_document_type_service

            dt_service = get_document_type_service()
            dt_service.initialize()
            basic_type_id = dt_service.bootstrap_repository(repository_id, created_by=owner)
        except Exception as exc:
            raise ServiceUnavailableError(
                "Repository creation failed while provisioning the Basic document type.",
                details={"repository_id": repository_id, "error": str(exc)},
            ) from exc

        if basic_type_id:
            raw_settings["document_type_id"] = basic_type_id

        try:
            record, _persisted_settings = store.create_repository_with_settings(
                repository_id=repository_id,
                name=name,
                owner_user_id=owner,
                owner_user_name=self._resolve_owner_user_name_for_create(payload, owner),
                weaviate_collection=weaviate_collection,
                default_tenant_id=default_tenant,
                status=str(payload.get("status") or "configuring").strip(),
                settings=raw_settings,
            )
        except DuplicateRepository as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        except DatabaseUnavailable as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        except PersistenceError as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc

        try:
            from src.features.users.infrastructure.user_repository import get_platform_security_store

            sec_store = get_platform_security_store()
            if sec_store is not None:
                sec_store.grant_repository_role(
                    repository_id=record.repository_id,
                    user_id=owner,
                    role="owner",
                    granted_by=owner,
                )
        except Exception:
            pass

        try:
            resolved_settings = resolve_for_repository(raw_settings)
        except Exception:
            # Do not fail repository creation if runtime env defaults are temporarily
            # misconfigured; persist user settings and return a lenient merged view.
            resolved_settings = merge_settings_lenient(raw_settings)
        self._audit_repository_change(
            "REPOSITORY_CREATED",
            repository_id=record.repository_id,
            old={},
            new={
                "name": record.name,
                "status": record.status,
                "owner_user_id": record.owner_user_id,
                "default_tenant_id": record.default_tenant_id,
                "weaviate_collection": record.weaviate_collection,
            },
            action="create",
            repository_name=record.name,
        )
        return self._repository_payload(
            record,
            settings=resolved_settings,
            include_document_count=True,
            include_embedding_model_options=True,
        )

    def list_repositories(
        self,
        *,
        owner_user_id: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        try:
            records = store.list_repositories(owner_user_id=owner_user_id, status=status)
        except DatabaseUnavailable as exc:
            raise self._map_persistence_error(exc) from exc
        platform_names = load_platform_display_names({record.owner_user_id for record in records})
        self._ensure_persisted_owner_user_names(store, records)
        items = [
            self._repository_payload(
                record,
                include_document_count=True,
                platform_names=platform_names,
            )
            for record in records
        ]
        return {"repositories": items, "count": len(items)}

    def get_repository(self, repository_id: str) -> dict[str, Any]:
        store = self._require_store()
        record = self._load_record(repository_id)
        try:
            stored = store.get_settings(repository_id)
        except DatabaseUnavailable as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        # Read paths must not fail on incomplete/invalid draft settings; strict validation
        # applies on activate, settings PATCH, and runtime resolve_settings().
        resolved = merge_settings_lenient(stored)
        platform_names = load_platform_display_names([record.owner_user_id])
        self._ensure_persisted_owner_user_names(store, [record])
        return self._repository_payload(
            record,
            settings=resolved,
            include_document_count=True,
            include_embedding_model_options=True,
            platform_names=platform_names,
        )

    def get_settings(self, repository_id: str) -> dict[str, Any]:
        store = self._require_store()
        record = self._load_record(repository_id)
        stored = store.get_settings(repository_id)
        settings = self._settings_view_for_record(record, stored)
        return {"repository_id": repository_id, "settings": settings}

    def update_owner_user_name(self, repository_id: str, owner_user_name: str) -> dict[str, Any]:
        return self.update_repository(repository_id, {"owner_user_name": owner_user_name})

    def update_repository(self, repository_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Update editable repository fields (and optional nested settings).

        Lifecycle status changes must use activate_repository / archive_repository /
        reactivate_repository. Immutable identity fields are rejected.
        """
        store = self._require_store()
        record = self._load_record(repository_id)
        patch = dict(patch or {})

        immutable_hits = sorted(
            key
            for key in ("repository_id", "name", "weaviate_collection", "owner_user_id", "created_at")
            if key in patch
        )
        if immutable_hits:
            raise ImmutableRepositorySetting(
                f"Field '{immutable_hits[0]}' is immutable.",
                details={"fields": immutable_hits, "repository_id": repository_id},
            )

        if "status" in patch:
            raise InvalidRepositoryStatusTransition(
                "Use activate_repository, archive_repository, or reactivate_repository for status changes.",
                details={
                    "repository_id": repository_id,
                    "current_status": record.status,
                    "requested_status": patch.get("status"),
                },
            )

        settings_nested = patch.pop("settings", None)
        owner_user_name = patch.pop("owner_user_name", ...)
        default_tenant_id = patch.pop("default_tenant_id", ...)

        unknown = sorted(patch.keys())
        if unknown:
            raise ValidationError(
                f"Unsupported repository update fields: {', '.join(unknown)}.",
                details={"fields": unknown, "repository_id": repository_id},
            )

        before = {
            "owner_user_name": record.owner_user_name,
            "default_tenant_id": record.default_tenant_id,
        }
        after = dict(before)

        if owner_user_name is not ...:
            name = str(owner_user_name or "").strip()
            if not name:
                raise ValidationError("owner_user_name is required.")
            if is_raw_user_identifier(name, record.owner_user_id):
                raise ValidationError(
                    "owner_user_name must be a display name, not a user id.",
                    details={"repository_id": repository_id, "owner_user_id": record.owner_user_id},
                )
            try:
                store.update_owner_user_name(repository_id, name)
            except PersistenceError as exc:
                raise self._map_persistence_error(exc, repository_id=repository_id) from exc
            after["owner_user_name"] = name

        if default_tenant_id is not ...:
            tenant = None if default_tenant_id is None else (str(default_tenant_id).strip() or None)
            try:
                store.update_repository(repository_id, default_tenant_id=tenant)
            except PersistenceError as exc:
                raise self._map_persistence_error(exc, repository_id=repository_id) from exc
            after["default_tenant_id"] = tenant

        if settings_nested is not None:
            if not isinstance(settings_nested, dict):
                raise ValidationError("settings must be an object.")
            if before != after:
                self._audit_repository_change(
                    "REPOSITORY_UPDATED",
                    repository_id=repository_id,
                    old=before,
                    new=after,
                    action="update",
                    repository_name=record.name,
                )
            return self.update_repository_settings(repository_id, settings_nested)

        if before != after:
            self._audit_repository_change(
                "REPOSITORY_UPDATED",
                repository_id=repository_id,
                old=before,
                new=after,
                action="update",
                repository_name=record.name,
            )
        return self.get_repository(repository_id)

    def update_repository_settings(self, repository_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist RepositorySettings only (no lifecycle status changes)."""
        patch = dict(patch or {})
        if "status" in patch:
            raise InvalidRepositoryStatusTransition(
                "Use activate_repository, archive_repository, or reactivate_repository for status changes.",
                details={"repository_id": repository_id, "requested_status": patch.get("status")},
            )
        return self.update_settings(repository_id, patch)

    def activate_repository(self, repository_id: str) -> dict[str, Any]:
        """Transition repository to ACTIVE after business activation checks. No Weaviate provision."""
        store = self._require_store()
        record = self._load_record(repository_id)
        if record.status == "active":
            return self.get_repository(repository_id)

        previous_status = record.status
        stored = store.get_settings(repository_id)
        settings = self._settings_view_for_record(record, stored)
        assert_activation(record, settings)
        assert_status_transition(record.status, "active")

        self._persist_status(repository_id, "active")
        try:
            store.lock_settings(repository_id)
        except PersistenceError as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        self._audit_repository_change(
            "REPOSITORY_ACTIVATED",
            repository_id=repository_id,
            old={"status": previous_status},
            new={"status": "active"},
            action="activate",
            repository_name=record.name,
        )
        return self.get_repository(repository_id)

    def archive_repository(self, repository_id: str) -> dict[str, Any]:
        """Soft-retire repository: ACTIVE → ARCHIVED (Phase 2)."""
        store = self._require_store()
        record = self._load_record(repository_id)
        if record.status == "archived":
            return self.get_repository(repository_id)

        previous_status = record.status
        assert_status_transition(record.status, "archived")
        try:
            store.archive_repository(repository_id)
        except PersistenceError as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        self._audit_repository_change(
            "REPOSITORY_STATUS_CHANGED",
            repository_id=repository_id,
            old={"status": previous_status},
            new={"status": "archived"},
            action="archive",
            repository_name=record.name,
        )
        return self.get_repository(repository_id)

    def reactivate_repository(self, repository_id: str) -> dict[str, Any]:
        """Reactivate soft-retired repository: ARCHIVED → ACTIVE."""
        record = self._load_record(repository_id)
        if record.status != "archived":
            raise InvalidRepositoryStatusTransition(
                f"Status transition '{record.status}' → 'active' via reactivate requires archived.",
                details={
                    "from": record.status,
                    "to": "active",
                    "operation": "reactivate_repository",
                },
            )
        return self.activate_repository(repository_id)

    def update_settings(self, repository_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        store = self._require_store()
        record = self._load_record(repository_id)

        settings_patch = {k: v for k, v in patch.items() if k != "status"}
        document_type_default_patch = self._is_document_type_default_patch(settings_patch)
        key_field_disable_recovery = self._is_key_field_disable_recovery_patch(settings_patch)
        assert_update_patch(
            record,
            patch,
            allow_document_type_default=document_type_default_patch,
            allow_key_field_disable_recovery=key_field_disable_recovery,
        )

        if key_field_disable_recovery:
            stored = dict(store.get_settings(repository_id) or {})
            before = {
                "key_field_extraction": stored.get("key_field_extraction"),
                "key_field_extraction_enabled": stored.get("key_field_extraction_enabled"),
            }
            stored["key_field_extraction"] = False
            stored["key_field_extraction_enabled"] = False
            try:
                store.upsert_settings(repository_id, stored)
            except PersistenceError as exc:
                raise self._map_persistence_error(exc, repository_id=repository_id) from exc
            self._audit_repository_change(
                "REPOSITORY_SETTINGS_UPDATED",
                repository_id=repository_id,
                old=before,
                new={
                    "key_field_extraction": False,
                    "key_field_extraction_enabled": False,
                },
                action="update",
                repository_name=record.name,
            )
            platform_names = load_platform_display_names([record.owner_user_id])
            return self._repository_payload(
                record,
                settings=merge_settings_lenient(stored),
                include_document_count=True,
                include_embedding_model_options=True,
                platform_names=platform_names,
            )

        if document_type_default_patch:
            self._assert_repository_document_type(
                repository_id,
                str(settings_patch["document_type_id"]).strip(),
            )

        stored = store.get_settings(repository_id)
        current = self._settings_view_for_record(record, stored)
        try:
            updated = patch_settings(current, settings_patch)
        except RepositoryError:
            raise
        except Exception as exc:
            raise InvalidSettingsError(str(exc)) from exc

        target_status = patch.get("status")
        activating = target_status == "active" and record.status != "active"

        if activating:
            assert_activation(record, updated)
            assert_status_transition(record.status, "active")

        try:
            store.upsert_settings(repository_id, updated)
        except PersistenceError as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc

        self._audit_repository_change(
            "REPOSITORY_SETTINGS_UPDATED",
            repository_id=repository_id,
            old=current,
            new=updated,
            action="update",
            repository_name=record.name,
        )

        if target_status is not None and str(target_status) in {"active", "archived", "configuring", "approved"}:
            status_value = str(target_status)
            if status_value == "active":
                return self.activate_repository(repository_id)
            if status_value == "archived":
                return self.archive_repository(repository_id)
            if status_value != record.status:
                assert_status_transition(record.status, status_value)
                previous_status = record.status
                self._persist_status(repository_id, status_value)
                self._audit_repository_change(
                    "REPOSITORY_STATUS_CHANGED",
                    repository_id=repository_id,
                    old={"status": previous_status},
                    new={"status": status_value},
                    action="update",
                    repository_name=record.name,
                )

        return self.get_repository(repository_id)

    def replace_settings(self, repository_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        store = self._require_store()
        record = self._load_record(repository_id)
        assert_update_patch(record, payload)
        assert_settings_business(payload, require_activation_fields=False)
        before = self._settings_view_for_record(record, store.get_settings(repository_id))
        try:
            updated = merge_settings(payload)
        except RepositoryError:
            raise
        except Exception as exc:
            raise InvalidSettingsError(str(exc)) from exc
        try:
            store.upsert_settings(repository_id, updated)
        except PersistenceError as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        self._audit_repository_change(
            "REPOSITORY_SETTINGS_UPDATED",
            repository_id=repository_id,
            old=before,
            new=updated,
            action="update",
            repository_name=record.name,
        )
        platform_names = load_platform_display_names([record.owner_user_id])
        return self._repository_payload(
            record,
            settings=updated,
            include_document_count=True,
            include_embedding_model_options=True,
            platform_names=platform_names,
        )

    def list_repository_key_fields(self, repository_id: str) -> dict[str, Any]:
        settings_payload = self.get_settings(repository_id)
        settings = settings_payload.get("settings") or {}
        fields = list(settings.get("key_fields") or [])
        return {"repository_id": repository_id, "key_fields": fields, "count": len(fields)}

    def create_repository_key_field(self, repository_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_settings(repository_id).get("settings") or {}
        key_fields = list(current.get("key_fields") or [])
        field_name = str(payload.get("name") or "").strip()
        if not field_name:
            raise ValidationError("name is required.")
        for existing in key_fields:
            if str(existing.get("name") or "").strip().lower() == field_name.lower():
                raise DuplicateNameError(
                    "A key field with this name already exists.",
                    details={"repository_id": repository_id, "name": field_name},
                )
        item = {
            "field_id": str(uuid.uuid4()),
            "name": field_name,
            "type": str(payload.get("type") or "string").strip().lower() or "string",
            "required": bool(payload.get("required", False)),
            "description": str(payload.get("description") or "").strip() or None,
        }
        key_fields.append(item)
        return self.update_settings(repository_id, {"key_fields": key_fields})

    def patch_repository_key_fields(self, repository_id: str, key_fields: list[dict[str, Any]]) -> dict[str, Any]:
        store = self._require_store()
        record = store.get_by_id(repository_id)
        if record is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})
        if not self._settings_are_mutable(record):
            raise SettingsLockedError(
                "Repository settings are read-only after activation.",
                details={"repository_id": repository_id, "status": record.status},
            )
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in key_fields:
            if not isinstance(raw, dict):
                raise ValidationError("Each key field must be an object.")
            field_name = str(raw.get("name") or "").strip()
            if not field_name:
                raise ValidationError("Each key field requires a non-empty name.")
            lowered = field_name.lower()
            if lowered in seen:
                raise DuplicateNameError(
                    "A key field with this name already exists.",
                    details={"repository_id": repository_id, "name": field_name},
                )
            seen.add(lowered)
            normalized.append(
                {
                    "field_id": str(raw.get("field_id") or "").strip() or str(uuid.uuid4()),
                    "name": field_name,
                    "type": str(raw.get("type") or "string").strip().lower() or "string",
                    "required": bool(raw.get("required", False)),
                    "description": str(raw.get("description") or "").strip() or None,
                }
            )
        stored = dict(store.get_settings(repository_id) or {})
        before_fields = list(stored.get("key_fields") or [])
        stored["key_fields"] = normalized
        store.upsert_settings(repository_id, stored)
        self._audit_repository_change(
            "REPOSITORY_SETTINGS_UPDATED",
            repository_id=repository_id,
            old={"key_fields": before_fields},
            new={"key_fields": normalized},
            action="update",
            repository_name=record.name,
        )
        return {
            "repository_id": repository_id,
            "key_fields": normalized,
            "count": len(normalized),
        }

    def delete_repository_key_field(self, repository_id: str, field_id: str) -> dict[str, Any]:
        current = self.get_settings(repository_id).get("settings") or {}
        key_fields = list(current.get("key_fields") or [])
        retained = [item for item in key_fields if str(item.get("field_id") or "") != field_id]
        if len(retained) == len(key_fields):
            raise NotFoundError(
                "Key field not found.",
                details={"repository_id": repository_id, "field_id": field_id},
            )
        return self.update_settings(repository_id, {"key_fields": retained})

    def delete_repository(self, repository_id: str) -> dict[str, Any]:
        store = self._require_store()
        record = store.get_by_id(repository_id)
        try:
            doc_count = store.count_documents(repository_id) if record is not None else 0
        except DatabaseUnavailable as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        assert_delete_allowed(record, document_count=doc_count)
        assert record is not None  # narrowed by assert_delete_allowed
        try:
            from src.features.document_types.application.document_type_service import get_document_type_service

            get_document_type_service().delete_repository_document_types(repository_id)
        except Exception:
            pass
        collection_name = record.weaviate_collection
        if collection_name:
            try:
                from src.features.repositories.infrastructure import weaviate_admin
                from src.features.repositories.infrastructure.weaviate_backend import backend

                if backend.ready:
                    client = backend.require_client()
                    if client.collections.exists(collection_name):
                        weaviate_admin.delete_collection(collection_name)
            except Exception as exc:
                _logger.error(
                    "repository_delete_weaviate_failed repository_id=%s collection=%s error=%s",
                    repository_id,
                    collection_name,
                    exc,
                    exc_info=exc,
                )
                raise ServiceUnavailableError(
                    "Repository search index could not be removed. Deletion was not completed.",
                    details={
                        "repository_id": repository_id,
                        "collection_name": collection_name,
                        "reason": str(exc),
                    },
                ) from exc
        try:
            deleted = store.delete_repository(repository_id)
        except PersistenceError as exc:
            raise self._map_persistence_error(exc, repository_id=repository_id) from exc
        if not deleted:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})
        self._audit_repository_change(
            "REPOSITORY_DELETED",
            repository_id=repository_id,
            old={
                "name": record.name,
                "status": record.status,
                "owner_user_id": record.owner_user_id,
                "weaviate_collection": collection_name,
            },
            new={},
            action="delete",
            repository_name=record.name,
        )
        return {
            "repository_id": repository_id,
            "name": record.name,
            "weaviate_collection": collection_name,
            "deleted": True,
        }

    def resolve_settings(self, repository_id: str) -> dict[str, Any]:
        """Runtime resolution for processor, retrieval, and document receiver.

        Delegates to SettingsResolver (Phase 3.5) — single source of truth for
        effective settings. Shape unchanged for existing callers.
        """
        return resolve_repository_context(repository_id, store=self._require_store())

    def validate_repository_active(self, repository_id: str) -> dict[str, Any]:
        resolved = self.resolve_settings(repository_id)
        if resolved["status"] != "active":
            raise RepositoryInactiveError(
                "Repository is not active.",
                details={"repository_id": repository_id, "status": resolved["status"]},
            )
        return resolved

    def get_repository_id_by_name(self, name: str) -> str | None:
        """Resolve a repository name to its ID."""
        store = self._require_store()
        record = store.get_by_name(name)
        return str(record.repository_id) if record else None

    def get_repository_id_for_document(self, document_id: str) -> str | None:
        """Resolve a document ID to its linked repository ID."""
        return self._require_store().get_repository_id_for_document(document_id)

    def is_document_linked_to_repository(self, repository_id: str, document_id: str) -> bool:
        """Check if a document is linked to a repository."""
        store = self._require_store()
        return document_id in store.list_document_ids(repository_id, limit=5000)

    def unlink_document_global(self, document_id: str) -> bool:
        """Remove document link from any repository it is linked to."""
        return bool(self._require_store().unlink_document(document_id))

    def link_document(self, document_id: str, repository_id: str) -> None:
        self.validate_repository_active(repository_id)
        self._require_store().link_document(document_id, repository_id)

    def list_documents(self, repository_id: str, *, limit: int = 500) -> dict[str, Any]:
        store = self._require_store()
        record = store.get_by_id(repository_id)
        if record is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})
        document_ids = store.list_document_ids(repository_id, limit=limit)
        documents = self._repository_document_entries(document_ids)
        return {
            "repository_id": repository_id,
            "document_count": store.count_documents(repository_id),
            "documents": documents,
            "returned": len(documents),
        }

    @staticmethod
    def _repository_document_entries(document_ids: list[str]) -> list[dict[str, Any]]:
        if not document_ids:
            return []
        from src.features.documents.application.document_service import DocumentReceiverService
        from src.infrastructure.database.document_jobs import get_document_job_store

        receiver = DocumentReceiverService()
        js = get_document_job_store()
        prefetched = js.fetch_for_documents(document_ids) if js else {}
        return [
            receiver.repository_document_entry(document_id, prefetched_jobs=prefetched)
            for document_id in document_ids
        ]

    def delete_repository_document(
        self,
        repository_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        store = self._require_store()
        record = store.get_by_id(repository_id)
        if record is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})
        if document_id not in store.list_document_ids(repository_id, limit=5000):
            raise NotFoundError(
                "Document is not linked to this repository.",
                details={"repository_id": repository_id, "document_id": document_id},
            )
        from src.features.documents.application.document_service import DocumentReceiverService

        receiver = DocumentReceiverService()
        if receiver.get_store().get_document(document_id) is None:
            from src.features.documents.application.document_purge_service import purge_document_job

            store.unlink_document(document_id)
            purge_document_job(document_id)
            return {
                "repository_id": repository_id,
                "document_id": document_id,
                "deleted": True,
                "orphan_cleanup": True,
            }

        return receiver.delete_document_cascade(
            document_id,
            repository_id=repository_id,
        )

    def unlink_document(self, repository_id: str, document_id: str) -> dict[str, Any]:
        store = self._require_store()
        record = store.get_by_id(repository_id)
        if record is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})
        if document_id not in store.list_document_ids(repository_id, limit=5000):
            raise NotFoundError(
                "Document is not linked to this repository.",
                details={"repository_id": repository_id, "document_id": document_id},
            )
        if not store.unlink_document(document_id):
            raise NotFoundError(
                "Document link not found.",
                details={"repository_id": repository_id, "document_id": document_id},
            )
        return {"repository_id": repository_id, "document_id": document_id, "unlinked": True}

    def collect_document_ids_for_repository(self, repository_id: str, *, limit: int = 5000) -> list[str]:
        """Union document ids from repository links, intake metadata, and orphan jobs."""
        store = self._require_store()
        if store.get_by_id(repository_id) is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})

        cap = max(1, min(limit, 5000))
        ids: set[str] = set(store.list_document_ids(repository_id, limit=cap))

        from src.features.documents.application.document_service import DocumentReceiverService

        receiver = DocumentReceiverService()
        for row in receiver.get_store().list_documents(repository_id=repository_id, limit=cap, offset=0):
            doc_id = str(row.get("document_id") or "").strip()
            if doc_id:
                ids.add(doc_id)

        from src.infrastructure.database.document_jobs import get_document_job_store

        js = get_document_job_store()
        if js is not None:
            for job in js.list_non_terminal_jobs(limit=cap):
                doc_id = str(job.get("document_id") or "").strip()
                if not doc_id or doc_id in ids:
                    continue
                if store.get_repository_id_for_document(doc_id) == repository_id:
                    ids.add(doc_id)

        return sorted(ids)

    def purge_all_documents(
        self,
        repository_id: str,
        *,
        delete_file: bool = True,
    ) -> dict[str, Any]:
        """Remove every document artifact tied to a repository (linked, intake-only, or job-only)."""
        store = self._require_store()
        if store.get_by_id(repository_id) is None:
            raise NotFoundError("Repository not found.", details={"repository_id": repository_id})

        from src.features.documents.application.document_service import DocumentReceiverService

        receiver = DocumentReceiverService()
        deleted = 0
        failed = 0
        errors: list[dict[str, str]] = []

        for _ in range(100):
            document_ids = self.collect_document_ids_for_repository(repository_id)
            if not document_ids:
                break
            progress = False
            for document_id in document_ids:
                try:
                    receiver.purge_document_for_repository(
                        document_id,
                        repository_id,
                        delete_file=delete_file,
                    )
                    deleted += 1
                    progress = True
                except Exception as exc:
                    failed += 1
                    errors.append({"document_id": document_id, "error": str(exc)})
            if not progress:
                break

        remaining = len(self.collect_document_ids_for_repository(repository_id))
        return {
            "repository_id": repository_id,
            "deleted": deleted,
            "failed": failed,
            "remaining": remaining,
            "errors": errors[:20],
        }


_default_service: RepositoryService | None = None


def get_repository_service() -> RepositoryService:
    global _default_service
    if _default_service is None:
        _default_service = RepositoryService()
    return _default_service
