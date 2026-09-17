from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.features.configuration.domain.config_exceptions import NotFoundError, ServiceUnavailableError, ValidationError
from src.features.configuration.application.config_provider import get_platform_config, reset_platform_config_provider
from src.features.configuration.application.schema_registry import CONFIG_REGISTRY, get_spec, resolve_env_value
from src.features.configuration.infrastructure.platform_config_repository import PlatformConfigStore, get_platform_config_store
from src.features.configuration.application.config_validator import validate_update

logger = logging.getLogger(__name__)


class PlatformConfigService:
    def __init__(self, store: PlatformConfigStore | None = None) -> None:
        self._store = store

    def _require_store(self) -> PlatformConfigStore:
        store = self._store or get_platform_config_store()
        if store is None:
            raise ServiceUnavailableError(
                "Platform Configuration Service persistence is unavailable.",
                details={"hint": "Configure DOCUMENT_JOBS_POSTGRES_HOST / POSTGRES_HOST or PLATFORM_CONFIG_POSTGRES_HOST."},
            )
        return store

    def initialize(self) -> None:
        self._require_store().ensure_schema()

    def seed_from_env(self, *, updated_by: str = "system:seed") -> dict[str, Any]:
        store = self._require_store()
        inserted = 0
        for spec in CONFIG_REGISTRY:
            if store.get_entry(spec.namespace, spec.key) is not None:
                continue
            value = resolve_env_value(spec)
            if value is None and spec.default is None:
                continue
            effective = value if value is not None else spec.default
            store.upsert_entry(
                namespace=spec.namespace,
                key=spec.key,
                value=effective,
                value_type=spec.value_type,
                reload_policy=spec.reload_policy,
                description=spec.description,
                is_sensitive=spec.is_sensitive,
                updated_by=updated_by,
                audit_action="create",
                change_comment="seed_from_env",
            )
            inserted += 1
        reset_platform_config_provider()
        get_platform_config().refresh()
        version, _ = store.get_meta()
        logger.info("Platform config seeded %d entries (config_version=%s).", inserted, version)
        return {"inserted": inserted, "config_version": version}

    def get_meta(self) -> dict[str, Any]:
        return get_platform_config().meta()

    def list_entries(self, namespace: str | None = None, *, include_sensitive: bool = False) -> dict[str, Any]:
        provider = get_platform_config()
        return {
            "config_version": provider.config_version,
            "entries": provider.list_stored_entries(namespace, include_sensitive=include_sensitive),
        }

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        provider = get_platform_config()
        values: dict[str, Any] = {}
        policies: dict[str, str] = {}
        for spec in CONFIG_REGISTRY:
            if spec.namespace != namespace:
                continue
            values[spec.key] = provider.get(namespace, spec.key, default=spec.default)
            policies[spec.key] = spec.reload_policy
        return {
            "namespace": namespace,
            "config_version": provider.config_version,
            "values": values,
            "effective_reload_policies": policies,
        }

    def get_entry(self, namespace: str, key: str, *, include_sensitive: bool = False) -> dict[str, Any]:
        store = self._require_store()
        row = store.get_entry(namespace, key)
        if row is None:
            raise NotFoundError(
                f"Configuration '{namespace}.{key}' is not stored.",
                details={"namespace": namespace, "key": key},
            )
        value = row.value
        if row.is_sensitive and not include_sensitive:
            value = "***REDACTED***"
        return {
            "namespace": row.namespace,
            "key": row.key,
            "value": value,
            "value_type": row.value_type,
            "reload_policy": row.reload_policy,
            "description": row.description,
            "is_sensitive": row.is_sensitive,
            "updated_at": row.updated_at.isoformat(),
            "updated_by": row.updated_by,
        }

    def patch(
        self,
        updates: list[dict[str, Any]],
        *,
        changed_by: str,
        request_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        pending_restart: list[str] = []
        applied = 0
        version = store.get_meta()[0]
        for item in updates:
            namespace = str(item.get("namespace") or "").strip()
            key = str(item.get("key") or "").strip()
            if not namespace or not key:
                raise ValidationError("Each update requires namespace and key.")
            spec, coerced = validate_update(namespace, key, item.get("value"))
            old = store.get_entry(namespace, key)
            old_value = old.value if old else None
            version = store.upsert_entry(
                namespace=namespace,
                key=key,
                value=coerced,
                value_type=spec.value_type,
                reload_policy=spec.reload_policy,
                description=spec.description,
                is_sensitive=spec.is_sensitive,
                updated_by=changed_by,
                audit_action="create" if old is None else "update",
                old_value=old_value,
                request_id=request_id,
                change_comment=comment,
            )
            applied += 1
            if spec.reload_policy == "cold":
                pending_restart.append(f"{namespace}.{key}")
        reset_platform_config_provider()
        get_platform_config().refresh()
        return {
            "config_version": version,
            "applied": applied,
            "pending_restart": pending_restart,
            "warnings": [],
        }

    def replace_namespace(
        self,
        namespace: str,
        values: dict[str, Any],
        *,
        preserve_keys: list[str] | None = None,
        changed_by: str,
        request_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        preserve = set(preserve_keys or [])
        store = self._require_store()
        existing = {entry.key: entry for entry in store.list_entries(namespace)}
        version = store.get_meta()[0]
        applied = 0
        pending_restart: list[str] = []

        for key, entry in existing.items():
            if key in preserve or key in values:
                continue
            version = store.delete_entry(
                namespace,
                key,
                updated_by=changed_by,
                old_value=entry.value,
                request_id=request_id,
                change_comment=comment,
            )
            applied += 1

        for key, raw in values.items():
            spec, coerced = validate_update(namespace, key, raw)
            if spec.namespace != namespace:
                raise ValidationError(f"Key '{key}' belongs to namespace '{spec.namespace}', not '{namespace}'.")
            old = store.get_entry(namespace, key)
            version = store.upsert_entry(
                namespace=namespace,
                key=key,
                value=coerced,
                value_type=spec.value_type,
                reload_policy=spec.reload_policy,
                description=spec.description,
                is_sensitive=spec.is_sensitive,
                updated_by=changed_by,
                audit_action="create" if old is None else "update",
                old_value=old.value if old else None,
                request_id=request_id,
                change_comment=comment,
            )
            applied += 1
            if spec.reload_policy == "cold":
                pending_restart.append(f"{namespace}.{key}")

        reset_platform_config_provider()
        get_platform_config().refresh()
        return {"config_version": version, "applied": applied, "pending_restart": pending_restart}

    def export_snapshot(self, *, include_sensitive: bool = False) -> dict[str, Any]:
        return get_platform_config().export_snapshot(include_sensitive=include_sensitive)

    def import_snapshot(
        self,
        payload: dict[str, Any],
        *,
        dry_run: bool = False,
        changed_by: str,
        request_id: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        namespaces = payload.get("namespaces") or {}
        if not isinstance(namespaces, dict):
            raise ValidationError("Import payload requires a 'namespaces' object.")
        planned: list[dict[str, Any]] = []
        for namespace, values in namespaces.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if value == "***REDACTED***":
                    continue
                planned.append({"namespace": namespace, "key": key, "value": value})
        if dry_run:
            return {"dry_run": True, "would_apply": len(planned)}
        return self.patch(planned, changed_by=changed_by, request_id=request_id, comment=comment or "import")

    def list_audit(
        self,
        *,
        namespace: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        store = self._require_store()
        since_dt = datetime.fromisoformat(since.replace("Z", "+00:00")) if since else None
        rows = store.list_audit(namespace=namespace, since=since_dt, limit=limit)
        return {
            "entries": [
                {
                    "audit_id": row.audit_id,
                    "namespace": row.namespace,
                    "key": row.config_key,
                    "action": row.action,
                    "old_value": row.old_value,
                    "new_value": row.new_value,
                    "changed_by": row.changed_by,
                    "changed_at": row.changed_at.isoformat(),
                    "request_id": row.request_id,
                    "change_comment": row.change_comment,
                }
                for row in rows
            ],
            "count": len(rows),
        }


_default_service: PlatformConfigService | None = None


def get_platform_config_service() -> PlatformConfigService:
    global _default_service
    if _default_service is None:
        _default_service = PlatformConfigService()
    return _default_service
