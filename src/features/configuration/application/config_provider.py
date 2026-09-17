from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from src.features.configuration.configuration.postgres_config import REFRESH_INTERVAL_SEC
from src.features.configuration.application.schema_registry import (
    CONFIG_REGISTRY,
    ConfigKeySpec,
    get_spec,
    namespace_chain,
    normalize_string_config_value,
    resolve_env_value,
)
from src.features.configuration.infrastructure.platform_config_repository import PlatformConfigStore, get_platform_config_store

logger = logging.getLogger(__name__)

REDACTED = "***REDACTED***"


class NamespaceView:
    def __init__(self, provider: PlatformConfigProvider, namespace: str) -> None:
        self._provider = provider
        self._namespace = namespace

    def get(self, key: str, default: Any = None) -> Any:
        return self._provider.get(self._namespace, key, default=default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def get_int(self, key: str, default: int = 0) -> int:
        value = self.get(key, default)
        if value is None:
            return default
        return int(value)

    def get_float(self, key: str, default: float = 0.0) -> float:
        value = self.get(key, default)
        if value is None:
            return default
        return float(value)

    def get_str(self, key: str, default: str | None = None) -> str | None:
        value = self.get(key, default)
        if value is None:
            return default
        return normalize_string_config_value(str(value))


class PlatformConfigProvider:
    def __init__(self, store: PlatformConfigStore | None = None) -> None:
        self._store = store
        self._lock = threading.RLock()
        self._local_version = -1
        self._config_version = 0
        self._updated_at: str | None = None
        self._snapshot: dict[tuple[str, str], Any] = {}
        self._last_version_check = 0.0
        self._listeners: list[Callable[[int], None]] = []

    @property
    def config_version(self) -> int:
        self._maybe_refresh(force=False)
        return self._config_version

    def subscribe(self, callback: Callable[[int], None]) -> None:
        self._listeners.append(callback)

    def invalidate(self) -> None:
        with self._lock:
            self._local_version = -1

    def _maybe_refresh(self, *, force: bool) -> None:
        store = self._store or get_platform_config_store()
        if store is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_version_check) < REFRESH_INTERVAL_SEC:
            return
        self._last_version_check = now
        try:
            remote_version, updated_at = store.get_meta()
        except Exception:
            logger.debug("Platform config version check failed", exc_info=True)
            return
        with self._lock:
            if not force and remote_version == self._local_version:
                return
            entries = store.list_entries()
            self._snapshot = {(entry.namespace, entry.key): entry.value for entry in entries}
            self._local_version = remote_version
            self._config_version = remote_version
            self._updated_at = updated_at.isoformat()
            for listener in self._listeners:
                try:
                    listener(remote_version)
                except Exception:
                    logger.debug("Platform config listener failed", exc_info=True)

    def refresh(self) -> None:
        self._last_version_check = 0.0
        self._maybe_refresh(force=True)

    def namespace(self, name: str) -> NamespaceView:
        return NamespaceView(self, name)

    def _normalize_value(self, namespace: str, key: str, value: Any) -> Any:
        spec = get_spec(namespace, key)
        if spec is not None and spec.value_type == "string":
            return normalize_string_config_value(value)
        return value

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        self._maybe_refresh(force=False)
        for ns in namespace_chain(namespace):
            if (ns, key) in self._snapshot:
                return self._normalize_value(ns, key, self._snapshot[(ns, key)])
        spec = get_spec(namespace, key)
        if spec is not None:
            env_value = resolve_env_value(spec)
            if env_value is not None:
                return env_value
            if spec.default is not None:
                return spec.default
        return default

    def get_spec_value(self, spec: ConfigKeySpec) -> Any:
        return self.get(spec.namespace, spec.key, default=spec.default)

    def list_stored_entries(self, namespace: str | None = None, *, include_sensitive: bool = False) -> list[dict[str, Any]]:
        self.refresh()
        store = self._store or get_platform_config_store()
        if store is None:
            return []
        rows = store.list_entries(namespace)
        result: list[dict[str, Any]] = []
        for row in rows:
            value = row.value
            if row.is_sensitive and not include_sensitive:
                value = REDACTED
            result.append(
                {
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
            )
        return result

    def meta(self) -> dict[str, Any]:
        self.refresh()
        from src.features.configuration.application.schema_registry import NAMESPACES

        return {
            "config_version": self._config_version,
            "updated_at": self._updated_at,
            "namespaces": list(NAMESPACES),
        }

    def export_snapshot(self, *, include_sensitive: bool = False) -> dict[str, Any]:
        self.refresh()
        grouped: dict[str, dict[str, Any]] = {}
        for spec in CONFIG_REGISTRY:
            ns = spec.namespace
            grouped.setdefault(ns, {})
            stored = self._snapshot.get((ns, spec.key))
            if stored is not None:
                value = stored
            else:
                value = resolve_env_value(spec)
            if spec.is_sensitive and not include_sensitive:
                value = REDACTED if value is not None else None
            grouped[ns][spec.key] = value
        return {"config_version": self._config_version, "namespaces": grouped}


_default_provider: PlatformConfigProvider | None = None


def get_platform_config() -> PlatformConfigProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = PlatformConfigProvider(get_platform_config_store())
    return _default_provider


def reset_platform_config_provider() -> None:
    global _default_provider
    _default_provider = None
