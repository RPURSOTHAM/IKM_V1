"""Plugin registry, DI container, and config-driven model loading."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from src.infrastructure.application_support.interfaces import COMPONENT_ROLES

try:
    from src.features.retrieval.domain.interfaces import ModelRegistry as _ModelRegistryBase
except Exception:  # pragma: no cover
    from abc import ABC

    class _ModelRegistryBase(ABC):  # type: ignore[no-redef]
        pass


T = TypeVar("T")
Factory = Callable[[], Any]


@dataclass
class PluginSpec:
    role: str
    name: str
    factory: Factory
    default: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Register replaceable implementations by role without hardcoding call sites."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._plugins: dict[str, dict[str, PluginSpec]] = {}
        self._defaults: dict[str, str] = {}

    def register(
        self,
        role: str,
        name: str,
        factory: Factory,
        *,
        default: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        role = role.strip().lower()
        name = name.strip()
        spec = PluginSpec(role=role, name=name, factory=factory, default=default, metadata=metadata or {})
        with self._lock:
            self._plugins.setdefault(role, {})[name] = spec
            if default or role not in self._defaults:
                self._defaults[role] = name

    def unregister(self, role: str, name: str) -> None:
        with self._lock:
            role_map = self._plugins.get(role, {})
            role_map.pop(name, None)
            if self._defaults.get(role) == name:
                self._defaults[role] = next(iter(role_map), "")

    def list(self, role: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            roles = [role] if role else sorted(self._plugins)
            out: list[dict[str, Any]] = []
            for r in roles:
                for name, spec in self._plugins.get(r, {}).items():
                    out.append(
                        {
                            "role": r,
                            "name": name,
                            "default": self._defaults.get(r) == name,
                            "metadata": dict(spec.metadata),
                        }
                    )
            return out

    def resolve_name(self, role: str, preferred: str | None = None) -> str:
        with self._lock:
            role = role.strip().lower()
            preferred = (preferred or "").strip()
            role_map = self._plugins.get(role, {})
            if preferred and preferred in role_map:
                return preferred
            if preferred:
                # Allow alias-like preferred ids that contain the plugin name.
                for name in role_map:
                    if preferred == name or preferred.endswith(name) or name in preferred:
                        return name
            default = self._defaults.get(role)
            if default and default in role_map:
                return default
            if role_map:
                return next(iter(role_map))
            raise KeyError(f"No plugin registered for role={role!r} preferred={preferred!r}")

    def create(self, role: str, preferred: str | None = None) -> Any:
        name = self.resolve_name(role, preferred)
        with self._lock:
            factory = self._plugins[role][name].factory
        return factory()

    def has(self, role: str, name: str | None = None) -> bool:
        with self._lock:
            role_map = self._plugins.get(role.strip().lower(), {})
            if name is None:
                return bool(role_map)
            return name in role_map


class ServiceContainer:
    """Minimal dependency-injection container with singleton and factory scopes."""

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        self.registry = registry or PluginRegistry()
        self._lock = threading.RLock()
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Factory] = {}
        self._singleton_flags: dict[str, bool] = {}

    def register_instance(self, key: str, instance: Any) -> None:
        with self._lock:
            self._singletons[key] = instance

    def register_factory(self, key: str, factory: Factory, *, singleton: bool = True) -> None:
        with self._lock:
            self._factories[key] = factory
            self._singleton_flags[key] = singleton
            if singleton:
                self._singletons.pop(key, None)

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._singletons:
                return self._singletons[key]
            factory = self._factories.get(key)
            if factory is None:
                raise KeyError(f"No DI binding for {key!r}")
            singleton = self._singleton_flags.get(key, True)
        instance = factory()
        if singleton:
            with self._lock:
                self._singletons[key] = instance
        return instance

    def get_plugin(self, role: str, preferred: str | None = None, *, singleton: bool = True) -> Any:
        cache_key = f"plugin:{role}:{preferred or 'default'}"
        if singleton:
            with self._lock:
                if cache_key in self._singletons:
                    return self._singletons[cache_key]
        instance = self.registry.create(role, preferred)
        if singleton:
            with self._lock:
                self._singletons[cache_key] = instance
        return instance

    def clear(self) -> None:
        with self._lock:
            self._singletons.clear()
            self._factories.clear()
            self._singleton_flags.clear()


# Role -> env var for preferred implementation / model id
_MODEL_ROLE_ENV: dict[str, tuple[str, ...]] = {
    "embedder": ("EMBEDDING_MODEL", "MODEL_DIR", "INFRA_EMBEDDER_MODEL"),
    "llm": ("GENERATION_MODEL_ID", "INFRA_LLM_MODEL"),
    "moderation": ("MODERATION_GUARD_MODEL", "INFRA_MODERATION_MODEL"),
    "retriever": ("RETRIEVAL_RERANKER_MODEL", "INFRA_RETRIEVER_MODEL"),
    "ner": ("INFRA_NER_MODEL",),
    "dlp": ("INFRA_DLP_BACKEND",),
    "parser": ("INFRA_PARSER_BACKEND",),
    "chunker": ("CHUNKING_STRATEGY", "INFRA_CHUNKER_BACKEND"),
    "vector_database": ("INFRA_VECTOR_DB_BACKEND", "VECTOR_DB_BACKEND"),
    "risk_engine": ("INFRA_RISK_ENGINE",),
}


class ConfigDrivenModelLoader(_ModelRegistryBase):
    """Resolve model/backend ids from configuration, then instantiate via PluginRegistry.

    Implements the retrieval ModelRegistry contract so call sites can swap models
    without changing business logic.
    """

    def __init__(
        self,
        registry: PluginRegistry,
        *,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self.registry = registry
        self.overrides = {k.lower(): v for k, v in (overrides or {}).items()}

    def preferred_for(self, role: str) -> str | None:
        role = role.strip().lower()
        if role in self.overrides and self.overrides[role]:
            return self.overrides[role]
        for env_name in _MODEL_ROLE_ENV.get(role, ()):
            value = os.getenv(env_name)
            if value and value.strip():
                return value.strip()
        # Generic INFRA_<ROLE>_MODEL
        generic = os.getenv(f"INFRA_{role.upper()}_MODEL") or os.getenv(f"INFRA_{role.upper()}_BACKEND")
        return generic.strip() if generic else None

    def resolve(self, role: str, preferred: str | None = None) -> str:
        choice = preferred or self.preferred_for(role)
        return self.registry.resolve_name(role, choice)

    def get(self, role: str, preferred: str | None = None) -> Any:
        choice = preferred or self.preferred_for(role)
        return self.registry.create(role, choice)

    def catalog(self) -> dict[str, Any]:
        return {
            "roles": list(COMPONENT_ROLES),
            "preferred": {role: self.preferred_for(role) for role in COMPONENT_ROLES},
            "plugins": self.registry.list(),
        }


_global_registry = PluginRegistry()
_global_container = ServiceContainer(_global_registry)
_global_loader = ConfigDrivenModelLoader(_global_registry)
_bootstrapped = False
_bootstrap_lock = threading.Lock()


def get_registry() -> PluginRegistry:
    return _global_registry


def get_container() -> ServiceContainer:
    return _global_container


def get_model_loader() -> ConfigDrivenModelLoader:
    return _global_loader


def reset_infra_globals() -> None:
    """Test helper: clear singleton state."""
    global _bootstrapped, _global_registry, _global_container, _global_loader
    with _bootstrap_lock:
        _global_registry = PluginRegistry()
        _global_container = ServiceContainer(_global_registry)
        _global_loader = ConfigDrivenModelLoader(_global_registry)
        _bootstrapped = False


def mark_bootstrapped() -> None:
    global _bootstrapped
    _bootstrapped = True


def is_bootstrapped() -> bool:
    return _bootstrapped
