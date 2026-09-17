from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.features.configuration.platform_settings import REPO_ROOT, settings
from src.features.repositories.configuration.chunking_strategy_catalog import (
    _CONFIG_FIELDS,
    default_chunking_config,
)
from src.features.repositories.infrastructure.collection_naming import (
    embedding_vector_dimensions,
    resolve_processor_model_name,
)
from src.features.repositories.domain.repository import RepositorySettings
from src.features.chunking.domain.chunking_strategy import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING_STRATEGY,
    normalize_chunking_strategy,
    resolve_chunking_strategy,
)
from src.features.document_processing.shared_processor.types import enabled_processor_types
from src.features.repositories.domain.repository_exceptions import InvalidSettingsError, ValidationError

CLOUD_PROVIDERS = frozenset({"openai", "azure_openai", "aws", "anthropic"})
RERANKER_MODEL_DIRS = frozenset(
    {"bge-reranker-base", "bge-reranker-large", "bge-reranker-v2-m3"}
)
SKIP_MODEL_DIRS = frozenset({".locks", "lfs", "__pycache__"})
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 2000


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bounded_chunk_size(value: Any, *, default: int = 512) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, MIN_CHUNK_SIZE), MAX_CHUNK_SIZE)


def _valid_chunk_size(value: Any) -> bool:
    return isinstance(value, int) and MIN_CHUNK_SIZE <= value <= MAX_CHUNK_SIZE


def _valid_chunk_overlap(value: Any, chunk_size: Any) -> bool:
    return (
        isinstance(value, int)
        and value >= 0
        and (not isinstance(chunk_size, int) or value < chunk_size)
    )

DEFAULT_EMBEDDING_MODEL: dict[str, Any] = {
    "provider": "local",
    "model_id": "all-MiniLM-L6-v2",
    "local_model_dir": "models--sentence-transformers--all-MiniLM-L6-v2",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "embedding_model": dict(DEFAULT_EMBEDDING_MODEL),
    "chunk_size": _bounded_chunk_size(_env_int("CHUNK_SIZE", 512)),
    "chunk_overlap": max(0, _env_int("CHUNK_OVERLAP_SENTENCES", 1)),
    "chunking_strategy": DEFAULT_CHUNKING_STRATEGY,
    "chunking_config": default_chunking_config(DEFAULT_CHUNKING_STRATEGY),
    "indexing_strategy": "weaviate_upsert",
    # Default-on matches create-UI; repositories may explicitly disable.
    "metadata_extraction": True,
    "key_field_extraction": False,
    "key_field_extraction_enabled": False,
    "key_fields": [],
    # Empty means logical folders automatically group shared extracted values.
    "folder_hierarchy": [],
    "strict_key_field_page_scope": False,
    "validation_enabled": False,
    "validation_confidence_threshold": 0.85,
    "citation_retainment": True,
    "template_extraction": False,
    "reference_document_extraction": False,
    "conversion_for_rendering": False,
    "intelligent_extraction": False,
    "content_intelligence": False,
    # None = unset → retrieval falls through to RERANKER_ENABLED / platform config.
    "reranking": None,
    "retrieval_search_mode": "hybrid",
    "lexical_composition": True,
    "document_type_id": None,
    "extraction_model": None,
}


def models_root() -> Path:
    configured = os.getenv("MODELS_ROOT") or os.getenv("SCHEDULER_MODELS_CONTAINER_PATH")
    if configured:
        return Path(configured)
    return REPO_ROOT / "src" / "models"


def discover_local_embedding_models() -> list[str]:
    root = models_root()
    if not root.is_dir():
        return []
    names: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name in SKIP_MODEL_DIRS or name in RERANKER_MODEL_DIRS:
            continue
        if name.startswith("."):
            continue
        names.append(name)
    return names


def validate_embedding_model(embedding_model: dict[str, Any], *, strict_local: bool = True) -> dict[str, Any]:
    if not isinstance(embedding_model, dict):
        raise InvalidSettingsError("embedding_model must be an object.")

    provider = str(embedding_model.get("provider", "")).strip().lower()
    model_id = str(embedding_model.get("model_id", "")).strip()
    if not provider:
        raise InvalidSettingsError("embedding_model.provider is required.")
    if not model_id:
        raise InvalidSettingsError("embedding_model.model_id is required.")

    normalized = dict(embedding_model)
    normalized["provider"] = provider
    normalized["model_id"] = model_id

    if provider == "local":
        local_dir = str(embedding_model.get("local_model_dir") or model_id).strip()
        if not local_dir:
            raise InvalidSettingsError("embedding_model.local_model_dir is required for local provider.")
        if local_dir in RERANKER_MODEL_DIRS:
            raise InvalidSettingsError(
                f"'{local_dir}' is a reranker model, not an embedding model.",
                details={"local_model_dir": local_dir},
            )
        path = models_root() / local_dir
        if not path.is_dir():
            if not strict_local:
                normalized["local_model_dir"] = local_dir
                return normalized
            available = discover_local_embedding_models()
            raise InvalidSettingsError(
                f"Local model directory '{local_dir}' was not found under mounted models root '{models_root()}'.",
                details={
                    "local_model_dir": local_dir,
                    "models_root": str(models_root()),
                    "available": available,
                },
            )
        normalized["local_model_dir"] = local_dir
        expected_dims = embedding_vector_dimensions(normalized)
        if expected_dims is None:
            raise InvalidSettingsError(
                f"Unknown embedding dimensions for local model '{local_dir}'.",
                details={"local_model_dir": local_dir, "model_id": model_id},
            )
        return normalized

    if provider not in CLOUD_PROVIDERS:
        raise InvalidSettingsError(
            f"Unsupported embedding_model.provider '{provider}'.",
            details={"allowed_providers": sorted(CLOUD_PROVIDERS | {"local"})},
        )
    if provider != "local" and not str(embedding_model.get("credential_ref") or "").strip():
        raise ValidationError(
            "embedding_model.credential_ref is required for cloud embedding providers.",
            details={"provider": provider},
        )
    return normalized


def merge_settings(partial: dict[str, Any] | None, *, strict_local: bool = True) -> dict[str, Any]:
    merged = dict(DEFAULT_SETTINGS)
    if not partial:
        return merged
    for key, value in partial.items():
        if value is None:
            continue
        merged[key] = value
    _normalize_key_field_aliases(merged, partial=partial)
    if "embedding_model" in partial and partial["embedding_model"] is not None:
        merged["embedding_model"] = validate_embedding_model(
            partial["embedding_model"],
            strict_local=strict_local,
        )
    elif isinstance(merged.get("embedding_model"), dict):
        merged["embedding_model"] = validate_embedding_model(merged["embedding_model"], strict_local=False)
    merged["chunking_strategy"] = normalize_chunking_strategy(merged.get("chunking_strategy"))
    if not isinstance(merged.get("chunking_config"), dict):
        merged["chunking_config"] = default_chunking_config(merged["chunking_strategy"])
    _validate_bool_flags(merged)
    _validate_key_field_config(merged)
    _validate_folder_hierarchy(merged)
    return merged


def merge_settings_lenient(partial: dict[str, Any] | None) -> dict[str, Any]:
    """Merge stored settings with defaults without validation (configuring repos)."""
    merged = dict(DEFAULT_SETTINGS)
    if partial:
        for key, value in partial.items():
            if value is not None:
                merged[key] = value
    _normalize_key_field_aliases(merged, partial=partial)
    if isinstance(merged.get("embedding_model"), dict):
        merged["embedding_model"] = dict(merged["embedding_model"])
    strategy = normalize_chunking_strategy(merged.get("chunking_strategy"))
    merged["chunking_strategy"] = strategy
    if not isinstance(merged.get("chunking_config"), dict):
        merged["chunking_config"] = default_chunking_config(strategy)
    if not isinstance(merged.get("key_fields"), list):
        merged["key_fields"] = []
    return merged


def patch_settings(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    updated = dict(current)
    for key, value in patch.items():
        if value is None:
            continue
        updated[key] = value
    _normalize_key_field_aliases(updated, partial=patch)
    _repair_unpatched_chunking_values(updated, patch)
    if "embedding_model" in patch:
        base = dict(current.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
        if isinstance(patch["embedding_model"], dict):
            base.update(patch["embedding_model"])
        updated["embedding_model"] = validate_embedding_model(base, strict_local=True)
    if "chunking_strategy" in patch and patch["chunking_strategy"] is not None:
        updated["chunking_strategy"] = normalize_chunking_strategy(patch["chunking_strategy"])
        if "chunking_config" not in patch:
            existing = updated.get("chunking_config") if isinstance(updated.get("chunking_config"), dict) else {}
            updated["chunking_config"] = {**default_chunking_config(updated["chunking_strategy"]), **existing}
    if "chunking_config" in patch and isinstance(patch["chunking_config"], dict):
        base = dict(current.get("chunking_config") or default_chunking_config(normalize_chunking_strategy(updated.get("chunking_strategy"))))
        base.update(patch["chunking_config"])
        updated["chunking_config"] = base
    _validate_bool_flags(updated)
    _validate_key_field_config(updated)
    _validate_folder_hierarchy(updated)
    return updated


def _repair_unpatched_chunking_values(updated: dict[str, Any], patch: dict[str, Any]) -> None:
    """Normalize legacy/default chunking values when a patch changes unrelated settings."""
    if "chunk_size" not in patch and not _valid_chunk_size(updated.get("chunk_size")):
        updated["chunk_size"] = DEFAULT_SETTINGS["chunk_size"]
    if "chunk_overlap" not in patch and not _valid_chunk_overlap(
        updated.get("chunk_overlap"),
        updated.get("chunk_size"),
    ):
        updated["chunk_overlap"] = DEFAULT_SETTINGS["chunk_overlap"]


def processing_hints_from_settings(resolved_settings: dict[str, Any]) -> dict[str, Any]:
    """Flatten repository settings for queue/processor payloads."""
    embedding = resolved_settings.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
    local_dir = str(embedding.get("local_model_dir") or embedding.get("model_id") or "").strip()
    return {
        "chunk_size": resolved_settings.get("chunk_size"),
        "chunk_overlap_sentences": resolved_settings.get("chunk_overlap"),
        "chunking_strategy": normalize_chunking_strategy(resolved_settings.get("chunking_strategy")),
        "chunking_config": resolved_settings.get("chunking_config") or {},
        "model_name": resolve_processor_model_name(embedding),
        "model_dir": local_dir or None,
        "citation_retainment": resolved_settings.get("citation_retainment", True),
        "enabled_processor_types": enabled_processor_types(resolved_settings),
        "document_type_id": resolved_settings.get("document_type_id"),
        "extraction_model": resolved_settings.get("extraction_model"),
        "metadata_extraction": resolved_settings.get("metadata_extraction", False),
        "key_field_extraction": resolved_settings.get(
            "key_field_extraction",
            resolved_settings.get("key_field_extraction_enabled", False),
        ),
        "key_field_extraction_enabled": resolved_settings.get(
            "key_field_extraction_enabled",
            resolved_settings.get("key_field_extraction", False),
        ),
        "key_fields": resolved_settings.get("key_fields") or [],
        "strict_key_field_page_scope": resolved_settings.get("strict_key_field_page_scope", False),
        "validation_enabled": resolved_settings.get("validation_enabled", False),
        "validation_confidence_threshold": resolved_settings.get("validation_confidence_threshold", 0.85),
        "template_extraction": resolved_settings.get("template_extraction", False),
        "reference_document_extraction": resolved_settings.get("reference_document_extraction", False),
        "conversion_for_rendering": resolved_settings.get("conversion_for_rendering", False),
        # Content Intelligence flags must be present so document scheduling can
        # recompute enabled_processor_types (including intelligence_extraction).
        "intelligent_extraction": bool(resolved_settings.get("intelligent_extraction", False)),
        "content_intelligence": bool(resolved_settings.get("content_intelligence", False)),
        "intelligence_extraction": bool(
            resolved_settings.get("intelligence_extraction")
            or resolved_settings.get("intelligent_extraction")
            or resolved_settings.get("content_intelligence")
            or False
        ),
    }


def resolve_for_repository(stored_settings: dict[str, Any] | None) -> dict[str, Any]:
    """Merge repository settings with platform env fallbacks for runtime consumers.

    Local embedding directories are not required to exist on the DMS host;
    processors load models from their own bind mount. Strict disk checks apply
    on create/patch via merge_settings(strict_local=True).
    """
    base = merge_settings(stored_settings, strict_local=False)
    if stored_settings and stored_settings.get("embedding_model"):
        return base

    env_model_dir = settings.retrieval_model_dir or os.getenv("MODEL_DIR")
    if env_model_dir:
        local_dir = Path(str(env_model_dir)).name
        base["embedding_model"] = validate_embedding_model(
            {
                "provider": "local",
                "model_id": settings.retrieval_model_name,
                "local_model_dir": local_dir,
            },
            strict_local=False,
        )
    return base


def _validate_bool_flags(settings_dict: dict[str, Any]) -> None:
    bool_keys = (
        "metadata_extraction",
        "key_field_extraction",
        "key_field_extraction_enabled",
        "strict_key_field_page_scope",
        "validation_enabled",
        "citation_retainment",
        "template_extraction",
        "reference_document_extraction",
        "conversion_for_rendering",
        "intelligent_extraction",
        "content_intelligence",
        "reranking",
        "lexical_composition",
    )
    for key in bool_keys:
        if key not in settings_dict:
            continue
        value = settings_dict[key]
        if value is None:
            continue
        if not isinstance(value, bool):
            raise InvalidSettingsError(f"Setting '{key}' must be a boolean.")
    threshold = settings_dict.get("validation_confidence_threshold")
    if threshold is not None:
        try:
            value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise InvalidSettingsError("validation_confidence_threshold must be a number.") from exc
        if value < 0.0 or value > 1.0:
            raise InvalidSettingsError("validation_confidence_threshold must be between 0 and 1.")
        settings_dict["validation_confidence_threshold"] = value
    _normalize_key_field_aliases(settings_dict)

    chunk_size = settings_dict.get("chunk_size")
    if chunk_size is not None:
        if not isinstance(chunk_size, int) or chunk_size < 200 or chunk_size > 2000:
            raise InvalidSettingsError("chunk_size must be an integer between 200 and 2000.")

    chunk_overlap = settings_dict.get("chunk_overlap")
    if chunk_overlap is not None:
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise InvalidSettingsError("chunk_overlap must be a non-negative integer.")
        if chunk_size is not None and chunk_overlap >= chunk_size:
            raise InvalidSettingsError("chunk_overlap must be smaller than chunk_size.")

    try:
        strategy = resolve_chunking_strategy(settings_dict.get("chunking_strategy"))
    except ValueError as exc:
        raise InvalidSettingsError(str(exc)) from exc
    settings_dict["chunking_strategy"] = strategy
    _validate_chunking_config(strategy, settings_dict)

    indexing = settings_dict.get("indexing_strategy")
    if indexing is not None and indexing not in {"weaviate_upsert"}:
        raise InvalidSettingsError("indexing_strategy must be 'weaviate_upsert'.")

    retrieval_search_mode = settings_dict.get("retrieval_search_mode")
    if retrieval_search_mode is not None:
        normalized_mode = str(retrieval_search_mode).strip().lower()
        if normalized_mode not in {"hybrid", "vector", "keyword"}:
            raise InvalidSettingsError("retrieval_search_mode must be one of: hybrid, vector, keyword.")
        settings_dict["retrieval_search_mode"] = normalized_mode


def _validate_chunking_config(strategy: str, settings_dict: dict[str, Any]) -> None:
    fields = _CONFIG_FIELDS.get(strategy, [])
    chunking_config = settings_dict.get("chunking_config")
    if chunking_config is not None and not isinstance(chunking_config, dict):
        raise InvalidSettingsError("chunking_config must be an object.")
    config = dict(chunking_config or {})

    for field in fields:
        key = field["key"]
        scope = field["scope"]
        required = bool(field.get("required"))
        value = settings_dict.get(key) if scope == "settings" else config.get(key)
        if value is None:
            if required:
                raise InvalidSettingsError(
                    f"Setting '{key}' is required for chunking_strategy '{strategy}'.",
                    details={"chunking_strategy": strategy, "field": key, "scope": scope},
                )
            continue
        value_type = field.get("value_type")
        if value_type == "integer":
            if not isinstance(value, int):
                raise InvalidSettingsError(f"'{key}' must be an integer.")
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            if minimum is not None and value < minimum:
                raise InvalidSettingsError(f"'{key}' must be >= {minimum}.")
            if maximum is not None and value > maximum:
                raise InvalidSettingsError(f"'{key}' must be <= {maximum}.")
        elif value_type == "number":
            if not isinstance(value, (int, float)):
                raise InvalidSettingsError(f"'{key}' must be a number.")
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            if minimum is not None and float(value) < float(minimum):
                raise InvalidSettingsError(f"'{key}' must be >= {minimum}.")
            if maximum is not None and float(value) > float(maximum):
                raise InvalidSettingsError(f"'{key}' must be <= {maximum}.")

    if strategy == "sliding-window":
        window = settings_dict.get("chunk_size")
        step = settings_dict.get("chunk_overlap")
        if isinstance(window, int) and isinstance(step, int) and step >= window:
            raise InvalidSettingsError("chunk_overlap (window step) must be smaller than chunk_size (window size).")


def _normalize_key_field_aliases(settings_dict: dict[str, Any], partial: dict[str, Any] | None = None) -> None:
    enabled_new = settings_dict.get("key_field_extraction_enabled")
    enabled_legacy = settings_dict.get("key_field_extraction")
    if partial:
        sent_new = "key_field_extraction_enabled" in partial and partial.get("key_field_extraction_enabled") is not None
        sent_legacy = "key_field_extraction" in partial and partial.get("key_field_extraction") is not None
        if sent_new and sent_legacy and bool(enabled_new) != bool(enabled_legacy):
            raise InvalidSettingsError(
                "key_field_extraction_enabled and key_field_extraction conflict.",
                details={
                    "key_field_extraction_enabled": enabled_new,
                    "key_field_extraction": enabled_legacy,
                },
            )
    canonical = (
        bool(enabled_new)
        if enabled_new is not None
        else bool(enabled_legacy) if enabled_legacy is not None else False
    )
    settings_dict["key_field_extraction"] = canonical
    settings_dict["key_field_extraction_enabled"] = canonical
    _normalize_content_intelligence_aliases(settings_dict, partial)


def _normalize_content_intelligence_aliases(
    settings_dict: dict[str, Any], partial: dict[str, Any] | None = None
) -> None:
    """Keep intelligent_extraction and content_intelligence in sync."""
    primary = settings_dict.get("intelligent_extraction")
    alias = settings_dict.get("content_intelligence")
    if partial:
        sent_primary = "intelligent_extraction" in partial and partial.get("intelligent_extraction") is not None
        sent_alias = "content_intelligence" in partial and partial.get("content_intelligence") is not None
        if sent_primary and sent_alias and bool(primary) != bool(alias):
            raise InvalidSettingsError(
                "intelligent_extraction and content_intelligence conflict.",
                details={
                    "intelligent_extraction": primary,
                    "content_intelligence": alias,
                },
            )
    canonical = (
        bool(primary)
        if primary is not None
        else bool(alias) if alias is not None else False
    )
    settings_dict["intelligent_extraction"] = canonical
    settings_dict["content_intelligence"] = canonical


def _validate_key_field_config(settings_dict: dict[str, Any]) -> None:
    fields = settings_dict.get("key_fields")
    if fields is None:
        fields = []
    if not isinstance(fields, list):
        raise InvalidSettingsError("key_fields must be an array.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_types = {"string", "number", "boolean", "date", "datetime", "text", "integer", "float"}
    for raw in fields:
        if not isinstance(raw, dict):
            raise InvalidSettingsError("Each key_fields item must be an object.")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise InvalidSettingsError("Each key field requires a non-empty 'name'.")
        key = name.lower()
        if key in seen:
            raise InvalidSettingsError(
                f"Duplicate key field name '{name}'.",
                details={"field_name": name},
            )
        seen.add(key)
        field_type = str(raw.get("type") or "string").strip().lower()
        if field_type not in allowed_types:
            raise InvalidSettingsError(
                f"Unsupported key field type '{field_type}'.",
                details={"field_name": name, "allowed_types": sorted(allowed_types)},
            )
        normalized.append(
            {
                "field_id": str(raw.get("field_id") or "").strip() or None,
                "name": name,
                "type": field_type,
                "required": bool(raw.get("required", False)),
                "description": str(raw.get("description") or "").strip() or None,
            }
        )
    settings_dict["key_fields"] = normalized
    enabled = bool(
        settings_dict.get("key_field_extraction_enabled")
        or settings_dict.get("key_field_extraction")
    )
    # Phase 3: repository document types supply effective fields when settings.key_fields is empty.
    has_document_type = bool(str(settings_dict.get("document_type_id") or "").strip())
    if enabled and not normalized and not has_document_type:
        raise InvalidSettingsError(
            "key_fields cannot be empty when key_field_extraction_enabled is true."
        )


def _validate_folder_hierarchy(settings_dict: dict[str, Any]) -> None:
    """Validate the optional logical-folder ordering without changing existing settings."""
    hierarchy = settings_dict.get("folder_hierarchy")
    if hierarchy is None:
        settings_dict["folder_hierarchy"] = []
        return
    if not isinstance(hierarchy, list):
        raise InvalidSettingsError("folder_hierarchy must be an array of key-field identifiers.")
    valid_ids = {
        str(field.get("field_id") or "").strip().lower()
        for field in settings_dict.get("key_fields") or []
        if isinstance(field, dict) and str(field.get("field_id") or "").strip()
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_id in hierarchy:
        field_id = str(raw_id or "").strip()
        key = field_id.lower()
        if not field_id:
            raise InvalidSettingsError("folder_hierarchy entries must be non-empty key-field identifiers.")
        if key in seen:
            raise InvalidSettingsError("folder_hierarchy cannot contain duplicate key-field identifiers.")
        if key not in valid_ids:
            raise InvalidSettingsError(
                "folder_hierarchy references a key field that is not configured on this repository.",
                details={"field_id": field_id},
            )
        seen.add(key)
        normalized.append(field_id)
    settings_dict["folder_hierarchy"] = normalized


# ---------------------------------------------------------------------------
# Phase 3.5 — SettingsResolver (effective settings SoT for runtime consumers)
#
# Precedence (existing behavior preserved; repository wins over system defaults):
#   1. System / platform defaults (DEFAULT_SETTINGS, incl. env CHUNK_SIZE /
#      CHUNK_OVERLAP_SENTENCES and default embedding / retrieval / processor flags)
#   2. Repository stored overrides (non-null keys from repository_settings.settings)
#   3. Platform embedding env fallback (retrieval_model_dir / MODEL_DIR) only when
#      the repository did not set embedding_model
#
# Does not persist. Does not call HTTP, Weaviate, Processing, or Retrieval.
# Stateless — no cache layer.
# ---------------------------------------------------------------------------


class SettingsResolverError(Exception):
    """Base error for SettingsResolver load/resolve failures (not SQL)."""

    code: str = "settings_resolver_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class RepositorySettingsNotFound(SettingsResolverError):
    """Repository row missing when resolving by id."""

    code = "repository_settings_not_found"


class SettingsStoreUnavailable(SettingsResolverError):
    """Persistence unavailable while loading settings for resolution."""

    code = "settings_store_unavailable"


@dataclass(frozen=True)
class EffectiveRepositorySettings:
    """Effective configuration consumed by Processing, Embedding, and Retrieval."""

    repository_id: str
    name: str
    weaviate_collection: str
    default_tenant_id: str | None
    status: str
    settings_locked: bool
    settings: dict[str, Any]
    processing_hints: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Wire shape compatible with RepositoryService.resolve_settings()."""
        return {
            "repository_id": self.repository_id,
            "name": self.name,
            "weaviate_collection": self.weaviate_collection,
            "default_tenant_id": self.default_tenant_id,
            "status": self.status,
            "settings_locked": self.settings_locked,
            "settings": dict(self.settings),
            "processing_hints": dict(self.processing_hints),
        }

    def processing(self) -> dict[str, Any]:
        """Processor-oriented view (queue / ProcessRequest hints)."""
        return dict(self.processing_hints)

    def retrieval(self) -> dict[str, Any]:
        """Retrieval-oriented subset of effective settings + binding fields."""
        s = self.settings
        return {
            "repository_id": self.repository_id,
            "weaviate_collection": self.weaviate_collection,
            "default_tenant_id": self.default_tenant_id,
            "status": self.status,
            "retrieval_search_mode": s.get("retrieval_search_mode"),
            "lexical_composition": s.get("lexical_composition"),
            "reranking": s.get("reranking"),
            "embedding_model": s.get("embedding_model"),
            "indexing_strategy": s.get("indexing_strategy"),
            "chunk_size": s.get("chunk_size"),
            "chunk_overlap": s.get("chunk_overlap"),
        }


class SettingsResolver:
    """Single source of truth for effective repository settings.

    May load via RepositoryStore. Must not call HTTP, Weaviate, Processing, or Retrieval.
    """

    def __init__(self, store: Any | None = None) -> None:
        self._store = store

    def _require_store(self) -> Any:
        if self._store is not None:
            return self._store
        from src.features.repositories.infrastructure.repository_repository import get_repository_store

        store = get_repository_store()
        if store is None:
            raise SettingsStoreUnavailable(
                "Repository settings store is unavailable.",
            )
        return store

    def resolve_from_stored(self, stored_settings: dict[str, Any] | None) -> dict[str, Any]:
        """Resolve effective settings dict from raw stored JSON (no repository row)."""
        return resolve_for_repository(stored_settings)

    def resolve(
        self,
        repository: Any,
        stored_settings: dict[str, Any] | RepositorySettings | None = None,
    ) -> EffectiveRepositorySettings:
        """Resolve EffectiveRepositorySettings from a repository record + optional stored JSON.

        ``repository`` may be a RepositoryRecord/Repository or a mapping with the same keys.
        """
        record = self._coerce_repository(repository)
        raw = self._coerce_stored_settings(stored_settings)
        if raw is None and record.repository_id:
            try:
                raw = self._require_store().get_settings(record.repository_id) or {}
            except SettingsResolverError:
                raise
            except Exception as exc:
                raise SettingsStoreUnavailable(
                    "Failed to load repository settings.",
                    details={"repository_id": record.repository_id, "error": str(exc)},
                ) from exc

        effective = self.resolve_from_stored(raw or {})
        hints = processing_hints_from_settings(effective)
        return EffectiveRepositorySettings(
            repository_id=record.repository_id,
            name=record.name,
            weaviate_collection=record.weaviate_collection,
            default_tenant_id=record.default_tenant_id,
            status=record.status,
            settings_locked=bool(record.settings_locked_at),
            settings=effective,
            processing_hints=hints,
        )

    def resolve_by_repository_id(self, repository_id: str) -> EffectiveRepositorySettings:
        """Load repository + settings from Store, then resolve effective configuration."""
        store = self._require_store()
        try:
            record = store.get_by_id(repository_id)
        except SettingsResolverError:
            raise
        except Exception as exc:
            raise SettingsStoreUnavailable(
                "Failed to load repository for settings resolution.",
                details={"repository_id": repository_id, "error": str(exc)},
            ) from exc
        if record is None:
            raise RepositorySettingsNotFound(
                "Repository not found.",
                details={"repository_id": repository_id},
            )
        try:
            stored = store.get_settings(repository_id) or {}
        except Exception as exc:
            raise SettingsStoreUnavailable(
                "Failed to load repository settings.",
                details={"repository_id": repository_id, "error": str(exc)},
            ) from exc
        return self.resolve(record, stored)

    def resolve_processing_settings(
        self,
        repository_or_id: str | EffectiveRepositorySettings | Any,
    ) -> dict[str, Any]:
        """Return processing_hints for queue/processor consumers."""
        effective = self._ensure_effective(repository_or_id)
        return effective.processing()

    def resolve_retrieval_settings(
        self,
        repository_or_id: str | EffectiveRepositorySettings | Any,
    ) -> dict[str, Any]:
        """Return retrieval-oriented effective settings."""
        effective = self._ensure_effective(repository_or_id)
        return effective.retrieval()

    def _ensure_effective(
        self,
        repository_or_id: str | EffectiveRepositorySettings | Any,
    ) -> EffectiveRepositorySettings:
        if isinstance(repository_or_id, EffectiveRepositorySettings):
            return repository_or_id
        if isinstance(repository_or_id, str):
            return self.resolve_by_repository_id(repository_or_id)
        return self.resolve(repository_or_id)

    @staticmethod
    def _coerce_stored_settings(
        stored_settings: dict[str, Any] | RepositorySettings | None,
    ) -> dict[str, Any] | None:
        if stored_settings is None:
            return None
        if isinstance(stored_settings, RepositorySettings):
            return stored_settings.copy_settings()
        return dict(stored_settings)

    @staticmethod
    def _coerce_repository(repository: Any) -> Any:
        from src.features.repositories.domain.repository import RepositoryRecord

        if isinstance(repository, RepositoryRecord):
            return repository
        if isinstance(repository, dict):
            return type(
                "_RepoView",
                (),
                {
                    "repository_id": str(repository.get("repository_id") or ""),
                    "name": str(repository.get("name") or ""),
                    "weaviate_collection": str(repository.get("weaviate_collection") or ""),
                    "default_tenant_id": repository.get("default_tenant_id"),
                    "status": str(repository.get("status") or ""),
                    "settings_locked_at": repository.get("settings_locked_at"),
                },
            )()
        return repository


def get_settings_resolver(store: Any | None = None) -> SettingsResolver:
    """Factory for a stateless SettingsResolver (no process-wide cache)."""
    return SettingsResolver(store=store)


def resolve_repository_context(
    repository_id: str,
    *,
    store: Any | None = None,
) -> dict[str, Any]:
    """Preferred runtime entry for Processing / Retrieval / Document Receiver.

    Returns the same dict shape as RepositoryService.resolve_settings().
    Maps resolver errors to repository NotFoundError / ServiceUnavailableError
    for backward-compatible callers.
    """
    from src.features.repositories.domain.repository_exceptions import NotFoundError, ServiceUnavailableError

    try:
        return get_settings_resolver(store).resolve_by_repository_id(repository_id).to_dict()
    except RepositorySettingsNotFound as exc:
        raise NotFoundError(exc.message, details=dict(exc.details)) from exc
    except SettingsStoreUnavailable as exc:
        raise ServiceUnavailableError(exc.message, details=dict(exc.details)) from exc
