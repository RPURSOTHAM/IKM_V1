"""Processor type identifiers shared across intake, scheduler, and worker containers."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ProcessorType(str, Enum):
    CHUNKING_VECTORIZING = "chunking_vectorizing"
    METADATA_EXTRACTION = "metadata_extraction"
    KEY_FIELD_EXTRACTION = "key_field_extraction"
    DOCUMENT_VALIDATION = "document_validation"
    TEMPLATE_EXTRACTION = "template_extraction"
    REFERENCE_DOCUMENT_EXTRACTION = "reference_document_extraction"
    REFERENCE_EXTRACTION = "reference_extraction"
    CONVERSION_FOR_RENDERING = "conversion_for_rendering"
    INTELLIGENCE_EXTRACTION = "intelligence_extraction"
    MEDIA_TRANSCRIPTION = "media_transcription"


PROCESSOR_STORAGE: dict[ProcessorType, str] = {
    ProcessorType.CHUNKING_VECTORIZING: "weaviate",
    ProcessorType.METADATA_EXTRACTION: "neo4j",
    ProcessorType.KEY_FIELD_EXTRACTION: "neo4j",
    ProcessorType.DOCUMENT_VALIDATION: "neo4j",
    ProcessorType.TEMPLATE_EXTRACTION: "neo4j",
    ProcessorType.REFERENCE_DOCUMENT_EXTRACTION: "neo4j",
    ProcessorType.REFERENCE_EXTRACTION: "neo4j",
    ProcessorType.CONVERSION_FOR_RENDERING: "redis",
    ProcessorType.INTELLIGENCE_EXTRACTION: "neo4j",
    ProcessorType.MEDIA_TRANSCRIPTION: "neo4j",
}

# Repository boolean flags that enable optional processor types (chunking is always on).
PROCESSOR_SETTING_FLAGS: dict[ProcessorType, str | None] = {
    ProcessorType.CHUNKING_VECTORIZING: None,
    ProcessorType.METADATA_EXTRACTION: "metadata_extraction",
    ProcessorType.KEY_FIELD_EXTRACTION: "key_field_extraction",
    ProcessorType.DOCUMENT_VALIDATION: "validation_enabled",
    ProcessorType.TEMPLATE_EXTRACTION: "template_extraction",
    ProcessorType.REFERENCE_DOCUMENT_EXTRACTION: "reference_document_extraction",
    ProcessorType.CONVERSION_FOR_RENDERING: "conversion_for_rendering",
    ProcessorType.INTELLIGENCE_EXTRACTION: "intelligent_extraction",
    # Auto-queued for audio/video uploads; optional repo flag also accepted.
    ProcessorType.MEDIA_TRANSCRIPTION: "media_transcription",
}

# Processors that must wait for listed predecessors to COMPLETE before queueing.
_CHUNKING_COMPLETE = (ProcessorType.CHUNKING_VECTORIZING.value,)
PROCESSOR_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    ProcessorType.METADATA_EXTRACTION.value: _CHUNKING_COMPLETE,
    ProcessorType.KEY_FIELD_EXTRACTION.value: _CHUNKING_COMPLETE,
    ProcessorType.TEMPLATE_EXTRACTION.value: _CHUNKING_COMPLETE,
    ProcessorType.REFERENCE_DOCUMENT_EXTRACTION.value: _CHUNKING_COMPLETE,
    ProcessorType.CONVERSION_FOR_RENDERING.value: _CHUNKING_COMPLETE,
    ProcessorType.INTELLIGENCE_EXTRACTION.value: _CHUNKING_COMPLETE,
    ProcessorType.DOCUMENT_VALIDATION.value: (ProcessorType.KEY_FIELD_EXTRACTION.value,),
}

# Soft deps: enforced only when the dependency was scheduled for this document.
# Media transcription must finish before any text-consuming processor runs on audio/video.
SOFT_PROCESSOR_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    ProcessorType.CHUNKING_VECTORIZING.value: (ProcessorType.MEDIA_TRANSCRIPTION.value,),
    ProcessorType.KEY_FIELD_EXTRACTION.value: (ProcessorType.MEDIA_TRANSCRIPTION.value,),
    ProcessorType.TEMPLATE_EXTRACTION.value: (ProcessorType.MEDIA_TRANSCRIPTION.value,),
    ProcessorType.REFERENCE_DOCUMENT_EXTRACTION.value: (ProcessorType.MEDIA_TRANSCRIPTION.value,),
    ProcessorType.CONVERSION_FOR_RENDERING.value: (ProcessorType.MEDIA_TRANSCRIPTION.value,),
    ProcessorType.INTELLIGENCE_EXTRACTION.value: (ProcessorType.MEDIA_TRANSCRIPTION.value,),
}

LEGACY_PROCESSOR_TYPE_ALIASES: dict[str, str] = {
    "reference_extraction": ProcessorType.REFERENCE_DOCUMENT_EXTRACTION.value,
    "validation_engine": ProcessorType.DOCUMENT_VALIDATION.value,
}

CANONICAL_PROCESSOR_TYPES: frozenset[str] = frozenset(pt.value for pt in ProcessorType)
ALL_PROCESSOR_TYPES: tuple[str, ...] = tuple(CANONICAL_PROCESSOR_TYPES) + tuple(
    alias for alias in LEGACY_PROCESSOR_TYPE_ALIASES if alias not in CANONICAL_PROCESSOR_TYPES
)

# Generic scheduler-managed worker — accepts any processor_type per job payload.
DOC_PROCESSOR = "doc_processor"
DOC_PROCESSOR_CONTAINER_PREFIX = "doc_processor"
REFERENCE_EXTRACTION_CONTAINER_PREFIX = "reference_extraction_processor"


def normalize_processor_type(value: str | None) -> str:
    raw = (value or ProcessorType.CHUNKING_VECTORIZING.value).strip().lower()
    raw = LEGACY_PROCESSOR_TYPE_ALIASES.get(raw, raw)
    if raw not in CANONICAL_PROCESSOR_TYPES:
        raise ValueError(f"Unsupported processor_type '{value}'.")
    return raw


def _filter_by_deployment_processors(processor_types: list[str]) -> list[str]:
    """Drop typed jobs whose deployment-level capability is disabled."""
    try:
        from src.features.document_processing.shared_processor.deployment import (
            DEPLOYMENT_TO_PROCESSOR_TYPES,
            EMBEDDING_DEPENDENT_PROCESSOR_TYPES,
            ProcessorConfigurationProvider,
        )
    except Exception:
        return processor_types

    provider = ProcessorConfigurationProvider.instance()
    blocked: set[str] = set()
    for deployment_id, typed in DEPLOYMENT_TO_PROCESSOR_TYPES.items():
        if not provider.is_processor_enabled(deployment_id):
            blocked.update(typed)
    if not provider.is_processor_enabled("embedding"):
        blocked.update(EMBEDDING_DEPENDENT_PROCESSOR_TYPES)
    if not blocked:
        return processor_types
    return [pt for pt in processor_types if pt not in blocked]


def _setting_flag_enabled(settings: dict[str, Any], flag: str) -> bool:
    """Treat common truthy encodings as enabled (bool True, 1, \"true\", \"yes\", \"on\")."""
    value = settings.get(flag)
    if value is True or value == 1:
        return True
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return False


def enabled_processor_types(settings: dict[str, Any] | None) -> list[str]:
    """Return processor types to queue for a repository's resolved settings.

    Repository flags select optional jobs; deployment processor configuration
    further restricts which capabilities this IKM instance may run.
    """
    merged = settings or {}
    enabled: list[str] = [ProcessorType.CHUNKING_VECTORIZING.value]
    for processor_type, flag in PROCESSOR_SETTING_FLAGS.items():
        if flag is None:
            continue
        if processor_type is ProcessorType.KEY_FIELD_EXTRACTION and (
            _setting_flag_enabled(merged, "key_field_extraction_enabled")
            or _setting_flag_enabled(merged, "key_field_extraction")
        ):
            enabled.append(processor_type.value)
            continue
        if processor_type is ProcessorType.INTELLIGENCE_EXTRACTION and (
            _setting_flag_enabled(merged, "intelligent_extraction")
            or _setting_flag_enabled(merged, "content_intelligence")
            or _setting_flag_enabled(merged, "intelligence_extraction")
        ):
            enabled.append(processor_type.value)
            continue
        if _setting_flag_enabled(merged, flag):
            enabled.append(processor_type.value)
    # Validation requires key-field extraction results; drop it when extraction is off.
    if ProcessorType.DOCUMENT_VALIDATION.value in enabled:
        has_key_fields = (
            ProcessorType.KEY_FIELD_EXTRACTION.value in enabled
            or _setting_flag_enabled(merged, "key_field_extraction")
            or _setting_flag_enabled(merged, "key_field_extraction_enabled")
        )
        if not has_key_fields:
            enabled = [pt for pt in enabled if pt != ProcessorType.DOCUMENT_VALIDATION.value]
    return _filter_by_deployment_processors(enabled)


def processor_dependencies_met(
    processor_type: str,
    processor_results: dict[str, Any] | None,
    *,
    enabled_types: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Return True when required predecessor processors have COMPLETED.

    Hard dependencies (``PROCESSOR_DEPENDENCIES``) always block until COMPLETED.
    Soft dependencies (``SOFT_PROCESSOR_DEPENDENCIES``) block only when that
    predecessor was scheduled for the document (present in ``enabled_types`` or
    already tracked in ``processor_results``).
    """
    key = str(processor_type or "").strip().lower()
    results = processor_results if isinstance(processor_results, dict) else {}
    enabled = {
        str(item or "").strip().lower()
        for item in (enabled_types or [])
        if str(item or "").strip()
    }

    for dep in PROCESSOR_DEPENDENCIES.get(key, ()):
        outcome = results.get(dep) or {}
        if str(outcome.get("status") or "").upper() != "COMPLETED":
            return False

    for dep in SOFT_PROCESSOR_DEPENDENCIES.get(key, ()):
        scheduled = (dep in enabled) if enabled else (dep in results)
        if not scheduled:
            continue
        outcome = results.get(dep) or {}
        status = str(outcome.get("status") or "").upper()
        # FAILED predecessor will never become COMPLETED — stop blocking forever so
        # the job can settle; dependents that need the artifact will fail once clearly.
        if status == "FAILED":
            continue
        if status != "COMPLETED":
            return False
    return True


def parse_processor_pool(raw: str | None, *, default_chunking_slots: int = 5) -> dict[str, int]:
    """Parse ``chunking_vectorizing:3,metadata_extraction:1`` pool configuration."""
    text = (raw or "").strip()
    if not text:
        return {ProcessorType.CHUNKING_VECTORIZING.value: max(1, default_chunking_slots)}
    pool: dict[str, int] = {}
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        name, _, count_text = piece.partition(":")
        name = normalize_processor_type(name.strip())
        count = int((count_text or "1").strip())
        pool[name] = max(1, count)
    if ProcessorType.CHUNKING_VECTORIZING.value not in pool:
        pool[ProcessorType.CHUNKING_VECTORIZING.value] = max(1, default_chunking_slots)
    return pool


def total_processor_slots(pool: dict[str, int] | None, *, default: int = 5) -> int:
    """Return the number of generic doc_processor containers to run."""
    if not pool:
        return max(1, default)
    return max(1, sum(pool.values()))
