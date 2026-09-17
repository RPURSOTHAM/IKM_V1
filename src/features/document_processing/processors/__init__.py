from __future__ import annotations

from collections.abc import Callable

from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_processing.shared_processor.types import ProcessorType, normalize_processor_type

# Lazy factories — processors are only constructed when first requested AND when
# the corresponding deployment capability (if any) is enabled.
_PROCESSOR_FACTORIES: dict[str, Callable[[], BaseProcessor]] = {}
_PROCESSORS: dict[str, BaseProcessor] = {}


def _register_factories() -> None:
    if _PROCESSOR_FACTORIES:
        return

    def _chunking():
        from src.features.document_processing.processors.chunking_vectorizing import ChunkingVectorizingProcessor

        return ChunkingVectorizingProcessor()

    def _metadata():
        from src.features.document_processing.processors.metadata_extraction import MetadataExtractionProcessor

        return MetadataExtractionProcessor()

    def _key_fields():
        from src.features.document_processing.processors.key_field_extraction import KeyFieldExtractionProcessor

        return KeyFieldExtractionProcessor()

    def _validation():
        from src.features.document_processing.processors.document_validation import DocumentValidationProcessor

        return DocumentValidationProcessor()

    def _template():
        from src.features.document_processing.processors.template_extraction import TemplateExtractionProcessor

        return TemplateExtractionProcessor()

    def _reference():
        from src.features.document_processing.processors.reference_extraction import (
            ReferenceDocumentExtractionProcessor,
        )

        return ReferenceDocumentExtractionProcessor()

    def _conversion():
        from src.features.document_processing.processors.conversion_rendering import ConversionForRenderingProcessor

        return ConversionForRenderingProcessor()

    def _intelligence():
        from src.features.document_processing.processors.intelligence_extraction import (
            IntelligenceExtractionProcessor,
        )

        return IntelligenceExtractionProcessor()

    def _media():
        from src.features.document_processing.processors.media_transcription import (
            MediaTranscriptionProcessor,
        )

        return MediaTranscriptionProcessor()

    _PROCESSOR_FACTORIES.update(
        {
            ProcessorType.CHUNKING_VECTORIZING.value: _chunking,
            ProcessorType.METADATA_EXTRACTION.value: _metadata,
            ProcessorType.KEY_FIELD_EXTRACTION.value: _key_fields,
            ProcessorType.DOCUMENT_VALIDATION.value: _validation,
            ProcessorType.TEMPLATE_EXTRACTION.value: _template,
            ProcessorType.REFERENCE_DOCUMENT_EXTRACTION.value: _reference,
            ProcessorType.CONVERSION_FOR_RENDERING.value: _conversion,
            ProcessorType.INTELLIGENCE_EXTRACTION.value: _intelligence,
            ProcessorType.MEDIA_TRANSCRIPTION.value: _media,
        }
    )


def _deployment_allows(processor_type: str) -> bool:
    """Return False when a typed job is gated off by deployment processor config."""
    try:
        from src.features.document_processing.shared_processor.deployment import (
            DEPLOYMENT_TO_PROCESSOR_TYPES,
            EMBEDDING_DEPENDENT_PROCESSOR_TYPES,
            ProcessorConfigurationProvider,
        )
    except Exception:
        return True

    provider = ProcessorConfigurationProvider.instance()
    if processor_type in EMBEDDING_DEPENDENT_PROCESSOR_TYPES and not provider.is_processor_enabled(
        "embedding"
    ):
        return False
    for deployment_id, typed in DEPLOYMENT_TO_PROCESSOR_TYPES.items():
        if processor_type in typed and not provider.is_processor_enabled(deployment_id):
            return False
    return True


def get_processor(processor_type: str) -> BaseProcessor:
    _register_factories()
    key = normalize_processor_type(processor_type)
    if not _deployment_allows(key):
        raise ValueError(
            f"Processor '{processor_type}' is disabled by deployment configuration "
            "and was never initialized."
        )
    processor = _PROCESSORS.get(key)
    if processor is None:
        factory = _PROCESSOR_FACTORIES.get(key)
        if factory is None:
            raise ValueError(f"No processor registered for type '{processor_type}'.")
        processor = factory()
        _PROCESSORS[key] = processor
    return processor


def container_processor_type() -> str | None:
    import os

    raw = (os.getenv("PROCESSOR_TYPE") or "").strip().lower()
    if raw in {"", "doc_processor", "generic", "any"}:
        return None
    return normalize_processor_type(os.getenv("PROCESSOR_TYPE"))


def is_generic_processor_container() -> bool:
    return container_processor_type() is None


def reset_processor_cache() -> None:
    """Test helper — drop lazily created BaseProcessor instances."""
    _PROCESSORS.clear()
