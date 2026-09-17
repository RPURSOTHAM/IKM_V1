"""Canonical deployment-level processor identifiers."""

from __future__ import annotations

from enum import Enum


class DeploymentProcessorId(str, Enum):
    """Processors gated at IKM deployment / licensing scope."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    MEDIA = "media"
    OCR = "ocr"
    SECURITY = "security"
    KEY_FIELDS = "key_fields"
    REFERENCE_EXTRACTION = "reference_extraction"
    IMAGE = "image"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


# Stable catalog order for logging and API responses.
DEPLOYMENT_PROCESSOR_ORDER: tuple[DeploymentProcessorId, ...] = (
    DeploymentProcessorId.PDF,
    DeploymentProcessorId.DOCX,
    DeploymentProcessorId.PPTX,
    DeploymentProcessorId.MEDIA,
    DeploymentProcessorId.OCR,
    DeploymentProcessorId.SECURITY,
    DeploymentProcessorId.KEY_FIELDS,
    DeploymentProcessorId.REFERENCE_EXTRACTION,
    DeploymentProcessorId.IMAGE,
    DeploymentProcessorId.EMBEDDING,
    DeploymentProcessorId.RERANKER,
)

DEPLOYMENT_PROCESSOR_DISPLAY_NAMES: dict[DeploymentProcessorId, str] = {
    DeploymentProcessorId.PDF: "PDF",
    DeploymentProcessorId.DOCX: "DOCX",
    DeploymentProcessorId.PPTX: "PowerPoint",
    DeploymentProcessorId.MEDIA: "Media Transcription",
    DeploymentProcessorId.OCR: "OCR",
    DeploymentProcessorId.SECURITY: "Security",
    DeploymentProcessorId.KEY_FIELDS: "Key Fields",
    DeploymentProcessorId.REFERENCE_EXTRACTION: "Reference Extraction",
    DeploymentProcessorId.IMAGE: "Image",
    DeploymentProcessorId.EMBEDDING: "Embedding",
    DeploymentProcessorId.RERANKER: "Reranker",
}

# Map deployment capabilities → typed ingest jobs they authorize.
DEPLOYMENT_TO_PROCESSOR_TYPES: dict[DeploymentProcessorId, tuple[str, ...]] = {
    DeploymentProcessorId.KEY_FIELDS: (
        "key_field_extraction",
        "document_validation",
    ),
    DeploymentProcessorId.REFERENCE_EXTRACTION: (
        "reference_document_extraction",
        "reference_extraction",
    ),
    DeploymentProcessorId.MEDIA: (
        "media_transcription",
    ),
}

# Typed jobs that require embedding capability at deployment scope.
EMBEDDING_DEPENDENT_PROCESSOR_TYPES: frozenset[str] = frozenset(
    {
        "chunking_vectorizing",
    }
)

CANONICAL_DEPLOYMENT_PROCESSOR_IDS: frozenset[str] = frozenset(
    pid.value for pid in DeploymentProcessorId
)


def display_name(processor_id: str | DeploymentProcessorId) -> str:
    if isinstance(processor_id, DeploymentProcessorId):
        return DEPLOYMENT_PROCESSOR_DISPLAY_NAMES[processor_id]
    try:
        return DEPLOYMENT_PROCESSOR_DISPLAY_NAMES[DeploymentProcessorId(processor_id)]
    except ValueError:
        return str(processor_id).replace("_", " ").title()
