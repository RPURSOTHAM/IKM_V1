"""Concrete deployment processor adapters.

Heavy dependencies are imported lazily inside initialize()/process() so
disabled processors never load associated models or libraries.
"""

from __future__ import annotations

import logging
from typing import Any

from src.features.document_processing.shared_processor.deployment.ids import DeploymentProcessorId
from src.features.document_processing.shared_processor.deployment.interface import DeploymentProcessor

logger = logging.getLogger(__name__)


class PdfProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.PDF

    def _do_initialize(self) -> None:
        # Import loader only when enabled — validates the capability is available.
        from src.features.document_processing.loaders.pdfloader import PdfLoader  # noqa: F401

        logger.debug("PDF processor initialized")

    def supports(self, context: dict[str, Any]) -> bool:
        path = str(context.get("document_path") or context.get("suffix") or "")
        return path.lower().endswith(".pdf") or str(context.get("file_extension") or "").lower() == "pdf"

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        from pathlib import Path

        from src.features.document_processing.loaders.pdfloader import PdfLoader

        document_path = Path(context["document_path"])
        blocks = PdfLoader().load(document_path)
        return {"blocks": blocks, "loader": "pdf"}


class DocxProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.DOCX

    def _do_initialize(self) -> None:
        from src.features.document_processing.loaders.docxloader import DocxLoader  # noqa: F401

        logger.debug("DOCX processor initialized")

    def supports(self, context: dict[str, Any]) -> bool:
        path = str(context.get("document_path") or context.get("suffix") or "")
        return path.lower().endswith(".docx") or str(context.get("file_extension") or "").lower() == "docx"

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        from pathlib import Path

        from src.features.document_processing.loaders.docxloader import DocxLoader

        document_path = Path(context["document_path"])
        blocks = DocxLoader().load(document_path)
        return {"blocks": blocks, "loader": "docx"}


class PptxProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.PPTX

    def _do_initialize(self) -> None:
        from src.features.document_processing.loaders.pptxloader import PptxLoader  # noqa: F401

        logger.debug("PPTX processor initialized")

    def supports(self, context: dict[str, Any]) -> bool:
        path = str(context.get("document_path") or context.get("suffix") or "").lower()
        ext = str(context.get("file_extension") or "").lower()
        return path.endswith((".pptx", ".ppt")) or ext in {"pptx", "ppt"}

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        from pathlib import Path

        from src.features.document_processing.loaders.document_text import _resolve_ppt_path
        from src.features.document_processing.loaders.pptxloader import PptxLoader

        document_path = _resolve_ppt_path(Path(context["document_path"]))
        blocks = PptxLoader().load(document_path)
        return {"blocks": blocks, "loader": "pptx"}


class MediaProcessor(DeploymentProcessor):
    """Deployment gate for audio/video transcription capability."""

    processor_id = DeploymentProcessorId.MEDIA

    def _do_initialize(self) -> None:
        logger.debug("Media transcription capability initialized")

    def supports(self, context: dict[str, Any]) -> bool:
        path = str(context.get("document_path") or context.get("suffix") or "").lower()
        ext = str(context.get("file_extension") or "").lower()
        media = {
            "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma",
            "mp4", "webm", "mov", "avi", "mkv", "mpeg", "m4v", "wmv",
        }
        return any(path.endswith(f".{m}") for m in media) or ext in media

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"media_transcription": True}


class OcrProcessor(DeploymentProcessor):
    """Tesseract OCR for uploaded images and scanned (image-only) PDFs."""

    processor_id = DeploymentProcessorId.OCR

    def _do_initialize(self) -> None:
        logger.debug("OCR processor initialized (Tesseract for images and scanned PDFs)")

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"ocr_applied": True, "ocr_status": "tesseract"}


class SecurityProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.SECURITY

    def _do_initialize(self) -> None:
        from src.features.security.dlp.policy_loader import validate_policy_configuration

        # Touch policy loader so missing configs fail at startup when security is enabled.
        try:
            validate_policy_configuration(force_reload=False)
        except Exception as exc:
            logger.warning("Security policy preload warning: %s", exc)
        logger.debug("Security processor initialized")

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        # Security remains gated at the /process entrypoint; this stage marks capability.
        return {"security_enabled": True}


class KeyFieldsProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.KEY_FIELDS

    def _do_initialize(self) -> None:
        from src.features.document_processing.processors.key_field_extraction import (  # noqa: F401
            KeyFieldExtractionProcessor,
        )

        logger.debug("Key Fields processor initialized")

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"key_fields_enabled": True}


class ReferenceExtractionProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.REFERENCE_EXTRACTION

    def _do_initialize(self) -> None:
        from src.features.document_processing.processors.reference_extraction import (  # noqa: F401
            ReferenceDocumentExtractionProcessor,
        )

        logger.debug("Reference Extraction processor initialized")

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"reference_extraction_enabled": True}


class ImageProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.IMAGE

    def _do_initialize(self) -> None:
        from src.features.document_processing.loaders.component_classification import (  # noqa: F401
            is_image_block,
        )

        logger.debug("Image processor initialized")

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        blocks = context.get("blocks") or []
        from src.features.document_processing.loaders.component_classification import is_image_block

        image_blocks = [b for b in blocks if is_image_block(b)]
        return {"image_blocks": image_blocks, "image_block_count": len(image_blocks)}


class EmbeddingProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.EMBEDDING
    _models_touched: bool = False

    def _do_initialize(self) -> None:
        # Do not preload the sentence-transformers model here — that remains lazy
        # on first embed. Initialization only proves the module is importable.
        from src.features.embeddings.application import embedding_service as _embedding  # noqa: F401

        self._models_touched = True
        logger.debug("Embedding processor initialized (model loads on first use)")

    def _do_shutdown(self) -> None:
        if not self._models_touched:
            return
        try:
            from src.features.embeddings.application.embedding_service import clear_embedding_model_cache

            clear_embedding_model_cache()
        except Exception:
            logger.debug("Embedding cache clear during shutdown failed.", exc_info=True)
        self._models_touched = False

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"embedding_enabled": True}


class RerankerProcessor(DeploymentProcessor):
    processor_id = DeploymentProcessorId.RERANKER
    _models_touched: bool = False

    def _do_initialize(self) -> None:
        # Reranking is retrieval-time functionality hosted in dms_service, not in the
        # processor image. Keep initialization lightweight here so processor workers do
        # not crash when the capability is enabled deployment-wide.
        self._models_touched = True
        logger.debug("Reranker processor initialized (runtime lives in dms-service)")

    def _do_shutdown(self) -> None:
        if not self._models_touched:
            return
        try:
            from src.features.retrieval.application.retrieval_service import clear_reranker_model_cache

            clear_reranker_model_cache()
        except Exception:
            logger.debug("Reranker cache clear during shutdown failed.", exc_info=True)
        self._models_touched = False

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"reranker_enabled": True}


# Factory map used by the registry — classes only, no instances yet.
PROCESSOR_CLASSES: dict[DeploymentProcessorId, type[DeploymentProcessor]] = {
    DeploymentProcessorId.PDF: PdfProcessor,
    DeploymentProcessorId.DOCX: DocxProcessor,
    DeploymentProcessorId.PPTX: PptxProcessor,
    DeploymentProcessorId.MEDIA: MediaProcessor,
    DeploymentProcessorId.OCR: OcrProcessor,
    DeploymentProcessorId.SECURITY: SecurityProcessor,
    DeploymentProcessorId.KEY_FIELDS: KeyFieldsProcessor,
    DeploymentProcessorId.REFERENCE_EXTRACTION: ReferenceExtractionProcessor,
    DeploymentProcessorId.IMAGE: ImageProcessor,
    DeploymentProcessorId.EMBEDDING: EmbeddingProcessor,
    DeploymentProcessorId.RERANKER: RerankerProcessor,
}
