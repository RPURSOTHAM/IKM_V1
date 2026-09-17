from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from src.features.document_processing.core.config import get_settings
from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.infrastructure.document_databases.neo4j_store import store_document_graph
from src.features.embeddings.application.embedding_service import encode_texts
from src.features.document_processing.key_fields.page_context import (
    filter_blocks_for_extraction,
    select_extraction_page_numbers,
)
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.document_processing.metadata.field_discovery import (
    discover_structured_fields,
    merge_extracted_with_discovered,
)
from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_processing.services.key_field_extractor import extract_key_fields
from src.features.document_processing.shared_processor.types import ProcessorType

logger = logging.getLogger(__name__)


class KeyFieldExtractionProcessor(BaseProcessor):
    processor_type = ProcessorType.KEY_FIELD_EXTRACTION

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        document_type_id = request.document_type_id
        if not document_type_id:
            set_status("skipped_no_document_type", 100.0)
            skip_message = "document_type_id is required for key field extraction."
            return ProcessorResult(
                processor_type=self.processor_type.value,
                document_id=request.document_id,
                storage_backend=self.storage_backend,
                result_location=None,
                document_metadata={
                    "skipped": True,
                    "skip_reason": "no_document_type_id",
                    "message": skip_message,
                    "extracted_field_count": 0,
                },
                artifacts={"fields": {}, "skip_reason": "no_document_type_id", "message": skip_message},
            )

        fields = list(request.key_fields or [])
        if not fields:
            fields = self._load_key_fields(document_type_id)
        if not fields:
            set_status("skipped_no_key_fields", 100.0)
            skip_message = "No effective key fields configured for document type."
            return ProcessorResult(
                processor_type=self.processor_type.value,
                document_id=request.document_id,
                storage_backend=self.storage_backend,
                result_location=None,
                document_metadata={
                    "skipped": True,
                    "skip_reason": "no_key_fields_to_extract",
                    "message": skip_message,
                    "document_type_id": document_type_id,
                    "extracted_field_count": 0,
                },
                artifacts={
                    "fields": {},
                    "skip_reason": "no_key_fields_to_extract",
                    "message": skip_message,
                },
            )

        set_status("loading_document", 10.0)
        blocks, doc_meta = load_document_blocks(document_path, document_id=request.document_id)
        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        page_count = int(doc_meta.get("page_count") or 1)
        source_pages = select_extraction_page_numbers(page_count)
        strict_page_scope = bool(getattr(request, "strict_key_field_page_scope", False))
        context_blocks = filter_blocks_for_extraction(
            blocks,
            page_count,
            strict_page_scope=strict_page_scope,
        )
        if strict_page_scope and not context_blocks:
            logger.warning(
                "Skipping key field extraction for document_id=%s due to missing page metadata with strict page scope enabled.",
                request.document_id,
            )
            set_status("skipped_missing_page_metadata", 100.0)
            return ProcessorResult(
                processor_type=self.processor_type.value,
                document_id=request.document_id,
                storage_backend=self.storage_backend,
                result_location=None,
                document_metadata={
                    **doc_meta,
                    "document_type_id": document_type_id,
                    "source_pages": source_pages,
                    "skipped": True,
                    "skip_reason": "missing_page_metadata_strict_scope",
                    "extracted_field_count": 0,
                },
                artifacts={
                    "fields": {},
                    "source_pages": source_pages,
                    "skip_reason": "missing_page_metadata_strict_scope",
                },
            )

        set_status("extracting_key_fields", 40.0)
        embed_fn = self._build_embed_fn(request)
        extracted, fields_map = extract_key_fields(
            context_blocks,
            fields,
            extraction_model=request.extraction_model,
            embed_fn=embed_fn,
        )
        # Keep configured key-field values intact and append generic structured
        # label/value fields for repository logical-folder assignment.
        extracted = merge_extracted_with_discovered(
            extracted,
            discover_structured_fields(context_blocks),
        )
        payload = {
            "document_type_id": document_type_id,
            "source": "key_field_extraction",
            "source_pages": source_pages,
            "page_count": page_count,
            "fields": fields_map,
        }

        set_status("storing", 85.0)
        location = store_document_graph(
            document_id=request.document_id,
            repository_id=request.repository_id,
            processor_type=self.processor_type.value,
            payload=payload,
            document_type_id=request.document_type_id,
            document_type_name=request.document_type_name,
        )
        extracted_count = len([item for item in extracted if item.get("value") is not None])
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata={
                **doc_meta,
                "document_type_id": document_type_id,
                "source_pages": source_pages,
                "extracted_field_count": extracted_count,
                "extracted_fields": extracted,
            },
            artifacts={
                "fields": fields_map,
                "extracted_fields": extracted,
                "source_pages": source_pages,
            },
        )

    def _build_embed_fn(self, request: ProcessRequest):
        extraction_model = request.extraction_model or {}
        provider = str(extraction_model.get("provider") or "").strip().lower()
        if provider in {"none", "disabled"}:
            return None
        if not extraction_model.get("model_id") and not extraction_model.get("local_model_dir"):
            return None

        settings = get_settings()
        model_name = extraction_model.get("model_id") or request.model_name or settings.model_name
        model_dir = extraction_model.get("local_model_dir") or request.model_dir or settings.model_dir
        if not model_name and not model_dir:
            return None

        def _embed(texts: list[str]):
            return encode_texts(texts, str(model_name), model_dir=model_dir)

        return _embed

    def _load_key_fields(self, document_type_id: str) -> list[dict[str, Any]]:
        try:
            from src.features.document_types.application.processing_key_fields import resolve_key_fields_for_type

            return resolve_key_fields_for_type(document_type_id)
        except Exception:
            return []
