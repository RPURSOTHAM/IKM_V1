from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.features.document_processing.core.config import get_settings
from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.infrastructure.document_databases.neo4j_store import store_document_graph
from src.features.embeddings.application.embedding_service import encode_texts
from src.features.document_processing.loaders.component_classification import blocks_for_chunking
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.document_processing.metadata.field_extractor import extract_metadata_fields
from src.features.document_processing.metadata.field_discovery import (
    discover_structured_fields,
    merge_extracted_with_discovered,
)
from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_processing.shared_processor.types import ProcessorType


class MetadataExtractionProcessor(BaseProcessor):
    processor_type = ProcessorType.METADATA_EXTRACTION

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        fields = list(request.metadata_fields or [])
        if not fields and request.document_type_id:
            fields = self._load_fields_from_document_type(request.document_type_id)

        set_status("loading_document", 10.0)
        blocks, doc_meta = load_document_blocks(document_path)
        blocks = blocks_for_chunking(blocks)
        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        set_status("extracting_metadata", 40.0)
        embed_fn = self._build_embed_fn(request)
        configured_extracted = (
            extract_metadata_fields(
                blocks,
                fields,
                extraction_model=request.extraction_model,
                embed_fn=embed_fn,
            )
            if fields
            else []
        )
        # Preserve configured extraction and append generic label/value fields
        # used by the repository-scoped logical-folder feature.
        extracted = merge_extracted_with_discovered(
            configured_extracted,
            discover_structured_fields(blocks),
        )
        payload = {
            "document_type_id": request.document_type_id,
            "fields": extracted,
            "source": "metadata_extraction",
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
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata={
                **doc_meta,
                "extracted_field_count": len([item for item in extracted if item.get("value") is not None]),
                "extracted_fields": extracted,
                "document_type_id": request.document_type_id,
            },
            artifacts={"extracted_fields": extracted},
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

    def _load_fields_from_document_type(self, document_type_id: str) -> list[dict]:
        try:
            from src.features.document_types.application.processing_fields import resolve_metadata_fields_for_type

            return resolve_metadata_fields_for_type(document_type_id)
        except Exception:
            try:
                from src.features.document_types.infrastructure.document_type_repository import get_document_type_store

                store = get_document_type_store()
                if not store:
                    return []
                bundle = store.resolve_metadata_bundle(document_type_id)
                from src.features.document_types.application.processing_fields import metadata_field_to_processing_dict

                return [
                    metadata_field_to_processing_dict(field)
                    for field in bundle.effective
                    if field.is_active
                ]
            except Exception:
                return []
