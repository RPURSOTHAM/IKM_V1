from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.features.document_processing.core.config import get_settings
from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.features.document_processing.core.logger import log
from src.infrastructure.document_databases.weaviate_store import store_in_weaviate
from src.features.citations.application.citation_anchor_builder import apply_citation_anchors
from src.features.citations.application.citation_metadata import clamp_chunks_to_document_page_count
from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy
from src.features.embeddings.application.embedding_service import embed_chunks, encode_texts
from src.features.document_processing.loaders.component_classification import (
    blocks_for_chunking,
    image_block_chunk_text,
    is_image_block,
)
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.document_processing.pipeline_tracking import (
    explain_zero_chunks,
    log_block_diagnostics,
    set_pipeline_stage,
    validate_pipeline_request,
)
from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_processing.processors.processing_settings import resolve_chunking_settings
from src.features.document_processing.utilities.chunk_classifier import apply_document_classification
from src.features.document_processing.shared_processor.types import ProcessorType


class ChunkingVectorizingProcessor(BaseProcessor):
    processor_type = ProcessorType.CHUNKING_VECTORIZING

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        settings = get_settings()
        resolved = resolve_chunking_settings(request)

        weaviate_url = request.weaviate_url or settings.weaviate_url
        weaviate_api_key = request.weaviate_api_key or settings.weaviate_api_key
        collection_name = request.collection_name or settings.collection_name
        doc_name = request.document_name or document_path.name
        display_name = request.original_file_name or doc_name

        set_pipeline_stage("validating_request", 2.0)
        set_status("validating_request", 2.0)
        validate_pipeline_request(
            document_path=document_path,
            chunking_strategy=resolved.chunking_strategy,
            model_name=resolved.model_name,
            model_dir=resolved.model_dir,
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key,
            collection_name=collection_name,
            check_neo4j=False,
        )

        set_pipeline_stage("loading_document", 5.0)
        set_status("loading_document", 5.0)
        blocks, doc_meta = load_document_blocks(
            document_path,
            document_id=request.document_id,
            citation_retainment=bool(resolved.citation_retainment),
        )
        try:
            from src.features.documents.application.document_extraction_cache import (
                build_extraction_payload,
                sections_from_blocks,
                write_extraction_sidecar,
            )

            suffix = document_path.suffix.lower()
            extraction_sections = sections_from_blocks(blocks, suffix=suffix)
            if extraction_sections:
                write_extraction_sidecar(
                    document_path,
                    build_extraction_payload(
                        document_id=request.document_id,
                        filename=display_name,
                        suffix=suffix,
                        sections=extraction_sections,
                        source="document_blocks",
                        file_extension=str(doc_meta.get("file_extension") or suffix.lstrip(".")),
                    ),
                )
        except Exception:
            log.debug("Extraction sidecar write skipped", exc_info=True)

        set_pipeline_stage("extracting_document_blocks", 8.0)
        set_status("extracting_document_blocks", 8.0)
        blocks = blocks_for_chunking(blocks)
        block_stats = log_block_diagnostics(blocks)
        has_text_or_table_content = any(
            (
                not is_image_block(block)
                and str(getattr(block, "text", "") or "").strip()
            )
            or (is_image_block(block) and bool(image_block_chunk_text(block)))
            for block in blocks
        )
        if not blocks or not has_text_or_table_content:
            reason = explain_zero_chunks(
                block_stats,
                had_blocks=bool(blocks),
                had_text=has_text_or_table_content,
            )
            raise RuntimeError(f"No chunkable content after excluding image blocks. {reason}")

        set_pipeline_stage("chunking", 10.0)
        set_status("chunking", 10.0)
        embed_fn = None
        if resolved.chunking_strategy in {"semantic", "semantic-hierarchy"}:
            embed_fn = lambda texts: encode_texts(
                texts,
                resolved.model_name,
                model_dir=resolved.model_dir,
            )
        chunks = chunk_document_with_strategy(
            blocks,
            display_name,
            strategy=resolved.chunking_strategy,
            chunk_size=resolved.chunk_size,
            overlap_sentences=resolved.chunk_overlap_sentences,
            min_content_words=resolved.min_content_words,
            chunking_config=resolved.chunking_config,
            embed_fn=embed_fn,
            document_id=request.document_id,
            citation_retainment=resolved.citation_retainment,
        )
        for chunk in chunks:
            chunk.strategy_name = resolved.chunking_strategy
            chunk.document_id = request.document_id
            chunk.document_name = display_name
            chunk.heading = chunk.heading or chunk.section_name
            chunk.topic = chunk.topic or chunk.category or "General"
            chunk.sensitivity = chunk.sensitivity or "unknown"
            chunk.embedding_version = resolved.model_name
        block_stats["chunk_count"] = len(chunks)
        log.info("Chunk count after chunking: %s", len(chunks))
        if check_stop():
            raise RuntimeError("Processing stopped by operator.")
        if not chunks:
            reason = explain_zero_chunks(
                block_stats,
                had_blocks=bool(blocks),
                had_text=has_text_or_table_content,
            )
            raise RuntimeError(f"No chunks produced for document. {reason}")
        if len(chunks) > resolved.max_chunks:
            raise RuntimeError(f"Document produced {len(chunks)} chunks; limit is {resolved.max_chunks}.")

        if not resolved.citation_retainment:
            for chunk in chunks:
                chunk.page = None
                chunk.end_page = None
                chunk.page_start = None
                chunk.page_end = None
                chunk.line_start = None
                chunk.line_end = None
                chunk.citation_anchor = None
        else:
            try:
                page_count = int(doc_meta.get("page_count") or 0) or None
                clamp_chunks_to_document_page_count(chunks, page_count)
                apply_citation_anchors(chunks)
            except Exception:
                log.exception("Citation anchor application failed; continuing without anchors.")

        apply_document_classification(chunks)

        set_pipeline_stage("embedding", 45.0)
        set_status("embedding", 45.0)
        from src.features.repositories.infrastructure.collection_naming import local_embedding_model_from_paths

        repository_embedding_model = local_embedding_model_from_paths(
            resolved.model_name,
            resolved.model_dir,
        )
        embed_chunks(
            chunks,
            resolved.model_name,
            rewrite_mode="rule_based",
            rewrite_workers=settings.max_workers,
            rewrite_max_tokens=700,
            rewrite_timeout_seconds=60,
            min_chunk_words=resolved.min_content_words,
            model_dir=resolved.model_dir,
            repository_embedding_model=repository_embedding_model,
        )
        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        embedded_count = len([c for c in chunks if c.embedding is not None and not c.is_duplicate])
        if embedded_count == 0:
            raise RuntimeError(
                f"Embedding produced zero vectors for {len(chunks)} chunk(s). "
                "Check embedding model configuration and chunk text content."
            )

        set_pipeline_stage("storing_weaviate", 80.0)
        set_status("storing_weaviate", 80.0)
        indexing_meta = dict(doc_meta or {})
        indexing_meta["document_id"] = request.document_id
        if request.repository_id:
            indexing_meta["repository_id"] = request.repository_id
        try:
            from src.features.logical_folders.application.folder_service import get_logical_folder_service

            assignment = get_logical_folder_service().assignment_for_document(request.document_id)
            if assignment:
                indexing_meta["logical_folder_id"] = assignment.get("folder_id")
                indexing_meta["logical_folder_path"] = assignment.get("path")
        except Exception:
            log.debug("logical folder lookup skipped during indexing", exc_info=True)
        store_in_weaviate(
            chunks,
            weaviate_url,
            collection_name,
            weaviate_api_key,
            tenant_id=request.tenant_id,
            document_metadata=indexing_meta,
        )

        set_pipeline_stage("completed", 100.0)
        citation_metadata = {
            **doc_meta,
            "citation_retainment": resolved.citation_retainment,
            "chunking_strategy": resolved.chunking_strategy,
            "chunk_count": len(chunks),
            "embedded_chunk_count": embedded_count,
            "collection_name": collection_name,
            "tenant_id": request.tenant_id,
            "document_id": request.document_id,
            "document_name": doc_name,
            "original_file_name": display_name,
            "embedding_model_name": resolved.model_name,
            "embedding_model_dir": resolved.model_dir,
            **block_stats,
        }
        location = f"weaviate://{collection_name}/{request.document_id}"
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata=citation_metadata,
            artifacts={
                "embedded_chunks": embedded_count,
                "num_chunks": len(chunks),
            },
        )
