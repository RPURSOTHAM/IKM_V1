from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.features.document_processing.core.logger import log
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.references.infrastructure.neo4j_reference_store import get_reference_store
from src.features.references.extraction.reference_extractor import (
    blocks_to_document_text,
    extract_pre_chunk_references,
    load_trigger_phrases,
)
from src.features.document_processing.shared_processor.types import ProcessorType


def run_pre_chunk_reference_extraction(
    request: ProcessRequest,
    document_path: Path,
    *,
    set_status: StatusCallback,
    check_stop: Callable[[], bool],
) -> ProcessorResult:
    display_name = request.original_file_name or request.document_name or document_path.name
    source_document_id = request.document_id
    tenant_id = request.tenant_id
    repository_id = request.repository_id

    set_status("loading_document", 10.0)
    blocks, doc_meta = load_document_blocks(document_path)
    document_text = blocks_to_document_text(blocks)
    if not document_text.strip():
        raise RuntimeError("No readable text found in document for reference extraction.")
    log.info("Document loaded: document_id=%s blocks=%s", source_document_id, len(blocks))

    if check_stop():
        raise RuntimeError("Processing stopped by operator.")

    set_status("detecting_trigger_phrases", 30.0)
    references = extract_pre_chunk_references(
        document_text,
        source_document_id=source_document_id,
        source_document_name=display_name,
        tenant_id=tenant_id,
        repository_id=repository_id,
        trigger_phrases=load_trigger_phrases(),
    )
    log.info("Regex match found: reference_count=%s document_id=%s", len(references), source_document_id)

    if check_stop():
        raise RuntimeError("Processing stopped by operator.")

    set_status("saving_to_neo4j", 70.0)
    store = get_reference_store()
    if not store.enabled:
        raise RuntimeError("Neo4j is not configured. Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.")

    store.register_document(
        document_id=source_document_id,
        document_name=display_name,
        source_title=display_name,
        tenant_id=tenant_id,
        repository_id=repository_id,
    )
    save_stats = store.save_document_reference_graph(
        source_document_id=source_document_id,
        source_title=display_name,
        tenant_id=tenant_id,
        repository_id=repository_id,
        references=references,
    )
    auto_resolved = store.resolve_pending_references(
        uploaded_document_id=source_document_id,
        uploaded_document_name=display_name,
        tenant_id=tenant_id,
        repository_id=repository_id,
    )
    if auto_resolved:
        log.info("Automatic resolution executed: resolved=%s", auto_resolved)

    set_status("completed", 100.0)
    log.info(
        "Completed reference extraction for document_id=%s saved=%s resolved=%s unresolved=%s auto_resolved=%s",
        source_document_id,
        save_stats.get("saved", 0),
        save_stats.get("resolved", 0),
        save_stats.get("unresolved", 0),
        auto_resolved,
    )
    return ProcessorResult(
        processor_type=ProcessorType.REFERENCE_DOCUMENT_EXTRACTION.value,
        document_id=source_document_id,
        storage_backend="neo4j",
        result_location=f"neo4j://Document/{source_document_id}/REFERENCES",
        document_metadata={
            **doc_meta,
            "document_id": source_document_id,
            "document_name": display_name,
            "tenant_id": tenant_id,
            "repository_id": repository_id,
        },
        artifacts={
            "reference_count": len(references),
            "saved_reference_count": int(save_stats.get("saved", 0)),
            "resolved_reference_count": int(save_stats.get("resolved", 0)),
            "unresolved_reference_count": int(save_stats.get("unresolved", 0)),
            "auto_resolved_reference_count": int(auto_resolved),
        },
    )
