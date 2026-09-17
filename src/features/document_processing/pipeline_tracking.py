"""Pipeline stage tracking, pre-flight validation, and failure snapshots."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.features.document_processing.core.logger import log
from src.features.document_processing.loaders.component_classification import is_image_block
from src.features.chunking.domain.chunking_strategy import resolve_chunking_strategy

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".docx",
        ".txt",
        ".text",
        ".pptx",
        ".ppt",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".wma",
        ".mp4",
        ".webm",
        ".mov",
        ".avi",
        ".mkv",
        ".mpeg",
        ".m4v",
        ".wmv",
    }
)

PIPELINE_STAGES = (
    "received_request",
    "validating_request",
    "loading_document",
    "extracting_document_blocks",
    "chunking",
    "extracting_references",
    "saving_to_neo4j",
    "embedding",
    "storing_weaviate",
    "completed",
)

StageCallback = Callable[[str, float | None], None]


@dataclass
class FailureSnapshot:
    failed_stage: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    traceback: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    document_path: str | None = None
    chunking_strategy: str | None = None
    processor_status: str | None = None
    block_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failed_stage": self.failed_stage,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "traceback": self.traceback,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_path": self.document_path,
            "chunking_strategy": self.chunking_strategy,
            "processor_status": self.processor_status,
            "block_stats": self.block_stats,
        }


_last_failure: FailureSnapshot | None = None
_stage_callback: StageCallback | None = None
_current_stage: str | None = None


def get_last_failure() -> dict[str, Any]:
    if _last_failure is None:
        return {}
    payload = _last_failure.to_dict()
    payload["exception"] = payload.get("exception_message")
    return payload


def clear_last_failure() -> None:
    global _last_failure
    _last_failure = None


def bind_stage_callback(callback: StageCallback | None) -> None:
    global _stage_callback
    _stage_callback = callback


def current_stage() -> str | None:
    return _current_stage


def set_pipeline_stage(stage: str, progress: float | None = None) -> None:
    global _current_stage
    _current_stage = stage
    log.info("Pipeline stage: %s", stage)
    if _stage_callback is not None:
        _stage_callback(stage, progress)


def record_failure(
    exc: BaseException,
    *,
    stage: str | None,
    document_id: str | None = None,
    document_name: str | None = None,
    document_path: str | None = None,
    chunking_strategy: str | None = None,
    processor_status: str | None = None,
    block_stats: dict[str, Any] | None = None,
) -> FailureSnapshot:
    global _last_failure
    tb = traceback.format_exc()
    snapshot = FailureSnapshot(
        failed_stage=stage or _current_stage,
        exception_type=type(exc).__name__,
        exception_message=str(exc) or repr(exc),
        traceback=tb,
        document_id=document_id,
        document_name=document_name,
        document_path=document_path,
        chunking_strategy=chunking_strategy,
        processor_status=processor_status or "failed",
        block_stats=dict(block_stats or {}),
    )
    _last_failure = snapshot
    log.exception(
        "PROCESSING FAILED stage=%s document_id=%s document_name=%s document_path=%s "
        "chunking_strategy=%s processor_status=%s",
        snapshot.failed_stage,
        document_id,
        document_name,
        document_path,
        chunking_strategy,
        processor_status,
    )
    log.error(tb)
    return snapshot


def coalesce_setting(*values: Any) -> Any:
    """Return the first non-None, non-blank value."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def summarize_blocks(blocks: list[Any]) -> dict[str, int]:
    pages = {getattr(block, "page", None) for block in blocks if getattr(block, "page", None)}
    paragraph_count = 0
    table_count = 0
    image_count = 0
    for block in blocks:
        block_type = str(getattr(block, "block_type", "text") or "text").lower()
        if block_type == "table":
            table_count += 1
        elif is_image_block(block):
            image_count += 1
        else:
            paragraph_count += 1
    return {
        "loaded_pages": len(pages) if pages else 0,
        "extracted_blocks": len(blocks),
        "paragraph_count": paragraph_count,
        "table_count": table_count,
        "image_count": image_count,
    }


def log_block_diagnostics(blocks: list[Any], *, prefix: str = "") -> dict[str, int]:
    stats = summarize_blocks(blocks)
    label = f"{prefix} " if prefix else ""
    log.info("%sLoaded pages: %s", label, stats["loaded_pages"])
    log.info("%sExtracted blocks: %s", label, stats["extracted_blocks"])
    log.info("%sNumber of paragraphs: %s", label, stats["paragraph_count"])
    log.info("%sNumber of tables: %s", label, stats["table_count"])
    log.info("%sNumber of images: %s", label, stats["image_count"])
    return stats


def explain_zero_chunks(block_stats: dict[str, int], *, had_blocks: bool, had_text: bool) -> str:
    if not had_blocks:
        return "No document blocks were extracted from the file."
    if not had_text:
        return (
            "Document blocks were extracted but none contained chunkable text "
            f"(paragraphs={block_stats.get('paragraph_count', 0)}, "
            f"tables={block_stats.get('table_count', 0)}, "
            f"images={block_stats.get('image_count', 0)})."
        )
    return "Chunking strategy produced zero chunks despite non-empty input blocks."


def validate_document_file(document_path: Path) -> None:
    if not document_path.exists():
        raise FileNotFoundError(f"Document does not exist: {document_path}")
    if not document_path.is_file():
        raise ValueError(f"Document path is not a readable file: {document_path}")
    try:
        with document_path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise ValueError(f"Document is not readable: {document_path}") from exc
    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document extension '{suffix}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def validate_chunking_strategy(raw_strategy: str | None) -> str:
    requested = raw_strategy
    resolved = resolve_chunking_strategy(raw_strategy)
    log.info("Requested strategy: %s", requested)
    log.info("Resolved strategy: %s", resolved)
    return resolved


def check_embedding_model_available(model_name: str, model_dir: str | None = None) -> None:
    from src.features.embeddings.application.embedding_service import get_embedding_model

    if not str(model_name or "").strip():
        raise RuntimeError(
            "Embedding model is not configured. Set model_name/model_dir on the request "
            "or configure MODEL_NAME / MODEL_DIR in the processor environment."
        )
    get_embedding_model(model_name, model_dir=model_dir)


def ping_neo4j() -> None:
    from src.features.references.infrastructure.neo4j_reference_store import get_reference_store

    store = get_reference_store()
    if not store.enabled:
        log.info("Neo4j reference store disabled; skipping connectivity check.")
        return
    driver = store._driver()
    try:
        with driver.session(**store._session_kwargs()) as session:
            session.execute_write(lambda tx: tx.run("RETURN 1 AS ok").consume())
    finally:
        driver.close()


def ping_weaviate(weaviate_url: str, weaviate_api_key: str | None, collection_name: str) -> None:
    from src.infrastructure.document_databases.weaviate_store import get_weaviate_client

    if not str(weaviate_url or "").strip():
        raise RuntimeError("Weaviate URL is not configured.")
    client = get_weaviate_client(weaviate_url, weaviate_api_key)
    if not client.is_ready():
        raise RuntimeError(f"Weaviate is not ready at {weaviate_url}")
    if not client.collections.exists(collection_name):
        log.warning(
            "Weaviate collection '%s' does not exist yet; it will be created during storage.",
            collection_name,
        )


def validate_pipeline_request(
    *,
    document_path: Path,
    chunking_strategy: str | None,
    model_name: str,
    model_dir: str | None,
    weaviate_url: str,
    weaviate_api_key: str | None,
    collection_name: str,
    check_neo4j: bool = True,
    check_weaviate: bool = True,
    check_embedding: bool = True,
) -> str:
    validate_document_file(document_path)
    resolved_strategy = validate_chunking_strategy(chunking_strategy)
    if check_embedding:
        check_embedding_model_available(model_name, model_dir=model_dir)
    if check_neo4j:
        ping_neo4j()
    if check_weaviate:
        ping_weaviate(weaviate_url, weaviate_api_key, collection_name)
    return resolved_strategy
