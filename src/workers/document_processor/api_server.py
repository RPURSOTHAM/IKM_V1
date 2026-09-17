from __future__ import annotations

import asyncio
import importlib.util
import os
import socket
import threading
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError as PydanticValidationError
from pydantic import BaseModel, Field

from src.features.document_processing.core.config import get_settings
from src.features.document_processing.core.contract import ProcessRequest
from src.features.document_processing.core.logger import log
from src.features.document_processing.pipeline_tracking import (
    bind_stage_callback,
    clear_last_failure,
    current_stage,
    get_last_failure,
    record_failure,
    set_pipeline_stage,
)
from src.features.document_processing.document_diagnostics import (
    debug_documents_payload,
    format_startup_document_diagnostics,
    resolve_processor_document_path,
)
from src.features.references.infrastructure.neo4j_reference_store import get_reference_store
from src.features.document_processing.processors import container_processor_type, get_processor
from src.features.security.classification.document_classifier import (
    classify_document,
    preload_document_classifier,
)

try:
    from db.document_jobs import get_document_job_store
except ImportError:
    from src.infrastructure.database.document_jobs import get_document_job_store
from src.infrastructure.document_databases.weaviate_store import close_weaviate_client
from src.features.embeddings.application.embedding_service import log_embedding_startup_diagnostics


@dataclass
class ProcessorState:
    current_status: str = "idle"
    error_log: list[str] = field(default_factory=list)
    progress_percentage: int = 0
    processor_type: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    collection_name: str | None = None
    tenant_id: str | None = None
    num_chunks: int = 0
    processed_chunks: int = 0
    process_time_seconds: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    # Last terminal outcome of the previous job (cleared on next /process) for clients polling /health.
    last_job_terminal_status: str | None = None
    last_job_document_id: str | None = None
    last_job_document_metadata: dict[str, Any] = field(default_factory=dict)
    result_location: str | None = None
    document_metadata: dict[str, Any] = field(default_factory=dict)
    document_path: str | None = None
    current_chunking_strategy: str | None = None
    pipeline_stage: str | None = None
    failed_stage: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    failure_traceback: str | None = None

    def summary(self) -> dict[str, Any]:
        payload = {
            "current_status": self.current_status,
            "processor_type": self.processor_type,
            "error_log": self.error_log,
            "progress_percentage": self.progress_percentage,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_path": self.document_path,
            "collection_name": self.collection_name,
            "tenant_id": self.tenant_id,
            "num_chunks": self.num_chunks,
            "processed_chunks": self.processed_chunks,
            "process_time_seconds": self.process_time_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_job_terminal_status": self.last_job_terminal_status,
            "last_job_document_id": self.last_job_document_id,
            "last_job_document_metadata": self.last_job_document_metadata,
            "result_location": self.result_location,
            "document_metadata": self.document_metadata,
            "pipeline_stage": self.pipeline_stage or current_stage(),
            "current_chunking_strategy": self.current_chunking_strategy,
            "processor_busy": _job_busy(),
        }
        if self.current_status == "failed" or self.last_job_terminal_status == "failed":
            failure = {
                "failed_stage": self.failed_stage,
                "exception_type": self.exception_type,
                "exception_message": self.exception_message,
                "traceback": self.failure_traceback,
            }
            if not failure["failed_stage"]:
                failure = get_last_failure()
            payload.update(failure)
        return payload


_state = ProcessorState()
_container_processor_type = container_processor_type()
_stop_event = threading.Event()
_process_task: asyncio.Task | None = None
_process_accept_lock = asyncio.Lock()

TERMINAL_STATUSES = frozenset({"idle", "completed", "failed", "stopped"})
JOB_HEARTBEAT_INTERVAL_SEC = float(os.getenv("PROCESSOR_JOB_HEARTBEAT_INTERVAL_SEC") or "12")

REQUIRED_MODULES = {
    "pdfplumber": "PDF ingestion",
    "docx": "DOCX ingestion",
}


class DocumentClassificationRequest(BaseModel):
    text: str = Field(..., description="Extracted document text to classify.")


class DocumentClassificationResponse(BaseModel):
    document_type: str
    confidence: float
    classification_time_ms: int
    all_scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(format_startup_document_diagnostics())
    store = get_reference_store()
    log.info("Document processor startup:")
    log.info("  Listening port: %s", os.getenv("PROCESSOR_PORT", "3100"))
    log.info("  Available endpoints: GET /health, GET /status, GET /debug/documents, POST /process, POST /stop")
    log.info("  Neo4j connection: uri=%s configured=%s", store.config.uri, store.enabled)
    log.info("  Weaviate connection: url=%s", os.getenv("WEAVIATE_URL", ""))
    log.info("  Scheduler registration: %s", "typed" if _container_processor_type else "generic")
    log.info("  Document root: %s", os.getenv("DOCUMENT_ROOT", "/app/documents"))
    log.info("  Processor type: %s", _container_processor_type or "generic")

    from src.features.document_processing.shared_processor.deployment import (
        ProcessorConfigurationError,
        ProcessorConfigurationProvider,
        bootstrap_deployment_processors,
    )

    try:
        bootstrap_deployment_processors(initialize=True, log=log, start_watcher=True)
    except ProcessorConfigurationError as exc:
        log.error("Deployment processor configuration error: %s", exc)
        raise

    if ProcessorConfigurationProvider.is_enabled("embedding"):
        log_embedding_startup_diagnostics()
    else:
        log.info("Embedding processor disabled — skipping embedding model diagnostics.")

    if (
        ProcessorConfigurationProvider.is_enabled("security")
        and os.getenv("DOCUMENT_CLASSIFIER_PRELOAD", "true").strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        try:
            preload_document_classifier()
        except Exception as exc:
            log.exception("Zero-shot document classifier preload failed: %s", exc)
    elif not ProcessorConfigurationProvider.is_enabled("security"):
        log.info("Security processor disabled — skipping document classifier preload.")
    yield
    close_weaviate_client()
    try:
        from src.features.document_processing.shared_processor.deployment import get_processor_registry, stop_processor_config_watcher

        stop_processor_config_watcher()
        get_processor_registry().shutdown_all()
    except Exception:
        log.exception("Failed to shut down deployment processors")


app = FastAPI(title="RAG Document Processor", version="1.0.0", lifespan=lifespan)

try:
    from src.features.observability.middleware.observability_middleware import ObservabilityMiddleware

    app.add_middleware(ObservabilityMiddleware)
except Exception as _obs_mw_exc:  # pragma: no cover - never block processor startup
    log.warning("ObservabilityMiddleware unavailable: %s", _obs_mw_exc)

try:
    from src.features.observability.middleware.correlation_id import CorrelationIdMiddleware

    app.add_middleware(CorrelationIdMiddleware)
except Exception as _corr_mw_exc:  # pragma: no cover
    log.warning("CorrelationIdMiddleware unavailable: %s", _corr_mw_exc)

# Enterprise infrastructure (JSON logs, plugins/DI, /ready, /metrics). Existing /health unchanged.
try:
    from src.infrastructure.application_support import HealthRegistry, attach_infra_routes, bootstrap_infra

    bootstrap_infra(service="processor-service")
    _infra_health = HealthRegistry("processor-service")
    _infra_health.add(
        "runtime_dependencies",
        lambda: {
            "ok": not bool(_missing_runtime_dependencies()),
            "missing": _missing_runtime_dependencies(),
        },
        critical=True,
    )
    attach_infra_routes(app, service="processor-service", health_registry=_infra_health)
except Exception as _infra_exc:  # pragma: no cover - never block processor startup
    import logging as _logging

    _logging.getLogger(__name__).warning("Infra bootstrap skipped: %s", _infra_exc)

from src.features.security.review.review_api import router as human_review_router

app.include_router(human_review_router, prefix="/api/v1")


def _missing_runtime_dependencies() -> dict[str, str]:
    missing: dict[str, str] = {}
    for module_name, purpose in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module_name) is None:
            missing[module_name] = purpose
    return missing


def _validate_document_dependencies(document_path: Path) -> None:
    suffix = document_path.suffix.lower()
    missing = _missing_runtime_dependencies()
    if suffix == ".pdf" and "pdfplumber" in missing:
        raise RuntimeError(
            "PDF support is unavailable because 'pdfplumber' is not installed in "
            "the src.processor_service runtime. Rebuild the processor image after "
            "installing src/processor_service (pip install -e src/processor_service)."
        )
    if suffix == ".docx" and "docx" in missing:
        raise RuntimeError(
            "DOCX support is unavailable because 'python-docx' is not installed in "
            "the src.processor_service runtime. Rebuild the processor image after "
            "installing src/processor_service (pip install -e src/processor_service)."
        )


def _reset_state() -> None:
    global _state, _stop_event
    _stop_event = threading.Event()
    _state = ProcessorState()


def _job_busy() -> bool:
    """True until the done-callback clears ``_process_task`` (avoids a race after ``task.done()``)."""
    return _process_task is not None


def _processor_identity() -> str:
    return (os.getenv("PROCESSOR_IDENTITY") or os.getenv("HOSTNAME") or socket.gethostname() or "processor").strip()


def _mysql_job_store():
    return get_document_job_store()


def _set_status(status: str) -> None:
    previous_status = _state.current_status
    _state.current_status = status
    log.info("Processor status: %s", status)
    log.info(
        "PIPELINE_TRACE %s",
        {
            "stage": "processor_status_update",
            "document_id": _state.document_id,
            "processor_type": _state.processor_type or _container_processor_type or "doc_processor",
            "processor_id": _processor_identity(),
            "current_status": previous_status,
            "next_status": status,
        },
    )
    store = _mysql_job_store()
    doc_id = _state.document_id
    if not store or not doc_id:
        return
    processor_type = _state.processor_type or _container_processor_type or "doc_processor"
    note_prefix = f"{processor_type}:"
    try:
        if status == "starting":
            store.update_started(doc_id, processor_identity=_processor_identity(), notes=f"{note_prefix}starting")
            log.info(
                "PIPELINE_TRACE %s",
                {
                    "stage": "processor_started",
                    "document_id": doc_id,
                    "processor_type": processor_type,
                    "processor_id": _processor_identity(),
                    "current_status": "ASSIGNED",
                    "next_status": "IN_PROGRESS",
                },
            )
        elif status == "completed":
            store.record_processor_outcome(
                doc_id,
                processor_type=processor_type,
                status="COMPLETED",
                result_location=_state.result_location,
                document_metadata=_state.document_metadata,
            )
            log.info(
                "PIPELINE_TRACE %s",
                {
                    "stage": f"{processor_type}_complete",
                    "document_id": doc_id,
                    "processor_type": processor_type,
                    "processor_id": _processor_identity(),
                    "current_status": "IN_PROGRESS",
                    "next_status": "COMPLETED",
                    "result_location": _state.result_location,
                },
            )
            if processor_type == "key_field_extraction":
                try:
                    meta = store._scheduling_metadata(doc_id)
                    follow_ups = [
                        pt
                        for pt in (meta.get("follow_up_processors") or [])
                        if pt in (meta.get("enabled_processor_types") or [])
                    ]
                    if follow_ups:
                        from src.features.documents.application.document_service import DocumentReceiverService

                        DocumentReceiverService().republish_processor_types(doc_id, follow_ups)
                except Exception:
                    log.debug("Could not enqueue follow-up processors for %s", doc_id, exc_info=True)
        elif status == "failed":
            err = "\n".join(str(x) for x in _state.error_log[-8:]) if _state.error_log else "failed"
            store.record_processor_outcome(
                doc_id,
                processor_type=processor_type,
                status="FAILED",
                error_details=err[:65000],
            )
        elif status == "stopped":
            msg = "\n".join(str(x) for x in _state.error_log[-4:]) if _state.error_log else "stopped"
            store.update_operator_stopped(doc_id, details=msg[:65000])
        elif status == "stopping":
            store.heartbeat(doc_id, notes=f"{note_prefix}stopping")
        else:
            pct = float(_state.progress_percentage or 0)
            store.update_in_progress(doc_id, progress_percent=pct, notes=f"{note_prefix}{status}")
    except Exception:
        log.exception("document_job MySQL update failed for status=%s", status)


def _report_progress(processed: int, total: int) -> None:
    if total > 0:
        _state.progress_percentage = int((processed / total) * 100)


def _check_stop() -> bool:
    if _stop_event.is_set():
        _state.error_log.append("Stop requested by /stop endpoint.")
        _set_status("stopped")
        return True
    return False


def _heartbeat_loop(stop_evt: threading.Event) -> None:
    """Periodically refresh ``document_job`` heartbeat while a job is non-terminal."""
    interval = max(3.0, JOB_HEARTBEAT_INTERVAL_SEC)
    while not stop_evt.wait(timeout=interval):
        try:
            store = _mysql_job_store()
            doc_id = _state.document_id
            if not store or not doc_id:
                continue
            if _state.current_status in TERMINAL_STATUSES:
                continue
            store.heartbeat(
                doc_id,
                progress_percent=float(_state.progress_percentage or 0),
                notes=_state.current_status,
            )
        except Exception:
            log.exception("processor job heartbeat failed")


def _resolve_document_path(request: ProcessRequest) -> Path:
    document_path, diagnostics = resolve_processor_document_path(
        document_path=request.document_path,
        document_name=request.document_name,
    )
    log.info(
        "Resolved document path: received=%s absolute=%s exists=%s is_file=%s",
        diagnostics.get("received_document_path"),
        diagnostics.get("absolute_resolved_path"),
        diagnostics.get("exists"),
        diagnostics.get("is_file"),
    )
    log.info("Document path diagnostics: %s", diagnostics)
    return document_path


@app.get("/health")
async def health() -> dict[str, Any]:
    """Scheduler-friendly liveness: ``health`` / ``status`` (service), ``state`` (idle|busy), plus job snapshot."""
    try:
        missing = _missing_runtime_dependencies()
        snapshot = _state.summary()
        svc = "unhealthy" if missing else "healthy"
        state = "busy" if _job_busy() else "idle"
        if missing:
            return {
                "health": svc,
                "status": svc,
                "state": state,
                "missing_dependencies": missing,
                **snapshot,
            }
        return {
            "health": svc,
            "status": svc,
            "state": state,
            "processor_mode": "generic" if _container_processor_type is None else "typed",
            "processor_type": _container_processor_type,
            **snapshot,
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("health endpoint failed: %s", exc)
        return {
            "health": "unhealthy",
            "status": "unhealthy",
            "state": "idle",
            "error": str(exc),
            "current_status": "unreachable",
        }


@app.get("/debug/last_failure")
async def debug_last_failure() -> dict[str, Any]:
    """Return the most recent processing failure snapshot."""
    payload = get_last_failure()
    if not payload:
        return {"status": "no_failures_recorded"}
    payload["processor_status"] = _state.last_job_terminal_status or _state.current_status
    return payload


@app.get("/debug/documents")
async def debug_documents() -> dict[str, Any]:
    """List mounted document directory contents for upload troubleshooting."""
    return debug_documents_payload()


@app.get("/status")
async def status() -> dict[str, Any]:
    """Scheduler compatibility alias for ``GET /health``."""
    return await health()


@app.post("/classify-document", response_model=DocumentClassificationResponse)
async def classify_document_endpoint(
    request: DocumentClassificationRequest,
) -> dict[str, Any]:
    """Classify extracted document text with the reusable zero-shot classifier."""
    return classify_document(request.text)


@app.get("/api/v1/documents/{document_id}/references")
async def document_references(document_id: str) -> dict[str, Any]:
    try:
        return {
            "document_id": document_id,
            "references": get_reference_store().references_for(document_id),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Neo4j references lookup failed for document_id=%s: %s", document_id, exc)
        return {"document_id": document_id, "references": [], "warning": "reference graph unavailable"}


@app.get("/api/v1/documents/{document_id}/referenced-by")
async def document_referenced_by(document_id: str) -> dict[str, Any]:
    try:
        return {
            "document_id": document_id,
            "referenced_by": get_reference_store().referenced_by(document_id),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Neo4j referenced-by lookup failed for document_id=%s: %s", document_id, exc)
        return {"document_id": document_id, "referenced_by": [], "warning": "reference graph unavailable"}


@app.get("/api/v1/documents/{document_id}/reference-graph")
async def document_reference_graph(document_id: str) -> dict[str, Any]:
    try:
        graph = get_reference_store().reference_graph(document_id)
        return {"document_id": document_id, **graph}
    except Exception as exc:  # noqa: BLE001
        log.warning("Neo4j reference graph lookup failed for document_id=%s: %s", document_id, exc)
        return {"document_id": document_id, "nodes": [], "edges": [], "warning": "reference graph unavailable"}


@app.post("/stop")
async def stop() -> dict[str, Any]:
    """Non-blocking: always acknowledge; stop signal is best-effort when a job is active."""
    try:
        stop_requested = False
        task = _process_task
        active_work = task is not None and not task.done()
        if active_work and _state.current_status not in TERMINAL_STATUSES:
            _stop_event.set()
            stop_requested = True
            asyncio.create_task(asyncio.to_thread(_set_status, "stopping"))
        return {"status": "acknowledged", "stop_requested": stop_requested}
    except Exception as exc:  # noqa: BLE001
        log.exception("stop endpoint failed: %s", exc)
        return {"status": "acknowledged", "stop_requested": False, "warning": str(exc)}


def _fail_processing(
    exc: BaseException,
    *,
    log_exc: bool = True,
    request: ProcessRequest | None = None,
    document_path: Path | None = None,
) -> None:
    """Move processor to a terminal failed state (never raises)."""
    try:
        err = str(exc) or repr(exc)
        tb = traceback.format_exc()
        _state.error_log.append(err)
        _state.error_log.append(tb)
        _state.failed_stage = current_stage() or _state.pipeline_stage or _state.current_status
        _state.exception_type = type(exc).__name__
        _state.exception_message = err
        _state.failure_traceback = tb
        record_failure(
            exc,
            stage=_state.failed_stage,
            document_id=_state.document_id,
            document_name=_state.document_name,
            document_path=str(document_path or _state.document_path or ""),
            chunking_strategy=_state.current_chunking_strategy,
            processor_status="failed",
        )
        _set_status("failed")
        if log_exc:
            log.exception("PROCESSING FAILED: %s", exc)
            log.error(tb)
        else:
            log.error("PROCESSING FAILED: %s", err)
            log.error(tb)
    except Exception as inner:  # noqa: BLE001 — last-resort; keep service alive
        log.critical("Could not record failure state: %s", inner, exc_info=True)


def _finalize_processing_timestamps() -> None:
    try:
        now = time.time()
        _state.finished_at = now
        start = _state.started_at or now
        _state.process_time_seconds = max(0.0, now - start)
    except Exception as inner:  # noqa: BLE001
        log.critical("Could not finalize processing timestamps: %s", inner, exc_info=True)


def _run_processing(request: ProcessRequest, document_path: Path) -> None:
    """Run the typed processor in a worker thread. Never raises to the caller."""
    context_factory = None
    try:
        from src.features.observability.hooks.processor_hooks import processor_job_context as context_factory
    except Exception:
        context_factory = None

    if context_factory is not None:
        with context_factory(
            document_id=request.document_id,
            repository_id=getattr(request, "repository_id", None),
        ):
            _run_processing_inner(request, document_path)
        return
    _run_processing_inner(request, document_path)


def _run_processing_inner(request: ProcessRequest, document_path: Path) -> None:
    """Inner processing loop (observability context optional)."""
    hb_stop = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(hb_stop,),
        daemon=True,
        name="document-job-heartbeat",
    )
    try:
        processor = get_processor(request.normalized_processor_type())
        _state.processor_type = request.normalized_processor_type()
        _state.document_id = request.document_id
        _state.document_name = request.document_name or document_path.name
        _state.document_path = str(document_path)
        _state.collection_name = request.collection_name
        _state.tenant_id = request.tenant_id
        _state.current_chunking_strategy = request.chunking_strategy
        _state.started_at = time.time()
        clear_last_failure()
        set_pipeline_stage("received_request", 0.0)
        def _on_pipeline_stage(stage: str, progress: float | None) -> None:
            _state.pipeline_stage = stage
            if progress is not None:
                _state.progress_percentage = int(progress)
            _set_status(stage)

        bind_stage_callback(_on_pipeline_stage)
        _set_status("starting")
        hb_thread.start()

        def set_status(phase: str, progress: float | None = None) -> None:
            _state.pipeline_stage = phase
            if progress is not None:
                _state.progress_percentage = int(progress)
            _set_status(phase)

        result = processor.run(
            request,
            document_path,
            set_status=set_status,
            check_stop=_check_stop,
        )
        _state.result_location = result.result_location
        _state.document_metadata = dict(result.document_metadata or {})
        artifacts = dict(result.artifacts or {})
        _state.num_chunks = int(
            artifacts.get("num_chunks")
            or _state.document_metadata.get("chunk_count")
            or 0
        )
        _state.processed_chunks = int(
            artifacts.get("embedded_chunks")
            or _state.document_metadata.get("embedded_chunk_count")
            or 0
        )
        _state.progress_percentage = 100
        _set_status("completed")
    except BaseException as exc:  # noqa: BLE001 — never let the worker thread crash the process
        if isinstance(exc, Exception):
            _fail_processing(exc, log_exc=True, request=request, document_path=document_path)
        else:
            log.critical("Non-Exception during processing: %s", exc, exc_info=True)
            try:
                _state.error_log.append(f"aborted: {exc!r}")
                _set_status("failed")
            except Exception:
                log.critical("Could not persist aborted job state", exc_info=True)
    finally:
        bind_stage_callback(None)
        hb_stop.set()
        hb_thread.join(timeout=5.0)
        _finalize_processing_timestamps()


def _on_process_task_done(task: asyncio.Task) -> None:
    """Retrieve task result so exceptions are never 'unhandled'; release slot for the next job."""
    global _process_task
    try:
        if task.cancelled():
            if _state.current_status not in {"completed", "failed", "stopped"}:
                _state.error_log.append("Processing task was cancelled before completion.")
                _fail_processing(RuntimeError("processing cancelled"), log_exc=False)
        else:
            exc = task.exception()
            if exc is not None:
                log.error("Background processing task ended with error: %s", exc, exc_info=exc)
                terminal = {"failed", "stopped", "completed"}
                if _state.current_status not in terminal:
                    _fail_processing(exc, log_exc=False)
        if _state.finished_at is None:
            _finalize_processing_timestamps()
    except asyncio.CancelledError:
        pass
    except Exception as inner:  # noqa: BLE001
        log.critical("process task done callback failed: %s", inner, exc_info=True)
    finally:
        if _process_task is task:
            _process_task = None
            try:
                last_term: str | None = None
                last_doc: str | None = None
                last_metadata: dict[str, Any] = {}
                last_num_chunks = _state.num_chunks
                last_processed_chunks = _state.processed_chunks
                last_process_time = _state.process_time_seconds
                if _state.current_status in {"completed", "failed", "stopped"}:
                    last_term = _state.current_status
                    last_doc = _state.document_id
                    last_metadata = dict(_state.document_metadata or {})
                _reset_state()
                if last_term:
                    _state.last_job_terminal_status = last_term
                    _state.last_job_document_id = last_doc
                    _state.last_job_document_metadata = last_metadata
                    _state.num_chunks = last_num_chunks
                    _state.processed_chunks = last_processed_chunks
                    _state.process_time_seconds = last_process_time
            except Exception:
                log.exception("processor slot reset after job failed")


async def _run_processing_task(request: ProcessRequest, document_path: Path) -> None:
    """Await worker thread; catch anything that escapes _run_processing (should be rare)."""
    try:
        await asyncio.to_thread(_run_processing, request, document_path)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        log.exception("Processing thread wrapper error: %s", exc)
        terminal = {"failed", "stopped", "completed"}
        if _state.current_status not in terminal and isinstance(exc, Exception):
            _fail_processing(exc, log_exc=False)
        if _state.finished_at is None:
            _finalize_processing_timestamps()


def _skip_security_when_prevalidated(request: ProcessRequest) -> bool:
    """DMS already scanned on upload; skip duplicate security work on every processor hop."""
    if not bool(getattr(request, "security_prevalidated", False)):
        return False
    flag = os.getenv("PROCESSOR_SKIP_SECURITY_WHEN_PREVALIDATED", "true").strip().lower()
    return flag in {"1", "true", "yes", "on"}


async def _evaluate_upload_security(request: ProcessRequest, document_path: Path) -> bool:
    """Run upload security gate. Returns True when document processing may continue."""
    from src.features.document_processing.shared_processor.deployment import ProcessorConfigurationProvider
    from src.features.security.application.upload_security_pipeline import run_security_pipeline
    from src.features.security.audit.security_event_logger import log_security_event

    store = _mysql_job_store()
    doc_id = str(request.document_id or "")

    if not ProcessorConfigurationProvider.is_enabled("security"):
        log.info("Security processor disabled for this deployment; skipping upload security gate.")
        return True

    if _skip_security_when_prevalidated(request):
        log.info(
            "Skipping processor security gate for document_id=%s (security_prevalidated=true).",
            request.document_id,
        )
        return True

    if bool(getattr(request, "security_prevalidated", False)):
        log_security_event(
            "UPLOAD_SCAN",
            decision="prevalidated_recheck",
            reason="Caller marked security_prevalidated; processor still re-validates.",
            severity="low",
            policy="processor_no_bypass",
            document_id=request.document_id,
            document_name=request.document_name,
        )

    scan_res = await asyncio.to_thread(
        run_security_pipeline,
        document_path,
        document_id=request.document_id,
        honor_reviewer_decision=bool(getattr(request, "security_prevalidated", False)),
    )
    status = str(scan_res.get("status") or "").lower()
    if status == "block":
        reason = str(scan_res.get("reason") or "upload blocked by security")
        if store and doc_id:
            store.update_failed(doc_id, error_details=reason, notes="security:blocked")
        _set_status("failed")
        return False

    if status == "human_review":
        from src.features.security.review.human_review_queue import get_reviewer_override_for_document

        override = get_reviewer_override_for_document(doc_id)
        pipeline = str((override or {}).get("pipeline_status") or "").lower()
        if pipeline == "block":
            reason = str((override or {}).get("reason") or scan_res.get("reason") or "Blocked by human review")
            if store and doc_id:
                store.update_failed(doc_id, error_details=reason, notes="security:reviewer_block")
            _set_status("failed")
            return False
        if pipeline not in {"allow", "mask_and_allow"}:
            if store and doc_id:
                store.update_human_review(
                    doc_id,
                    notes=f"security:human_review:{scan_res.get('reason') or 'pending review'}",
                )
            return False
        log_security_event(
            "UPLOAD_SCAN",
            decision="human_review_continue_after_reviewer",
            reason=str(
                (override or {}).get("reason")
                or scan_res.get("reason")
                or "DLP human_review cleared by reviewer Allow/Mask"
            ),
            severity=str(scan_res.get("severity") or "medium"),
            policy="reviewer_override_continue",
            document_id=request.document_id,
            document_name=request.document_name,
        )
    return True


async def _run_security_then_processing_task(request: ProcessRequest, document_path: Path) -> None:
    """Security gate then processing pipeline; keeps POST /process fast for the scheduler."""
    try:
        if not await _evaluate_upload_security(request, document_path):
            return
        await _run_processing_task(request, document_path)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        log.exception("Security/processing task failed: %s", exc)
        terminal = {"failed", "stopped", "completed"}
        if _state.current_status not in terminal and isinstance(exc, Exception):
            _fail_processing(exc, log_exc=False)
        if _state.finished_at is None:
            _finalize_processing_timestamps()


@app.post("/process", status_code=202)
async def process_document(request: ProcessRequest) -> dict[str, Any]:
    """Accept one job at a time; returns immediately while work runs in the background."""
    global _process_task
    try:
        if (
            _container_processor_type is not None
            and request.normalized_processor_type() != _container_processor_type
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"This container handles processor_type='{_container_processor_type}' only; "
                    f"received '{request.normalized_processor_type()}'."
                ),
            )

        async with _process_accept_lock:
            if _job_busy():
                raise HTTPException(
                    status_code=409,
                    detail="Processor is busy processing another document.",
                )
            try:
                document_path = _resolve_document_path(request)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            try:
                _validate_document_dependencies(document_path)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

            # Accept immediately (202) so the scheduler is not blocked by scanned-PDF OCR
            # inside the security gate. Security + processing run in the background task.
            _reset_state()
            _state.processor_type = request.normalized_processor_type()
            _state.document_id = request.document_id
            _state.document_name = request.document_name or document_path.name
            _state.collection_name = request.collection_name
            _state.tenant_id = request.tenant_id
            _state.started_at = time.time()
            _state.current_status = "starting"
            _state.last_job_terminal_status = None
            _state.last_job_document_id = None

            store = _mysql_job_store()
            if store and request.document_id:
                try:
                    store.update_started(
                        str(request.document_id),
                        processor_identity=_processor_identity(),
                        notes=f"{request.normalized_processor_type()}:accepted",
                    )
                except Exception:
                    log.debug("Could not mark job started on /process accept", exc_info=True)

            _process_task = asyncio.create_task(
                _run_security_then_processing_task(request, document_path)
            )
            _process_task.add_done_callback(_on_process_task_done)

        return {
            "status": "accepted",
            "message": "Document processing started in the background.",
            "document_id": request.document_id,
            "processor_type": request.normalized_processor_type(),
            "current_status": "starting",
        }
    except HTTPException:
        raise
    except PydanticValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("POST /process failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal error while accepting process request.",
        ) from exc
