from __future__ import annotations

import asyncio
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

from src.features.document_processing.core.contract import ProcessRequest
from src.features.document_processing.core.logger import log
from src.features.document_processing.document_diagnostics import (
    debug_documents_payload,
    format_startup_document_diagnostics,
    resolve_processor_document_path,
)
from src.features.references.infrastructure.neo4j_reference_store import get_reference_store
from src.workers.reference_processor.processor import ReferenceExtractionProcessor


@dataclass
class ReferenceProcessorState:
    current_status: str = "idle"
    error_log: list[str] = field(default_factory=list)
    progress_percentage: int = 0
    document_id: str | None = None
    document_name: str | None = None
    document_path: str | None = None
    reference_count: int = 0
    saved_reference_count: int = 0
    resolved_reference_count: int = 0
    unresolved_reference_count: int = 0
    process_time_seconds: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    last_job_terminal_status: str | None = None
    last_job_document_id: str | None = None
    failed_stage: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    failure_traceback: str | None = None

    def summary(self) -> dict[str, Any]:
        payload = {
            "state": "busy" if _job_busy() else "idle",
            "current_status": self.current_status,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_path": self.document_path,
            "reference_count": self.reference_count,
            "saved_reference_count": self.saved_reference_count,
            "resolved_reference_count": self.resolved_reference_count,
            "unresolved_reference_count": self.unresolved_reference_count,
            "process_time_seconds": self.process_time_seconds,
            "last_job_terminal_status": self.last_job_terminal_status,
            "last_job_document_id": self.last_job_document_id,
            "error_log": self.error_log,
            "processor_busy": _job_busy(),
        }
        if self.current_status == "failed" or self.last_job_terminal_status == "failed":
            payload.update(
                {
                    "failed_stage": self.failed_stage,
                    "exception_type": self.exception_type,
                    "exception_message": self.exception_message,
                    "traceback": self.failure_traceback,
                }
            )
        return payload


_state = ReferenceProcessorState()
_process_task: asyncio.Task | None = None
_process_accept_lock = asyncio.Lock()
_stop_event = threading.Event()
_processor = ReferenceExtractionProcessor()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info(format_startup_document_diagnostics())
    store = get_reference_store()
    log.info("Reference extraction processor startup:")
    log.info("  Listening port: %s", os.getenv("REFERENCE_EXTRACTION_PORT", os.getenv("PROCESSOR_PORT", "3110")))
    log.info("  Available endpoints: GET /health, GET /status, GET /debug/documents, POST /process, POST /stop")
    log.info("  Neo4j connection: uri=%s configured=%s", store.config.uri, store.enabled)
    log.info("  Weaviate connection: not used by reference extraction")
    log.info("  Scheduler registration: standalone HTTP processor")
    log.info("  Document root: %s", os.getenv("DOCUMENT_ROOT", "/app/documents"))
    log.info("  Processor type: reference_extraction")
    yield


app = FastAPI(title="Reference Extraction Processor", version="1.0.0", lifespan=lifespan)


def _reset_state() -> None:
    global _state, _stop_event
    _stop_event = threading.Event()
    _state = ReferenceProcessorState()


def _job_busy() -> bool:
    return _process_task is not None


def _check_stop() -> bool:
    if _stop_event.is_set():
        _state.error_log.append("Stop requested.")
        _state.current_status = "stopped"
        return True
    return False


def _fail_processing(exc: BaseException) -> None:
    err = str(exc) or repr(exc)
    tb = traceback.format_exc()
    _state.error_log.extend([err, tb])
    _state.failed_stage = _state.current_status
    _state.exception_type = type(exc).__name__
    _state.exception_message = err
    _state.failure_traceback = tb
    _state.current_status = "failed"
    log.exception("Reference extraction failed: %s", exc)
    log.error(tb)


def _run_processing(request: ProcessRequest, document_path: Path) -> None:
    try:
        _state.document_id = request.document_id
        _state.document_name = request.document_name or document_path.name
        _state.document_path = str(document_path)
        _state.started_at = time.time()
        _state.current_status = "starting"

        def set_status(phase: str, progress: float | None = None) -> None:
            _state.current_status = phase
            if progress is not None:
                _state.progress_percentage = int(progress)
            log.info("Reference processor status: %s", phase)

        result = _processor.run(
            request,
            document_path,
            set_status=set_status,
            check_stop=_check_stop,
        )
        artifacts = dict(result.artifacts or {})
        _state.reference_count = int(artifacts.get("reference_count", 0))
        _state.saved_reference_count = int(artifacts.get("saved_reference_count", 0))
        _state.resolved_reference_count = int(artifacts.get("resolved_reference_count", 0))
        _state.unresolved_reference_count = int(artifacts.get("unresolved_reference_count", 0))
        _state.current_status = "completed"
    except BaseException as exc:  # noqa: BLE001
        _fail_processing(exc)
    finally:
        now = time.time()
        _state.finished_at = now
        start = _state.started_at or now
        _state.process_time_seconds = max(0.0, now - start)


def _on_process_task_done(task: asyncio.Task) -> None:
    global _process_task
    try:
        exc = task.exception()
        if exc is not None and _state.current_status not in {"completed", "failed", "stopped"}:
            _fail_processing(exc)
    finally:
        if _process_task is task:
            _process_task = None
            last_status = _state.current_status
            last_doc = _state.document_id
            last_ref = _state.reference_count
            last_saved = _state.saved_reference_count
            last_resolved = _state.resolved_reference_count
            last_unresolved = _state.unresolved_reference_count
            last_time = _state.process_time_seconds
            _reset_state()
            if last_status in {"completed", "failed", "stopped"}:
                _state.last_job_terminal_status = last_status
                _state.last_job_document_id = last_doc
                _state.reference_count = last_ref
                _state.saved_reference_count = last_saved
                _state.resolved_reference_count = last_resolved
                _state.unresolved_reference_count = last_unresolved
                _state.process_time_seconds = last_time


async def _run_processing_task(request: ProcessRequest, document_path: Path) -> None:
    await asyncio.to_thread(_run_processing, request, document_path)


@app.get("/health")
@app.get("/status")
async def status() -> dict[str, Any]:
    return _state.summary()


@app.get("/debug/documents")
async def debug_documents() -> dict[str, Any]:
    return debug_documents_payload()


@app.get("/debug/last_failure")
async def debug_last_failure() -> dict[str, Any]:
    if _state.last_job_terminal_status != "failed":
        return {"status": "no_failures_recorded"}
    return {
        "failed_stage": _state.failed_stage,
        "exception": _state.exception_message,
        "traceback": _state.failure_traceback,
        "document_id": _state.last_job_document_id,
        "processor_status": _state.last_job_terminal_status,
    }


@app.get("/debug/resolve/{target_id}")
async def debug_resolve(target_id: str) -> dict[str, Any]:
    return get_reference_store().debug_resolve_target(target_id)


@app.get("/debug/references/{document_id}")
async def debug_references(document_id: str) -> dict[str, Any]:
    references = get_reference_store().debug_references_for(document_id)
    return {
        "document_id": document_id,
        "references": references,
    }


@app.get("/api/v1/documents/{document_id}/references")
async def document_references(document_id: str) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "references": get_reference_store().references_for(document_id),
    }


@app.post("/process", status_code=202)
async def process_document(request: ProcessRequest) -> dict[str, Any]:
    global _process_task
    async with _process_accept_lock:
        if _job_busy():
            raise HTTPException(status_code=409, detail="Processor is busy processing another document.")
        try:
            document_path, _diagnostics = resolve_processor_document_path(
                document_path=request.document_path,
                document_name=request.document_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Run secure compliance validation scan
        from src.features.security.application.upload_security_pipeline import run_security_pipeline
        scan_res = run_security_pipeline(document_path, document_id=request.document_id)
        if scan_res.get("status") == "block":
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=400,
                content={
                    "status": "upload_blocked",
                    "reason": scan_res.get("reason"),
                    "severity": scan_res.get("severity"),
                    "detected_categories": scan_res.get("detected_categories"),
                    "detections": scan_res.get("detections"),
                    "document_type": scan_res.get("document_type"),
                    "topics": scan_res.get("topics")
                }
            )
        elif scan_res.get("status") == "human_review":
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=202,
                content={
                    "status": "human_review",
                    "reason": "Pending human review: " + scan_res.get("reason"),
                    "severity": scan_res.get("severity"),
                    "detected_categories": scan_res.get("detected_categories"),
                    "detections": scan_res.get("detections"),
                    "document_type": scan_res.get("document_type"),
                    "topics": scan_res.get("topics")
                }
            )

        _reset_state()
        _state.document_id = request.document_id
        _state.document_name = request.document_name or document_path.name
        _state.current_status = "starting"
        _process_task = asyncio.create_task(_run_processing_task(request, document_path))
        _process_task.add_done_callback(_on_process_task_done)

    return {
        "status": "accepted",
        "message": "Reference extraction started in the background.",
        "document_id": request.document_id,
        "processor_type": "reference_extraction",
        "current_status": "starting",
    }


@app.post("/stop")
async def stop() -> dict[str, Any]:
    stop_requested = False
    if _job_busy() and _state.current_status not in {"completed", "failed", "stopped", "idle"}:
        _stop_event.set()
        stop_requested = True
    return {"status": "acknowledged", "stop_requested": stop_requested}
