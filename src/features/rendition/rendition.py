"""Backend-Only Document Rendition Engine.

This module provides document rendition capabilities for converting uploaded documents
(starting with PDF via PyMuPDF) into a viewable representation (rendered page images).

Key Design Principles:
1. Pure Rendition Engine: Strictly handles format identification, page rendering,
   storage of page representations, and status tracking.
2. Separation of Security: Security policy decisions (sensitive content detection,
   risk scoring, DLP blocking, human review) are managed externally by the security pipeline.
3. Original File Integrity: Original uploaded documents are NEVER modified or overwritten.
   Generated page images are stored separately under ``DOCUMENT_ROOT/renditions/<rendition_id>/``.
4. Extensible Modular Architecture: Uses a handler-dispatcher pattern (`BaseRenditionHandler`)
   designed for seamless future extension to DOCX, PPTX, TXT, Video, and Audio.
5. Persistent State & Swagger API: Provides REST APIs (/api/v1/rendering/*) backed by thread-safe
   persistent metadata storage for integration and Swagger testing.
"""

from __future__ import annotations

import abc
import io
import json
import logging
import shutil
import threading
import time
import uuid
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from fastapi import APIRouter, File, Form, HTTPException, Path as FastAPIPath, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from src.features.configuration.platform_settings import settings

_logger = logging.getLogger(__name__)


# =============================================================================
# 1. ENUMS AND SCHEMAS
# =============================================================================

class RenditionStatus(str, Enum):
    """Lifecycle status of a document rendition operation."""
    NOT_STARTED = "NOT_STARTED"
    QUEUED = "QUEUED"
    RENDERING = "RENDERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PageDetail(BaseModel):
    """Details of a single rendered page."""
    page_number: int = Field(..., description="1-indexed page number")
    page_path: str = Field(..., description="Relative or absolute path to rendered page image")
    width: int = Field(..., description="Page width in pixels")
    height: int = Field(..., description="Page height in pixels")


class RenditionStatusResponse(BaseModel):
    """Status summary response for a rendition."""
    rendition_id: str
    document_id: Optional[str] = None
    status: RenditionStatus
    progress: Optional[float] = Field(default=None, description="Progress ratio (0.0 to 1.0) if available")
    error: Optional[str] = Field(default=None, description="Error message if rendition failed")


class RenditionDetails(BaseModel):
    """Detailed response for a completed or failed document rendition."""
    rendition_id: str
    document_id: Optional[str] = None
    repository_id: Optional[str] = None
    document_name: str
    original_file_name: Optional[str] = None
    document_type: str
    status: RenditionStatus
    page_count: int = 0
    created_at: float
    pages: List[PageDetail] = Field(default_factory=list)
    error: Optional[str] = None
    source_pdf_path: Optional[str] = Field(
        default=None,
        description="Path to preserved original PDF copy under the rendition folder.",
    )


# =============================================================================
# 2. MODULAR RENDITION HANDLER ABSTRACTION
# =============================================================================

class BaseRenditionHandler(abc.ABC):
    """Abstract base class for document rendition handlers.
    
    Future handlers (DOCX, PPTX, TXT, Video, Audio) will inherit from this
    class and implement ``can_handle`` and ``render_pages``.
    """

    @abc.abstractmethod
    def can_handle(self, document_type: str, file_path: Path) -> bool:
        """Return True if this handler supports the given file format."""
        pass

    @abc.abstractmethod
    def render_pages(self, input_path: Path, output_dir: Path) -> Tuple[int, List[PageDetail]]:
        """Process the input file, render pages into output_dir, and return (page_count, pages)."""
        pass


class PDFRenditionHandler(BaseRenditionHandler):
    """PDF Rendition Handler using PyMuPDF (fitz)."""

    def can_handle(self, document_type: str, file_path: Path) -> bool:
        doc_type = (document_type or "").strip().lower().lstrip(".")
        ext = file_path.suffix.lower().lstrip(".")
        return doc_type == "pdf" or ext == "pdf"

    def render_pages(self, input_path: Path, output_dir: Path) -> Tuple[int, List[PageDetail]]:
        if not input_path.exists():
            raise FileNotFoundError(f"Source document not found: {input_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(input_path))
        except Exception as exc:
            _logger.warning("Failed to open PDF %s: %s", input_path, exc)
            raise ValueError(f"Could not open or parse PDF document: {exc}") from exc

        try:
            page_count = len(doc)
            if page_count == 0:
                raise ValueError("PDF document contains 0 pages.")

            pages: List[PageDetail] = []
            # Standard rendering matrix (150 DPI preview quality)
            zoom = 150 / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for idx in range(page_count):
                page_number = idx + 1
                page = doc.load_page(idx)
                rect = page.rect
                width = int(rect.width * zoom)
                height = int(rect.height * zoom)

                pix = page.get_pixmap(matrix=matrix, alpha=False)
                out_filename = f"page_{page_number}.png"
                out_file_path = output_dir / out_filename
                pix.save(str(out_file_path))

                pages.append(
                    PageDetail(
                        page_number=page_number,
                        page_path=str(out_file_path.resolve()),
                        width=width,
                        height=height,
                    )
                )

            return page_count, pages
        finally:
            doc.close()


# =============================================================================
# 3. RENDITION PERSISTENCE & ENGINE CORE
# =============================================================================

class RenditionStore:
    """Thread-safe JSON-backed persistence store for rendition metadata."""

    def __init__(self, db_file_path: Path):
        self.path = db_file_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._write_unlocked({})

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            text = self.path.read_text(encoding="utf-8")
            return json.loads(text) if text.strip() else {}
        except Exception as exc:
            _logger.error("Error reading rendition store JSON: %s", exc)
            return {}

    def _write_unlocked(self, data: Dict[str, Any]) -> None:
        text = json.dumps(data, indent=2)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(self.path)

    def save(self, rendition: RenditionDetails) -> None:
        with self._lock:
            data = self._read_unlocked()
            data[rendition.rendition_id] = rendition.model_dump()
            self._write_unlocked(data)

    def get(self, rendition_id: str) -> Optional[RenditionDetails]:
        with self._lock:
            data = self._read_unlocked()
            raw = data.get(rendition_id)
            if not raw:
                return None
            try:
                return RenditionDetails(**raw)
            except Exception as exc:
                _logger.error("Failed to parse stored rendition %s: %s", rendition_id, exc)
                return None


class RenditionEngine:
    """Core Rendition Engine singleton managing document conversion lifecycle."""

    def __init__(self):
        root_dir = Path(settings.upload_dir)
        self.base_renditions_dir = root_dir / "renditions"
        self.base_renditions_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = self.base_renditions_dir / "renditions_db.json"
        self.store = RenditionStore(db_path)
        self.handlers: List[BaseRenditionHandler] = [PDFRenditionHandler()]

    def register_handler(self, handler: BaseRenditionHandler) -> None:
        """Register a new format handler (extensibility point for DOCX/PPTX/etc.)."""
        self.handlers.insert(0, handler)

    def _get_handler(self, document_type: str, file_path: Path) -> Optional[BaseRenditionHandler]:
        for handler in self.handlers:
            if handler.can_handle(document_type, file_path):
                return handler
        return None

    def render_document(
        self,
        input_path: Path,
        original_filename: str,
        document_type: str,
        document_id: Optional[str] = None,
        repository_id: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> RenditionDetails:
        """Execute document rendition synchronously and update status lifecycle."""
        rendition_id = f"ren_{uuid.uuid4()}"
        created_at = time.time()
        output_dir = self.base_renditions_dir / rendition_id
        display_name = (document_name or original_filename or "document").strip()

        # 1. Initial State: QUEUED
        record = RenditionDetails(
            rendition_id=rendition_id,
            document_id=document_id,
            repository_id=repository_id,
            document_name=display_name,
            original_file_name=original_filename or display_name,
            document_type=document_type.lower(),
            status=RenditionStatus.QUEUED,
            page_count=0,
            created_at=created_at,
            pages=[],
            error=None,
        )
        self.store.save(record)

        # 2. State Transition: RENDERING
        record.status = RenditionStatus.RENDERING
        self.store.save(record)

        # Check format support
        handler = self._get_handler(document_type, input_path)
        if not handler:
            error_msg = (
                f"Unsupported document type '{document_type}'. "
                "Currently supported types: PDF"
            )
            record.status = RenditionStatus.FAILED
            record.error = error_msg
            self.store.save(record)
            return record

        # 3. Perform Rendering
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Preserve original PDF for PDF-view / download (never modify source file).
            source_copy = output_dir / "original.pdf"
            try:
                shutil.copy2(str(input_path), str(source_copy))
                record.source_pdf_path = str(source_copy.resolve())
            except Exception as copy_exc:
                _logger.warning(
                    "Could not preserve source PDF for rendition_id=%s: %s",
                    rendition_id,
                    copy_exc,
                )
                record.source_pdf_path = None

            page_count, pages = handler.render_pages(input_path, output_dir)
            record.page_count = page_count
            record.pages = pages
            record.status = RenditionStatus.COMPLETED
            record.error = None
        except Exception as exc:
            _logger.exception("Rendering failed for rendition_id=%s", rendition_id)
            record.status = RenditionStatus.FAILED
            record.error = str(exc)
        
        self.store.save(record)
        return record

    def resolve_source_pdf(self, rendition_id: str) -> Optional[Path]:
        """Return path to preserved original PDF for a rendition, if available."""
        record = self.get_rendition(rendition_id)
        if not record:
            return None
        candidates: List[Path] = []
        if record.source_pdf_path:
            candidates.append(Path(record.source_pdf_path))
        candidates.append(self.base_renditions_dir / rendition_id / "original.pdf")
        for path in candidates:
            if path.is_file():
                return path
        return None

    def get_rendition(self, rendition_id: str) -> Optional[RenditionDetails]:
        return self.store.get(rendition_id)

    def get_status(self, rendition_id: str) -> Optional[RenditionStatusResponse]:
        record = self.get_rendition(rendition_id)
        if not record:
            return None
        progress = 1.0 if record.status == RenditionStatus.COMPLETED else (0.0 if record.status == RenditionStatus.FAILED else 0.5)
        return RenditionStatusResponse(
            rendition_id=record.rendition_id,
            document_id=record.document_id,
            status=record.status,
            progress=progress,
            error=record.error,
        )


# Global Engine Instance
rendition_engine = RenditionEngine()


# =============================================================================
# 4. FASTAPI ROUTER ENDPOINTS
# =============================================================================

router = APIRouter(prefix="/rendering", tags=["Document Rendition"])


def _parse_submit_for_processing(value: Any) -> bool:
    """Parse multipart boolean; default False for rendition (store without queue unless asked)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "null", "none"}:
        return False
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    return False


def _parse_page_selection(pages: Optional[str], page_count: int) -> List[int]:
    """Parse ``pages`` query: ``all``, ``1,2,3``, or ``1-3``. Empty → all pages."""
    raw = (pages or "").strip().lower()
    if not raw or raw in {"all", "*"}:
        return list(range(1, page_count + 1))

    selected: List[int] = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid page range '{token}'. Use forms like 1-3 or 1,2,5.",
                ) from exc
            if start > end:
                start, end = end, start
            selected.extend(range(start, end + 1))
        else:
            try:
                selected.append(int(token))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid page number '{token}'.",
                ) from exc

    if not selected:
        raise HTTPException(status_code=400, detail="No valid page numbers provided.")

    unique = sorted(set(selected))
    out_of_bounds = [n for n in unique if n < 1 or n > page_count]
    if out_of_bounds:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Page number(s) out of bounds: {out_of_bounds}. "
                f"Valid range is 1–{page_count}."
            ),
        )
    return unique


def _build_pages_zip(rendition: RenditionDetails, page_numbers: List[int]) -> bytes:
    """Pack selected rendered PNG pages into a ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for page_num in page_numbers:
            matching = next((p for p in rendition.pages if p.page_number == page_num), None)
            if not matching:
                raise HTTPException(
                    status_code=404,
                    detail=f"Page {page_num} detail not found.",
                )
            image_path = Path(matching.page_path)
            if not image_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Rendered image file for page {page_num} missing on disk.",
                )
            zf.write(image_path, arcname=f"page_{page_num}.png")
    return buf.getvalue()


def _build_selected_pages_pdf(pdf_path: Path, page_numbers: List[int]) -> bytes:
    """Build a new PDF containing only the selected 1-indexed pages (in order)."""
    try:
        src = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open source PDF: {exc}") from exc
    out = None
    try:
        out = fitz.open()
        for page_num in page_numbers:
            if page_num < 1 or page_num > len(src):
                raise HTTPException(
                    status_code=404,
                    detail=f"Page number {page_num} out of bounds. Total pages available: {len(src)}",
                )
            # insert_pdf copies a contiguous range; call once per selected page to preserve order.
            out.insert_pdf(src, from_page=page_num - 1, to_page=page_num - 1)
        return out.tobytes()
    finally:
        if out is not None:
            out.close()
        src.close()


@router.post(
    "/render",
    response_model=RenditionDetails,
    summary="Create a document rendition",
    description=(
        "Create page-image renditions from a PDF.\n\n"
        "**Repository upload:** pass `file` + `repository_id` to ingest into the DMS repository "
        "first, then render (response includes `document_id`).\n\n"
        "**Existing document:** pass `document_id` alone.\n\n"
        "**Preview-only:** pass `file` without `repository_id` (temp storage, not in a repository).\n\n"
        "Original files are NEVER modified."
    ),
)
async def create_rendition(
    file: Optional[UploadFile] = File(default=None, description="PDF file to render"),
    document_id: Optional[str] = Form(default=None, description="Existing ingested document UUID"),
    repository_id: Optional[str] = Form(
        default=None,
        description=(
            "When uploading a file, optional repository UUID. "
            "Use GET /api/v1/repositories/options. Document is stored in the repository then rendered."
        ),
    ),
    document_type_id: Optional[str] = Form(
        default=None,
        description="Optional document type UUID when uploading into a repository.",
    ),
    submit_for_processing: str = Form(
        default="false",
        description=(
            "When uploading into a repository: queue processors after upload. "
            "Default false (store + render only). Set true to also submit for processing."
        ),
    ),
) -> RenditionDetails:
    """Create a new rendition from an uploaded file or an ingested document_id."""
    if not file and not document_id:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' upload or 'document_id' must be provided.",
        )

    repo_id = (str(repository_id).strip() if repository_id else "") or None
    type_id = (str(document_type_id).strip() if document_type_id else "") or None
    if type_id and type_id.lower() in {"null", "none", "undefined", "string"}:
        type_id = None
    if repo_id and repo_id.lower() in {"null", "none", "undefined", "string"}:
        repo_id = None

    resolved_path: Optional[Path] = None
    original_filename: str = ""
    display_document_name: Optional[str] = None
    doc_type: str = "pdf"
    target_document_id: Optional[str] = (str(document_id).strip() if document_id else "") or None
    if target_document_id and target_document_id.lower() in {"null", "none", "undefined", "string"}:
        target_document_id = None

    from src.features.documents.application.document_service import DocumentReceiverService

    service = DocumentReceiverService()

    # Path A: existing document_id → resolve from store (already in a repository).
    if target_document_id and not file:
        try:
            path, filename, _media_type = service.resolve_original_file(target_document_id)
            resolved_path = path
            original_filename = filename
            doc_type = Path(filename).suffix.lstrip(".").lower() or "pdf"
            # Prefer repository / display name from the document record when available.
            try:
                record = service.get_document(target_document_id)
                repo_id = repo_id or getattr(record, "repository_id", None)
                display_document_name = (
                    str(getattr(record, "document_name", None) or "").strip() or None
                )
                original_from_record = str(
                    getattr(record, "original_file_name", None) or ""
                ).strip()
                if original_from_record:
                    original_filename = original_from_record
            except Exception:
                pass
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to locate existing document '{target_document_id}': {exc}",
            ) from exc

    # Path B: file + repository_id → ingest into repository, then render stored original.
    elif file and repo_id:
        original_filename = file.filename or "uploaded_document.pdf"
        display_document_name = original_filename
        ext = Path(original_filename).suffix.lstrip(".").lower() or "pdf"
        if ext != "pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{ext}'. Currently supported types: PDF",
            )
        try:
            upload_resp = await service.upload_documents(
                [file],
                repository_id=repo_id,
                document_type_id=type_id,
                submit_for_processing=_parse_submit_for_processing(submit_for_processing),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload document into repository: {exc}",
            ) from exc

        docs = list(getattr(upload_resp, "documents", None) or [])
        if not docs:
            raise HTTPException(
                status_code=500,
                detail="Repository upload succeeded but returned no document record.",
            )
        target_document_id = str(docs[0].document_id)
        repo_id = getattr(docs[0], "repository_id", None) or repo_id
        display_document_name = (
            str(getattr(docs[0], "document_name", None) or "").strip() or original_filename
        )
        original_filename = (
            str(getattr(docs[0], "original_file_name", None) or "").strip()
            or original_filename
        )
        path, filename, _media_type = service.resolve_original_file(target_document_id)
        resolved_path = path
        if not original_filename:
            original_filename = filename
        doc_type = Path(filename).suffix.lstrip(".").lower() or "pdf"

    # Path C: file only → temp preview (not stored in a repository).
    elif file:
        original_filename = file.filename or "uploaded_document.pdf"
        display_document_name = original_filename
        ext = Path(original_filename).suffix.lstrip(".").lower() or "pdf"
        doc_type = ext

        if ext != "pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{ext}'. Currently supported types: PDF",
            )

        temp_dir = Path(settings.upload_dir) / "renditions_intake"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / f"upload_{uuid.uuid4().hex}_{original_filename}"

        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            temp_file_path.write_bytes(content)
            resolved_path = temp_file_path
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to write uploaded file to disk: {exc}",
            ) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide file (optionally with repository_id) or document_id.",
        )

    if not resolved_path or not resolved_path.is_file():
        raise HTTPException(status_code=400, detail="Unable to access source document file.")

    rendition = rendition_engine.render_document(
        input_path=resolved_path,
        original_filename=original_filename,
        document_type=doc_type,
        document_id=target_document_id,
        repository_id=repo_id,
        document_name=display_document_name,
    )

    if rendition.status == RenditionStatus.FAILED and rendition.error:
        if "Could not open or parse PDF" in rendition.error or "0 pages" in rendition.error:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted PDF document: {rendition.error}",
            )

    return rendition


@router.get(
    "/{rendition_id}/status",
    response_model=RenditionStatusResponse,
    summary="Get rendition status",
    description="Returns current rendition lifecycle status (QUEUED, RENDERING, COMPLETED, FAILED).",
)
def get_rendition_status(rendition_id: str) -> RenditionStatusResponse:
    status_resp = rendition_engine.get_status(rendition_id)
    if not status_resp:
        raise HTTPException(
            status_code=404,
            detail=f"Rendition ID '{rendition_id}' not found.",
        )
    return status_resp


@router.get(
    "/{rendition_id}",
    response_model=RenditionDetails,
    summary="Get rendition details",
    description="Returns complete details for a rendition including metadata and page list.",
)
def get_rendition_details(rendition_id: str) -> RenditionDetails:
    rendition = rendition_engine.get_rendition(rendition_id)
    if not rendition:
        raise HTTPException(
            status_code=404,
            detail=f"Rendition ID '{rendition_id}' not found.",
        )
    return rendition


@router.get(
    "/{rendition_id}/download",
    summary="Download selected pages as ZIP (PNG) or multi-page PDF",
    description=(
        "Download one or more pages.\n\n"
        "**Page selection (`pages`):**\n"
        "- `pages=all` (default): all pages\n"
        "- `pages=1,3,5` or `pages=6,9,12`: specific pages\n"
        "- `pages=2-9` or `pages=1-3`: inclusive range\n"
        "- Mix: `pages=1,3,5-8`\n\n"
        "**Format (`format`):**\n"
        "- `format=zip` (default): rendered page PNGs in a ZIP\n"
        "- `format=pdf`: one PDF containing only the selected pages\n\n"
        "For a single page as a raw PNG (not ZIP), use "
        "`GET /api/v1/rendering/{rendition_id}/pages/{page_number}`.\n"
        "For the full original PDF, use `GET /api/v1/rendering/{rendition_id}/pdf`."
    ),
)
def download_rendered_pages(
    rendition_id: str,
    pages: Optional[str] = Query(
        default="all",
        description="Page selection: all | 1,2,5 | 2-9 | 6,9,12 (comma and/or ranges).",
    ),
    format: Optional[str] = Query(
        default="zip",
        description="Output format: zip (PNG pages) or pdf (selected pages as one PDF).",
    ),
) -> Response:
    rendition = rendition_engine.get_rendition(rendition_id)
    if not rendition:
        raise HTTPException(
            status_code=404,
            detail=f"Rendition ID '{rendition_id}' not found.",
        )
    if rendition.status != RenditionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Rendition is in '{rendition.status.value}' state. Pages are not available.",
        )
    if rendition.page_count < 1:
        raise HTTPException(status_code=400, detail="Rendition has no pages to download.")

    selected = _parse_page_selection(pages, rendition.page_count)
    fmt = (format or "zip").strip().lower()
    safe_name = Path(rendition.document_name or rendition.original_file_name or "document").stem or "document"
    pages_label = "all" if (pages or "all").strip().lower() == "all" else "-".join(str(n) for n in selected[:8])
    if len(selected) > 8:
        pages_label += f"_plus{len(selected) - 8}"

    if fmt in {"pdf", "application/pdf"}:
        pdf_path = rendition_engine.resolve_source_pdf(rendition_id)
        if not pdf_path:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Original PDF is not available; cannot export selected pages as PDF. "
                    "Create a new rendition to preserve the source PDF."
                ),
            )
        pdf_bytes = _build_selected_pages_pdf(pdf_path, selected)
        filename = f"{safe_name}_pages_{pages_label}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if fmt not in {"zip", "png", "png-zip", "application/zip"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Use format=zip (PNG ZIP) or format=pdf (selected pages PDF).",
        )

    zip_bytes = _build_pages_zip(rendition, selected)
    filename = f"{safe_name}_{rendition_id}_pages.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{rendition_id}/pdf",
    summary="Download or view original PDF",
    description=(
        "Returns the preserved original PDF for this rendition "
        "(for PDF view / full-document download)."
    ),
)
def get_rendition_pdf(rendition_id: str) -> Response:
    rendition = rendition_engine.get_rendition(rendition_id)
    if not rendition:
        raise HTTPException(
            status_code=404,
            detail=f"Rendition ID '{rendition_id}' not found.",
        )
    pdf_path = rendition_engine.resolve_source_pdf(rendition_id)
    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=(
                "Original PDF is not available for this rendition. "
                "Create a new rendition to enable PDF view/download."
            ),
        )
    safe_name = Path(
        rendition.original_file_name or rendition.document_name or "document.pdf"
    ).name
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=safe_name,
        content_disposition_type="inline",
        headers={
            # Allow Streamlit (other localhost ports) to embed / fetch the real PDF.
            "Content-Security-Policy": "frame-ancestors *",
            "X-Content-Type-Options": "nosniff",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get(
    "/{rendition_id}/pages/{page_number}/pdf",
    summary="Download a single page as PDF",
    description="Extracts one page from the original PDF and returns it as a single-page PDF file.",
)
def get_rendition_page_as_pdf(
    rendition_id: str,
    page_number: int = FastAPIPath(..., ge=1, description="1-indexed page number"),
) -> Response:
    rendition = rendition_engine.get_rendition(rendition_id)
    if not rendition:
        raise HTTPException(
            status_code=404,
            detail=f"Rendition ID '{rendition_id}' not found.",
        )
    if rendition.status != RenditionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Rendition is in '{rendition.status.value}' state. Pages are not available.",
        )
    if page_number < 1 or page_number > rendition.page_count:
        raise HTTPException(
            status_code=404,
            detail=f"Page number {page_number} out of bounds. Total pages available: {rendition.page_count}",
        )
    pdf_path = rendition_engine.resolve_source_pdf(rendition_id)
    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail="Original PDF is not available; cannot export a single-page PDF.",
        )
    try:
        src = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open source PDF: {exc}") from exc
    try:
        if page_number > len(src):
            raise HTTPException(
                status_code=404,
                detail=f"Page number {page_number} out of bounds. Total pages available: {len(src)}",
            )
        out = fitz.open()
        out.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)
        pdf_bytes = out.tobytes()
        out.close()
    finally:
        src.close()

    safe_stem = Path(rendition.original_file_name or rendition.document_name or "document").stem
    filename = f"{safe_stem}_page_{page_number}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{rendition_id}/pages",
    summary="Get full rendered document view or specific page",
    description="Returns an HTML document displaying all rendered pages sequentially if page_number is omitted, or returns a specific PNG page image if page_number is provided.",
)
def get_rendered_pages_full(
    rendition_id: str,
    page_number: Optional[int] = Query(default=None, description="Optional 1-indexed page number. If left empty, displays full document view."),
) -> Response:
    if page_number is not None:
        return get_rendered_page(rendition_id=rendition_id, page_number=str(page_number))

    rendition = rendition_engine.get_rendition(rendition_id)
    if not rendition:
        raise HTTPException(
            status_code=404,
            detail=f"Rendition ID '{rendition_id}' not found.",
        )

    if rendition.status != RenditionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Rendition is in '{rendition.status.value}' state. Pages are not available.",
        )

    sorted_pages = sorted(rendition.pages, key=lambda p: p.page_number)
    api_prefix = settings.api_prefix.rstrip("/")
    download_all = f"{api_prefix}/rendering/{rendition_id}/download?pages=all"

    html_items = []
    for p in sorted_pages:
        img_src = f"{api_prefix}/rendering/{rendition_id}/pages/{p.page_number}"
        page_dl = f"{api_prefix}/rendering/{rendition_id}/download?pages={p.page_number}"
        html_items.append(
            f'<div style="margin-bottom: 24px; text-align: center;">'
            f'<img src="{img_src}" alt="Page {p.page_number}" style="max-width: 100%; height: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.4); border-radius: 4px;" />'
            f'<div style="margin-top: 8px; color: #888888; font-size: 14px;">'
            f'Page {p.page_number} of {rendition.page_count} · '
            f'<a href="{img_src}" download style="color:#8ab4f8;">PNG</a> · '
            f'<a href="{page_dl}" style="color:#8ab4f8;">ZIP</a>'
            f"</div>"
            f"</div>"
        )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{rendition.document_name} - Rendition View</title>
  <style>
    body {{
      background-color: #121212;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .header {{
      margin-bottom: 24px;
      text-align: center;
    }}
    .header h2 {{
      margin: 0 0 8px 0;
      color: #ffffff;
    }}
    .header p {{
      margin: 0;
      color: #aaaaaa;
      font-size: 14px;
    }}
    a {{ color: #8ab4f8; }}
  </style>
</head>
<body>
  <div class="header">
    <h2>{rendition.document_name}</h2>
    <p>Rendition ID: <code>{rendition.rendition_id}</code> | Total Pages: {rendition.page_count}</p>
    <p style="margin-top:12px;"><a href="{download_all}">Download all pages (ZIP)</a></p>
  </div>
  {"".join(html_items)}
</body>
</html>"""

    return HTMLResponse(content=html_content)


@router.get(
    "/{rendition_id}/pages/{page_number}",
    summary="Get a rendered page image or full document view",
    description="Returns the rendered PNG image for a specific page number, or full document view if page_number is left empty.",
)
def get_rendered_page(
    rendition_id: str,
    page_number: str = FastAPIPath(..., description="1-indexed page number. If left empty, returns full document view."),
) -> Response:
    raw_page = (page_number or "").strip()
    if not raw_page or not raw_page.isdigit():
        return get_rendered_pages_full(rendition_id=rendition_id)

    page_num = int(raw_page)
    rendition = rendition_engine.get_rendition(rendition_id)
    if not rendition:
        raise HTTPException(
            status_code=404,
            detail=f"Rendition ID '{rendition_id}' not found.",
        )

    if rendition.status != RenditionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Rendition is in '{rendition.status.value}' state. Pages are not available.",
        )

    if page_num < 1 or page_num > rendition.page_count:
        raise HTTPException(
            status_code=404,
            detail=f"Page number {page_num} out of bounds. Total pages available: {rendition.page_count}",
        )

    matching_page = next((p for p in rendition.pages if p.page_number == page_num), None)
    if not matching_page:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} detail not found.",
        )

    image_path = Path(matching_page.page_path)
    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Rendered image file for page {page_num} missing on disk.",
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=f"{rendition_id}_page_{page_num}.png",
    )
