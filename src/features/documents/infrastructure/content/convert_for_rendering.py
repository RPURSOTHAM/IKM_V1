"""Convert uploaded documents into render-ready HTML or PDF artifacts."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.features.documents.infrastructure.content.render_html import blocks_to_html, wrap_render_document_html

_logger = logging.getLogger(__name__)

SUPPORTED_RENDER_SUFFIXES = frozenset({
    ".docx",
    ".pdf",
    ".txt",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    # Audio/video preview uses transcript text blocks → HTML
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
})
DOCX_SUFFIX = ".docx"
PDF_SUFFIX = ".pdf"
TXT_SUFFIX = ".txt"
PPTX_SUFFIXES = frozenset({".pptx", ".ppt"})
MEDIA_RENDER_SUFFIXES = frozenset({
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v", ".wmv",
})
IMAGE_RENDER_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


@dataclass(frozen=True)
class RenderConversionResult:
    format: str
    media_type: str
    conversion_method: str
    content: str | None = None
    content_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_soffice_binary() -> str | None:
    configured = (os.getenv("SOFFICE_PATH") or os.getenv("LIBREOFFICE_PATH") or "").strip()
    if configured and Path(configured).is_file():
        return configured
    for candidate in ("soffice", "libreoffice"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    for candidate in (
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def docx_to_pdf(document_path: Path, *, output_dir: Path, timeout_seconds: int = 180) -> Path | None:
    """Convert DOCX to PDF using LibreOffice headless when available."""
    soffice = _resolve_soffice_binary()
    if not soffice:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = output_dir / ".lo_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    user_installation = profile_dir.resolve().as_uri()
    command = [
        soffice,
        f"-env:UserInstallation={user_installation}",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--norestore",
        "--convert-to",
        "pdf:writer_pdf_Export",
        "--outdir",
        str(output_dir),
        str(document_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _logger.warning("DOCX to PDF conversion failed for %s: %s", document_path.name, exc)
        return None

    if completed.returncode != 0:
        _logger.warning(
            "LibreOffice returned %s for %s: %s",
            completed.returncode,
            document_path.name,
            (completed.stderr or completed.stdout or "").strip()[:500],
        )
        return None

    expected = output_dir / f"{document_path.stem}.pdf"
    if expected.is_file() and expected.stat().st_size > 0:
        return expected
    return None


def render_output_dir(documents_root: Path, document_id: str) -> Path:
    return documents_root / "render" / document_id


def convert_document_for_rendering(
    document_path: Path,
    *,
    document_id: str,
    title: str,
    blocks: list[Any] | None,
    doc_meta: dict[str, Any],
    documents_root: Path | None = None,
) -> RenderConversionResult:
    """Convert a supported document into HTML or PDF for frontend preview."""
    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_RENDER_SUFFIXES:
        raise ValueError(f"Unsupported document extension for rendering: {suffix or '(none)'}")

    page_count = int(doc_meta.get("page_count") or 0) or None
    root = documents_root or Path(os.getenv("DOCUMENT_ROOT") or "/app/documents")

    if suffix == DOCX_SUFFIX:
        render_dir = render_output_dir(root, document_id)
        pdf_path = docx_to_pdf(document_path, output_dir=render_dir)
        if pdf_path is not None:
            return RenderConversionResult(
                format="pdf",
                media_type="application/pdf",
                content_path=pdf_path,
                conversion_method="libreoffice_docx_to_pdf",
                metadata={**doc_meta, "render_format": "pdf"},
            )

        raise RuntimeError(
            "DOCX preview requires LibreOffice (soffice). "
            "Install LibreOffice or set SOFFICE_PATH / LIBREOFFICE_PATH."
        )

    if suffix == PDF_SUFFIX:
        render_dir = render_output_dir(root, document_id)
        render_dir.mkdir(parents=True, exist_ok=True)
        target = render_dir / f"{document_id}.pdf"
        if not target.is_file() or target.stat().st_size == 0:
            shutil.copy2(document_path, target)
        return RenderConversionResult(
            format="pdf",
            media_type="application/pdf",
            content_path=target,
            conversion_method="native_pdf",
            metadata={**doc_meta, "render_format": "pdf"},
        )

    # TXT + PowerPoint + media transcript: HTML preview from extracted text blocks.
    if blocks is None:
        raise RuntimeError("HTML rendering requires loaded document blocks.")
    html = wrap_render_document_html(
        blocks_to_html(blocks, page_count=page_count),
        title=title,
    )
    if suffix in PPTX_SUFFIXES:
        method = "pptx_blocks_to_html"
    elif suffix in MEDIA_RENDER_SUFFIXES:
        method = "media_transcript_blocks_to_html"
    elif suffix in IMAGE_RENDER_SUFFIXES:
        method = "image_ocr_blocks_to_html"
    else:
        method = "txt_blocks_to_html"
    return RenderConversionResult(
        format="html",
        media_type="text/html; charset=utf-8",
        content=html,
        conversion_method=method,
        metadata={**doc_meta, "render_format": "html"},
    )
