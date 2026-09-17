"""Read page counts from document files when loader metadata under-reports pages."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

_logger = logging.getLogger(__name__)

_APP_XML_NS = {"ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"}


def docx_page_count_from_app_properties(document_path: Path) -> int | None:
    """Return Word's stored page count from docProps/app.xml when present."""
    try:
        with zipfile.ZipFile(document_path) as archive:
            if "docProps/app.xml" not in archive.namelist():
                return None
            root = ET.fromstring(archive.read("docProps/app.xml"))
            for elem in root.iter():
                if elem.tag.endswith("Pages") and elem.text and str(elem.text).strip().isdigit():
                    value = int(str(elem.text).strip())
                    return value if value > 0 else None
    except Exception:
        _logger.debug("Could not read DOCX page count from %s", document_path, exc_info=True)
    return None


def docx_word_count_from_file(document_path: Path) -> int:
    """Count words in a DOCX by reading all text runs from document.xml."""
    try:
        with zipfile.ZipFile(document_path) as archive:
            if "word/document.xml" not in archive.namelist():
                return 0
            root = ET.fromstring(archive.read("word/document.xml"))
        parts: list[str] = []
        for elem in root.iter():
            if elem.tag.endswith("}t") and elem.text:
                parts.append(elem.text)
        return len(" ".join(parts).split())
    except Exception:
        _logger.debug("Could not count DOCX words from %s", document_path, exc_info=True)
        return 0


def estimate_docx_page_count(*, word_count: int, block_count: int) -> int:
    """Estimate printed pages when Word metadata does not include a page count."""
    if word_count <= 0:
        return max(1, block_count // 40)
    return max(1, (word_count + 199) // 200)


def pdf_page_count(document_path: Path) -> int | None:
    """Return PDF page count using pdfplumber when available."""
    try:
        import pdfplumber
    except ImportError:
        return None
    try:
        with pdfplumber.open(str(document_path)) as pdf:
            count = len(pdf.pages)
            return count if count > 0 else None
    except Exception:
        _logger.debug("Could not read PDF page count from %s", document_path, exc_info=True)
        return None


def page_count_from_file(document_path: Path, *, word_count: int | None = None, block_count: int | None = None) -> int | None:
    """Best-effort page count from the original file on disk."""
    suffix = document_path.suffix.lower()
    if suffix == ".docx":
        stored = docx_page_count_from_app_properties(document_path)
        if stored is not None:
            return stored
        resolved_word_count = word_count
        if resolved_word_count is None:
            resolved_word_count = docx_word_count_from_file(document_path)
        if resolved_word_count > 0:
            return estimate_docx_page_count(
                word_count=resolved_word_count,
                block_count=block_count or 0,
            )
        return None
    if suffix == ".pdf":
        return pdf_page_count(document_path)
    return None
