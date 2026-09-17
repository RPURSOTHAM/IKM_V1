"""Load PDFs through the vendored pdf_reader_engine extractor."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdf_reader_adapter import (
    adapt_pdf_reader_content_to_blocks,
)
from src.features.document_processing.loaders.pdf_reader_engine import (
    DocumentContent,
    extract_document,
)


def pdf_reader_engine_enabled() -> bool:
    engine = (os.getenv("PDF_LOADER_ENGINE") or "pdf_reader").strip().lower()
    return engine in {"pdf_reader", "reader", "pymupdf"}


def load_pdf_with_reader(
    document_path: Path,
    *,
    image_dir: Path | None = None,
) -> DocumentContent:
    if image_dir is None:
        with tempfile.TemporaryDirectory(prefix="pdf-reader-images-") as tmp:
            return extract_document(str(document_path), image_dir=tmp)
    image_dir.mkdir(parents=True, exist_ok=True)
    return extract_document(str(document_path), image_dir=image_dir)


def load_pdf_blocks_with_reader(document_path: Path) -> list[Block]:
    content = load_pdf_with_reader(document_path)
    return adapt_pdf_reader_content_to_blocks(content)
