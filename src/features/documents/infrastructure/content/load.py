from __future__ import annotations

from pathlib import Path

from src.features.documents.infrastructure.content.blocks import Block


from src.features.documents.infrastructure.content.page_count import page_count_from_file


def document_facts(document_path: Path, blocks: list[Block]) -> dict:
    pages = {block.page for block in blocks if block.page}
    full_text = "\n".join(block.text for block in blocks if block.text)
    suffix = document_path.suffix.lower()
    file_size_bytes: int | None = None
    try:
        file_size_bytes = int(document_path.stat().st_size)
    except OSError:
        file_size_bytes = None
    page_count = max(pages) if pages else 1
    word_count = len(full_text.split())
    from_file = page_count_from_file(document_path, word_count=word_count, block_count=len(blocks))
    if from_file is not None and from_file > page_count:
        page_count = from_file
    return {
        "original_file_name": document_path.name,
        "page_count": page_count,
        "character_count": len(full_text),
        "line_count": len(blocks),
        "file_extension": suffix.lstrip("."),
        "file_size_bytes": file_size_bytes,
    }


def _load_txt(document_path: Path) -> list[Block]:
    text = document_path.read_text(encoding="utf-8", errors="replace")
    return [Block(text=line.strip(), page=1) for line in text.splitlines() if line.strip()]


def _load_pdf(document_path: Path) -> list[Block]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "PDF preview requires pdfplumber. Install dms-service with the preview extra."
        ) from exc
    blocks: list[Block] = []
    with pdfplumber.open(str(document_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            for line in page_text.splitlines():
                cleaned = line.strip()
                if cleaned:
                    blocks.append(Block(text=cleaned, page=page_num))
    return blocks


def _load_docx(document_path: Path) -> list[Block]:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise RuntimeError(
            "DOCX preview requires python-docx. Install dms-service with the preview extra."
        ) from exc
    doc = DocxDocument(str(document_path))
    blocks: list[Block] = []
    page = 1
    for paragraph in doc.paragraphs:
        text = (paragraph.text or "").strip()
        if not text:
            continue
        bold = any(run.bold for run in paragraph.runs if run.text)
        blocks.append(Block(text=text, page=page, bold=bool(bold)))
    return blocks


def _shared_load(document_path: Path) -> tuple[list[Block], dict]:
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        blocks = _load_pdf(document_path)
    elif suffix == ".docx":
        blocks = _load_docx(document_path)
    elif suffix in {".txt", ".text"}:
        blocks = _load_txt(document_path)
    else:
        raise ValueError(f"Unsupported document extension for preview: {suffix}")
    return blocks, document_facts(document_path, blocks)


def load_document_blocks(document_path: Path) -> tuple[list[Block], dict]:
    """Load document blocks; uses processor loaders when available, else shared loaders."""
    try:
        from src.features.document_processing.loaders.document_text import load_document_blocks as processor_load

        blocks, meta = processor_load(document_path)
        return blocks, meta
    except Exception:
        return _shared_load(document_path)
