from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

REPO_TMP = Path(__file__).resolve().parents[3] / ".pytest-extraction-tmp"


def _scratch() -> Path:
    REPO_TMP.mkdir(parents=True, exist_ok=True)
    return REPO_TMP

from src.features.documents.application.document_extraction_cache import (
    build_extraction_payload,
    load_extraction_sidecar,
    sections_from_blocks,
    sections_from_chunk_rows,
    write_extraction_sidecar,
)
from src.features.document_processing.loaders.loader import Block


def test_write_and_load_extraction_sidecar() -> None:
    pdf_path = _scratch() / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    payload = build_extraction_payload(
        document_id="doc-1",
        filename="sample.pdf",
        suffix=".pdf",
        sections=[{"index": 1, "page": 1, "text": "Hello", "kind": "page"}],
        source="document_blocks",
    )
    write_extraction_sidecar(pdf_path, payload)
    loaded = load_extraction_sidecar(pdf_path)
    assert loaded is not None
    assert loaded["sections"][0]["text"] == "Hello"


def test_sections_from_blocks_groups_by_page() -> None:
    blocks = [
        Block(text="Page one", page=1, line_number=1),
        Block(text="More page one", page=1, line_number=2),
        Block(text="Page two", page=2, line_number=1),
    ]
    sections = sections_from_blocks(blocks, suffix=".pdf")
    assert len(sections) == 2
    assert "More page one" in sections[0]["text"]


def test_sections_from_chunk_rows_groups_by_page() -> None:
    chunks = [
        {"page": 58, "text": "Figure caption"},
        {"page": 58, "text": "BMab1200 45 mg PFS"},
        {"page": 59, "text": "Next page"},
    ]
    sections = sections_from_chunk_rows(chunks, suffix=".pdf")
    assert len(sections) == 2
    assert sections[0]["page"] == 58
    assert "BMab1200" in sections[0]["text"]


def test_get_document_extraction_uses_sidecar_without_live_reload() -> None:
    from src.features.documents.application.document_service import DocumentReceiverService

    pdf_path = _scratch() / "cached.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    write_extraction_sidecar(
        pdf_path,
        build_extraction_payload(
            document_id="doc-abc",
            filename="cached.pdf",
            suffix=".pdf",
            sections=[{"index": 1, "page": 1, "text": "Cached text", "kind": "page"}],
            source="document_blocks",
        ),
    )
    service = DocumentReceiverService()
    with (
        patch.object(service, "get_document_record", return_value={"document_id": "doc-abc"}),
        patch("src.features.documents.application.document_metadata_service.check_document_access"),
        patch.object(service, "resolve_original_file", return_value=(pdf_path, "cached.pdf", "application/pdf")),
        patch(
            "src.features.document_processing.loaders.document_text.load_document_blocks",
        ) as live_loader,
    ):
        payload = service.get_document_extraction("doc-abc")

    live_loader.assert_not_called()
    assert payload["source"] == "document_blocks"
    assert payload["sections"][0]["text"] == "Cached text"
