"""Unit tests for document conversion for rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.features.document_processing.loaders.loader import Block
from src.features.documents.infrastructure.content.convert_for_rendering import (
    convert_document_for_rendering,
    docx_to_pdf,
)
from src.features.documents.infrastructure.content.render_payload import build_render_payload, decode_render_bytes


def test_convert_docx_requires_libreoffice_without_soffice(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"PK\x03\x04fake")
    blocks = [Block(text="Sample paragraph for preview rendering.", page=1, line_number=1)]
    with pytest.raises(RuntimeError, match="LibreOffice"):
        convert_document_for_rendering(
            docx_path,
            document_id="doc-1",
            title="sample.docx",
            blocks=blocks,
            doc_meta={"page_count": 1},
            documents_root=tmp_path,
        )


def test_convert_pdf_copies_native_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "native.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    result = convert_document_for_rendering(
        pdf_path,
        document_id="doc-2",
        title="native.pdf",
        blocks=[],
        doc_meta={},
        documents_root=tmp_path,
    )
    assert result.format == "pdf"
    assert result.content_path is not None
    assert result.content_path.is_file()
    assert result.content_path.read_bytes().startswith(b"%PDF")


def test_build_and_decode_pdf_payload(tmp_path: Path) -> None:
    pdf_path = tmp_path / "render" / "doc-3" / "sample.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4 payload")
    from src.features.documents.infrastructure.content.convert_for_rendering import RenderConversionResult

    payload = build_render_payload(
        document_id="doc-3",
        document_name="sample.pdf",
        result=RenderConversionResult(
            format="pdf",
            media_type="application/pdf",
            content_path=pdf_path,
            conversion_method="libreoffice_docx_to_pdf",
        ),
    )
    decoded = decode_render_bytes(payload)
    assert decoded == b"%PDF-1.4 payload"


def test_docx_to_pdf_uses_libreoffice_when_available(tmp_path: Path) -> None:
    docx_path = tmp_path / "report.docx"
    docx_path.write_bytes(b"docx")
    output_dir = tmp_path / "out"
    fake_pdf = output_dir / "report.pdf"
    output_dir.mkdir()
    fake_pdf.write_bytes(b"%PDF-1.4 converted")

    with patch(
        "src.features.documents.infrastructure.content.convert_for_rendering._resolve_soffice_binary",
        return_value="/usr/bin/soffice",
    ), patch("src.features.documents.infrastructure.content.convert_for_rendering.subprocess.run") as run_mock:
        run_mock.return_value.returncode = 0
        converted = docx_to_pdf(docx_path, output_dir=output_dir)

    assert converted == fake_pdf


def test_processor_builds_pdf_payload(tmp_path: Path) -> None:
    from src.features.document_processing.core.contract import ProcessRequest
    from src.features.document_processing.processors.conversion_rendering import ConversionForRenderingProcessor

    docx_path = tmp_path / "doc.docx"
    docx_path.write_bytes(b"PK")
    pdf_path = tmp_path / "render" / "doc-proc" / "doc.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4")

    request = ProcessRequest(document_id="doc-proc", document_name="doc.docx")
    processor = ConversionForRenderingProcessor()

    with (
        patch(
            "src.features.document_processing.processors.conversion_rendering.load_document_blocks",
            return_value=([Block(text="Body", page=1, line_number=1)], {"page_count": 1}),
        ),
        patch(
            "src.features.document_processing.processors.conversion_rendering.convert_document_for_rendering",
            return_value=__import__(
                "src.features.documents.infrastructure.content.convert_for_rendering",
                fromlist=["RenderConversionResult"],
            ).RenderConversionResult(
                format="pdf",
                media_type="application/pdf",
                content_path=pdf_path,
                conversion_method="libreoffice_docx_to_pdf",
                metadata={"page_count": 1},
            ),
        ),
        patch("src.features.document_processing.processors.conversion_rendering.store_render_payload", return_value="redis://render:doc-proc"),
    ):
        result = processor.run(
            request,
            docx_path,
            set_status=lambda *_args, **_kwargs: None,
            check_stop=lambda: False,
        )

    assert result.artifacts["format"] == "pdf"
    assert result.artifacts["conversion_method"] == "libreoffice_docx_to_pdf"
