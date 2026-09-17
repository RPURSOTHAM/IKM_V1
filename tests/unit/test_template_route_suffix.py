"""Routing tests for TemplateExtractionProcessor DOCX/PDF selection."""

from __future__ import annotations

from pathlib import Path

from src.features.document_processing.core.contract import ProcessRequest
from src.features.document_processing.processors.template_extraction import resolve_template_route_suffix


def test_route_prefers_path_suffix() -> None:
    req = ProcessRequest(document_id="d1", document_name="d1.bin", original_file_name="x.txt")
    suffix, source = resolve_template_route_suffix(req, Path("/app/documents/d1.docx"))
    assert suffix == ".docx"
    assert source == "document_path"


def test_route_falls_back_to_document_name() -> None:
    req = ProcessRequest(document_id="d1", document_name="d1.pdf", original_file_name=None)
    suffix, source = resolve_template_route_suffix(req, Path("/app/documents/d1"))
    assert suffix == ".pdf"
    assert source == "document_name"


def test_route_falls_back_to_original_file_name() -> None:
    req = ProcessRequest(
        document_id="d1",
        document_name="d1",
        original_file_name="report.docx",
    )
    suffix, source = resolve_template_route_suffix(req, Path("/tmp/upload.bin"))
    assert suffix == ".docx"
    assert source == "original_file_name"
