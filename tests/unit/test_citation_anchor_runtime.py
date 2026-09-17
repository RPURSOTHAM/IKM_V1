"""Focused runtime-path tests for citation_anchor parse → ChunkOut fields."""

from __future__ import annotations

import json

from src.features.retrieval.application.retrieval_service import _citation_fields_from_properties
from src.features.citations.application.citation_anchor_builder import parse_citation_anchor


def _sample_anchor() -> dict:
    return {
        "chunk_id": "08283082-d8e3-4eea-b31d-b2591aac1a05",
        "document_id": "5b4f5738-9fa3-45e6-8dde-99b3e066d45d",
        "document_name": "GL-CQA-ANN-0427.pdf",
        "section_name": "ANNEXURE",
        "chunk_type": "paragraph",
        "start_page": 2,
        "end_page": 3,
        "start_line": 1,
        "end_line": 4,
    }


def test_parse_citation_anchor_from_json_string() -> None:
    raw = json.dumps(_sample_anchor())
    parsed = parse_citation_anchor(raw)
    assert parsed is not None
    for key in (
        "chunk_id",
        "start_page",
        "end_page",
        "document_id",
        "document_name",
        "section_name",
        "chunk_type",
    ):
        assert parsed.get(key) == _sample_anchor()[key]


def test_parse_citation_anchor_from_dict() -> None:
    parsed = parse_citation_anchor(_sample_anchor())
    assert parsed is not None
    assert parsed["document_name"] == "GL-CQA-ANN-0427.pdf"


def test_parse_citation_anchor_rejects_invalid() -> None:
    assert parse_citation_anchor("") is None
    assert parse_citation_anchor(None) is None
    assert parse_citation_anchor("{not-json") is None


def test_citation_fields_from_properties_exposes_anchor() -> None:
    props = {
        "page": 2,
        "citation_anchor": json.dumps(_sample_anchor()),
    }
    fields = _citation_fields_from_properties(props)
    assert isinstance(fields["citation_anchor"], dict)
    assert fields["citation_anchor"]["chunk_id"] == _sample_anchor()["chunk_id"]
    assert fields["page"] == 2
    assert fields["page_end"] == 3
    assert fields["line_start"] == 1
    assert fields["line_end"] == 4
    assert fields["line_range"] == "1-4"
