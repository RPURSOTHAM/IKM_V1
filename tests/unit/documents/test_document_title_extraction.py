"""Regression tests for document_title extraction (punctuation / multi-span titles)."""

from __future__ import annotations

from src.features.document_types.application.processing_key_fields import key_field_to_processing_dict
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.metadata.field_extractor import (
    _is_usable_document_title_value,
    extract_metadata_fields,
)


def _title_field() -> dict:
    return key_field_to_processing_dict(
        {"field_name": "document_title", "field_type": "string", "description": "Document Title"}
    )


def test_punctuation_only_rejected_as_usable_document_title() -> None:
    for junk in (":", "-", "_", "/", "|", "...", "—", "·"):
        assert _is_usable_document_title_value(junk) is False


def test_legitimate_titles_with_punctuation_still_accepted() -> None:
    assert _is_usable_document_title_value("SOP: Cleaning Procedure") is True
    assert _is_usable_document_title_value("QA/QC Batch Record") is True
    assert _is_usable_document_title_value("C++ Coding Standard") is True


def test_unicode_titles_accepted_as_usable() -> None:
    assert _is_usable_document_title_value("設備洗浄手順") is True
    assert _is_usable_document_title_value("Процедура очистки") is True
    assert _is_usable_document_title_value("إجراءات التشغيل") is True
    assert _is_usable_document_title_value("製造バッチ記録") is True


def test_key_field_extraction_rejects_punctuation_only_document_title() -> None:
    fields = [_title_field()]
    blocks = [
        Block(text="Title:", page=1, line_number=1, component_type="body"),
        Block(text=":", page=1, line_number=2, component_type="title"),
        Block(text="Manufacturing Batch Record", page=1, line_number=3, component_type="title"),
        Block(text="SOP Number: SOP-100", page=1, line_number=4, component_type="body"),
    ]
    extracted = extract_metadata_fields(blocks, fields)
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["document_title"]["value"] != ":"
    assert by_name["document_title"]["value"] == "Manufacturing Batch Record"


def test_key_field_extraction_prefers_meaningful_over_separator_token() -> None:
    fields = [_title_field()]
    blocks = [
        Block(text="Document Title: -", page=1, line_number=1),
        Block(text="Equipment Cleaning SOP", page=1, line_number=2, component_type="title"),
    ]
    extracted = extract_metadata_fields(blocks, fields)
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["document_title"]["value"] == "Equipment Cleaning SOP"


def test_key_field_extraction_merges_multi_span_title_blocks() -> None:
    fields = [_title_field()]
    blocks = [
        Block(text="Manufacturing", page=1, line_number=1, component_type="title"),
        Block(text="Batch Record", page=1, line_number=2, component_type="title"),
        Block(text="Version: 1.0", page=1, line_number=3, component_type="body"),
    ]
    extracted = extract_metadata_fields(blocks, fields)
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["document_title"]["value"] == "Manufacturing Batch Record"


def test_key_field_extraction_keeps_title_with_embedded_punctuation() -> None:
    fields = [_title_field()]
    blocks = [
        Block(text="Document Title: SOP: Cleaning & Sanitization", page=1, line_number=1),
    ]
    extracted = extract_metadata_fields(blocks, fields)
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["document_title"]["value"] == "SOP: Cleaning & Sanitization"


def test_other_key_fields_unchanged_when_title_is_punctuation() -> None:
    fields = [
        _title_field(),
        key_field_to_processing_dict(
            {"field_name": "sop_number", "field_type": "string", "description": "SOP Number"}
        ),
    ]
    blocks = [
        Block(text="Title: :", page=1, line_number=1),
        Block(text="SOP Number: SOP-100", page=1, line_number=2),
        Block(text="Widget Assembly Guide", page=1, line_number=3, component_type="title"),
    ]
    extracted = extract_metadata_fields(blocks, fields)
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["sop_number"]["value"] == "SOP-100"
    assert by_name["document_title"]["value"] == "Widget Assembly Guide"
