"""Unit tests for Phase 2 key field extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.features.documents.application.document_metadata_service import (
    _normalize_key_field_map,
    resolve_key_fields,
)
from src.features.document_types.application.processing_key_fields import key_field_to_processing_dict
from src.features.document_processing.key_fields.page_context import filter_blocks_for_extraction, select_extraction_page_numbers
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.metadata.field_extractor import extract_metadata_fields
from src.features.document_processing.shared_processor.types import ProcessorType, enabled_processor_types


def test_key_field_extraction_disabled_does_not_schedule_processor() -> None:
    enabled = enabled_processor_types({"key_field_extraction": False, "metadata_extraction": False})
    assert ProcessorType.KEY_FIELD_EXTRACTION.value not in enabled
    assert ProcessorType.CHUNKING_VECTORIZING.value in enabled


def test_key_field_extraction_enabled_schedules_processor() -> None:
    enabled = enabled_processor_types({"key_field_extraction": True})
    assert ProcessorType.KEY_FIELD_EXTRACTION.value in enabled


def test_key_field_extraction_enabled_alias_schedules_processor() -> None:
    enabled = enabled_processor_types({"key_field_extraction_enabled": True})
    assert ProcessorType.KEY_FIELD_EXTRACTION.value in enabled


def test_select_extraction_page_numbers_short_document() -> None:
    assert select_extraction_page_numbers(1) == [1]
    assert select_extraction_page_numbers(3) == [1, 2, 3]


def test_select_extraction_page_numbers_long_document() -> None:
    assert select_extraction_page_numbers(10) == [1, 2, 10]


def test_filter_blocks_for_extraction_avoids_duplicate_pages() -> None:
    blocks = [
        Block(text="Page 1", page=1, line_number=1),
        Block(text="Page 2", page=2, line_number=1),
        Block(text="Page 3", page=3, line_number=1),
        Block(text="Page 4", page=4, line_number=1),
    ]
    filtered = filter_blocks_for_extraction(blocks, page_count=4)
    pages = {block.page for block in filtered}
    assert pages == {1, 2, 4}


def test_filter_blocks_for_extraction_strict_scope_skips_when_page_metadata_missing() -> None:
    blocks = [
        Block(text="Untyped block A", page=None, line_number=1),
        Block(text="Untyped block B", page=None, line_number=2),
    ]
    filtered = filter_blocks_for_extraction(blocks, page_count=5, strict_page_scope=True)
    assert filtered == []


def test_filter_blocks_for_extraction_non_strict_falls_back_to_all_blocks() -> None:
    blocks = [
        Block(text="Untyped block A", page=None, line_number=1),
        Block(text="Untyped block B", page=None, line_number=2),
    ]
    filtered = filter_blocks_for_extraction(blocks, page_count=5, strict_page_scope=False)
    assert len(filtered) == 2


def test_key_field_to_processing_dict_uses_description_as_alias() -> None:
    payload = key_field_to_processing_dict(
        {
            "id": "field-1",
            "field_name": "batch_number",
            "field_type": "string",
            "description": "Batch Number",
        }
    )
    assert payload["field_name"] == "batch_number"
    assert payload["display_label"] == "Batch Number"
    assert "Batch Number" in payload["aliases"]


def test_effective_inherited_fields_are_extracted() -> None:
    fields = [
        key_field_to_processing_dict(
            {"field_name": "document_title", "field_type": "string", "description": "Document Title"}
        ),
        key_field_to_processing_dict(
            {"field_name": "sop_number", "field_type": "string", "description": "SOP Number"}
        ),
        key_field_to_processing_dict(
            {"field_name": "batch_number", "field_type": "string", "description": "Batch Number"}
        ),
    ]
    blocks = [
        Block(text="Document Title: Manufacturing Batch Record", page=1, line_number=1),
        Block(text="SOP Number: SOP-100", page=1, line_number=2),
        Block(text="Batch Number: BN-42", page=2, line_number=1),
    ]
    extracted = extract_metadata_fields(blocks, fields)
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["document_title"]["value"] == "Manufacturing Batch Record"
    assert by_name["sop_number"]["value"] == "SOP-100"
    assert by_name["batch_number"]["value"] == "BN-42"


def test_key_field_neo4j_payload_format() -> None:
    extracted = [{"field_name": "document_title", "value": "Widget SOP", "confidence": 0.9}]
    fields_map = {
        item["field_name"]: {"value": item["value"], "confidence": item["confidence"]}
        for item in extracted
        if item.get("field_name")
    }
    payload = {
        "document_type_id": "type-1",
        "source": "key_field_extraction",
        "fields": fields_map,
    }
    assert payload["source"] == "key_field_extraction"
    assert payload["document_type_id"] == "type-1"
    assert payload["fields"]["document_title"]["value"] == "Widget SOP"


def test_normalize_key_field_map() -> None:
    normalized = _normalize_key_field_map(
        {
            "document_title": {"value": "Title A", "confidence": 0.91},
            "version": "1.0",
        }
    )
    assert normalized["document_title"]["value"] == "Title A"
    assert normalized["document_title"]["confidence"] == 0.91
    assert normalized["version"]["value"] == "1.0"


@patch("src.features.documents.application.document_metadata_service.fetch_key_fields_from_neo4j")
def test_resolve_key_fields_returns_neo4j_payload(mock_fetch: MagicMock) -> None:
    mock_fetch.return_value = {
        "document_type_id": "type-abc",
        "fields": {"document_title": {"value": "Hello", "confidence": 0.9}},
        "source": "neo4j",
        "updated_at": "2026-07-21T10:00:00Z",
    }
    payload = resolve_key_fields({"document_id": "doc-abc", "metadata": {}})
    assert payload["document_id"] == "doc-abc"
    assert payload["document_type_id"] == "type-abc"
    assert payload["fields"]["document_title"]["value"] == "Hello"
    assert payload["updated_at"] == "2026-07-21T10:00:00Z"


@patch("src.features.documents.application.document_service.DocumentReceiverService.publisher", new_callable=MagicMock)
@patch("src.features.documents.application.document_service.DocumentReceiverService._job_store")
@patch("src.features.documents.application.document_service.DocumentReceiverService.get_store")
@patch("src.features.documents.application.document_service.DocumentReceiverService._attach_metadata_fields")
def test_submit_to_queue_skips_key_field_when_disabled(
    mock_attach_metadata,
    mock_get_store,
    mock_job_store,
    mock_publisher,
) -> None:
    from src.features.documents.application.document_service import DocumentReceiverService

    mock_attach_metadata.side_effect = lambda processing: processing
    mock_job_store.return_value = MagicMock()
    mock_get_store.return_value = MagicMock()

    record = {
        "document_id": "doc-1",
        "document_name": "file.pdf",
        "collection_name": "Docs",
        "processing": {
            "enabled_processor_types": ["chunking_vectorizing"],
            "key_field_extraction": False,
        },
    }
    DocumentReceiverService().submit_to_queue(record)
    published_types = [
        call.args[0]["processor_type"]
        for call in mock_publisher.publish_document_job.call_args_list
    ]
    assert ProcessorType.KEY_FIELD_EXTRACTION.value not in published_types
    assert ProcessorType.CHUNKING_VECTORIZING.value in published_types


@patch("src.features.documents.application.document_service.DocumentReceiverService.publisher", new_callable=MagicMock)
@patch("src.features.documents.application.document_service.DocumentReceiverService._job_store")
@patch("src.features.documents.application.document_service.DocumentReceiverService.get_store")
@patch("src.features.documents.application.document_service.DocumentReceiverService._attach_metadata_fields")
def test_submit_to_queue_schedules_key_field_when_enabled(
    mock_attach_metadata,
    mock_get_store,
    mock_job_store,
    mock_publisher,
) -> None:
    from src.features.documents.application.document_service import DocumentReceiverService

    mock_attach_metadata.side_effect = lambda processing: processing
    mock_job_store.return_value = MagicMock()
    mock_get_store.return_value = MagicMock()

    record = {
        "document_id": "doc-2",
        "document_name": "file.pdf",
        "collection_name": "Docs",
        "processing": {
            "enabled_processor_types": ["chunking_vectorizing", "key_field_extraction"],
            "key_field_extraction": True,
            "document_type_id": "type-basic",
        },
    }
    DocumentReceiverService().submit_to_queue(record)
    published_types = [
        call.args[0]["processor_type"]
        for call in mock_publisher.publish_document_job.call_args_list
    ]
    assert ProcessorType.KEY_FIELD_EXTRACTION.value in published_types
