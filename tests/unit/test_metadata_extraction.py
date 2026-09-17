"""Unit tests for metadata field extraction."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.features.document_processing.loaders.component_classification import COMPONENT_IMAGE, COMPONENT_TABLE
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.metadata.field_extractor import extract_metadata_fields
from src.features.document_processing.processors.metadata_extraction import MetadataExtractionProcessor


METADATA_FIELDS = [
    {"field_name": "Equipment_Name", "display_label": "Equipment Name", "data_type": "string"},
    {"field_name": "Document_Number", "display_label": "Document Number", "data_type": "string"},
    {"field_name": "Effective_Date", "display_label": "Effective Date", "data_type": "date"},
]


def test_extract_metadata_fields_from_label_value_lines() -> None:
    blocks = [
        Block(text="Equipment_Name: Centrifuge Model X100", page=1, line_number=1),
        Block(text="Document_Number: PROC-001", page=1, line_number=2),
        Block(text="Effective_Date: 2026-01-01", page=1, line_number=3),
    ]
    extracted = extract_metadata_fields(blocks, METADATA_FIELDS)
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["Equipment_Name"]["value"] == "Centrifuge Model X100"
    assert by_name["Document_Number"]["value"] == "PROC-001"
    assert by_name["Effective_Date"]["value"] == "2026-01-01"
    assert by_name["Equipment_Name"]["confidence"] >= 0.9


def test_extract_metadata_fields_supports_spaced_labels() -> None:
    blocks = [Block(text="Equipment Name: Mixer 2000", page=1, line_number=1)]
    extracted = extract_metadata_fields(blocks, METADATA_FIELDS[:1])
    assert extracted[0]["value"] == "Mixer 2000"


def test_extract_metadata_fields_uses_next_line_values() -> None:
    blocks = [
        Block(text="Document Number:", page=1, line_number=1),
        Block(text="SOP-2026-014", page=1, line_number=2),
    ]
    extracted = extract_metadata_fields(
        blocks,
        [{"field_name": "Document_Number", "display_label": "Document Number", "data_type": "string"}],
    )
    assert extracted[0]["value"] == "SOP-2026-014"
    assert extracted[0]["extraction_method"] in {"adjacent_line", "pattern_pair", "rule_pattern"}


def test_extract_metadata_fields_from_table_rows() -> None:
    blocks = [
        Block(
            text="Table:\nEquipment Name | Centrifuge X100\nDocument Number | DOC-44",
            page=1,
            line_number=1,
            block_type="table",
            component_type=COMPONENT_TABLE,
        )
    ]
    extracted = extract_metadata_fields(blocks, METADATA_FIELDS[:2])
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["Equipment_Name"]["value"] == "Centrifuge X100"
    assert by_name["Document_Number"]["value"] == "DOC-44"
    assert by_name["Equipment_Name"]["extraction_method"] in {"table_cell", "rule_pattern", "pattern_pair"}


def test_extract_metadata_fields_skips_image_blocks() -> None:
    blocks = [
        Block(text="Equipment_Name: Visible Value", page=1, line_number=1),
        Block(
            text="Image 1 on page 1",
            page=1,
            line_number=2,
            block_type="image",
            component_type=COMPONENT_IMAGE,
        ),
    ]
    extracted = extract_metadata_fields(blocks, METADATA_FIELDS[:1])
    assert extracted[0]["value"] == "Visible Value"


def test_extract_metadata_fields_uses_embedding_fallback() -> None:
    blocks = [Block(text="Asset identifier for sterilizer unit A-17", page=1, line_number=1)]

    def fake_embed(texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 2), dtype=float)
        for idx, text in enumerate(texts):
            vectors[idx, 0] = 1.0 if "equipment" in text.lower() else 0.2
            vectors[idx, 1] = 1.0 if "sterilizer" in text.lower() else 0.1
        return vectors

    extracted = extract_metadata_fields(
        blocks,
        METADATA_FIELDS[:1],
        embed_fn=fake_embed,
    )
    assert extracted[0]["value"]
    assert extracted[0]["extraction_method"] == "embedding_similarity"


def test_legacy_find_field_value_matches_underscore_label() -> None:
    text = "Equipment_Name: Centrifuge Model X100\nDocument_Number: PROC-001"
    blocks = [Block(text=text, page=1, line_number=1)]
    extracted = extract_metadata_fields(blocks, METADATA_FIELDS[:2])
    by_name = {item["field_name"]: item for item in extracted}
    assert by_name["Equipment_Name"]["value"] == "Centrifuge Model X100"
    assert by_name["Document_Number"]["value"] == "PROC-001"


def test_process_request_accepts_null_metadata_fields() -> None:
    from src.features.document_processing.core.contract import ProcessRequest

    request = ProcessRequest.model_validate(
        {
            "document_id": "doc-1",
            "document_name": "sample.txt",
            "metadata_fields": None,
        }
    )
    assert request.metadata_fields == []


def test_processor_skips_when_no_metadata_fields(tmp_path) -> None:
    document_path = tmp_path / "meta.txt"
    document_path.write_text("Equipment_Name: Centrifuge Model X100\n", encoding="utf-8")

    from src.features.document_processing.core.contract import ProcessRequest

    request = ProcessRequest(
        document_id="doc-meta-skip",
        document_name="meta.txt",
        document_type_id=None,
        metadata_fields=[],
    )
    processor = MetadataExtractionProcessor()
    statuses: list[str] = []

    result = processor.run(
        request,
        document_path,
        set_status=lambda phase, *_args, **_kwargs: statuses.append(phase),
        check_stop=lambda: False,
    )

    assert statuses == ["skipped_no_metadata_fields"]
    assert result.document_metadata["skipped"] is True
    assert result.document_metadata["skip_reason"] == "no_metadata_fields_to_extract"
    assert result.document_metadata["extracted_fields"] == []
    assert result.artifacts["extracted_fields"] == []


def test_processor_run_with_resolved_fields(tmp_path) -> None:
    document_path = tmp_path / "meta.txt"
    document_path.write_text("Equipment_Name: Centrifuge Model X100\nDocument_Number: PROC-001\n", encoding="utf-8")

    from src.features.document_processing.core.contract import ProcessRequest

    request = ProcessRequest(
        document_id="doc-meta-1",
        document_name="meta.txt",
        document_type_id="type-1",
        metadata_fields=METADATA_FIELDS[:2],
    )
    processor = MetadataExtractionProcessor()

    with patch("src.features.document_processing.processors.metadata_extraction.store_document_graph", return_value="neo4j://test"):
        result = processor.run(
            request,
            document_path,
            set_status=lambda *_args, **_kwargs: None,
            check_stop=lambda: False,
        )

    fields = result.artifacts["extracted_fields"]
    by_name = {item["field_name"]: item for item in fields}
    assert by_name["Equipment_Name"]["value"] == "Centrifuge Model X100"
    assert by_name["Document_Number"]["value"] == "PROC-001"
