"""Tests for DOCX table recognition, extraction, and table-aware chunking."""

from __future__ import annotations

from src.features.chunking.application.chunking_service import _append_table_block_chunks, chunk_document_blocks
from src.features.document_processing.loaders.table_heuristics import looks_like_form_table, looks_like_specification_table
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.component_classification import COMPONENT_TABLE


def test_spec_table_with_numeric_ranges_is_not_skipped_as_form() -> None:
    text = (
        "Sl. No. | Parameters | Requirement / Specifications\n"
        "7.1 Utilities available\n"
        "| Electricity | RA W/UPS: 1Φ, 230 ± 10%V and 50 ± 5% Hz\n"
        "| Compressed Air | 0 to 6 bar\n"
        "| Process Water | 0 to 2 bar\n"
        "| Chilled Water | 10 to 15oC at 0 to 2.5 bar"
    )
    assert looks_like_specification_table(text, column_count=3) is True
    assert looks_like_form_table(text, row_count=7, column_count=3, empty_ratio=0.1) is False


def test_sign_off_table_is_still_skipped() -> None:
    text = "Department | Name | Designation | Sign & date\nPrepared by\nReviewed by\nApproved by"
    assert looks_like_form_table(text, row_count=4, column_count=4, empty_ratio=0.5) is True


def test_table_block_chunks_preserve_header_and_rows() -> None:
    block = Block(
        text=(
            "Table:\n"
            "Parameters | Requirements\n"
            "Electricity | 230V single phase supply\n"
            "Compressed Air | 0 to 6 bar\n"
            "Process Water | 0 to 2 bar"
        ),
        page=2,
        line_number=10,
        block_type="table",
        component_type=COMPONENT_TABLE,
        metadata={"component_type": COMPONENT_TABLE, "source_line_end": 13},
    )
    chunks = []
    _append_table_block_chunks(
        chunks,
        doc_name="sample.docx",
        section_name="Utilities",
        section_path="Utilities",
        block=block,
        chunk_size=40,
        min_content_words=60,
    )
    assert chunks
    assert all("Parameters | Requirements" not in chunk.text for chunk in chunks)
    assert any("Compressed Air" in chunk.text for chunk in chunks)
    assert all(chunk.content_types == [COMPONENT_TABLE] for chunk in chunks)



def test_chunk_document_blocks_treats_tables_as_first_class() -> None:
    blocks = [
        Block(text="3.0 Utilities", page=1, line_number=1, style="Heading 1", bold=True),
        Block(
            text="Table:\nParameters | Requirements\nElectricity | 230V\nCompressed Air | 0 to 6 bar",
            page=1,
            line_number=2,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE},
        ),
    ]
    chunks = chunk_document_blocks(blocks, doc_name="sample.docx", chunk_size=80, min_content_words=8)
    assert chunks
    assert any("Electricity" in chunk.text for chunk in chunks)
    assert any(COMPONENT_TABLE in (chunk.content_types or []) for chunk in chunks)
