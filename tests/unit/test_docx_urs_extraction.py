"""Regression tests for URS-style DOCX tables and page count estimation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.features.document_processing.loaders.table_heuristics import looks_like_form_table
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.chunking.application.chunking_service import chunk_document_blocks
from src.features.document_processing.loaders.component_classification import COMPONENT_TABLE
from src.features.documents.infrastructure.content.page_count import estimate_docx_page_count

URS_DOCX = Path(
    r"c:\Users\Adhisivan Ragunathan\OneDrive - Rudhra info solution\Shared\Cleaned\Machinery & Equipments\AUTOCLAVE\HPHV Steam Sterilizer URS (AutoClave).docx"
)


@pytest.mark.skipif(not URS_DOCX.is_file(), reason="URS sample DOCX not available on this machine")
def test_urs_spec_tables_are_not_treated_as_blank_forms() -> None:
    text = (
        "Sl. No. | Parameters | Requirement / Specifications\n"
        "1 | System Description | HPHV Steam Sterilizers are used for sterilization of various solid materials, "
        "liquid material, porous loads and heat stable material in pharmaceutical production areas."
    )
    assert looks_like_form_table(text, row_count=28, column_count=3, empty_ratio=0.27) is False


@pytest.mark.skipif(not URS_DOCX.is_file(), reason="URS sample DOCX not available on this machine")
def test_urs_docx_produces_substantial_chunks() -> None:
    blocks, meta = load_document_blocks(URS_DOCX)
    table_blocks = [block for block in blocks if block.block_type == "table"]
    assert len(blocks) >= 50
    assert len(table_blocks) >= 12
    assert sum(len(block.text) for block in blocks) >= 20000
    assert meta["page_count"] >= 10

    chunks = chunk_document_blocks(
        blocks,
        doc_name=URS_DOCX.name,
        chunk_size=150,
        min_content_words=60,
    )
    assert len(chunks) >= 20
    table_chunks = [chunk for chunk in chunks if COMPONENT_TABLE in (chunk.content_types or [])]
    assert len(table_chunks) >= 8


def test_estimate_docx_page_count_from_words() -> None:
    assert estimate_docx_page_count(word_count=5200, block_count=50) >= 18
