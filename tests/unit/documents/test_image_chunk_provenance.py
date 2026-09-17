from __future__ import annotations

from src.features.chunking.strategies.chunking_strategies import (
    _block_coordinates,
    chunk_paragraph_based,
)
from src.features.document_processing.loaders.loader import Block


def test_image_chunk_includes_provenance_and_coordinates() -> None:
    block = Block(
        text="SC CE SDS NgHC BMab1200 45 mg PFS",
        page=58,
        line_number=2,
        block_type="image",
        component_type="image",
        metadata={
            "image_index": 1,
            "x0": 40.0,
            "top": 120.0,
            "x1": 560.0,
            "bottom": 420.0,
            "ocr_text": "SC CE SDS NgHC BMab1200 45 mg PFS",
            "embedded_image_ocr": True,
            "image_text_source": "tesseract_ocr",
            "content_type": "image_ocr",
            "image_caption": "SC CE SDS NgHC BMab1200 45 mg PFS",
        },
    )
    chunks = chunk_paragraph_based(
        [block],
        "reference.pdf",
        max_paragraphs_per_chunk=3,
        chunk_size=500,
        min_content_words=5,
        document_id="doc-123",
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_type == "image_caption"
    assert chunk.document_id == "doc-123"
    assert chunk.extraction_metadata["image_reference"]["page_number"] == 58
    assert chunk.coordinates
    assert chunk.coordinates[0].x0 == 40.0


def test_block_coordinates_reads_image_bbox_fields() -> None:
    block = Block(
        text="",
        page=10,
        line_number=1,
        block_type="image",
        metadata={"x0": 1.0, "top": 2.0, "x1": 3.0, "bottom": 4.0},
    )
    coords = _block_coordinates(block)
    assert len(coords) == 1
    assert coords[0].page == 10
    assert coords[0].y0 == 2.0
