from __future__ import annotations

from unittest.mock import patch

from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy
from src.features.document_processing.loaders.component_classification import (
    COMPONENT_FOOTER,
    COMPONENT_IMAGE,
    COMPONENT_PARAGRAPH,
    COMPONENT_SUBTITLE,
    COMPONENT_TABLE,
    COMPONENT_TITLE,
)
from src.features.document_processing.loaders.loader import Block
from src.shared.semantic.models import SemanticChunk


def test_semantic_hierarchy_preserves_structural_units_and_required_metadata() -> None:
    blocks = [
        Block("1.0 Procedure", page=1, line_number=1, component_type=COMPONENT_TITLE, heading_level=1),
        Block("1.1 Preparation", page=1, line_number=2, component_type=COMPONENT_SUBTITLE, heading_level=2),
        Block(
            "Operators validate the batch before production.",
            page=1,
            line_number=3,
            component_type=COMPONENT_PARAGRAPH,
            metadata={"bounds": [10, 20, 300, 60]},
        ),
        Block(
            "Parameter | Limit\nTemperature | 25 C",
            page=2,
            line_number=1,
            block_type="table",
            component_type=COMPONENT_TABLE,
        ),
        Block(
            "Image 1 on page 2",
            page=2,
            line_number=2,
            block_type="image",
            component_type=COMPONENT_IMAGE,
            metadata={"caption": "Figure 1: Approved equipment layout"},
        ),
    ]

    with patch(
        "src.features.chunking.strategies.chunking_strategies._enrich_semantic_chunk",
        side_effect=lambda chunk: None,
    ):
        chunks = chunk_document_with_strategy(
            blocks,
            "procedure.pdf",
            document_id="DOC-123",
            strategy="semantic-hierarchy",
            chunk_size=512,
            overlap_sentences=0,
            min_content_words=1,
        )

    assert all(isinstance(chunk, SemanticChunk) for chunk in chunks)
    assert [chunk.extraction_metadata["semantic_element_type"] for chunk in chunks] == [
        "paragraph",
        "table",
        "image_caption",
    ]
    assert all(chunk.document_id == "DOC-123" for chunk in chunks)
    assert all(chunk.heading == "1.1 Preparation" for chunk in chunks)
    assert all(chunk.section_path == "1.0 Procedure > 1.1 Preparation" for chunk in chunks)
    assert chunks[0].page == 1
    assert chunks[0].coordinates[0].x0 == 10
    assert chunks[1].table_count == 1
    assert chunks[2].image_count == 1
    assert chunks[2].text == "Figure 1: Approved equipment layout"
    assert len({chunk.chunk_id for chunk in chunks}) == 3
    assert all(chunk.topic and chunk.sensitivity for chunk in chunks)


def test_semantic_hierarchy_drops_pdf_margin_text_and_title_fragments_from_paths() -> None:
    """PDF extraction artifacts must not become inherited table context."""
    blocks = [
        # These blocks can be bold/centred and pre-labelled as titles by a PDF
        # extractor, but their region provenance makes them non-structural.
        Block(
            "Corporate Quality s",
            page=1,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"region": "header"},
        ),
        Block(
            "Assurance T",
            page=1,
            line_number=2,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"region": "header"},
        ),
        Block(
            "5 OF TREND)",
            page=1,
            line_number=3,
            component_type=COMPONENT_TITLE,
            heading_level=1,
        ),
        Block("1.0 Procedure", page=1, line_number=4, component_type=COMPONENT_TITLE, heading_level=1),
        Block("1.1 Investigation", page=1, line_number=5, component_type=COMPONENT_SUBTITLE, heading_level=2),
        Block(
            "Result | Approved\nBatch 1 | Yes",
            page=1,
            line_number=6,
            block_type="table",
            component_type=COMPONENT_TABLE,
        ),
        Block(
            "Page 1 of 1",
            page=1,
            line_number=7,
            component_type=COMPONENT_FOOTER,
            metadata={"region": "footer"},
        ),
    ]

    with patch(
        "src.features.chunking.strategies.chunking_strategies._enrich_semantic_chunk",
        side_effect=lambda chunk: None,
    ):
        chunks = chunk_document_with_strategy(
            blocks,
            "oos.pdf",
            document_id="DOC-456",
            strategy="semantic-hierarchy",
            chunk_size=512,
            overlap_sentences=0,
            min_content_words=1,
        )

    assert len(chunks) == 1
    assert chunks[0].section_name.startswith("1.1")
    assert chunks[0].section_path == "1.0 Procedure > 1.1 Investigation"
    assert "Corporate Quality" not in chunks[0].section_path
    assert "OF TREND" not in chunks[0].section_path


def test_hierarchical_drops_pdf_margin_text_and_title_fragments_from_table_paths() -> None:
    """The non-semantic hierarchical strategy must protect table context too."""
    blocks = [
        Block(
            "Corporate Quality s",
            page=1,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"region": "header"},
        ),
        Block(
            "Assurance T",
            page=1,
            line_number=2,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"region": "header"},
        ),
        Block(
            "5 OF TREND)",
            page=1,
            line_number=3,
            component_type=COMPONENT_TITLE,
            heading_level=1,
        ),
        Block("1.0 Procedure", page=1, line_number=4, component_type=COMPONENT_TITLE, heading_level=1),
        Block("1.1 Investigation", page=1, line_number=5, component_type=COMPONENT_SUBTITLE, heading_level=2),
        Block(
            "Result | Approved\nBatch 1 | Yes",
            page=1,
            line_number=6,
            block_type="table",
            component_type=COMPONENT_TABLE,
        ),
    ]

    chunks = chunk_document_with_strategy(
        blocks,
        "oos.pdf",
        document_id="DOC-789",
        strategy="hierarchical",
        chunk_size=512,
        overlap_sentences=0,
        min_content_words=1,
    )

    assert len(chunks) == 1
    # Citation metadata represents numbered subsection names by their stable
    # numeric identifier, while the full human-readable label stays in path.
    assert chunks[0].section_name == "1.1"
    assert chunks[0].section_path == "1.0 Procedure > 1.1 Investigation"
    assert "Corporate Quality" not in chunks[0].section_path
    assert "OF TREND" not in chunks[0].section_path
