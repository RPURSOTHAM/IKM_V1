"""Tests for structural component handling during chunking."""

from __future__ import annotations

from src.features.chunking.application.chunking_service import chunk_document_blocks
from src.features.document_processing.loaders.component_classification import (
    COMPONENT_FOOTER,
    COMPONENT_HEADER,
    COMPONENT_IMAGE,
    COMPONENT_PARAGRAPH,
    COMPONENT_TABLE,
    COMPONENT_TITLE,
)
from src.features.document_processing.loaders.loader import Block


def _sample_blocks() -> list[Block]:
    return [
        Block(
            text="Standard Operating Procedure | SOP-001",
            page=1,
            line_number=1,
            component_type=COMPONENT_HEADER,
            metadata={"component_type": COMPONENT_HEADER, "region": "header"},
        ),
        Block(
            text="1.0 OBJECTIVE",
            page=1,
            line_number=2,
            style="Heading 1",
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="This procedure defines the cleaning requirements for equipment.",
            page=1,
            line_number=3,
            component_type=COMPONENT_PARAGRAPH,
        ),
        Block(
            text="Table:\nSolvent | Quantity\nIPA | 500 ml",
            page=1,
            line_number=4,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE, "source_line_end": 5},
        ),
        Block(
            text="Image 1 on page 1",
            page=1,
            line_number=6,
            block_type="image",
            component_type=COMPONENT_IMAGE,
            metadata={"component_type": COMPONENT_IMAGE, "image_index": 1},
        ),
        Block(
            text="Page 1 of 7",
            page=1,
            line_number=7,
            component_type=COMPONENT_FOOTER,
            metadata={"component_type": COMPONENT_FOOTER, "region": "footer"},
        ),
    ]


def test_chunk_document_includes_structural_components() -> None:
    chunks = chunk_document_blocks(_sample_blocks(), doc_name="sample.pdf", chunk_size=200, min_content_words=10)
    assert chunks

    all_types: set[str] = set()
    all_text = " ".join(chunk.text for chunk in chunks)
    for chunk in chunks:
        all_types.update(chunk.content_types or [])
        all_types.update((chunk.extraction_metadata or {}).get("component_types") or [])

    assert COMPONENT_HEADER not in all_types
    assert COMPONENT_FOOTER not in all_types
    assert "Standard Operating Procedure" not in all_text
    assert "Page 1 of 7" not in all_text
    assert COMPONENT_TABLE in all_types or "Solvent" in all_text
    assert "Image 1" not in all_text
    assert sum(chunk.image_count for chunk in chunks) == 1
    section_names = [chunk.section_name for chunk in chunks]
    assert any("OBJECTIVE" in name for name in section_names) or any(
        COMPONENT_TITLE in (chunk.content_types or []) for chunk in chunks
    )


def test_chunk_document_excludes_cover_page_metadata() -> None:
    blocks = [
        Block(
            text=(
                "GLOBAL OPERATING PROCEDURE\n"
                "Title: DEVIATION MANAGEMENT\n"
                "Document No.: GL-CQA-GOP-0030 | Version No.: 1.0\n"
                "Effective Date: 15/10/2025 | Review Date: 15/10/2027"
            ),
            page=1,
            line_number=1,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE, "source_line_end": 4},
        ),
        Block(
            text="1.0 PURPOSE",
            page=2,
            line_number=1,
            style="Heading 1",
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="To define a standardized procedure for reporting, documenting, investigating, assessing and closing deviations.",
            page=2,
            line_number=2,
            component_type=COMPONENT_PARAGRAPH,
        ),
    ]

    chunks = chunk_document_blocks(blocks, doc_name="deviation.pdf", chunk_size=200, min_content_words=5)
    all_text = " ".join(chunk.text for chunk in chunks)

    assert "Document No." not in all_text
    assert "Version No." not in all_text
    assert "Effective Date" not in all_text
    assert "DEVIATION MANAGEMENT" not in all_text
    assert "standardized procedure for reporting" in all_text


def test_chunk_document_preserves_business_metadata_from_cover_tables() -> None:
    blocks = [
        Block(
            text=(
                "GLOBAL OPERATING PROCEDURE | Title: MANAGEMENT OF OOS | "
                "Document No.: GL-CQA-GOP-0042 | Version No.: 1.0 | "
                "Facility | Global | Department | Corporate Quality Assurance | "
                "Sub Department | Corporate Quality Assurance | Effective Date | 01/01/2026"
            ),
            page=1,
            line_number=1,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE, "source_line_end": 4},
        ),
        Block(
            text="Facility Global Department Corporate Quality Assurance Sub Department Corporate Quality Assurance",
            page=1,
            line_number=5,
            component_type=COMPONENT_HEADER,
            metadata={"component_type": COMPONENT_HEADER, "region": "header"},
        ),
        Block(
            text="Facility Global Department Corporate Quality AssuraSnuceb Department Corporate Quality AssuraT*nce",
            page=1,
            line_number=6,
            component_type=COMPONENT_PARAGRAPH,
        ),
        Block(
            text="1.0 PURPOSE",
            page=2,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="To define the process for investigation and closure of quality events.",
            page=2,
            line_number=2,
            component_type=COMPONENT_PARAGRAPH,
        ),
    ]

    chunks = chunk_document_blocks(blocks, doc_name="metadata.pdf", chunk_size=200, min_content_words=5)
    all_text = " ".join(chunk.text for chunk in chunks)

    assert "Facility: Global" in all_text
    assert "Department: Corporate Quality Assurance" in all_text
    assert "Sub Department: Corporate Quality Assurance" in all_text
    assert "Document No." not in all_text
    assert "Effective Date" not in all_text
    assert any(chunk.section_name == "Document Metadata" for chunk in chunks)


def test_chunk_document_excludes_pdf_running_header_fragments() -> None:
    blocks = [
        Block(
            text="GLOBAL OPERATING PROCEDURE Title: MANAGEMENT OF OOS (OUT OF SPECIFICATION) AND OOT (OUT",
            page=3,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="5 OF TREND):7",
            page=3,
            line_number=2,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="Release Pending Title: MANAGEMENT OF OOS (OUT OF SPECIFICATION) AND OOT (OUT:7",
            page=3,
            line_number=3,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="1.0 PURPOSE",
            page=3,
            line_number=4,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text=(
                "Release Pending Title: MANAGEMENT OF OOS (OUT OF SPECIFICATION) AND OOT (OUT:7 "
                "To define a procedure for reporting, documenting, investigating, assessing and closing "
                "Out of Specification and Out of Trend results."
            ),
            page=3,
            line_number=5,
            component_type=COMPONENT_PARAGRAPH,
        ),
        Block(
            text="jar n. Specification gis",
            page=3,
            line_number=6,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="jar o. Resample gis",
            page=3,
            line_number=7,
            component_type=COMPONENT_PARAGRAPH,
        ),
    ]

    chunks = chunk_document_blocks(blocks, doc_name="oos.pdf", chunk_size=200, min_content_words=5)
    all_text = " ".join(chunk.text for chunk in chunks)
    section_names = {chunk.section_name for chunk in chunks}

    assert "GLOBAL OPERATING PROCEDURE" not in all_text
    assert "5 OF TREND" not in all_text
    assert "Release Pending Title" not in all_text
    assert "jar" not in all_text
    assert "gis" not in all_text
    assert "reporting, documenting, investigating" in all_text
    assert "o. Resample" in all_text
    assert "5 OF TREND)" not in section_names
    assert "jar n. Specification gis" not in section_names


def test_chunk_document_keeps_responsibility_and_definition_tables() -> None:
    blocks = [
        Block(
            text=(
                "Table:\n"
                "3.2. Deviation owner | Initiate the deviation by collecting information related to deviation. "
                "Perform initial impact assessment. Apply for cancellation if required.\n"
                "3.3. Section QA | Perform initial and final QA review. Approve 1st extension request."
            ),
            page=3,
            line_number=1,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE, "source_line_end": 3},
        ),
        Block(
            text=(
                "Table:\n"
                "Impact Assessment | Process used to determine actual or potential effects of a change, deviation, "
                "non-conformance, quality events on product, process, data, patient, regulatory compliance and business operations.\n"
                "Investigation | Structured scientific documented process to determine root cause of a quality-related event."
            ),
            page=4,
            line_number=1,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE, "source_line_end": 3},
        ),
    ]

    chunks = chunk_document_blocks(blocks, doc_name="deviation.pdf", chunk_size=200, min_content_words=5)
    all_text = " ".join(chunk.text for chunk in chunks)

    assert "Deviation owner" in all_text
    assert "Perform initial impact assessment" in all_text
    assert "Section QA" in all_text
    assert "Impact Assessment" in all_text
    assert "determine actual or potential effects" in all_text


def test_chunk_document_keeps_single_line_table_after_table_label() -> None:
    blocks = [
        Block(
            text=(
                "6.1.14. Revision to Limit of Specifications: "
                "a. Do not revise limits during an investigation. "
                "b. Any post-investigation changes to specifications or testing must be supported by scientific rationale."
            ),
            page=20,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text=(
                "Table 1: Sl. No. | Activity 6.1.14. | Revision to Limit of Specifications: "
                "a. Do not revise limits during an investigation. "
                "b. Any post-investigation changes to specifications or testing must be supported by scientific rationale."
            ),
            page=20,
            line_number=19,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE, "source_line_end": 45},
        ),
    ]

    chunks = chunk_document_blocks(blocks, doc_name="oos.pdf", chunk_size=200, min_content_words=5)
    all_text = " ".join(chunk.text for chunk in chunks)

    assert "Do not revise limits during an investigation" in all_text
    assert "post-investigation changes to specifications" in all_text
    assert "Table 1:" not in all_text
    assert any(chunk.page == 20 for chunk in chunks)


def test_chunk_document_keeps_revision_history_summary_changes() -> None:
    blocks = [
        Block(
            text="REVISION HISTORY:",
            page=1,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text=(
                "Table:\n"
                "Version Number | Effective date | Change Control Number | Summary of changes\n"
                "1.0 | 16/10/2025 | 243731 | New annexure introduction due to BioQuest implementation\n"
                "2.0 | Refer Header Section | CC-002873 | Medical devices removed and Manufacturing department is included for product related rejections."
            ),
            page=1,
            line_number=2,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={"component_type": COMPONENT_TABLE, "source_line_end": 4},
        ),
    ]

    chunks = chunk_document_blocks(blocks, doc_name="annexure.pdf", chunk_size=200, min_content_words=5)
    all_text = " ".join(chunk.text for chunk in chunks)

    assert "Summary of changes" not in all_text
    assert "New annexure introduction due to BioQuest implementation" in all_text
    assert "Medical devices removed" in all_text
    assert all(chunk.section_name == "REVISION HISTORY" for chunk in chunks)


def test_chunk_document_keeps_annexure_instructions_and_note_with_pages() -> None:
    blocks = [
        Block(
            text="Instructions for performing Periodic GXP Training Completion Verification",
            page=2,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="1. From the report presented by CQA, as pre-work, DTC shall identify whether all personnel are featured.",
            page=2,
            line_number=2,
            component_type=COMPONENT_PARAGRAPH,
        ),
        Block(
            text="Note:",
            page=2,
            line_number=20,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="No remarks required as the attached report shall present these details.",
            page=2,
            line_number=21,
            component_type=COMPONENT_PARAGRAPH,
        ),
        Block(
            text="DEVIATION RECURRENCE CHECK INSTRUCTIONS",
            page=3,
            line_number=1,
            component_type=COMPONENT_TITLE,
            heading_level=1,
            metadata={"component_type": COMPONENT_TITLE, "heading_level": 1},
        ),
        Block(
            text="A. Recurrence Check to identify OOAC: Run Recurrence Check based on the frequency of the impacted activity.",
            page=3,
            line_number=2,
            component_type=COMPONENT_PARAGRAPH,
        ),
        Block(
            text="Export Date: 12/06/2026 09:00",
            page=3,
            line_number=3,
            component_type=COMPONENT_PARAGRAPH,
        ),
    ]

    chunks = chunk_document_blocks(blocks, doc_name="annexure.pdf", chunk_size=200, min_content_words=5)
    all_text = " ".join(chunk.text for chunk in chunks)
    sections = {(chunk.section_name, chunk.page) for chunk in chunks}

    assert "From the report presented by CQA" in all_text
    assert "No remarks required" in all_text
    assert "Recurrence Check to identify OOAC" in all_text
    assert "Export Date" not in all_text
    assert ("Instructions for performing Periodic GXP Training Completion Verification", 2) in sections
    assert ("Note", 2) in sections
    assert ("DEVIATION RECURRENCE CHECK INSTRUCTIONS", 3) in sections
