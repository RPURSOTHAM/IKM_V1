"""Validation tests for citation metadata across chunking strategies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.features.citations.resolution.citation_resolver import (
    build_llm_citation_context,
    citation_quality_score,
    format_citation_line,
    resolve_citation,
)
from src.features.citations.application.citation_anchor_builder import CitationAnchor, apply_citation_anchors
from src.features.citations.application.citation_metadata import (
    compute_page_range,
    finalize_chunks_for_citation,
    infer_section,
    leaf_section_labels,
    split_section_number_and_title,
)
from src.features.chunking.strategies.chunking_strategies import (
    chunk_document_with_strategy,
    chunk_fixed_overlap_based,
    chunk_sentence_based,
    chunk_sliding_window,
)
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.processors.processing_settings import resolve_chunking_settings
from src.features.document_processing.core.contract import ProcessRequest


def _sop_blocks() -> list[Block]:
  return [
      Block(text="5.0 PROCEDURE", page=4, line_number=1, style="Heading 1", component_type="title"),
      Block(text="5.1.4 Applying and releasing vacuum", page=4, line_number=8, style="Heading 2", component_type="subtitle"),
      Block(
          text="Connect the vacuum hose to the reactor port and verify the pressure gauge reads zero.",
          page=4,
          line_number=9,
          component_type="paragraph",
      ),
      Block(text="5.1.5 Sample verification", page=5, line_number=1, style="Heading 2", component_type="subtitle"),
      Block(
          text="Collect representative samples after completion of the vacuum cycle.",
          page=5,
          line_number=2,
          component_type="paragraph",
      ),
      Block(
          text="Table 1: Critical process parameters\nParameter | Limit\nPressure | 0-5 psi",
          page=5,
          line_number=10,
          block_type="table",
          component_type="table",
          metadata={"table_index": 1, "source_line_end": 12},
      ),
  ]


@pytest.mark.parametrize(
    "strategy",
    [
        "section-based",
        "hierarchical",
        "sentence-based",
        "paragraph-based",
        "fixed-overlap-based",
        "sliding-window",
    ],
)
def test_strategies_emit_citation_metadata(strategy: str) -> None:
    blocks = _sop_blocks()
    chunks = chunk_document_with_strategy(
        blocks,
        "SOP-02214.docx",
        strategy=strategy,
        chunk_size=40,
        overlap_sentences=1,
        min_content_words=8,
        document_id="doc-sop-1",
        citation_retainment=True,
    )
    assert chunks, f"{strategy} produced no chunks"
    for chunk in chunks:
        assert chunk.strategy_name == strategy
        assert chunk.document_id == "doc-sop-1"
        assert chunk.page_start is not None
        assert chunk.page_end is not None
        assert chunk.page_start <= chunk.page_end
        assert chunk.section_name.lower() not in {"", "document", "unknown"}
        assert chunk.section_path


def test_compute_page_range_uses_all_blocks() -> None:
    blocks = _sop_blocks()
    start, end = compute_page_range(blocks)
    assert start == 4
    assert end == 5


def _pharma_blocks(section: str, title: str, body: str, *, page: int = 5) -> list[Block]:
    return [
        Block(text=f"{section} {title}", page=page, line_number=1, style="Heading 2", component_type="subtitle"),
        Block(text=body, page=page, line_number=2, component_type="paragraph"),
    ]


@pytest.mark.parametrize(
    "section,title,keyword",
    [
        ("5.5.3.1", "Working Standard Qualification", "working standard"),
        ("6.2.1", "Out of Specification (OOS)", "investigation"),
        ("7.1", "CAPA", "corrective action"),
        ("8.3", "Reserve Samples", "retention"),
    ],
)
def test_pharma_sections_preserve_number_and_title(section: str, title: str, keyword: str) -> None:
    blocks = _pharma_blocks(section, title, f"Operators must follow the {keyword} procedure documented herein.")
    number, parsed_title = split_section_number_and_title(f"{section} {title}")
    assert number == section
    assert parsed_title == title
    chunks = chunk_document_with_strategy(
        blocks,
        "SOP-00003.docx",
        strategy="sentence-based",
        chunk_size=30,
        overlap_sentences=0,
        min_content_words=6,
        document_id="doc-pharma",
        citation_retainment=True,
    )
    assert chunks
    assert chunks[0].section_name == section
    assert title.lower() in chunks[0].title.lower() or title.lower() in chunks[0].section_path.lower()


def test_infer_section_from_numbered_heading() -> None:
    blocks = _sop_blocks()[1:3]
    name, path, parent, title = infer_section(blocks)
    assert "5.1.4" in path or "5.1.4" in name
    assert parent
    assert "vacuum" in title.lower() or "Applying" in title


def test_leaf_section_labels_splits_number_and_title() -> None:
    section, path, title = leaf_section_labels("5.0 PROCEDURE > 5.5.3.1 Working Standard Qualification")
    assert section == "5.5.3.1"
    assert title == "Working Standard Qualification"


def test_format_citation_line_matches_expected_pattern() -> None:
    citation = resolve_citation(
        {
            "section_name": "5.5.3.1",
            "title": "Working Standard Qualification",
            "page": 5,
            "page_end": 5,
            "document_name": "SOP-00003",
        },
        document_display_name="SOP-00003",
    )
    line = format_citation_line(citation)
    assert line == "SOP-00003 | Section 5.5.3.1 | Page 5"


def test_llm_citation_context_forbids_hallucination() -> None:
    context = build_llm_citation_context(
        [resolve_citation({"section_name": "5.1.4", "page": 4, "document_name": "SOP-00003"})]
    )
    assert "Do not invent page numbers" in context
    assert "SOP-00003" in context


def test_multi_page_section_preserves_page_span() -> None:
    blocks = _sop_blocks()
    chunks = chunk_sentence_based(
        blocks,
        "SOP-02214.docx",
        max_sentences_per_chunk=4,
        overlap_sentences=0,
        min_content_words=6,
    )
    finalized = finalize_chunks_for_citation(
        chunks,
        blocks,
        strategy_name="sentence-based",
        document_id="doc-1",
        min_words=6,
        chunk_size=40,
    )
    pages = {chunk.page_start for chunk in finalized}
    assert 4 in pages or 5 in pages


def test_table_chunk_has_table_metadata() -> None:
    blocks = _sop_blocks()
    chunks = chunk_document_with_strategy(
        blocks,
        "SOP-02214.docx",
        strategy="section-based",
        chunk_size=80,
        overlap_sentences=1,
        min_content_words=6,
        citation_retainment=True,
    )
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table" or chunk.table_count]
    assert table_chunks
    assert any(chunk.table_count > 0 for chunk in table_chunks)
    assert any(chunk.chunk_type == "table" for chunk in table_chunks)
    assert any(chunk.source_blocks for chunk in table_chunks)


def test_orphan_chunks_merge_within_section() -> None:
    blocks = [
        Block(text="5.2 Short note.", page=6, line_number=1, component_type="paragraph"),
        Block(text="Additional supporting detail for the same section.", page=6, line_number=2, component_type="paragraph"),
    ]
    chunks = chunk_sliding_window(
        blocks,
        "SOP.docx",
        window_size=8,
        step_size=4,
        min_content_words=20,
    )
    finalized = finalize_chunks_for_citation(
        chunks,
        blocks,
        strategy_name="sliding-window",
        document_id="doc-2",
        min_words=20,
        chunk_size=8,
    )
    assert len(finalized) <= len(chunks)


def test_citation_resolver_uses_metadata_only() -> None:
    props = {
        "chunk_id": "chunk-1",
        "doc_name": "internal-id.docx",
        "section_name": "5.1.4",
        "section_path": "5.0 > 5.1.4",
        "parent_section": "5.0",
        "title": "Applying and releasing vacuum",
        "page": 6,
        "page_end": 6,
        "line_start": 12,
        "line_end": 15,
        "chunk_type": "paragraph",
        "strategy_name": "section-based",
        "document_id": "doc-1",
    }
    citation = resolve_citation(props, document_display_name="SOP-02214")
    assert citation["section"] == "5.1.4"
    assert citation["page_start"] == 6
    assert citation["title"] == "Applying and releasing vacuum"
    assert citation["document"] == "SOP-02214"


def test_citation_quality_prefers_complete_metadata() -> None:
    weak = citation_quality_score({"section_name": "Document"})
    strong = citation_quality_score(
        {
            "page": 4,
            "section_name": "5.1.5",
            "parent_section": "5.1",
            "title": "Sample verification",
            "line_start": 2,
            "line_end": 4,
        }
    )
    assert strong > weak


def test_anchor_round_trip_includes_strategy_and_pages() -> None:
    chunk = SimpleNamespace(
        id="chunk-9",
        page=4,
        end_page=5,
        page_start=4,
        page_end=5,
        line_start=2,
        line_end=6,
        section_name="5.1.5",
        section_path="5.0 > 5.1.5",
        parent_section="5.0",
        title="Sample verification",
        table_name="",
        chunk_type="paragraph",
        strategy_name="section-based",
        document_id="doc-9",
        raw_text="Collect representative samples.",
        text="Collect representative samples.",
        extraction_metadata={},
        source_blocks=[],
    )
    anchor = CitationAnchor.from_chunk(chunk)
    assert anchor.start_page == 4
    assert anchor.end_page == 5
    assert anchor.strategy_name == "section-based"


def test_clamp_chunk_pages_to_document_caps_overflow() -> None:
    from src.features.citations.application.citation_metadata import clamp_chunk_pages_to_document
    from src.features.chunking.application.chunking_service import Chunk

    chunk = Chunk(
        id="c1",
        doc_name="sop.docx",
        page=11,
        end_page=11,
        page_start=9,
        page_end=11,
        section_name="5.1.4",
        text="Sample text for citation clamping validation.",
    )
    clamp_chunk_pages_to_document(chunk, 7)
    assert chunk.page_start == 7
    assert chunk.page_end == 7
    assert chunk.page == 7


def test_processing_settings_keeps_semantic_with_citation() -> None:
    resolved = resolve_chunking_settings(
        ProcessRequest(
            document_id="doc-3",
            document_name="doc.pdf",
            chunking_strategy="semantic",
            citation_retainment=True,
        )
    )
    assert resolved.chunking_strategy == "semantic"


def test_missing_request_strategy_uses_repository_paragraph_based() -> None:
    resolved = resolve_chunking_settings(
        ProcessRequest(
            document_id="doc-para",
            document_name="doc.docx",
            chunking_strategy=None,
            repository_settings={"chunking_strategy": "paragraph-based", "chunk_size": 400},
        )
    )
    assert resolved.chunking_strategy == "paragraph-based"


def test_blank_request_strategy_does_not_force_section_based() -> None:
    request = ProcessRequest(
        document_id="doc-blank",
        document_name="doc.docx",
        chunking_strategy="",
        repository_settings={"chunking_strategy": "paragraph-based"},
    )
    assert request.chunking_strategy is None
    resolved = resolve_chunking_settings(request)
    assert resolved.chunking_strategy == "paragraph-based"


def test_paragraph_based_chunks_record_paragraph_strategy_name() -> None:
    chunks = chunk_document_with_strategy(
        _sop_blocks(),
        "SOP-02214.docx",
        strategy="paragraph-based",
        chunk_size=80,
        overlap_sentences=0,
        min_content_words=5,
        chunking_config={"max_paragraphs_per_chunk": 2},
        document_id="doc-para-1",
        citation_retainment=True,
    )
    assert chunks
    assert all(chunk.strategy_name == "paragraph-based" for chunk in chunks)
    assert any(chunk.chunk_type == "paragraph" for chunk in chunks)
