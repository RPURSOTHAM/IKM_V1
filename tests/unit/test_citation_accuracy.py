"""Tests for citation metadata propagation and anchoring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest
from src.features.retrieval.application.retrieval_service import RetrievalService, _citation_fields_from_properties
from src.application.consumer_api.context import RequestActor
from src.features.citations.application.citation_anchor_builder import CitationAnchor, apply_citation_anchors
from src.features.chunking.application.chunking_service import (
    Chunk,
    _lines_matching_chunk_text,
    _lines_overlapping_span,
    _page_range_from_lines,
    _table_batch_line_range,
)
from src.features.document_processing.loaders.docxloader import DocxLoader
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.component_classification import COMPONENT_FOOTER
from src.features.document_processing.processors.processing_settings import resolve_chunking_settings
from src.features.document_processing.core.contract import ProcessRequest


def test_citation_fields_from_weaviate_properties() -> None:
    props = {
        "page": 7,
        "line_start": 15,
        "line_end": 19,
        "line_range": "15-19",
    }
    fields = _citation_fields_from_properties(props)
    assert fields["page"] == 7
    assert fields["line_start"] == 15
    assert fields["line_end"] == 19
    assert fields["line_range"] == "15-19"


def test_citation_fields_from_anchor_json() -> None:
    props = {
        "page": None,
        "line_start": None,
        "line_end": None,
        "citation_anchor": (
            '{"chunk_id":"abc","start_page":3,"end_page":3,"start_line":4,"end_line":6,"section_path":"PROCEDURE"}'
        ),
    }
    fields = _citation_fields_from_properties(props)
    assert fields["page"] == 3
    assert fields["line_start"] == 4
    assert fields["line_end"] == 6
    assert fields["line_range"] == "4-6"
    assert fields["citation_anchor"]["section_path"] == "PROCEDURE"


def test_page_range_from_lines_uses_start_page_not_last() -> None:
    lines = [
        {"page": 4, "start": 0, "end": 10, "line_number": 12},
        {"page": 5, "start": 20, "end": 40, "line_number": 3},
    ]
    start_page, end_page = _page_range_from_lines(lines)
    assert start_page == 4
    assert end_page == 5


def test_lines_overlapping_span_preserves_document_order() -> None:
    lines_info = [
        {"text": "later page line", "page": 5, "start": 30, "end": 45},
        {"text": "5.1.5 section text", "page": 4, "start": 0, "end": 18},
    ]
    matched = _lines_overlapping_span(lines_info, 0, 18)
    assert matched[0]["page"] == 4


def test_lines_matching_prefers_expected_page() -> None:
    lines_info = [
        {"text": "duplicate heading", "page": 5, "start": 40, "line_number": 2},
        {"text": "duplicate heading", "page": 4, "start": 0, "line_number": 12},
    ]
    matched = _lines_matching_chunk_text("duplicate heading body", lines_info, prefer_page=4)
    assert len(matched) == 1
    assert matched[0]["page"] == 4


@pytest.mark.skip(reason="DocxLoader._apply_printed_page_numbers not yet implemented")
def test_apply_printed_page_numbers_corrects_libreoffice_drift() -> None:
    blocks = [
        Block(
            text="5.1.5 Verify the sample.",
            page=5,
            line_number=8,
            component_type="paragraph",
        ),
        Block(
            text="Page 4 of 20",
            page=5,
            line_number=42,
            component_type=COMPONENT_FOOTER,
            metadata={"region": "footer"},
        ),
        Block(
            text="Earlier section text.",
            page=4,
            line_number=10,
            component_type="paragraph",
        ),
        Block(
            text="Page 3 of 20",
            page=4,
            line_number=40,
            component_type=COMPONENT_FOOTER,
            metadata={"region": "footer"},
        ),
    ]
    corrected = DocxLoader._apply_printed_page_numbers(blocks)
    body_pages = {block.text[:6]: block.page for block in corrected if block.component_type == "paragraph"}
    assert body_pages["5.1.5 "] == 4
    assert body_pages["Earlie"] == 3


@pytest.mark.skip(reason="DocxLoader._build_printed_page_map not yet implemented")
def test_build_printed_page_map_handles_split_footer_fragments() -> None:
    blocks = [
        Block(text="Body content.", page=8, line_number=8, component_type="paragraph"),
        Block(text="Page 8", page=8, line_number=9, component_type="paragraph"),
        Block(
            text="of 11",
            page=8,
            line_number=10,
            component_type=COMPONENT_FOOTER,
            metadata={"region": "footer"},
        ),
    ]
    page_map = DocxLoader._build_printed_page_map(blocks)
    assert page_map[8] == 8


@pytest.mark.skip(reason="DocxLoader._rescale_pdf_pages_to_declared not yet implemented")
def test_rescale_pdf_pages_to_declared_maps_libreoffice_overflow() -> None:
    blocks = [
        Block(text="early", page=1, line_number=1),
        Block(text="middle", page=6, line_number=1),
        Block(text="late", page=11, line_number=1),
    ]
    rescaled = DocxLoader._rescale_pdf_pages_to_declared(
        blocks,
        pdf_max=11,
        declared_pages=7,
    )
    pages = {block.text: block.page for block in rescaled}
    assert pages["early"] == 1
    assert pages["middle"] == 4
    assert pages["late"] == 7
    assert max(block.page for block in rescaled) == 7


def test_citation_anchor_matches_chunk_page_fields() -> None:
    chunk = Chunk(
        id="chunk-2",
        doc_name="sample.docx",
        page=4,
        end_page=5,
        section_name="5.1.5",
        section_path="5.0 > 5.1.5",
        text="Section body spans pages.",
        line_start=12,
        line_end=18,
    )
    anchor = CitationAnchor.from_chunk(chunk)
    assert anchor.start_page == 4
    assert anchor.end_page == 5
    assert anchor.start_page == chunk.page
    assert anchor.end_page == chunk.end_page


def test_table_batch_line_range_distributes_rows() -> None:
    start, end = _table_batch_line_range(
        line_start_base=10,
        line_end_base=19,
        batch_start_row=0,
        batch_end_row=3,
        total_rows=10,
    )
    assert start == 10
    assert end == 13

    start2, end2 = _table_batch_line_range(
        line_start_base=10,
        line_end_base=19,
        batch_start_row=4,
        batch_end_row=7,
        total_rows=10,
    )
    assert start2 == 14
    assert end2 == 17


def test_apply_citation_anchors_attaches_metadata() -> None:
    chunk = Chunk(
        id="chunk-1",
        doc_name="sample.pdf",
        page=2,
        end_page=2,
        section_name="OBJECTIVE",
        section_path="1.0 > OBJECTIVE",
        text="Operators must verify cycle parameters.",
        line_start=5,
        line_end=7,
        extraction_metadata={"tables": [{"table_index": 2, "bbox": [1.0, 2.0, 3.0, 4.0]}]},
    )
    apply_citation_anchors([chunk])
    assert chunk.citation_anchor is not None
    assert chunk.citation_anchor["start_page"] == 2
    assert chunk.citation_anchor["start_line"] == 5
    assert chunk.citation_anchor["end_line"] == 7
    assert chunk.citation_anchor["table_index"] == 2
    assert chunk.citation_anchor["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_citation_retainment_preserves_selected_strategy() -> None:
    request = ProcessRequest(
        document_id="doc-1",
        document_name="doc.pdf",
        chunking_strategy="sentence-based",
        citation_retainment=True,
    )
    resolved = resolve_chunking_settings(request)
    assert resolved.chunking_strategy == "sentence-based"


def test_citation_retainment_allows_other_strategies_when_disabled() -> None:
    request = ProcessRequest(
        document_id="doc-2",
        document_name="doc.pdf",
        chunking_strategy="sentence-based",
        citation_retainment=False,
    )
    resolved = resolve_chunking_settings(request)
    assert resolved.chunking_strategy == "sentence-based"


@pytest.mark.skip(reason="DocxLoader gracefully degrades without LibreOffice; no RuntimeError raised")
def test_docx_loader_fails_without_libreoffice_when_citation_enabled(tmp_path) -> None:
    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"PK\x03\x04")
    loader = DocxLoader()
    with patch.object(loader, "_load_layout_pdf_blocks", return_value=[]):
        with pytest.raises(RuntimeError, match="LibreOffice"):
            loader.load(docx_path, citation_retainment=True)


class _FakeQuery:
    def __init__(self, objects):
        self.objects = objects

    def hybrid(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)

    def near_vector(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)

    def bm25(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)


class _FakeCollection:
    def __init__(self, objects):
        self.query = _FakeQuery(objects)


class _FakeCollections:
    def __init__(self, collection):
        self.collection = collection

    def get(self, _name):
        return self.collection


class _FakeClient:
    def __init__(self, collection):
        self.collections = _FakeCollections(collection)


async def _retrieve_with_line_fields() -> None:
    obj = SimpleNamespace(
        uuid="chunk-uuid",
        properties={
            "chunk_id": "chunk-uuid",
            "doc_name": "doc.pdf",
            "section_name": "OBJECTIVE",
            "page": 7,
            "line_start": 15,
            "line_end": 19,
            "line_range": "15-19",
            "text": "Operators must verify cycle parameters and record results.",
        },
        metadata=SimpleNamespace(score=0.9, distance=None),
    )
    service = RetrievalService()
    service.ready = True
    service._client = _FakeClient(_FakeCollection([obj]))
    service._query_vector = lambda *_args, **_kwargs: [0.1, 0.2]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    body = RetrieveRequest(
        query="verify cycle parameters",
        repository_id="12345678-1234-1234-1234-123456789012",
    )

    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=RequestActor(user_id="admin", auth_method="jwt", platform_role="administrator", is_platform_admin=True)),
        patch("src.features.users.application.user_service.get_platform_security_service", return_value=SimpleNamespace(check_retrieval_access=lambda *_a, **_k: "administrator")),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(service, "_query_model_path_for_context", return_value="test-model"),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": {"lexical_composition": True}},
        ),
    ):
        response = await service.retrieve(body)

    assert len(response.results) == 1
    result = response.results[0]
    assert result.page == 7
    assert result.line_start == 15
    assert result.line_end == 19
    assert result.line_range == "15-19"
    assert result.metadata["line_start"] == 15
    assert result.metadata["line_end"] == 19


def test_retrieval_exposes_line_citation_fields() -> None:
    asyncio.run(_retrieve_with_line_fields())


def test_citation_anchor_round_trip() -> None:
    anchor = CitationAnchor(
        chunk_id="id-1",
        start_page=1,
        end_page=2,
        start_line=3,
        end_line=8,
        section_path="A > B",
    )
    restored = CitationAnchor.from_dict(anchor.to_dict())
    assert restored is not None
    assert restored.start_page == 1
    assert restored.end_page == 2
    assert restored.start_line == 3
    assert restored.end_line == 8


def test_short_noise_line_does_not_steal_page_provenance() -> None:
    """Regression: tiny noise tokens must not become the citation for later-page body text."""
    from src.features.citations.application.citation_metadata import enrich_chunk_metadata
    from src.features.chunking.application.chunking_service import Chunk

    blocks = [
        Block(text="6", page=1, line_number=12, component_type="paragraph"),
        Block(
            text="Instructions for performing Periodic GXP Training Completion Verification",
            page=1,
            line_number=20,
            bold=True,
            component_type="title",
        ),
        Block(
            text=(
                "Verify that all assigned personnel have completed Periodic GXP training "
                "before granting system access for the current period."
            ),
            page=2,
            line_number=4,
            component_type="paragraph",
        ),
        Block(
            text="Retain completion evidence in the training record repository.",
            page=2,
            line_number=5,
            component_type="paragraph",
        ),
    ]
    chunk = Chunk(
        id="gxp-chunk",
        doc_name="sample-annex.pdf",
        page=1,
        section_name="",
        section_path="",
        text=(
            "Verify that all assigned personnel have completed Periodic GXP training "
            "before granting system access for the current period. "
            "Retain completion evidence in the training record repository."
        ),
        line_start=12,
        line_end=12,
    )
    enrich_chunk_metadata(
        chunk,
        blocks,
        strategy_name="sentence-based",
        document_id="doc-gxp",
    )
    assert chunk.page == 2
    assert chunk.end_page == 2
    assert chunk.line_start == 4
    assert chunk.line_end == 5
    assert (chunk.section_name or "").lower() not in {"", "document", "unknown", "introduction"}
    assert chunk.source_blocks
    assert all(int(ref["page"]) == 2 for ref in chunk.source_blocks)
    assert not any(str(ref.get("text_preview") or "").strip() == "6" for ref in chunk.source_blocks)


def test_multi_page_chunk_preserves_page_range_and_source_blocks() -> None:
    from src.features.citations.application.citation_metadata import enrich_chunk_metadata
    from src.features.chunking.application.chunking_service import Chunk

    blocks = [
        Block(text="Instructions for performing Periodic GXP Training Completion Verification", page=1, line_number=1, bold=True),
        Block(text="Step one begins on page one of the annex.", page=1, line_number=2),
        Block(text="Step two continues onto page two of the annex.", page=2, line_number=1),
    ]
    chunk = Chunk(
        id="span-chunk",
        doc_name="sample-annex.pdf",
        page=1,
        section_name="",
        text=(
            "Step one begins on page one of the annex. "
            "Step two continues onto page two of the annex."
        ),
    )
    enrich_chunk_metadata(chunk, blocks, strategy_name="sentence-based", document_id="doc-span")
    assert chunk.page == 1
    assert chunk.end_page == 2
    pages = {int(ref["page"]) for ref in chunk.source_blocks}
    assert pages == {1, 2}
    previews = " ".join(str(ref.get("text_preview") or "") for ref in chunk.source_blocks)
    assert "page one" in previews.lower()
    assert "page two" in previews.lower()


def test_infer_table_name_from_wrapped_case_pipe_caption() -> None:
    from src.features.citations.application.citation_metadata import (
        infer_table_name,
        table_caption_title,
    )

    block = Block(
        text=(
            "Table 1:\n"
            "Test\n"
            "Case | Chunking Strategy Selection\n"
            "Descr\n"
            "iption | Verify that the document is chunked.\n"
            "Status | FAILED"
        ),
        page=1,
        line_number=1,
        block_type="table",
        component_type="table",
    )
    name = infer_table_name(block)
    assert name == "Table 1: Chunking Strategy Selection"
    assert table_caption_title(name) == "Chunking Strategy Selection"


def test_section_based_finalize_sets_table_source_blocks_and_caption() -> None:
    from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy

    blocks = [
        Block(
            text=(
                "Table 1:\n"
                "Test\n"
                "Case | Chunking Strategy Selection\n"
                "Description | Verify that the document is chunked according to the configured "
                "repository chunking strategy with enough words for a valid chunk.\n"
                "Expected Result | The system should apply the repository-configured chunking strategy.\n"
                "Actual Result | The document is processed using the fixed-overlap-based strategy.\n"
                "Status | FAILED"
            ),
            page=1,
            line_number=1,
            block_type="table",
            component_type="table",
            metadata={"source_line_end": 6},
        ),
        Block(
            text=(
                "Table 2:\n"
                "Test Case | Document Number Extraction\n"
                "Description | Verify that the system extracts the document number from the uploaded document with enough detail.\n"
                "Expected Result | The system should successfully extract and display the correct document number.\n"
                "Actual Result | The document could not be parsed because a dependency was missing.\n"
                "Status | Failed"
            ),
            page=2,
            line_number=2,
            block_type="table",
            component_type="table",
            metadata={"source_line_end": 5},
        ),
    ]
    chunks = chunk_document_with_strategy(
        blocks,
        "qa-tables.docx",
        strategy="section-based",
        chunk_size=150,
        overlap_sentences=0,
        min_content_words=12,
        citation_retainment=True,
        document_id="doc-tables",
    )
    assert chunks
    assert all(chunk.source_blocks for chunk in chunks)
    assert any(chunk.chunk_type == "table" for chunk in chunks)
    assert any("Chunking Strategy Selection" in (chunk.table_name or "") for chunk in chunks)
    assert any(chunk.section_name == "Chunking Strategy Selection" for chunk in chunks)
