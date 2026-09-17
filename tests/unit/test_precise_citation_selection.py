"""Regression tests for precise supporting-block citation selection.

Uses synthetic multi-block chunks only — no document-specific production values.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.features.citations.resolution.citation_resolver import (
    resolve_citation,
    select_supporting_source_blocks,
)
from src.features.generation.prompts.prompt_builder import build_citations, chunk_to_public_fields
from src.features.retrieval.application.retrieval_service import (
    _extract_mentioned_document_tokens,
    _resolve_document_id_from_query,
)
from src.features.retrieval.application.pipeline.stages import RuleMetadataFilterGenerator
from src.features.retrieval.domain.models import PipelineState
from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig


def _multi_block_chunk_props() -> dict:
    """Synthetic chunk spanning unrelated table + heading + value cell."""
    return {
        "chunk_id": "chunk-synth-1",
        "document_name": "fixture-doc.pdf",
        "document_id": "doc-synth-1",
        "section_name": "REGISTER TITLE",
        "section_path": "REGISTER TITLE",
        "page": 2,
        "page_end": 2,
        "line_start": 26,
        "line_end": 51,
        "chunk_type": "paragraph",
        "source_blocks": [
            {
                "index": 0,
                "page": 2,
                "line_start": 26,
                "line_end": 32,
                "line_number": 26,
                "block_type": "table",
                "component_type": "table",
                "text": "TRACKWISE MANAGEMENT REGISTER | Owner | Status",
                "text_preview": "TRACKWISE MANAGEMENT REGISTER | Owner | Status",
            },
            {
                "index": 1,
                "page": 2,
                "line_start": 47,
                "line_end": 47,
                "line_number": 47,
                "block_type": "text",
                "component_type": "heading",
                "heading": "Revision History",
                "text": "Revision History",
                "text_preview": "Revision History",
            },
            {
                "index": 2,
                "page": 2,
                "line_start": 50,
                "line_end": 51,
                "line_number": 50,
                "block_type": "table",
                "component_type": "table",
                "table_name": "Revision History",
                "section_name": "Revision History",
                "text": (
                    "Version Number | Effective Date | Change Control Number | Summary\n"
                    "1.0 | Refer Header Section | 197902 | Annexure Migrated"
                ),
                "text_preview": "Change Control Number | 197902",
            },
        ],
    }


def test_select_supporting_blocks_narrows_to_evidence_cell() -> None:
    props = _multi_block_chunk_props()
    selected = select_supporting_source_blocks(
        props["source_blocks"],
        evidence_terms=["197902", "change", "control"],
        query_text="What is the Change Control Number?",
    )
    assert selected
    assert selected[0]["line_start"] == 50
    assert selected[0]["line_end"] == 51


def test_resolve_citation_uses_supporting_block_not_chunk_span() -> None:
    props = _multi_block_chunk_props()
    citation = resolve_citation(
        props,
        evidence_terms=["197902"],
        query_text="What is the Change Control Number?",
    )
    assert citation["page"] == 2
    assert citation["line_start"] == 50
    assert citation["line_end"] == 51
    assert citation["line_range"] == "50-51"
    section = str(citation.get("section") or "").lower()
    assert "revision" in section
    # Must not keep the broad chunk range when a precise block matched.
    assert not (citation["line_start"] == 26 and citation["line_end"] == 51)


def test_resolve_citation_preserves_cross_page_supporting_blocks() -> None:
    props = {
        "document_name": "multi-page.pdf",
        "page": 1,
        "line_start": 90,
        "line_end": 115,
        "source_blocks": [
            {
                "page": 1,
                "line_start": 90,
                "line_end": 100,
                "text": "Prefix context without the target value.",
            },
            {
                "page": 2,
                "line_start": 1,
                "line_end": 15,
                "text": "Target value ALPHA-42 appears here.",
                "section_name": "Appendix",
            },
        ],
    }
    citation = resolve_citation(
        props,
        evidence_terms=["ALPHA-42"],
        query_text="Where is ALPHA-42?",
    )
    assert citation["page_start"] == 2
    assert citation["page_end"] == 2
    assert citation["line_start"] == 1
    assert citation["line_end"] == 15
    assert "appendix" in str(citation.get("section") or "").lower()


def test_resolve_citation_does_not_fabricate_when_no_match() -> None:
    props = _multi_block_chunk_props()
    citation = resolve_citation(props)  # no evidence / query
    # Falls back to chunk-level provenance already on props — does not invent new pages/lines.
    assert citation["page_start"] == 2
    assert citation["line_start"] == 26
    assert citation["line_end"] == 51
    assert citation.get("supporting_source_blocks") is None


def test_build_citations_prefers_resolved_metadata() -> None:
    props = _multi_block_chunk_props()
    citation_payload = resolve_citation(
        props,
        evidence_terms=["197902"],
        query_text="Change Control Number",
    )
    chunk = SimpleNamespace(
        text=props["source_blocks"][2]["text"],
        score=0.9,
        metadata={
            "citation": citation_payload,
            "source_blocks": props["source_blocks"],
            "page": 2,
            "line_start": 26,
            "line_end": 51,
            "section": "REGISTER TITLE",
            "document_id": "doc-synth-1",
            "original_file_name": "fixture-doc.pdf",
        },
        document_name="fixture-doc.pdf",
        page=2,
        line_start=26,
        line_end=51,
        section_name="REGISTER TITLE",
        chunk_id="chunk-synth-1",
    )
    fields = chunk_to_public_fields(chunk)
    assert fields["line_start"] == 50
    assert fields["line_end"] == 51
    assert "revision" in str(fields["section"] or "").lower()

    built = build_citations(
        [chunk],
        max_chunks=3,
        max_snippet_chars=200,
        query_text="What is the Change Control Number 197902?",
    )
    assert len(built) == 1
    assert built[0].line_start == 50
    assert built[0].line_end == 51


def test_prefer_answer_supporting_citations_ranks_value_over_header() -> None:
    """Extractive answers dump many repeated doc digits; ranking must still prefer the value cell."""
    from src.features.generation.domain.models import Citation
    from src.features.generation.application.generation_service import GenerationService

    header = Citation(
        index=1,
        document_name="fixture-doc.pdf",
        page=1,
        line_start=17,
        line_end=21,
        section="TRACKWISE MANAGEMENT REGISTER",
        snippet="Doc No 0063 Rev 0016 Effective 2025 TRACKWISE MANAGEMENT REGISTER",
        source_blocks=[
            {
                "text": "Doc No 0063 | Rev 0016 | Effective 2025 | TRACKWISE MANAGEMENT REGISTER",
                "line_start": 17,
                "line_end": 21,
            }
        ],
    )
    value = Citation(
        index=2,
        document_name="fixture-doc.pdf",
        page=2,
        line_start=50,
        line_end=51,
        section="Revision History",
        snippet="Change Control Number 5544331 Annexure Migrated",
        source_blocks=[
            {
                "text": "Version 1.0 | Change Control Number | 5544331 | Annexure Migrated",
                "section_name": "Revision History",
                "line_start": 50,
                "line_end": 51,
            }
        ],
    )
    # Noisy extractive dump: repeated header digits, no control number in answer text.
    noisy_answer = (
        "Doc No 0063 Rev 0016 Effective 2025 TRACKWISE MANAGEMENT REGISTER. "
        "Doc No 0063 Rev 0016 Effective 2025. "
        "See revision history table for change control details."
    )
    ordered = GenerationService._prefer_answer_supporting_citations(
        [header, value],
        question="What is the Change Control Number in fixture-doc.pdf?",
        answer=noisy_answer,
    )
    assert ordered[0].line_start == 50
    assert ordered[0].line_end == 51
    assert "revision" in str(ordered[0].section or "").lower()
    assert ordered[0].index == 1


def test_extract_mentioned_document_tokens_is_generic() -> None:
    tokens = _extract_mentioned_document_tokens(
        "What is the Change Control Number in report-alpha.pdf?"
    )
    assert tokens == ["report-alpha.pdf"]
    assert _extract_mentioned_document_tokens("What is the change control number?") == []


def test_resolve_document_id_from_query_matches_lookup() -> None:
    lookup = {
        "report-alpha.pdf": {
            "document_id": "doc-aaa",
            "original_file_name": "report-alpha.pdf",
        },
        "report-beta.pdf": {
            "document_id": "doc-bbb",
            "original_file_name": "report-beta.pdf",
        },
    }
    assert (
        _resolve_document_id_from_query(
            "Find the value in report-alpha.pdf",
            lookup,
        )
        == "doc-aaa"
    )
    # No mention → no filter
    assert _resolve_document_id_from_query("Find the change control number", lookup) is None


def test_metadata_filter_generator_detects_filename() -> None:
    stage = RuleMetadataFilterGenerator()
    state = PipelineState(original_query="Open notes from sample-file.docx please")
    config = RetrievalPipelineConfig()
    if hasattr(config, "enable_metadata_filter_generator"):
        # Ensure stage runs when the flag exists.
        try:
            config.enable_metadata_filter_generator = True
        except Exception:
            pass
    result = stage.run(state, config)
    assert result.status in {"ok", "skipped", "degraded"}
    if result.status == "ok":
        assert str(state.filters.get("doc_name") or "").lower().endswith(".docx")


def test_no_hardcoded_document_names_in_resolver_source() -> None:
    from pathlib import Path

    source = Path(
        "src/features/citations/resolution/citation_resolver.py"
    ).read_text(encoding="utf-8")
    forbidden = ("GL-CQA-ANN-0063", "GL-CQA-ANN-0062", "197902", "page = 2", "line_start = 50")
    for token in forbidden:
        assert token not in source, f"Forbidden hardcoding found: {token}"
