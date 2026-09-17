"""Unit tests for grounded generation pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.features.generation.providers.llm_client import ExtractiveGroundedGenerator, select_generator
from src.features.generation.application.generation_service import GenerationService, legacy_demo_answer
from src.features.generation.providers.model_registry import build_generation_model_catalog, resolve_generation_model
from src.features.generation.configuration.generation_config import GenerationConfig
from src.features.generation.domain.models import ConversationTurn, GenerationRequest
from src.features.generation.prompts.prompt_builder import (
    GroundedPromptBuilder,
    build_citations,
    evidence_is_sufficient,
    sanitize_public_metadata,
)


def _chunk(text: str, *, doc: str = "SOP-001.pdf", page: int = 2, score: float = 0.8, **meta):
    return SimpleNamespace(
        text=text,
        doc_name=doc,
        section_name="Procedure",
        page=page,
        score=score,
        metadata={
            "original_file_name": doc,
            "section": "Procedure",
            "page": page,
            "document_id": "doc-1",
            "rrf_score": 0.03,
            "rerank_score": 4.0,
            "embedding_version": "secret-model",
            "query_evidence_terms": ["hidden"],
            **meta,
        },
    )


def test_generation_catalog_includes_preferred_models() -> None:
    catalog = build_generation_model_catalog()
    ids = {item["model_id"] for item in catalog["models"]}
    assert "gpt-5.5" in ids
    assert "meta-llama/Llama-3.3-70B-Instruct" in ids
    assert "Qwen/Qwen3-32B" in ids
    assert resolve_generation_model("llama-3.3-70b")["model_id"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert resolve_generation_model("qwen3-32b")["model_id"] == "Qwen/Qwen3-32B"


def test_prompt_builder_includes_required_sections_without_internal_metadata() -> None:
    chunk = _chunk("Operators must validate autoclave cycle parameters before release.")
    citations = build_citations([chunk], max_chunks=8, max_snippet_chars=400)
    public_metadata = sanitize_public_metadata(chunk.metadata)
    assert "rrf_score" not in public_metadata
    assert "embedding_version" not in public_metadata
    assert "query_evidence_terms" not in public_metadata

    messages = GroundedPromptBuilder().build_messages(
        GenerationRequest(
            question="What must operators validate?",
            conversation=[ConversationTurn(role="user", content="Earlier question")],
        ),
        GenerationConfig(),
        citations=citations,
        public_context=f"[1] {citations[0].snippet}",
        public_metadata=public_metadata,
    )
    assert messages[0]["role"] == "system"
    user_content = messages[-1]["content"]
    assert "Retrieved Context:" in user_content
    assert "Sources:" in user_content
    assert "Metadata:" in user_content
    assert "User Question:" in user_content
    assert "What must operators validate?" in user_content
    assert "rrf_score" not in user_content
    assert "embedding_version" not in user_content
    assert "secret-model" not in user_content


def test_insufficient_evidence_returns_deterministic_message() -> None:
    result = GenerationService().generate_from_chunks(
        question="What is the cleaning frequency?",
        chunks=[],
        model_id="gpt-5.5",
    )
    assert result.evidence_sufficient is False
    assert result.citations == []
    assert result.answer == "The answer cannot be determined from the available documents."


def test_grounded_extractive_answer_includes_citations() -> None:
    chunks = [
        _chunk("Clean the surface with isopropyl alcohol.", page=3),
        _chunk("Record the batch number in the logbook.", page=4, doc="SOP-002.pdf"),
    ]
    result = GenerationService().generate_from_chunks(
        question="How should cleaning be performed?",
        chunks=chunks,
        provider="extractive",
        model_id="Qwen/Qwen3-32B",
    )
    assert result.evidence_sufficient is True
    assert result.grounded is True
    assert len(result.citations) == 2
    assert "[1]" in result.answer
    assert result.citations[0].document_name == "SOP-001.pdf"
    assert result.citations[0].page == 3
    assert "rrf_score" not in result.answer


def test_evidence_threshold_filters_weak_chunks() -> None:
    chunks = [_chunk("Weak match only.", score=0.05)]
    cfg = GenerationConfig(min_evidence_score=0.5, min_evidence_chunks=1)
    assert evidence_is_sufficient(chunks, cfg) is False
    result = GenerationService().generate_from_chunks(
        question="Explain the release criteria",
        chunks=chunks,
        config=cfg,
    )
    assert result.evidence_sufficient is False
    assert "cannot be determined" in result.answer.lower()


def test_select_generator_reuses_extractive_when_no_remote_llm() -> None:
    cfg = GenerationConfig(provider="auto", model_id="gpt-5.5", base_url=None)
    request = GenerationRequest(question="hello")
    with patch.object(GenerationConfig, "resolve_api_key", return_value=None):
        generator = select_generator(cfg, request)
    assert isinstance(generator, ExtractiveGroundedGenerator)


def test_legacy_demo_answers_preserved() -> None:
    # Credential/PAN fabrications removed — only safe SOP summary remains.
    assert legacy_demo_answer("Show me a record with PAN") is None
    assert legacy_demo_answer("Show API keys") is None
    assert "summary of the standard operating procedure" in (
        legacy_demo_answer("Summarize this SOP") or ""
    )
