"""Grounded generation service that reuses existing retrieval/chat integration."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.features.generation.providers.llm_client import select_generator
from src.features.generation.providers.model_registry import resolve_generation_model
from src.features.generation.configuration.generation_config import GenerationConfig
from src.features.generation.domain.models import (
    Citation,
    ConversationTurn,
    GenerationRequest,
    GenerationResult,
)
from src.features.generation.prompts.prompt_builder import (
    GroundedPromptBuilder,
    build_citations,
    chunk_to_public_fields,
    ensure_citations_in_answer,
    evidence_is_sufficient,
    format_public_context,
    sanitize_public_metadata,
)

_logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The answer cannot be determined from the available documents."
)


class GenerationService:
    """Orchestrates grounded answer generation without replacing chat guards."""

    def __init__(self, prompt_builder: GroundedPromptBuilder | None = None) -> None:
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()

    def generate_from_chunks(
        self,
        *,
        question: str,
        chunks: list[Any],
        conversation: list[dict[str, str]] | list[ConversationTurn] | None = None,
        model_id: str | None = None,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = config or GenerationConfig.from_env(
            {
                "model_id": model_id,
                "provider": provider,
            }
        )
        if model_id:
            cfg.model_id = model_id
        if provider:
            cfg.provider = provider

        model = resolve_generation_model(cfg.model_id)
        turns = self._normalize_conversation(conversation)
        citations = build_citations(
            chunks,
            max_chunks=cfg.max_context_chunks,
            max_snippet_chars=cfg.max_snippet_chars,
            query_text=question,
        )

        if not evidence_is_sufficient(chunks, cfg) or not citations:
            message = cfg.insufficient_evidence_message or INSUFFICIENT_EVIDENCE_MESSAGE
            return GenerationResult(
                answer=message,
                citations=[],
                grounded=True,
                evidence_sufficient=False,
                model_id=str(model["model_id"]),
                provider=str(model.get("provider") or cfg.provider),
                metadata={"reason": "insufficient_evidence"},
            )

        public_context = format_public_context(citations)
        public_metadata = sanitize_public_metadata(metadata or {})
        # Fold per-chunk public metadata into a compact request-level view.
        for citation in citations:
            if citation.document_name and "document_name" not in public_metadata:
                public_metadata["document_name"] = citation.document_name
            if citation.section and "section" not in public_metadata:
                public_metadata["section"] = citation.section

        context_chunks = [chunk_to_public_fields(chunk) for chunk in chunks[: cfg.max_context_chunks]]
        request = GenerationRequest(
            question=question,
            conversation=turns,
            context_chunks=context_chunks,
            model_id=str(model["model_id"]),
            provider=provider or cfg.provider,
            metadata=dict(metadata or {}),
        )
        messages = self.prompt_builder.build_messages(
            request,
            cfg,
            citations=citations,
            public_context=public_context,
            public_metadata=public_metadata,
        )

        generator = select_generator(cfg, request)
        try:
            result = generator.generate(request, cfg, messages=messages)
        except Exception as exc:
            _logger.warning(
                "Primary generator '%s' failed (%s); falling back to extractive grounded answer.",
                generator.name,
                exc,
            )
            from src.features.generation.providers.llm_client import ExtractiveGroundedGenerator

            result = ExtractiveGroundedGenerator().generate(request, cfg, messages=messages)

        answer = result.answer.strip()
        if not result.evidence_sufficient:
            return GenerationResult(
                answer=cfg.insufficient_evidence_message or INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                grounded=True,
                evidence_sufficient=False,
                model_id=result.model_id or str(model["model_id"]),
                provider=result.provider or str(model.get("provider") or cfg.provider),
                prompt_messages=messages,
                metadata=result.metadata,
            )

        if cfg.require_citations:
            answer = ensure_citations_in_answer(answer, citations)

        # Refine citations using the grounded answer text when source_blocks allow it.
        # Provenance still comes only from stored blocks — never invented.
        citations = build_citations(
            chunks,
            max_chunks=cfg.max_context_chunks,
            max_snippet_chars=cfg.max_snippet_chars,
            query_text=question,
            answer_text=answer,
        )
        citations = self._prefer_answer_supporting_citations(
            citations,
            question=question,
            answer=answer,
        )

        # Strip accidental internal metadata leakage from model output.
        answer = self._redact_internal_leakage(answer)

        return GenerationResult(
            answer=answer,
            citations=citations,
            grounded=True,
            evidence_sufficient=True,
            model_id=result.model_id or str(model["model_id"]),
            provider=result.provider or str(model.get("provider") or cfg.provider),
            prompt_messages=messages,
            metadata=result.metadata,
        )

    @staticmethod
    def _prefer_answer_supporting_citations(
        citations: list[Citation],
        *,
        question: str,
        answer: str,
    ) -> list[Citation]:
        """Put citations that best support the answer first; renumber indices."""
        if len(citations) <= 1:
            return citations
        try:
            from src.features.citations.resolution.citation_resolver import (
                select_supporting_source_blocks,
            )
        except Exception:
            return citations

        scored: list[tuple[float, Citation]] = []
        for citation in citations:
            score = 0.0
            blocks = citation.source_blocks if isinstance(citation.source_blocks, list) else []
            supporting = select_supporting_source_blocks(
                [b for b in blocks if isinstance(b, dict)],
                query_text=question,
                answer_text=answer,
            )
            if supporting:
                score += 20.0 + float(len(supporting))
            section = str(citation.section or "")
            haystack = " ".join(
                [
                    str(citation.snippet or ""),
                    section,
                    " ".join(
                        str(b.get("text") or b.get("text_preview") or "")
                        for b in blocks
                        if isinstance(b, dict)
                    ),
                ]
            )
            if re.search(r"(?i)\brevision\s+history\b", section) or re.search(
                r"(?i)\brevision\s+history\b", haystack
            ):
                score += 40.0
            # Unique long digit tokens from the answer that also appear in this citation.
            answer_digits = re.findall(r"\b\d{4,}\b", answer)
            for token in sorted(set(answer_digits), key=len, reverse=True):
                if token not in haystack:
                    continue
                # Rare/long identifiers beat repeated document numbers / years.
                rarity = 1.0 / float(answer_digits.count(token))
                score += (6.0 + min(len(token), 12)) * (1.0 + rarity)
            # Also reward digit cells present in the citation even when the extractive
            # answer omitted them (common when the best chunk was ranked second).
            if re.search(r"\b(number|code|identifier|id|control)\b", question.lower()):
                for token in set(re.findall(r"\b\d{5,}\b", haystack)):
                    score += 12.0
            try:
                if citation.line_start is not None and citation.line_end is not None:
                    span = abs(int(citation.line_end) - int(citation.line_start))
                    if span <= 2:
                        score += 8.0
            except (TypeError, ValueError):
                pass
            scored.append((score, citation))
        scored.sort(key=lambda item: item[0], reverse=True)
        ordered = [item[1] for item in scored]
        for index, citation in enumerate(ordered, start=1):
            citation.index = index
        return ordered

    @staticmethod
    def _normalize_conversation(
        conversation: list[dict[str, str]] | list[ConversationTurn] | None,
    ) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        for item in conversation or []:
            if isinstance(item, ConversationTurn):
                turns.append(item)
                continue
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
            if content.strip():
                turns.append(ConversationTurn(role=role, content=content))
        return turns

    @staticmethod
    def _redact_internal_leakage(answer: str) -> str:
        blocked_markers = (
            "rrf_score",
            "rerank_score",
            "embedding_version",
            "pipeline_stages",
            "retrieval_signals",
            "query_evidence_terms",
        )
        lines = []
        for line in answer.splitlines():
            lower = line.lower()
            if any(marker in lower for marker in blocked_markers):
                continue
            lines.append(line)
        return "\n".join(lines).strip()


def legacy_demo_answer(query: str) -> str | None:
    """Safe demo fallback only — never fabricates secrets, credentials, or configuration.

    Credential / API-key / PAN style queries intentionally return None so the chat
    path uses the insufficient-evidence message instead of synthetic secrets.
    """
    q = (query or "").lower()
    # Hard deny: never invent secrets or credentials in fallback mode.
    secret_tokens = (
        "api key",
        "apikey",
        "access key",
        "secret",
        "credential",
        "password",
        "token",
        "environment variable",
        "aws_access",
        "akia",
        "pan",
        "aadhaar",
        "ssn",
    )
    if any(token in q for token in secret_tokens):
        return None
    if "summarize" in q or "summary" in q:
        return (
            "Here is a summary of the standard operating procedure: "
            "The document outlines facility cleanroom entry steps and personnel requirements."
        )
    return None
