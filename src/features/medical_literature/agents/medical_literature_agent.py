"""PubMed AI Agent — repository-scoped grounded chat with validated citations."""

from __future__ import annotations

import logging
from typing import Any

from src.features.medical_literature.citations.literature_citation_service import (
    PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE,
    PUBMED_SYSTEM_PROMPT,
    citations_to_public_dicts,
    ensure_pubmed_citations_in_answer,
    format_pubmed_context,
    format_pubmed_sources,
    validate_pubmed_citations,
)
from src.features.medical_literature.schemas.literature_schemas import PubmedAgentRequest, PubmedAgentResponse
from src.features.generation.application.generation_service import GenerationService
from src.features.generation.providers.llm_client import ExtractiveGroundedGenerator, select_generator
from src.features.generation.configuration.generation_config import GenerationConfig
from src.features.generation.domain.models import ConversationTurn, GenerationRequest
from src.features.generation.prompts.prompt_builder import (
    GroundedPromptBuilder,
    build_citations,
    chunk_to_public_fields,
    evidence_is_sufficient,
    sanitize_public_metadata,
)

logger = logging.getLogger(__name__)


class PubmedPromptBuilder(GroundedPromptBuilder):
    def build_messages(
        self,
        request: GenerationRequest,
        config: GenerationConfig,
        *,
        citations,
        public_context: str,
        public_metadata: dict[str, Any],
    ) -> list[dict[str, str]]:
        repository_id = str(
            (request.metadata or {}).get("repository_id")
            or public_metadata.get("repository_id")
            or "REPO"
        )
        sources = format_pubmed_sources(citations, repository_id=repository_id)
        context = format_pubmed_context(citations, repository_id=repository_id)
        metadata_lines = [f"- {k}: {v}" for k, v in sorted(public_metadata.items())]
        metadata_block = "\n".join(metadata_lines) if metadata_lines else "None"
        user_payload = (
            f"Retrieved Context:\n{context or 'None'}\n\n"
            f"Sources:\n{sources or 'None'}\n\n"
            f"Metadata:\n{metadata_block}\n\n"
            f"User Question:\n{request.question.strip()}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": config.system_prompt}]
        for turn in request.conversation:
            role = turn.role if isinstance(turn, ConversationTurn) else str(turn.get("role", "user"))
            content = (
                turn.content if isinstance(turn, ConversationTurn) else str(turn.get("content", ""))
            )
            role = role.strip().lower()
            if role == "system" or not content.strip():
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": content.strip()})
        messages.append({"role": "user", "content": user_payload})
        return messages


class PubmedAgentService:
    """Grounded repository chat with mandatory [REPO:DOCUMENT:PAGE] citations."""

    def __init__(self) -> None:
        self.prompt_builder = PubmedPromptBuilder()
        self.generation = GenerationService(prompt_builder=self.prompt_builder)

    async def answer(
        self,
        *,
        repository_id: str,
        body: PubmedAgentRequest,
        chunks: list[Any],
    ) -> PubmedAgentResponse:
        provider = (body.provider or "").strip() or None
        overrides: dict[str, Any] = {
            "model_id": body.model_id,
            "provider": provider,
            "system_prompt": PUBMED_SYSTEM_PROMPT,
            "insufficient_evidence_message": PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE,
            "require_citations": True,
            "enable_legacy_demo_fallback": False,
        }
        if body.top_k is not None:
            overrides["max_context_chunks"] = body.top_k
        gen_cfg = GenerationConfig.from_env(overrides)
        if not provider:
            # Prefer Azure when an Azure key/endpoint is present.
            import os

            if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_ENDPOINT"):
                gen_cfg.provider = "azure_openai"
                if os.getenv("AZURE_OPENAI_ENDPOINT"):
                    gen_cfg.base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
                deployment = (
                    body.model_id
                    or os.getenv("AZURE_OPENAI_DEPLOYMENT")
                    or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                )
                if deployment:
                    gen_cfg.model_id = deployment
                gen_cfg.api_key_env = "AZURE_OPENAI_API_KEY"
            elif gen_cfg.provider in {"auto", ""}:
                gen_cfg.provider = "openai"

        citations = build_citations(
            chunks,
            max_chunks=gen_cfg.max_context_chunks,
            max_snippet_chars=gen_cfg.max_snippet_chars,
        )
        if not evidence_is_sufficient(chunks, gen_cfg) or not citations:
            return PubmedAgentResponse(
                allowed=True,
                answer=PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                grounded=True,
                evidence_sufficient=False,
                model_id=gen_cfg.model_id,
                provider=gen_cfg.provider,
                repository_id=repository_id,
                action="refuse",
                reason="out_of_corpus",
            )

        # Prefer documents with document_id for citation contract.
        usable = [c for c in citations if c.document_id]
        if not usable:
            return PubmedAgentResponse(
                allowed=True,
                answer=PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE,
                citations=[],
                grounded=True,
                evidence_sufficient=False,
                model_id=gen_cfg.model_id,
                provider=gen_cfg.provider,
                repository_id=repository_id,
                action="refuse",
                reason="missing_document_identity",
            )
        citations = usable

        public_metadata = sanitize_public_metadata({"repository_id": repository_id})
        public_metadata["repository_id"] = repository_id
        context_chunks = [chunk_to_public_fields(c) for c in chunks[: gen_cfg.max_context_chunks]]
        request = GenerationRequest(
            question=body.query,
            conversation=[
                ConversationTurn(role=str(t.get("role") or "user"), content=str(t.get("content") or ""))
                for t in (body.conversation or [])
                if str(t.get("content") or "").strip()
            ],
            context_chunks=context_chunks,
            model_id=gen_cfg.model_id,
            provider=gen_cfg.provider,
            metadata={"repository_id": repository_id, "pubmed_agent": True},
        )
        messages = self.prompt_builder.build_messages(
            request,
            gen_cfg,
            citations=citations,
            public_context=format_pubmed_context(citations, repository_id=repository_id),
            public_metadata=public_metadata,
        )

        generator = select_generator(gen_cfg, request)
        try:
            result = generator.generate(request, gen_cfg, messages=messages)
            answer = (result.answer or "").strip()
            model_id = result.model_id or gen_cfg.model_id
            provider_out = result.provider or gen_cfg.provider
        except Exception as exc:
            logger.warning("pubmed_generator_failed falling_back_extractive error=%s", type(exc).__name__)
            result = ExtractiveGroundedGenerator().generate(request, gen_cfg, messages=messages)
            answer = ensure_pubmed_citations_in_answer(
                result.answer,
                citations,
                repository_id=repository_id,
            )
            # Rewrite [1] style from extractive into PubMed citation markers
            answer = ensure_pubmed_citations_in_answer(answer, citations, repository_id=repository_id)
            model_id = result.model_id or "extractive-grounded"
            provider_out = "local"

        answer = ensure_pubmed_citations_in_answer(answer, citations, repository_id=repository_id)
        ok, _found, invalid = validate_pubmed_citations(
            answer,
            repository_id=repository_id,
            citations=citations,
        )
        if not ok:
            # One regeneration attempt via extractive grounded rewrite
            answer = ensure_pubmed_citations_in_answer(
                PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE
                if "insufficient" in (answer or "").lower()
                else (
                    "Based on the retrieved repository evidence:\n"
                    + format_pubmed_context(citations, repository_id=repository_id)
                ),
                citations,
                repository_id=repository_id,
            )
            ok, _found, invalid = validate_pubmed_citations(
                answer,
                repository_id=repository_id,
                citations=citations,
            )
            if not ok:
                return PubmedAgentResponse(
                    allowed=True,
                    answer=PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE,
                    citations=[],
                    grounded=True,
                    evidence_sufficient=False,
                    model_id=model_id,
                    provider=provider_out,
                    repository_id=repository_id,
                    action="refuse",
                    reason="invalid_citation",
                    safe_message=f"Citation validation failed: {', '.join(invalid[:5])}",
                )

        return PubmedAgentResponse(
            allowed=True,
            answer=answer,
            citations=citations_to_public_dicts(citations, repository_id=repository_id),
            grounded=True,
            evidence_sufficient=True,
            model_id=model_id,
            provider=provider_out,
            repository_id=repository_id,
            action="allow",
        )
