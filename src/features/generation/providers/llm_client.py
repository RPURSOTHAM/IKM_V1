"""OpenAI-compatible generation client used by GPT / Llama / Qwen deployments."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from src.features.generation.providers.model_registry import resolve_generation_model
from src.features.generation.configuration.generation_config import GenerationConfig
from src.features.generation.domain.interfaces import Generator
from src.features.generation.domain.models import GenerationRequest, GenerationResult

_logger = logging.getLogger(__name__)


class OpenAICompatibleGenerator(Generator):
    """Calls /chat/completions on OpenAI or OpenAI-compatible servers.

    This is additive infrastructure. Existing chat guards and retrieval remain upstream.
    """

    name = "openai_compatible"

    def generate(
        self,
        request: GenerationRequest,
        config: GenerationConfig,
        *,
        messages: list[dict[str, str]],
    ) -> GenerationResult:
        model = resolve_generation_model(request.model_id or config.model_id)
        model_id = str(model["model_id"])
        api_key = config.resolve_api_key()
        base_url = (
            request.metadata.get("base_url")
            or config.base_url
            or model.get("default_base_url")
            or "https://api.openai.com/v1"
        )
        base_url = str(base_url).rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": float(config.temperature),
            "max_tokens": int(config.max_output_tokens),
        }
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=float(config.timeout_seconds)) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            answer = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                or ""
            ).strip()
            if not answer:
                raise RuntimeError("Empty completion content returned by generation provider.")
            return GenerationResult(
                answer=answer,
                grounded=True,
                evidence_sufficient=True,
                model_id=model_id,
                provider=str(model.get("provider") or "openai_compatible"),
                prompt_messages=messages,
                metadata={"endpoint": endpoint, "transport": "openai_compatible"},
            )
        except Exception as exc:
            _logger.warning("OpenAI-compatible generation failed: %s", exc)
            raise


class ExtractiveGroundedGenerator(Generator):
    """Reuse existing extractive chat behavior as a grounded offline fallback.

    This preserves the repository's current non-remote generation path instead of
    replacing it with a hard dependency on an external LLM.
    """

    name = "extractive_grounded"

    def generate(
        self,
        request: GenerationRequest,
        config: GenerationConfig,
        *,
        messages: list[dict[str, str]],
    ) -> GenerationResult:
        # Prefer explicit context chunks carried on the request.
        snippets: list[str] = []
        for index, chunk in enumerate(request.context_chunks[: config.max_context_chunks], start=1):
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            doc = str(chunk.get("document_name") or "document")
            page = chunk.get("page")
            page_bit = f", p.{page}" if page is not None else ""
            snippets.append(f"[{index}] ({doc}{page_bit}) {text}")
        if not snippets:
            answer = config.insufficient_evidence_message
            return GenerationResult(
                answer=answer,
                grounded=True,
                evidence_sufficient=False,
                model_id="extractive-grounded",
                provider="local",
                prompt_messages=messages,
                metadata={"implementation": "extractive_grounded"},
            )
        joined = " ".join(snippets)
        answer = (
            "Based on the retrieved documents:\n"
            f"{joined}\n\n"
            f"Sources: {' '.join(f'[{i}]' for i in range(1, len(snippets) + 1))}"
        )
        return GenerationResult(
            answer=answer.strip(),
            grounded=True,
            evidence_sufficient=True,
            model_id="extractive-grounded",
            provider="local",
            prompt_messages=messages,
            metadata={"implementation": "extractive_grounded"},
        )


def select_generator(config: GenerationConfig, request: GenerationRequest) -> Generator:
    """Prefer remote OpenAI-compatible clients when configured; else reuse extractive path."""
    provider = (request.provider or config.provider or "auto").strip().lower()
    has_remote = bool(config.resolve_api_key() or config.base_url or request.metadata.get("base_url"))
    if provider in {"extractive", "local"}:
        return ExtractiveGroundedGenerator()
    if provider in {"openai", "azure_openai", "openai_compatible", "vllm", "ollama"}:
        if has_remote:
            return OpenAICompatibleGenerator()
        return ExtractiveGroundedGenerator()
    # auto
    if has_remote:
        return OpenAICompatibleGenerator()
    return ExtractiveGroundedGenerator()
