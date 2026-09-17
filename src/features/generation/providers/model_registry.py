"""Catalog of supported generation models. Additive; does not replace existing settings."""

from __future__ import annotations

from typing import Any


GENERATION_MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "provider": "openai",
        "model_id": "gpt-5.5",
        "label": "GPT-5.5",
        "transport": "openai_compatible",
        "description": "OpenAI GPT-5.5 chat model.",
        "default_base_url": "https://api.openai.com/v1",
    },
    {
        "provider": "openai_compatible",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "label": "Llama 3.3 70B",
        "transport": "openai_compatible",
        "description": "Llama 3.3 70B Instruct via OpenAI-compatible endpoint (vLLM/Ollama/proxy).",
        "aliases": ["llama-3.3-70b", "Llama-3.3-70B", "llama3.3:70b"],
    },
    {
        "provider": "openai_compatible",
        "model_id": "Qwen/Qwen3-32B",
        "label": "Qwen3 32B",
        "transport": "openai_compatible",
        "description": "Qwen3 32B via OpenAI-compatible endpoint (vLLM/Ollama/proxy).",
        "aliases": ["qwen3-32b", "Qwen3-32B", "qwen3:32b"],
    },
]


def build_generation_model_catalog() -> dict[str, Any]:
    return {
        "default": "gpt-5.5",
        "models": list(GENERATION_MODEL_CATALOG),
        "count": len(GENERATION_MODEL_CATALOG),
    }


def resolve_generation_model(model_id: str | None) -> dict[str, Any]:
    raw = str(model_id or "gpt-5.5").strip()
    lowered = raw.lower()
    for entry in GENERATION_MODEL_CATALOG:
        if entry["model_id"].lower() == lowered or entry["label"].lower() == lowered:
            return dict(entry)
        for alias in entry.get("aliases") or []:
            if str(alias).lower() == lowered:
                return dict(entry)
    # Unknown IDs remain configurable passthrough for OpenAI-compatible deployments.
    return {
        "provider": "openai_compatible",
        "model_id": raw,
        "label": raw,
        "transport": "openai_compatible",
        "description": "Custom generation model.",
    }
