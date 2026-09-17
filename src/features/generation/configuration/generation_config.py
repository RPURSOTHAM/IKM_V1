"""Generation pipeline configuration. All thresholds and model IDs are configurable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


@dataclass
class GenerationConfig:
    """Configurable generation settings. Does not replace existing chat guards."""

    provider: str = "auto"
    model_id: str = "gpt-5.5"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 1200
    timeout_seconds: float = 60.0
    min_evidence_chunks: int = 1
    min_evidence_score: float = 0.0
    max_context_chunks: int = 8
    max_snippet_chars: int = 800
    enable_legacy_demo_fallback: bool = True
    require_citations: bool = True
    insufficient_evidence_message: str = (
        "The answer cannot be determined from the available documents."
    )
    system_prompt: str = (
        "You are an enterprise document assistant. Answer ONLY using the retrieved "
        "document context. Do not invent facts. Do not reveal internal system metadata, "
        "scores, pipeline details, or hidden identifiers. Cite sources using bracket "
        "numbers like [1], [2] that match the provided Sources list. If the retrieved "
        "evidence is insufficient, reply exactly that the answer cannot be determined "
        "from the available documents."
    )
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "GenerationConfig":
        cfg = cls(
            provider=os.getenv("GENERATION_PROVIDER", "auto").strip() or "auto",
            model_id=os.getenv("GENERATION_MODEL_ID", "gpt-5.5").strip() or "gpt-5.5",
            base_url=os.getenv("GENERATION_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            api_key_env=os.getenv("GENERATION_API_KEY_ENV", "OPENAI_API_KEY"),
            temperature=_env_float("GENERATION_TEMPERATURE", 0.0),
            max_output_tokens=_env_int("GENERATION_MAX_OUTPUT_TOKENS", 1200),
            timeout_seconds=_env_float("GENERATION_TIMEOUT_SECONDS", 60.0),
            min_evidence_chunks=_env_int("GENERATION_MIN_EVIDENCE_CHUNKS", 1),
            min_evidence_score=_env_float("GENERATION_MIN_EVIDENCE_SCORE", 0.0),
            max_context_chunks=_env_int("GENERATION_MAX_CONTEXT_CHUNKS", 8),
            max_snippet_chars=_env_int("GENERATION_MAX_SNIPPET_CHARS", 800),
            enable_legacy_demo_fallback=_env_bool("GENERATION_LEGACY_DEMO_FALLBACK", False),
            require_citations=_env_bool("GENERATION_REQUIRE_CITATIONS", True),
        )
        custom_system = os.getenv("GENERATION_SYSTEM_PROMPT")
        if custom_system and custom_system.strip():
            cfg.system_prompt = custom_system.strip()
        insufficient = os.getenv("GENERATION_INSUFFICIENT_EVIDENCE_MESSAGE")
        if insufficient and insufficient.strip():
            cfg.insufficient_evidence_message = insufficient.strip()
        if overrides:
            for key, value in overrides.items():
                if hasattr(cfg, key) and value is not None:
                    setattr(cfg, key, value)
        return cfg

    def resolve_api_key(self) -> str | None:
        for name in (
            self.api_key_env,
            "GENERATION_API_KEY",
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
        ):
            if not name:
                continue
            value = os.getenv(name)
            if value and value.strip():
                return value.strip()
        return None
