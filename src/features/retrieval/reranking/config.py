"""Configuration for cross-encoder reranking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


@dataclass
class RerankerConfig:
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    batch_size: int = 32
    top_k_before: int = 20
    top_k_after: int = 10
    score_threshold: float = 0.0

    @classmethod
    def from_env(cls, overrides: dict | None = None) -> "RerankerConfig":
        deployment_enabled = True
        try:
            from src.features.document_processing.shared_processor.deployment import ProcessorConfigurationProvider

            deployment_enabled = ProcessorConfigurationProvider.is_enabled("reranker")
        except Exception:
            deployment_enabled = True

        cfg = cls(
            enabled=deployment_enabled and _env_bool("RERANKER_ENABLED", True),
            model=os.getenv("RERANKER_MODEL", os.getenv("RETRIEVAL_RERANKER_MODEL", "BAAI/bge-reranker-base")),
            batch_size=_env_int("RERANKER_BATCH_SIZE", 32),
            top_k_before=_env_int("TOP_K_BEFORE", _env_int("RETRIEVAL_RERANK_TOP_K", 20)),
            top_k_after=_env_int("TOP_K_AFTER", _env_int("RETRIEVAL_FINAL_TOP_K", 10)),
            score_threshold=_env_float("RETRIEVAL_RERANK_SCORE_THRESHOLD", 0.0),
        )
        if overrides:
            for key, value in overrides.items():
                if value is not None and hasattr(cfg, key):
                    setattr(cfg, key, value)
        # Deployment gate always wins — disabled processors never load models.
        if not deployment_enabled:
            cfg.enabled = False
        return cfg


def resolve_rerank_enabled(
    *,
    use_rerank: bool | None = None,
    repo_settings: dict[str, Any] | None = None,
    config: RerankerConfig | None = None,
) -> bool:
    """Resolve whether retrieval should run the cross-encoder reranker.

    Precedence (highest → lowest after the platform master switch):
      1. Platform master switch: ``RerankerConfig.enabled``
         (``RERANKER_ENABLED`` ∧ deployment ``reranker`` processor). When off,
         reranking never runs and the model is not loaded.
      2. Request ``use_rerank`` when explicitly set
      3. Repository ``settings.reranking`` when explicitly set (bool)
      4. Platform enabled default (True when master switch is on)

    Repository default ``None`` means "unset" and falls through to the platform
    env. Explicit repository ``False`` opts out; explicit ``True`` opts in
    (still requires the platform master switch).
    """
    cfg = config or RerankerConfig.from_env()
    if not cfg.enabled:
        return False
    if use_rerank is not None:
        return bool(use_rerank)
    if repo_settings is not None and "reranking" in repo_settings:
        repo_val = repo_settings.get("reranking")
        if isinstance(repo_val, bool):
            return repo_val
    return True
