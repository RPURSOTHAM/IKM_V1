"""Configurable thresholds and preferred model IDs for the retrieval pipeline."""

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
class RetrievalPipelineConfig:
    """All retrieval-stage thresholds are configuration-driven."""

    enable_prompt_guard: bool = True
    enable_query_rewrite: bool = True
    enable_intent_classifier: bool = True
    enable_metadata_filter_generator: bool = True
    enable_hybrid_rrf: bool = False
    enable_rerank: bool = True
    enable_context_compression: bool = True
    enable_prompt_builder: bool = True

    # Preferred models (do not replace production defaults unless configured).
    preferred_prompt_guard_model: str = "meta-llama/Llama-Guard-3-8B"
    preferred_query_rewriter_model: str = "Qwen/Qwen3-8B"
    preferred_embedding_model: str = "BAAI/bge-m3"
    preferred_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    preferred_compression_model: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

    # Candidate / ranking thresholds
    candidate_k: int = 40
    rerank_top_k: int = 20
    final_top_k: int = 10
    dense_limit: int = 40
    sparse_limit: int = 40
    rrf_k: int = 60
    min_score: float = 0.0
    rerank_score_threshold: float = 0.0
    hybrid_alpha: float = 0.75
    prompt_guard_block_on_high: bool = True
    compression_max_tokens: int = 1200
    compression_keep_sentences: int = 8
    intent_confidence_threshold: float = 0.3

    stage_timeout_seconds: float = 30.0
    never_block_on_optional_failure: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "RetrievalPipelineConfig":
        cfg = cls(
            enable_prompt_guard=_env_bool("RETRIEVAL_ENABLE_PROMPT_GUARD", True),
            enable_query_rewrite=_env_bool("RETRIEVAL_ENABLE_QUERY_REWRITE", True),
            enable_intent_classifier=_env_bool("RETRIEVAL_ENABLE_INTENT", True),
            enable_metadata_filter_generator=_env_bool("RETRIEVAL_ENABLE_METADATA_FILTERS", True),
            enable_hybrid_rrf=_env_bool("RETRIEVAL_ENABLE_HYBRID_RRF", False),
            enable_rerank=_env_bool("RETRIEVAL_ENABLE_RERANK", True),
            enable_context_compression=_env_bool("RETRIEVAL_ENABLE_COMPRESSION", True),
            enable_prompt_builder=_env_bool("RETRIEVAL_ENABLE_PROMPT_BUILDER", True),
            preferred_prompt_guard_model=os.getenv(
                "RETRIEVAL_PROMPT_GUARD_MODEL", "meta-llama/Llama-Guard-3-8B"
            ),
            preferred_query_rewriter_model=os.getenv(
                "RETRIEVAL_QUERY_REWRITER_MODEL", "Qwen/Qwen3-8B"
            ),
            preferred_embedding_model=os.getenv("RETRIEVAL_EMBEDDING_MODEL", "BAAI/bge-m3"),
            preferred_reranker_model=os.getenv(
                "RETRIEVAL_RERANKER_MODEL",
                os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            ),
            preferred_compression_model=os.getenv(
                "RETRIEVAL_COMPRESSION_MODEL",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            ),
            candidate_k=_env_int("RETRIEVAL_CANDIDATE_K", 40),
            rerank_top_k=_env_int("RETRIEVAL_RERANK_TOP_K", 20),
            final_top_k=_env_int("RETRIEVAL_FINAL_TOP_K", 10),
            dense_limit=_env_int("RETRIEVAL_DENSE_LIMIT", 40),
            sparse_limit=_env_int("RETRIEVAL_SPARSE_LIMIT", 40),
            rrf_k=_env_int("RETRIEVAL_RRF_K", 60),
            min_score=_env_float("RETRIEVAL_MIN_SCORE", 0.0),
            rerank_score_threshold=_env_float("RETRIEVAL_RERANK_SCORE_THRESHOLD", 0.0),
            hybrid_alpha=_env_float("HYBRID_ALPHA", 0.75),
            prompt_guard_block_on_high=_env_bool("RETRIEVAL_PROMPT_GUARD_BLOCK", True),
            compression_max_tokens=_env_int("RETRIEVAL_COMPRESSION_MAX_TOKENS", 1200),
            compression_keep_sentences=_env_int("RETRIEVAL_COMPRESSION_KEEP_SENTENCES", 8),
            intent_confidence_threshold=_env_float("RETRIEVAL_INTENT_CONFIDENCE", 0.3),
            stage_timeout_seconds=_env_float("RETRIEVAL_STAGE_TIMEOUT_SECONDS", 30.0),
            never_block_on_optional_failure=_env_bool("RETRIEVAL_NEVER_BLOCK_OPTIONAL", True),
        )
        if overrides:
            for key, value in overrides.items():
                if hasattr(cfg, key) and value is not None:
                    setattr(cfg, key, value)
        # Deployment-level processor gate — disabled reranker never runs / loads models.
        try:
            from src.features.document_processing.shared_processor.deployment import ProcessorConfigurationProvider

            if not ProcessorConfigurationProvider.is_enabled("reranker"):
                cfg.enable_rerank = False
        except Exception:
            pass
        return cfg

    def with_request(
        self,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        use_rerank: bool | None = None,
        expand_query: bool | None = None,
        hybrid_alpha: float | None = None,
    ) -> "RetrievalPipelineConfig":
        cfg = RetrievalPipelineConfig(**{**self.__dict__, "extra": dict(self.extra)})
        if top_k is not None:
            cfg.final_top_k = int(top_k)
            cfg.rerank_top_k = max(cfg.rerank_top_k, cfg.final_top_k)
            cfg.candidate_k = max(cfg.candidate_k, cfg.rerank_top_k)
            cfg.dense_limit = max(cfg.dense_limit, cfg.candidate_k)
            cfg.sparse_limit = max(cfg.sparse_limit, cfg.candidate_k)
        if min_score is not None:
            cfg.min_score = float(min_score)
        if use_rerank is not None:
            cfg.enable_rerank = bool(use_rerank)
        if expand_query is not None:
            cfg.enable_query_rewrite = bool(expand_query)
        if hybrid_alpha is not None:
            cfg.hybrid_alpha = float(hybrid_alpha)
        try:
            from src.features.document_processing.shared_processor.deployment import ProcessorConfigurationProvider

            if not ProcessorConfigurationProvider.is_enabled("reranker"):
                cfg.enable_rerank = False
        except Exception:
            pass
        return cfg


def resolve_candidate_pool_k(
    *,
    final_top_k: int,
    rerank_top_k: int | None = None,
    candidate_k: int | None = None,
    pipeline_cfg: RetrievalPipelineConfig | None = None,
) -> int:
    """Resolve how many candidates retrieval should gather before reranking."""
    cfg = pipeline_cfg or RetrievalPipelineConfig.from_env()
    final_k = max(1, int(final_top_k))
    rerank_k = max(final_k, int(rerank_top_k if rerank_top_k is not None else cfg.rerank_top_k))
    pool_k = max(rerank_k, int(candidate_k if candidate_k is not None else cfg.candidate_k))
    return pool_k
