"""Interfaces for modular retrieval pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
from src.features.retrieval.domain.models import PipelineState, StageResult


class RetrievalStage(ABC):
    """Every retrieval stage exposes confidence, execution time, and metadata."""

    name: str = "stage"

    @abstractmethod
    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        raise NotImplementedError


class PromptGuardStage(RetrievalStage):
    name = "prompt_guard"


class QueryRewriterStage(RetrievalStage):
    name = "query_rewriter"


class IntentClassifierStage(RetrievalStage):
    name = "intent_classifier"


class MetadataFilterGeneratorStage(RetrievalStage):
    name = "metadata_filter_generator"


class HybridRetrieverStage(RetrievalStage):
    name = "hybrid_retrieval"


class RankFusionStage(RetrievalStage):
    name = "reciprocal_rank_fusion"


class RerankerStage(RetrievalStage):
    name = "cross_encoder_reranker"


class ContextCompressionStage(RetrievalStage):
    name = "context_compression"


class PromptBuilderStage(RetrievalStage):
    name = "prompt_builder"


class ModelRegistry(ABC):
    """Resolve replaceable models by role without hardcoding call sites."""

    @abstractmethod
    def resolve(self, role: str, preferred: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(self, role: str, preferred: str | None = None) -> Any:
        raise NotImplementedError
