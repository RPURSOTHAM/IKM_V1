"""Orchestrate Prompt Guard → Prompt Builder retrieval stages."""

from __future__ import annotations

from typing import Iterable

from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
from src.features.retrieval.domain.interfaces import RetrievalStage
from src.features.retrieval.domain.models import PipelineState, StageResult
from src.features.retrieval.application.pipeline.stages import (
    ExistingCrossEncoderReranker,
    ExistingPromptGuardAdapter,
    HeuristicQueryRewriter,
    KeywordIntentClassifier,
    LLMLinguaOrExtractiveCompressor,
    ReciprocalRankFusionStage,
    RuleMetadataFilterGenerator,
    TemplatePromptBuilder,
)


class RetrievalPipeline:
    def __init__(self, stages: Iterable[RetrievalStage]) -> None:
        self.stages = list(stages)

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> PipelineState:
        for stage in self.stages:
            result: StageResult = stage.run(state, config)
            state.record(result)
            if state.blocked:
                break
            if result.status == "error" and stage.name == "hybrid_retrieval":
                break
        return state


def build_default_pipeline(
    *,
    hybrid_retriever: RetrievalStage,
    reranker: ExistingCrossEncoderReranker,
) -> RetrievalPipeline:
    return RetrievalPipeline(
        [
            ExistingPromptGuardAdapter(),
            HeuristicQueryRewriter(),
            KeywordIntentClassifier(),
            RuleMetadataFilterGenerator(),
            hybrid_retriever,
            ReciprocalRankFusionStage(),
            reranker,
            LLMLinguaOrExtractiveCompressor(),
            TemplatePromptBuilder(),
        ]
    )


__all__ = [
    "RetrievalPipeline",
    "build_default_pipeline",
    "ExistingCrossEncoderReranker",
    "ExistingPromptGuardAdapter",
    "HeuristicQueryRewriter",
    "KeywordIntentClassifier",
    "LLMLinguaOrExtractiveCompressor",
    "ReciprocalRankFusionStage",
    "RuleMetadataFilterGenerator",
    "TemplatePromptBuilder",
]
