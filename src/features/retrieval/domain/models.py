"""Shared result and state contracts for retrieval pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    stage: str
    status: str
    confidence: float = 1.0
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped", "degraded"}


@dataclass
class RetrievalCandidate:
    chunk_id: str
    text: str
    doc_name: str = ""
    section_name: str = ""
    page: int | None = None
    score: float = 0.0
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_signals: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> str:
        return self.chunk_id or f"{self.doc_name}:{self.page}:{hash(self.text)}"


@dataclass
class PipelineState:
    original_query: str
    rewritten_query: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    intent: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    dense_candidates: list[RetrievalCandidate] = field(default_factory=list)
    sparse_candidates: list[RetrievalCandidate] = field(default_factory=list)
    fused_candidates: list[RetrievalCandidate] = field(default_factory=list)
    reranked_candidates: list[RetrievalCandidate] = field(default_factory=list)
    final_candidates: list[RetrievalCandidate] = field(default_factory=list)
    compressed_context: str = ""
    prompt: str = ""
    blocked: bool = False
    block_reason: str | None = None
    stages: list[StageResult] = field(default_factory=list)

    def record(self, result: StageResult) -> None:
        self.stages.append(result)

    @property
    def pipeline_stages_executed(self) -> list[str]:
        return [stage.stage for stage in self.stages if stage.status != "skipped"]

    @property
    def pipeline_trace(self) -> dict[str, float]:
        return {stage.stage: round(stage.execution_time_ms, 2) for stage in self.stages}
