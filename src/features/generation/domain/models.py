"""Shared generation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Fields that must never appear in prompts or public citations.
INTERNAL_METADATA_KEYS = frozenset(
    {
        "score",
        "original_score",
        "rerank_score",
        "rrf_score",
        "fused_score",
        "distance",
        "query_evidence_terms",
        "query_evidence_count",
        "embedding_version",
        "sensitivity",
        "file_size_bytes",
        "category_confidence",
        "coordinates",
        "extraction_metadata",
        "semantic_metadata",
        "pipeline_stages",
        "retrieval_signals",
        "tenant_id",
        "match_pct",
        "cluster_id",
        "is_duplicate",
        "dense_score",
        "bm25_score",
        "dense_rank",
        "bm25_rank",
        "rerank_model",
    }
)


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class Citation:
    index: int
    document_name: str
    page: int | None = None
    section: str | None = None
    document_id: str | None = None
    snippet: str = ""
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    chunk_id: str | None = None
    section_path: str | None = None
    source_blocks: list[dict[str, Any]] | None = None
    citation_anchor: dict[str, Any] | None = None

    def to_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "document_name": self.document_name,
            "page": self.page,
            "section": self.section,
            "document_id": self.document_id,
            "snippet": self.snippet,
        }
        if self.page_end is not None:
            payload["page_end"] = self.page_end
        if self.line_start is not None:
            payload["line_start"] = self.line_start
        if self.line_end is not None:
            payload["line_end"] = self.line_end
        if self.chunk_id:
            payload["chunk_id"] = self.chunk_id
        if self.section_path:
            payload["section_path"] = self.section_path
        if self.source_blocks:
            payload["source_blocks"] = self.source_blocks
        if self.citation_anchor:
            payload["citation_anchor"] = self.citation_anchor
        return payload


@dataclass
class GenerationRequest:
    question: str
    conversation: list[ConversationTurn] = field(default_factory=list)
    context_chunks: list[dict[str, Any]] = field(default_factory=list)
    model_id: str | None = None
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    evidence_sufficient: bool = True
    model_id: str = ""
    provider: str = ""
    prompt_messages: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
