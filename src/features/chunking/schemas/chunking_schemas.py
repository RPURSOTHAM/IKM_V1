"""Pydantic models for the chunking catalog and preview API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.features.repositories.schemas.repository_schemas import (
    ChunkingConfigFieldSpec,
    ChunkingStrategyCatalogResponse,
    ChunkingStrategyId,
    ChunkingStrategyOption,
)

__all__ = [
    "ChunkingConfigFieldSpec",
    "ChunkingPreviewChunk",
    "ChunkingPreviewRequest",
    "ChunkingPreviewResponse",
    "ChunkingStrategyCatalogResponse",
    "ChunkingStrategyId",
    "ChunkingStrategyOption",
]


class ChunkingPreviewRequest(BaseModel):
    """Run a chunking strategy against arbitrary text without indexing."""

    strategy: ChunkingStrategyId | str = Field(
        ...,
        description="Canonical strategy id (or legacy alias) from GET /api/v1/chunking/strategies.",
    )
    text: str = Field(..., min_length=1, description="Plain text to chunk for a live preview.")
    chunk_size: int | None = Field(
        default=None,
        ge=1,
        description="Word budget / window size. Defaults from the strategy catalog when omitted.",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Overlap or step size depending on strategy "
            "(sentence overlap, word overlap, or sliding-window step)."
        ),
    )
    min_content_words: int | None = Field(
        default=None,
        ge=1,
        description="Minimum words required to keep a preview chunk.",
    )
    chunking_config: dict[str, Any] | None = Field(
        default=None,
        description="Strategy-specific knobs (max_sentences_per_chunk, similarity_threshold, …).",
    )
    document_name: str | None = Field(
        default="preview.txt",
        max_length=256,
        description="Display name attached to preview chunks.",
    )
    citation_retainment: bool = Field(
        default=True,
        description="When true, attach citation metadata (pages/lines/source_blocks) to preview chunks.",
    )


class ChunkingPreviewChunk(BaseModel):
    index: int
    chunk_id: str
    text: str
    word_count: int
    page: int | None = None
    page_end: int | None = None
    section_name: str = ""
    section_path: str = ""
    parent_section: str = ""
    title: str = ""
    chunk_type: str = "paragraph"
    table_name: str = ""
    line_start: int | None = None
    line_end: int | None = None
    strategy_name: str = ""
    source_blocks: list[dict[str, Any]] = Field(default_factory=list)


class ChunkingPreviewResponse(BaseModel):
    strategy_id: ChunkingStrategyId | str
    document_name: str
    chunk_count: int
    parameters: dict[str, Any]
    chunks: list[ChunkingPreviewChunk]
