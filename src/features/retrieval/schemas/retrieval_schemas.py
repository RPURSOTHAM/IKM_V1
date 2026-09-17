from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class RetrieveRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=2048)
    query: str | None = Field(default=None, min_length=1, max_length=2048)
    query_text: str | None = Field(default=None, min_length=1, max_length=2048)
    domain_name: str = Field(default="default", min_length=1, max_length=128)
    complexity_level: Literal["low", "high", "complex"] = "complex"
    search_mode: Literal["hybrid", "vector", "keyword"] | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    num_chunks: int | None = Field(default=None, ge=1, le=10)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    document_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        description="When set, restrict retrieval to chunks from this document (merged into filters).",
    )
    folder_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        description=(
            "Optional automatically created logical folder ID from "
            "GET /api/v1/repositories/{repository_id}/folders. "
            "When set, only chunks whose indexed logical_folder_id matches this folder "
            "participate in retrieval. Omit to search the entire repository. "
            "The folder must belong to repository_id."
        ),
        json_schema_extra={"format": "uuid"},
    )
    folder_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of automatically created logical folder IDs from "
            "GET /api/v1/repositories/{repository_id}/folders. Combined with folder_id "
            "when both are set (union / OR). Each ID must belong to repository_id. "
            "Omit, together with folder_id, to search the entire repository."
        ),
        json_schema_extra={"items": {"type": "string", "minLength": 36, "maxLength": 36, "format": "uuid"}},
    )
    repository_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        description="Repository that scopes retrieval. Required when folder_id or folder_ids is set.",
    )
    repository_ids: list[str] | None = Field(
        default=None,
        description="Search across multiple repositories. Overrides repository_id when set.",
    )
    search_all_repositories: bool = Field(
        default=False,
        description="When true and no repository_id/repository_ids, search all active repositories.",
    )
    model_name: str | None = Field(
        default=None,
        max_length=512,
        description="Optional query embedding model override (Hub id or local path).",
    )
    model_dir: str | None = Field(
        default=None,
        max_length=512,
        description="Optional local model directory under src/models for query embeddings.",
    )
    hybrid_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override alpha for hybrid retrieval fusion. 0=BM25 only, 1=vector only, 0.75=default.",
    )
    alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Alias for hybrid_alpha (API compatibility).",
    )
    use_graph: bool | None = None
    use_rerank: bool | None = None
    expand_query: bool | None = Field(
        default=None,
        description=(
            "When false, query expansion is skipped and expanded_queries is empty. "
            "When true, a configured rewrite provider may return additional queries. "
            "If no rewrite provider/model is available, the original query is used and "
            "expanded_queries remains empty (documented fallback; no synthetic expansions)."
        ),
    )
    use_production_pipeline: bool | None = Field(
        default=None,
        description="When true, run Prompt Guard→RRF→Top-K rerank pipeline. Defaults to platform config.",
    )
    candidate_k: int | None = Field(default=None, ge=1, le=200)
    rerank_top_k: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int | None = Field(default=None, ge=1, le=200)
    explain: bool = False
    debug_retrieval: bool = Field(
        default=False,
        description="When true, include full retrieval metrics, pipeline trace, and candidate statistics.",
    )

    @field_validator("folder_ids")
    @classmethod
    def validate_folder_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            if len(text) != 36:
                raise ValueError(
                    "folder_ids entries must be 36-character logical folder IDs from "
                    "GET /api/v1/repositories/{repository_id}/folders."
                )
            cleaned.append(text)
        return cleaned or None

    @model_validator(mode="after")
    def validate_query_text(self):
        if not self.text and not self.query and not self.query_text:
            raise ValueError("One of 'text', 'query', or 'query_text' is required.")
        if self.repository_id and self.repository_ids:
            raise ValueError("Use either repository_id or repository_ids, not both.")
        # Top-level document_id is authoritative when provided.
        if self.document_id:
            merged = dict(self.filters or {})
            merged["document_id"] = self.document_id
            self.filters = merged
        if self.alpha is not None:
            if self.hybrid_alpha is not None and self.hybrid_alpha != self.alpha:
                raise ValueError("alpha and hybrid_alpha conflict; provide only one value.")
            self.hybrid_alpha = self.alpha
        return self

    @property
    def resolved_query(self) -> str:
        return (self.text or self.query or self.query_text or "").strip()

    @property
    def resolved_top_k(self) -> int:
        return self.num_chunks if self.num_chunks is not None else self.top_k


class ChunkOut(BaseModel):
    chunk_id: str
    id: str
    text: str
    doc_name: str
    section_name: str
    page: int | None
    line_start: int | None = None
    line_end: int | None = None
    line_range: str | None = None
    citation_anchor: dict[str, Any] | None = None
    citation_text: str | None = None
    document_name: str | None = None
    title: str | None = None
    page_end: int | None = None
    score: float
    source: str
    graph_context: list[str]
    highlight_spans: list[tuple[int, int]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    grounding_score: float = 0.0
    quality_score: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    hop: int = 0
    retrieval_signals: dict[str, Any] = Field(default_factory=dict)
    repository_id: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    domain: str = "default"
    complexity_level: str = "complex"
    expanded_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Additional queries produced by a rewrite provider. Empty when expansion is "
            "disabled or when no rewrite provider is configured (fallback to original query)."
        ),
    )
    sub_queries: list[str]
    results: list[ChunkOut]
    total_candidates: int
    latency_ms: float
    pipeline_stages_executed: list[str] = Field(default_factory=list)
    pipeline_trace: dict[str, float] | None = None
    retrieval_summary: dict[str, Any] | None = None
    grounding_summary: dict[str, Any] | None = None
    quality_summary: dict[str, Any] | None = None
    intent: dict[str, Any] | None = None
    compressed_context: str | None = None
    prompt: str | None = None
    security: dict[str, Any] | None = None
    blocked: bool = False
    block_reason: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunks(self) -> list[ChunkOut]:
        """Alias for clients that expect a ``chunks`` key."""
        return self.results


class IndexedDocumentMetadata(BaseModel):
    indexed: bool = False
    chunk_count: int = 0
    embedded_chunk_count: int = 0
    category: str | None = None
    category_summary: str | None = None
    category_confidence: float | None = None
    is_duplicate: bool | None = None
    file_size_bytes: int | None = None


class DocumentCatalogItem(BaseModel):
    document_id: str
    document_name: str
    original_file_name: str
    document_type: str
    status: str
    repository_id: str | None = None
    collection_name: str
    tenant_id: str | None = None
    upload_timestamp: float | None = None
    queue_submission_timestamp: float | None = None
    processing_completion_timestamp: float | None = None
    job_id: str | None = None
    batch_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    indexed: IndexedDocumentMetadata = Field(default_factory=IndexedDocumentMetadata)
    relevance_score: float | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentCatalogItem]
    count: int
    total: int
    limit: int
    offset: int
    collection_name: str | None = None
    repository_id: str | None = None


class DocumentSearchRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=2048)
    query: str | None = Field(default=None, min_length=1, max_length=2048)
    query_text: str | None = Field(default=None, min_length=1, max_length=2048)
    search_mode: Literal["hybrid", "vector", "keyword"] | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: dict[str, Any] = Field(default_factory=dict)
    repository_id: str | None = Field(default=None, min_length=36, max_length=36)

    @model_validator(mode="after")
    def validate_query_text(self):
        if not self.text and not self.query and not self.query_text:
            raise ValueError("One of 'text', 'query', or 'query_text' is required.")
        return self

    @property
    def resolved_query(self) -> str:
        return (self.text or self.query or self.query_text or "").strip()


class DocumentSearchResponse(BaseModel):
    query: str
    search_mode: str
    documents: list[DocumentCatalogItem]
    count: int
    total_candidates: int
    latency_ms: float
    collection_name: str | None = None
    repository_id: str | None = None


class DocumentChunkItem(BaseModel):
    chunk_id: str
    document_id: str
    page: int | None = None
    section: str | None = None
    chunk_index: int
    text: str = ""
    token_count: int = 0
    embedding_exists: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunksResponse(BaseModel):
    document_id: str
    repository_id: str
    chunk_count: int
    chunks: list[DocumentChunkItem] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_id": "030d8045-6b2a-4f1e-9c3d-8a7b6c5d4e3f",
                    "repository_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "chunk_count": 2,
                    "chunks": [
                        {
                            "chunk_id": "c1111111-1111-4111-8111-111111111111",
                            "document_id": "030d8045-6b2a-4f1e-9c3d-8a7b6c5d4e3f",
                            "page": 1,
                            "section": "Introduction",
                            "chunk_index": 0,
                            "text": "This SOP defines the batch validation procedure.",
                            "token_count": 8,
                            "embedding_exists": True,
                            "metadata": {
                                "section_path": "1 Introduction",
                                "line_start": 1,
                                "line_end": 4,
                                "category": "procedure",
                            },
                        },
                        {
                            "chunk_id": "c2222222-2222-4222-8222-222222222222",
                            "document_id": "030d8045-6b2a-4f1e-9c3d-8a7b6c5d4e3f",
                            "page": 2,
                            "section": "Batch Release",
                            "chunk_index": 1,
                            "text": "Release criteria must be verified before shipment.",
                            "token_count": 8,
                            "embedding_exists": True,
                            "metadata": {
                                "section_path": "2 Batch Release",
                                "line_start": 12,
                                "line_end": 18,
                                "category": "procedure",
                            },
                        },
                    ],
                }
            ]
        }
    }
