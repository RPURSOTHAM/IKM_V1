"""Canonical semantic document and chunk representations shared across tiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SEMANTIC_DOCUMENT_VERSION = "1.0"
SEMANTIC_CHUNK_VERSION = "1.0"


@dataclass
class Coordinates:
    page: int | None = None
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None


@dataclass
class DocumentMetadata:
    document_id: str = ""
    document_name: str = ""
    original_file_name: str = ""
    repository_id: str | None = None
    tenant_id: str | None = None
    mime_type: str | None = None
    page_count: int = 0
    file_size_bytes: int | None = None
    content_hash: str | None = None


@dataclass
class ParserMetadata:
    parser_id: str = ""
    parser_version: str = ""
    backend: str = ""
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class SecurityMetadata:
    sensitivity: str = "unknown"
    decision: str = "unscanned"
    confidence: float = 0.0
    policy_version: str = ""
    detected_categories: list[str] = field(default_factory=list)


@dataclass
class EmbeddingMetadata:
    provider: str = "local"
    model_id: str = ""
    model_revision: str = ""
    dimensions: int | None = None
    normalized: bool = True

    @property
    def version(self) -> str:
        return f"{self.model_id}@{self.model_revision}" if self.model_revision else self.model_id


@dataclass
class RetrievalMetadata:
    retrieval_mode: str = ""
    original_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = None
    query_id: str | None = None
    pipeline_stages: list[str] = field(default_factory=list)


@dataclass
class VersionMetadata:
    semantic_schema_version: str = SEMANTIC_DOCUMENT_VERSION
    content_version: str = "1"
    ingestion_generation: int = 1
    producer_version: str = ""


@dataclass
class ProcessingHistory:
    stage: str
    status: str
    execution_time_ms: int = 0
    implementation_id: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class SemanticElement:
    element_id: str
    text: str
    page: int = 1
    line_start: int | None = None
    line_end: int | None = None
    coordinates: Coordinates | None = None
    source: str = ""


@dataclass
class SemanticParagraph(SemanticElement):
    heading: str = ""
    heading_path: list[str] = field(default_factory=list)


@dataclass
class SemanticTable(SemanticElement):
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""


@dataclass
class SemanticImage(SemanticElement):
    caption: str = ""
    media_reference: str = ""
    ocr_text: str = ""


@dataclass
class SemanticSection:
    section_id: str
    title: str
    level: int = 1
    path: list[str] = field(default_factory=list)
    paragraphs: list[SemanticParagraph] = field(default_factory=list)
    tables: list[SemanticTable] = field(default_factory=list)
    images: list[SemanticImage] = field(default_factory=list)


@dataclass
class SemanticDocument:
    document: DocumentMetadata
    sections: list[SemanticSection] = field(default_factory=list)
    paragraphs: list[SemanticParagraph] = field(default_factory=list)
    tables: list[SemanticTable] = field(default_factory=list)
    images: list[SemanticImage] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    captions: list[SemanticElement] = field(default_factory=list)
    footnotes: list[SemanticElement] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    language: str = "und"
    source: str = ""
    parser_metadata: ParserMetadata = field(default_factory=ParserMetadata)
    security_metadata: SecurityMetadata = field(default_factory=SecurityMetadata)
    embedding_metadata: EmbeddingMetadata | None = None
    retrieval_metadata: RetrievalMetadata | None = None
    version: VersionMetadata = field(default_factory=VersionMetadata)
    processing_history: list[ProcessingHistory] = field(default_factory=list)

    def full_text(self) -> str:
        elements: list[SemanticElement] = [*self.paragraphs, *self.tables]
        return "\n".join(element.text for element in elements if element.text.strip())


@dataclass
class SemanticChunk:
    """Canonical mutable chunk contract; legacy ``Chunk`` imports alias this class."""

    id: str
    doc_name: str
    page: int | None
    section_name: str
    text: str
    line_start: int | None = None
    line_end: int | None = None
    section_path: str = ""
    embedding: Any | None = None
    cluster_id: int | None = None
    match_pct: float | None = None
    is_duplicate: bool = False
    raw_text: str = ""
    rewritten_text: str = ""
    normalized_text: str = ""
    category: str = ""
    category_confidence: float = 0.0
    category_keywords: list[str] = field(default_factory=list)
    category_summary: str = ""
    content_types: list[str] = field(default_factory=list)
    table_count: int = 0
    image_count: int = 0
    extraction_metadata: dict[str, Any] = field(default_factory=dict)
    end_page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    document_id: str = ""
    document_name: str = ""
    strategy_name: str = ""
    chunk_type: str = "paragraph"
    parent_section: str = ""
    table_name: str = ""
    title: str = ""
    source_blocks: list[dict[str, Any]] = field(default_factory=list)
    citation_anchor: dict[str, Any] | None = None
    heading: str = ""
    topic: str = "General"
    entities: list[str] = field(default_factory=list)
    sensitivity: str = "unknown"
    embedding_version: str = ""
    retrieval_metadata: RetrievalMetadata = field(default_factory=RetrievalMetadata)
    coordinates: list[Coordinates] = field(default_factory=list)
    source: str = ""
    version: VersionMetadata = field(default_factory=VersionMetadata)
    processing_history: list[ProcessingHistory] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.id

    def __post_init__(self) -> None:
        if not self.raw_text:
            self.raw_text = self.text
        if not self.heading:
            self.heading = self.section_name
        if self.page_start is None and self.page is not None:
            self.page_start = self.page
        if self.page_end is None:
            self.page_end = self.end_page if self.end_page is not None else self.page
        if self.end_page is None:
            self.end_page = self.page_end
        if not self.document_name:
            self.document_name = self.doc_name
