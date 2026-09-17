"""Retrieval strategy interface and shared context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.features.retrieval.schemas.retrieval_schemas import ChunkOut
from src.features.retrieval.strategies.models import StrategyResult


@dataclass
class RetrievalStrategyContext:
    """Inputs shared by all retrieval strategies."""

    repository_id: str | None
    query: str
    top_k: int
    collection: Any
    query_vector: list[float] | None
    common: dict[str, Any]
    hybrid_alpha: float
    citation_lookup: dict[str, dict[str, str]]
    query_terms: list[str]
    min_score: float
    require_evidence: bool
    rrf_k: int = 60
    vector_limit: int = 40
    bm25_limit: int = 40
    use_rrf: bool = False
    fetch_candidates: Any = field(repr=False, default=None)


class RetrievalStrategy(Protocol):
    """Common retrieval strategy contract."""

    @property
    def name(self) -> str: ...

    def search(self, ctx: RetrievalStrategyContext) -> StrategyResult: ...


def merged_to_chunk_out(
    item: Any,
    *,
    final_score: float,
    signals: dict[str, Any],
    query_terms: list[str] | None = None,
    repository_id: str | None = None,
) -> ChunkOut:
    """Map a merged/fused candidate to the public ``ChunkOut`` contract."""
    from src.features.retrieval.scoring.quality_scorer import enrich_chunk_scores

    metadata = dict(getattr(item, "metadata", None) or {})
    document_id = getattr(item, "document_id", None) or metadata.get("document_id")
    if document_id and not metadata.get("document_id"):
        metadata["document_id"] = document_id
    # Prefer chunk/document metadata over the request-scoped fallback so
    # multi-repository results expose each chunk's actual repository_id.
    resolved_repository_id = (
        metadata.get("repository_id")
        or getattr(item, "repository_id", None)
        or repository_id
    )
    if resolved_repository_id and not metadata.get("repository_id"):
        metadata["repository_id"] = resolved_repository_id
    section = (
        getattr(item, "section", None)
        or getattr(item, "section_name", None)
        or metadata.get("section")
        or metadata.get("section_name")
        or ""
    )
    doc_name = (
        getattr(item, "doc_name", None)
        or str(metadata.get("doc_name") or metadata.get("source") or "")
    )
    page = getattr(item, "page", None)
    line_range = str(metadata.get("line_range") or "").strip() or None
    citation_anchor = metadata.get("citation_anchor") if isinstance(metadata.get("citation_anchor"), dict) else None
    text = str(getattr(item, "text", "") or "")
    highlights, grounding, quality, flags = enrich_chunk_scores(
        text=text,
        retrieval_score=float(final_score),
        metadata=metadata,
        query_terms=query_terms or [],
    )
    return ChunkOut(
        chunk_id=item.chunk_id,
        id=item.chunk_id,
        text=text,
        doc_name=str(doc_name),
        document_name=metadata.get("original_file_name") or metadata.get("document_name"),
        section_name=str(section),
        title=metadata.get("title"),
        page=page if isinstance(page, int) else (
            int(page) if isinstance(page, float) and page.is_integer() else None
        ),
        page_end=metadata.get("page_end") if isinstance(metadata.get("page_end"), int) else None,
        line_start=metadata.get("line_start") if isinstance(metadata.get("line_start"), int) else None,
        line_end=metadata.get("line_end") if isinstance(metadata.get("line_end"), int) else None,
        line_range=line_range,
        citation_anchor=citation_anchor,
        citation_text=metadata.get("citation_text"),
        score=round(float(final_score), 4),
        source=str(doc_name),
        graph_context=[],
        highlight_spans=highlights,
        metadata=metadata,
        grounding_score=grounding,
        quality_score=quality,
        quality_flags=flags,
        hop=0,
        retrieval_signals=signals,
        repository_id=str(resolved_repository_id) if resolved_repository_id else None,
    )
