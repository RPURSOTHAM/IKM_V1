"""Quality and candidate analysis helpers for retrieval metrics."""

from __future__ import annotations

import statistics
from typing import Any

from src.features.retrieval.metrics.models import CandidateAnalysis, QualityMetrics
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut


def _signal_values(chunks: list[ChunkOut], key: str) -> list[float]:
    values: list[float] = []
    for chunk in chunks:
        raw = (chunk.retrieval_signals or {}).get(key)
        if raw is not None:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
    return values


def _estimate_tokens(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))


def _document_key(chunk: ChunkOut) -> str:
    metadata = chunk.metadata or {}
    doc_id = metadata.get("document_id")
    if doc_id:
        return str(doc_id)
    if chunk.doc_name:
        return str(chunk.doc_name)
    return str(chunk.chunk_id)


def compute_quality_metrics(chunks: list[ChunkOut]) -> QualityMetrics:
    vector_scores = _signal_values(chunks, "vector_score")
    bm25_scores = _signal_values(chunks, "bm25_score")
    rrf_scores = _signal_values(chunks, "rrf_score")
    rerank_scores = _signal_values(chunks, "rerank_score")

    lengths = [len(chunk.text or "") for chunk in chunks]
    tokens = [_estimate_tokens(chunk.text or "") for chunk in chunks]
    doc_counts: dict[str, int] = {}
    for chunk in chunks:
        key = _document_key(chunk)
        doc_counts[key] = doc_counts.get(key, 0) + 1

    avg_doc_hits = (sum(doc_counts.values()) / len(doc_counts)) if doc_counts else 0.0

    return QualityMetrics(
        average_vector_score=(sum(vector_scores) / len(vector_scores)) if vector_scores else None,
        average_bm25_score=(sum(bm25_scores) / len(bm25_scores)) if bm25_scores else None,
        average_rrf_score=(sum(rrf_scores) / len(rrf_scores)) if rrf_scores else None,
        average_rerank_score=(sum(rerank_scores) / len(rerank_scores)) if rerank_scores else None,
        highest_rerank_score=max(rerank_scores) if rerank_scores else None,
        lowest_rerank_score=min(rerank_scores) if rerank_scores else None,
        median_rerank_score=statistics.median(rerank_scores) if rerank_scores else None,
        average_chunk_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        average_chunk_tokens=(sum(tokens) / len(tokens)) if tokens else 0.0,
        average_document_hits=avg_doc_hits,
    )


def compute_candidate_analysis(
    *,
    vector_candidates: int,
    bm25_candidates: int,
    merged_candidates: int,
    chunks: list[ChunkOut],
) -> CandidateAnalysis:
    duplicate_chunks_removed = max(0, vector_candidates + bm25_candidates - merged_candidates)
    shared_hits = duplicate_chunks_removed
    vector_only_hits = max(0, vector_candidates - shared_hits)
    bm25_only_hits = max(0, bm25_candidates - shared_hits)

    returned_vector_only = 0
    returned_bm25_only = 0
    returned_hybrid = 0
    documents: set[str] = set()
    sections: set[str] = set()
    pages: set[Any] = set()

    for chunk in chunks:
        signals = chunk.retrieval_signals or {}
        has_vector = signals.get("vector_score") is not None
        has_bm25 = signals.get("bm25_score") is not None
        if has_vector and has_bm25:
            returned_hybrid += 1
        elif has_vector:
            returned_vector_only += 1
        elif has_bm25:
            returned_bm25_only += 1

        documents.add(_document_key(chunk))
        section = chunk.section_name or (chunk.metadata or {}).get("section")
        if section:
            sections.add(str(section))
        if isinstance(chunk.page, int):
            pages.add((_document_key(chunk), chunk.page))

    denominator = max(vector_candidates, bm25_candidates, 1)
    overlap_pct = (shared_hits / denominator) * 100.0

    return CandidateAnalysis(
        duplicate_chunks_removed=duplicate_chunks_removed,
        unique_documents=len(documents),
        unique_sections=len(sections),
        vector_only_hits=vector_only_hits,
        bm25_only_hits=bm25_only_hits,
        hybrid_hits=shared_hits,
        shared_hits=shared_hits,
        candidate_overlap_percentage=overlap_pct,
    )


def count_documents_and_pages(chunks: list[ChunkOut]) -> tuple[int, int]:
    documents: set[str] = set()
    pages: set[tuple[str, int]] = set()
    for chunk in chunks:
        doc = _document_key(chunk)
        documents.add(doc)
        if isinstance(chunk.page, int):
            pages.add((doc, chunk.page))
    return len(documents), len(pages)
