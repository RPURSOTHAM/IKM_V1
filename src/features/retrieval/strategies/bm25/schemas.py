"""Pydantic schemas for standalone BM25 search API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Bm25SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048, examples=["batch release"])
    top_k: int = Field(default=20, ge=1, le=100)


class Bm25SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str | None = None
    bm25_score: float
    page: int | None = None
    section: str | None = None
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Bm25SearchMetrics(BaseModel):
    bm25_search_ms: float = 0.0
    total_ms: float = 0.0
    chunks_examined: int = 0
    chunks_returned: int = 0


class Bm25SearchResponse(BaseModel):
    query: str
    repository_id: str
    result_count: int
    results: list[Bm25SearchResultItem] = Field(default_factory=list)
    metrics: Bm25SearchMetrics = Field(default_factory=Bm25SearchMetrics)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "batch release",
                    "repository_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "result_count": 2,
                    "results": [
                        {
                            "chunk_id": "c1111111-1111-4111-8111-111111111111",
                            "document_id": "030d8045-6b2a-4f1e-9c3d-8a7b6c5d4e3f",
                            "bm25_score": 18.9,
                            "page": 4,
                            "section": "Batch Release",
                            "text": "Release criteria must be verified before shipment.",
                            "metadata": {"category": "procedure"},
                        }
                    ],
                    "metrics": {
                        "bm25_search_ms": 12.4,
                        "total_ms": 18.7,
                        "chunks_examined": 2,
                        "chunks_returned": 2,
                    },
                }
            ]
        }
    }
