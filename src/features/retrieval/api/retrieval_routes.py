"""FastAPI routes for retrieval inspection and debugging."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from src.application.consumer_api.security import security_scheme

from src.features.repositories.domain.repository_exceptions import RepositoryError
from src.features.repositories.api.repository_routes import _repo_error
from src.features.retrieval.strategies.bm25.schemas import Bm25SearchRequest, Bm25SearchResponse
from src.features.retrieval.strategies.bm25.bm25_search_service import Bm25SearchService, get_bm25_search_service
from src.features.retrieval.schemas.retrieval_schemas import DocumentChunksResponse
from src.features.retrieval.application.retrieval_service import RetrievalService, get_retrieval_service

router = APIRouter(tags=["Retrieval"])


def _retrieval_svc() -> RetrievalService:
    return get_retrieval_service()


def _bm25_svc() -> Bm25SearchService:
    return get_bm25_search_service()


@router.get(
    "/repositories/{repository_id}/documents/{document_id}/chunks",
    summary="List all chunks for a repository document",
    operation_id="getRepositoryDocumentChunks",
    response_model=DocumentChunksResponse,
    dependencies=[Depends(security_scheme)],
    responses={
        200: {
            "description": "All indexed chunks for the document in ingestion order.",
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "030d8045-6b2a-4f1e-9c3d-8a7b6c5d4e3f",
                        "repository_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "chunk_count": 241,
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
                            }
                        ],
                    }
                }
            },
        },
        404: {"description": "Repository not found, document not found, or document not in repository."},
    },
)
def get_repository_document_chunks(
    repository_id: str,
    document_id: str,
    service: RetrievalService = Depends(_retrieval_svc),
) -> dict[str, Any]:
    """
    Return every Weaviate chunk stored for a document.

    Intended for debugging, validation, and UI inspection. Does not run vector search,
    similarity scoring, or reranking. Embeddings are not loaded.
    """
    try:
        return service.get_document_chunks(repository_id, document_id)
    except RepositoryError as exc:
        raise _repo_error(exc) from exc


@router.post(
    "/repositories/{repository_id}/search/bm25",
    summary="BM25 keyword search within a repository",
    operation_id="searchRepositoryBm25",
    response_model=Bm25SearchResponse,
    dependencies=[Depends(security_scheme)],
    responses={
        200: {
            "description": "BM25-ranked chunks for the query.",
            "content": {
                "application/json": {
                    "example": {
                        "query": "batch release",
                        "repository_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "result_count": 1,
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
                            "chunks_examined": 1,
                            "chunks_returned": 1,
                        },
                    }
                }
            },
        },
        404: {"description": "Repository or search index not found."},
    },
)
def search_repository_bm25(
    repository_id: str,
    body: Bm25SearchRequest,
    service: Bm25SearchService = Depends(_bm25_svc),
) -> dict[str, Any]:
    """
    Run pure BM25 lexical search against the repository's Weaviate inverted index.

    Does not compute embeddings, hybrid fusion, or reranking. For semantic/hybrid search
    use ``POST /api/v1/retrieve`` instead.
    """
    return service.search_bm25(repository_id, body.query, top_k=body.top_k)
