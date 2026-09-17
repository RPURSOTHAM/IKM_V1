"""Chunking strategy catalog and live preview API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, status

from src.application.consumer_api.security import security_scheme
from src.features.chunking.application.chunking_preview_service import preview_chunking
from src.features.chunking.schemas.chunking_schemas import (
    ChunkingPreviewRequest,
    ChunkingPreviewResponse,
    ChunkingStrategyCatalogResponse,
    ChunkingStrategyOption,
)
from src.features.repositories.configuration.chunking_strategy_catalog import (
    build_chunking_strategy_catalog,
    get_chunking_strategy,
)

router = APIRouter(
    prefix="/chunking",
    tags=["Chunking"],
    dependencies=[Depends(security_scheme)],
)


@router.get(
    "/strategies",
    summary="List chunking strategies",
    description="Catalog of all supported chunking strategies and their default parameters.",
    operation_id="listChunkingStrategyCatalog",
    response_model=ChunkingStrategyCatalogResponse,
    responses={200: {"description": "Chunking strategy catalog"}},
)
def list_chunking_strategies() -> dict[str, Any]:
    return build_chunking_strategy_catalog()


@router.get(
    "/strategies/{strategy_id}",
    summary="Get chunking strategy details",
    description="Return label, description, config fields, and example settings for one strategy.",
    operation_id="getChunkingStrategy",
    response_model=ChunkingStrategyOption,
    responses={
        200: {"description": "Strategy details"},
        404: {"description": "Unknown strategy id"},
    },
)
def get_chunking_strategy_details(
    strategy_id: str = Path(..., min_length=1, description="Canonical strategy id or legacy alias"),
) -> dict[str, Any]:
    try:
        return get_chunking_strategy(strategy_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "chunking_strategy_not_found", "message": str(exc)},
        ) from exc


@router.post(
    "/preview",
    summary="Preview chunking on custom text",
    description=(
        "Test a chunking strategy on arbitrary text and return live preview chunks "
        "without uploading or indexing a document."
    ),
    operation_id="previewChunking",
    response_model=ChunkingPreviewResponse,
    responses={
        200: {"description": "Preview chunks"},
        400: {"description": "Invalid strategy or empty text"},
        404: {"description": "Unknown strategy id"},
    },
)
def preview_chunking_endpoint(body: ChunkingPreviewRequest) -> dict[str, Any]:
    try:
        return preview_chunking(
            strategy=str(body.strategy),
            text=body.text,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
            min_content_words=body.min_content_words,
            chunking_config=body.chunking_config,
            document_name=body.document_name or "preview.txt",
            citation_retainment=body.citation_retainment,
        )
    except ValueError as exc:
        message = str(exc)
        code = (
            "chunking_strategy_not_found"
            if "Unsupported chunking strategy" in message
            else "chunking_preview_invalid"
        )
        status_code = (
            status.HTTP_404_NOT_FOUND
            if code == "chunking_strategy_not_found"
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(
            status_code=status_code,
            detail={"code": code, "message": message},
        ) from exc
