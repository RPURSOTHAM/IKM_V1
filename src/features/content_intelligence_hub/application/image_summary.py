"""CIH image summary service — Ollama VLM over uploaded or stored images."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from src.features.content_intelligence_hub.application.ollama_vlm import summarize_image_bytes, vlm_enabled

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def is_image_filename(name: str) -> bool:
    return Path(name or "").suffix.lower() in _IMAGE_SUFFIXES


async def summarize_upload(
    file: UploadFile,
    *,
    prompt: str | None = None,
) -> dict[str, Any]:
    if not vlm_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Image summary requires OPEN_WEIGHT_VLM_ENABLED=true and "
                "IMAGE_DESCRIPTION_ENABLED=true (Ollama gemma3 / OPEN_WEIGHT_VLM_MODEL)."
            ),
        )
    filename = file.filename or "image"
    if not is_image_filename(filename) and not str(file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file ({', '.join(sorted(_IMAGE_SUFFIXES))}). Got: {filename}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        result = summarize_image_bytes(
            data,
            filename=filename,
            prompt=prompt,
            mime_type=file.content_type or mimetypes.guess_type(filename)[0],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


def summarize_document_image(document_id: str, *, prompt: str | None = None) -> dict[str, Any]:
    from src.features.documents.application.document_service import DocumentReceiverService

    if not vlm_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "Image summary requires OPEN_WEIGHT_VLM_ENABLED=true and "
                "IMAGE_DESCRIPTION_ENABLED=true."
            ),
        )
    svc = DocumentReceiverService()
    path, filename, media_type = svc.resolve_original_file(document_id)
    if not is_image_filename(filename) and not str(media_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Document {document_id} is not an image ({filename}). "
                "Upload a PNG/JPG (or use POST /api/v1/cih/image-summary)."
            ),
        )
    data = path.read_bytes()
    try:
        result = summarize_image_bytes(
            data,
            filename=filename,
            prompt=prompt,
            mime_type=media_type or mimetypes.guess_type(filename)[0],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    result["document_id"] = document_id
    result["original_file_name"] = filename
    return result
