"""Serialize and deserialize render cache payloads for Redis."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from src.features.documents.infrastructure.content.convert_for_rendering import RenderConversionResult


def build_render_payload(
    *,
    document_id: str,
    document_name: str,
    result: RenderConversionResult,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "document_id": document_id,
        "document_name": document_name,
        "format": result.format,
        "media_type": result.media_type,
        "conversion_method": result.conversion_method,
        "metadata": result.metadata,
    }
    if result.format == "html":
        payload["content"] = result.content or ""
        return payload

    if result.content_path and result.content_path.is_file():
        payload["content_path"] = str(result.content_path)
        payload["content_bytes_b64"] = base64.b64encode(result.content_path.read_bytes()).decode("ascii")
        return payload

    raise RuntimeError(f"Render conversion for format '{result.format}' did not produce storable content.")


def decode_render_bytes(payload: dict[str, Any]) -> bytes | None:
    encoded = payload.get("content_bytes_b64")
    if isinstance(encoded, str) and encoded.strip():
        return base64.b64decode(encoded)
    content_path = payload.get("content_path")
    if content_path:
        path = Path(str(content_path))
        if path.is_file():
            return path.read_bytes()
    return None


def decode_render_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    return str(content or "")
