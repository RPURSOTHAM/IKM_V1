"""Read-only document preview and render helpers for the consumer API."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.shared.errors import (
    COMPONENT_DOCUMENT_PREVIEW,
    raise_client_error,
    raise_service_error,
)

from src.features.documents.infrastructure.content.convert_for_rendering import convert_document_for_rendering
from src.features.documents.infrastructure.content.load import load_document_blocks
from src.features.documents.infrastructure.content.render_cache import get_render_payload, store_render_payload
from src.features.documents.infrastructure.content.render_html import wrap_render_document_html
from src.features.documents.infrastructure.content.render_payload import build_render_payload, decode_render_bytes, decode_render_text

_logger = logging.getLogger(__name__)


def _read_only_headers(*, media_type: str) -> dict[str, str]:
    return {
        "Content-Disposition": "inline",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, max-age=300",
        "X-DMS-Preview-Mode": "read-only",
        "Content-Type": media_type,
    }


def _convert_for_preview(
    document_path: Path,
    *,
    document_id: str,
    document_name: str,
) -> dict[str, Any]:
    try:
        blocks, doc_meta = load_document_blocks(document_path)
    except ValueError as exc:
        raise_client_error(
            COMPONENT_DOCUMENT_PREVIEW,
            code="unsupported_media_type",
            http_status=415,
            user_message="This document format is not supported for preview.",
            reason=f"Unsupported preview format: {exc}",
            cause=exc,
        )
    except RuntimeError as exc:
        raise_service_error(
            COMPONENT_DOCUMENT_PREVIEW,
            code="preview_unavailable",
            http_status=503,
            user_message="Document preview is temporarily unavailable. Please try again shortly.",
            reason=f"Preview dependencies unavailable: {exc}",
            cause=exc,
        )
    except Exception as exc:
        raise_service_error(
            COMPONENT_DOCUMENT_PREVIEW,
            code="preview_conversion_failed",
            http_status=500,
            user_message="Document preview could not be generated. Please try again later.",
            reason=f"Preview conversion failed for document_id={document_id}",
            cause=exc,
        )

    if not blocks and document_path.suffix.lower() != ".pdf":
        raise_client_error(
            COMPONENT_DOCUMENT_PREVIEW,
            code="empty_document",
            http_status=422,
            user_message="This document does not contain readable content for preview.",
            reason=f"No readable blocks for document_id={document_id}",
        )

    result = convert_document_for_rendering(
        document_path,
        document_id=document_id,
        title=document_name,
        blocks=blocks,
        doc_meta=doc_meta,
    )
    payload = build_render_payload(
        document_id=document_id,
        document_name=document_name,
        result=result,
    )
    try:
        ttl = int(os.getenv("RENDER_CACHE_TTL_SECONDS") or "86400")
        store_render_payload(document_id=document_id, payload=payload, ttl_seconds=ttl)
    except Exception:
        _logger.debug("Render cache write skipped for document_id=%s", document_id, exc_info=True)
    return payload


def _payload_to_preview_dict(
    payload: dict[str, Any],
    *,
    document_id: str,
    document_name: str,
    filename: str,
    source: str,
    document_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    render_format = str(payload.get("format") or "html").lower()
    media_type = str(payload.get("media_type") or "text/html; charset=utf-8")
    merged_info = dict(document_info or {})
    merged_info.update({k: v for k, v in (payload.get("metadata") or {}).items() if v is not None})

    if render_format == "pdf":
        return {
            "document_id": document_id,
            "document_name": document_name,
            "original_file_name": filename,
            "format": "pdf",
            "content": None,
            "content_bytes_b64": payload.get("content_bytes_b64"),
            "content_path": payload.get("content_path"),
            "media_type": media_type,
            "read_only": True,
            "source": source,
            "metadata": merged_info,
            "document_info": merged_info,
            "conversion_method": payload.get("conversion_method"),
        }

    content = decode_render_text(payload)
    if content and not content.lstrip().lower().startswith("<!doctype"):
        content = wrap_render_document_html(content, title=document_name)
    return {
        "document_id": document_id,
        "document_name": document_name,
        "original_file_name": filename,
        "format": "html",
        "content": content,
        "media_type": media_type,
        "read_only": True,
        "source": source,
        "metadata": merged_info,
        "document_info": merged_info,
        "conversion_method": payload.get("conversion_method"),
    }


def resolve_document_preview(
    record: dict[str, Any],
    *,
    media_type: str,
    path: Path,
    filename: str,
    document_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only preview payload, converting to PDF/HTML when needed."""
    document_id = str(record["document_id"])
    document_name = str(record.get("document_name") or path.name)

    cached: dict[str, Any] | None = None
    try:
        cached = get_render_payload(document_id=document_id)
    except Exception:
        cached = None

    if cached and (cached.get("content") or cached.get("content_bytes_b64") or cached.get("content_path")):
        cached_format = str(cached.get("format") or "").lower()
        if path.suffix.lower() == ".docx" and cached_format == "html":
            _logger.info(
                "Ignoring stale HTML render cache for DOCX document_id=%s; reconverting to PDF.",
                document_id,
            )
            cached = None
        else:
            return _payload_to_preview_dict(
                cached,
                document_id=document_id,
                document_name=document_name,
                filename=filename,
                source="redis_cache",
                document_info=document_info,
            )

    converted = _convert_for_preview(path, document_id=document_id, document_name=document_name)
    return _payload_to_preview_dict(
        converted,
        document_id=document_id,
        document_name=document_name,
        filename=filename,
        source="inline_conversion",
        document_info=document_info,
    )


def resolve_document_render_bytes(payload: dict[str, Any]) -> tuple[bytes, str]:
    render_format = str(payload.get("format") or "html").lower()
    media_type = str(payload.get("media_type") or "text/html; charset=utf-8")
    if render_format == "pdf":
        pdf_bytes = decode_render_bytes(payload)
        if not pdf_bytes:
            raise_client_error(
                COMPONENT_DOCUMENT_PREVIEW,
                code="preview_not_found",
                http_status=404,
                user_message="PDF preview is not available for this document.",
                reason=f"Missing PDF render bytes for document_id={payload.get('document_id')}",
            )
        return pdf_bytes, media_type

    content = decode_render_text(payload)
    if not content:
        raise_client_error(
            COMPONENT_DOCUMENT_PREVIEW,
            code="preview_not_found",
            http_status=404,
            user_message="Preview content is not available for this document.",
            reason=f"Missing HTML render content for document_id={payload.get('document_id')}",
        )
    if not content.lstrip().lower().startswith("<!doctype"):
        content = wrap_render_document_html(content, title=str(payload.get("document_name") or "Document preview"))
    return content.encode("utf-8"), media_type


def preview_response_headers(media_type: str) -> dict[str, str]:
    return _read_only_headers(media_type=media_type)
