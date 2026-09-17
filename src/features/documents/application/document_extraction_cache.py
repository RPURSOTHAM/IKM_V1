"""Persist and load CIH document extraction without re-running the PDF pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXTRACTION_JSON_SUFFIX = ".extraction.json"
PPT_SUFFIXES = {".pptx", ".ppt"}


def extraction_sidecar_path(document_path: Path) -> Path:
    return document_path.with_suffix(document_path.suffix + EXTRACTION_JSON_SUFFIX)


def sections_from_blocks(blocks: list[Any], *, suffix: str) -> list[dict[str, Any]]:
    by_page: dict[int, list[str]] = {}
    for block in blocks:
        text = str(getattr(block, "text", "") or "").strip()
        if not text:
            continue
        page = int(getattr(block, "page", 0) or 1)
        by_page.setdefault(page, []).append(text)
    kind = "slide" if suffix in PPT_SUFFIXES else "page"
    sections: list[dict[str, Any]] = []
    for page in sorted(by_page):
        text = "\n".join(line for line in by_page[page] if line).strip()
        if not text:
            continue
        sections.append(
            {
                "index": page,
                "page": page,
                "text": text,
                "kind": kind,
            }
        )
    return sections


def build_extraction_payload(
    *,
    document_id: str,
    filename: str,
    suffix: str,
    sections: list[dict[str, Any]],
    source: str,
    file_extension: str | None = None,
) -> dict[str, Any]:
    ext = (file_extension or suffix.lstrip(".") or source).strip()
    return {
        "document_id": document_id,
        "original_file_name": filename,
        "file_extension": ext,
        "source": source,
        "section_count": len(sections),
        "sections": sections,
        "full_text": "\n\n".join(str(section.get("text") or "") for section in sections),
    }


def write_extraction_sidecar(document_path: Path, payload: dict[str, Any]) -> Path:
    json_path = extraction_sidecar_path(document_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def load_extraction_sidecar(document_path: Path) -> dict[str, Any] | None:
    json_path = extraction_sidecar_path(document_path)
    if not json_path.is_file():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed reading extraction sidecar %s", json_path, exc_info=True)
        return None
    return payload if isinstance(payload, dict) else None


def sections_from_chunk_rows(chunks: list[dict[str, Any]], *, suffix: str) -> list[dict[str, Any]]:
    by_page: dict[int, list[str]] = {}
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            props = chunk.get("properties") if isinstance(chunk.get("properties"), dict) else {}
            text = str(props.get("text") or props.get("raw_text") or "").strip()
        if not text:
            continue
        page_raw = chunk.get("page")
        if page_raw is None:
            props = chunk.get("properties") if isinstance(chunk.get("properties"), dict) else {}
            page_raw = props.get("page")
        page = int(page_raw or 1)
        by_page.setdefault(page, []).append(text)
    kind = "slide" if suffix in PPT_SUFFIXES else "page"
    sections: list[dict[str, Any]] = []
    for page in sorted(by_page):
        text = "\n\n".join(by_page[page]).strip()
        if not text:
            continue
        sections.append(
            {
                "index": page,
                "page": page,
                "text": text,
                "kind": kind,
            }
        )
    return sections


def live_extraction_reload_enabled() -> bool:
    return str(os.getenv("CIH_EXTRACTION_FORCE_LIVE", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
