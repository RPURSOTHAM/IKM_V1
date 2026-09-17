"""Content fingerprints for duplicate detection across file formats."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

_logger = logging.getLogger(__name__)

_MIN_NORMALIZED_CHARS = 80
_WS_RE = re.compile(r"[^a-z0-9]+")


def normalize_document_text(text: str) -> str:
    """Collapse extracted text so PDF/DOCX/TXT of the same wording can match."""
    folded = str(text or "").lower().replace("\u00a0", " ")
    collapsed = _WS_RE.sub(" ", folded)
    return collapsed.strip()


def hash_normalized_text(text: str) -> str | None:
    """Return SHA-256 of normalized text, or None when the body is too short to compare."""
    normalized = normalize_document_text(text)
    if len(normalized) < _MIN_NORMALIZED_CHARS:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_text_for_fingerprint(path: Path) -> str:
    """Best-effort extractable text for duplicate hashing. Never raises."""
    suffix = path.suffix.lower()
    if suffix in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
                  ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v", ".wmv"}:
        return ""
    try:
        from src.features.document_processing.loaders.document_text import load_document_blocks

        blocks, _meta = load_document_blocks(path, mask_sensitive=False)
        return " ".join(str(getattr(block, "text", "") or "") for block in blocks)
    except Exception:
        _logger.debug("Fingerprint extract via loaders failed for %s", path, exc_info=True)
    if suffix in {".txt", ".text"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    return ""


def compute_content_text_hash(path: Path) -> str | None:
    """Hash extractable wording of a saved upload, if enough text is present."""
    return hash_normalized_text(extract_text_for_fingerprint(path))
