"""Resolve chunking and embedding settings from request, repository, and platform config.

Effective repository settings are produced upstream by
``repository_service.settings_resolver.SettingsResolver`` (via document-receiver
``processing_hints``). This module only coalesces already-resolved request fields
with processor platform env defaults — it must not load MySQL settings directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.features.document_processing.core.config import get_settings
from src.features.document_processing.core.contract import ProcessRequest
from src.features.chunking.domain.chunking_strategy import DEFAULT_CHUNKING_STRATEGY, resolve_chunking_strategy


@dataclass(frozen=True)
class ResolvedChunkingSettings:
    model_name: str
    model_dir: str | None
    chunk_size: int
    chunk_overlap_sentences: int
    min_content_words: int
    citation_retainment: bool
    chunking_strategy: str
    chunking_config: dict[str, Any]
    max_chunks: int


def _coalesce(*values: Any) -> Any:
    """Return the first non-None, non-blank value."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _first(*values: Any) -> Any:
    return _coalesce(*values)


def _platform_config():
    try:
        from src.features.configuration.application import get_shared_platform_config

        return get_shared_platform_config().namespace("processor")
    except Exception:
        return None


def resolve_chunking_settings(request: ProcessRequest) -> ResolvedChunkingSettings:
    """Merge top-level request fields, nested repository settings, and platform defaults."""
    settings = get_settings()
    proc_cfg = _platform_config()
    repo = request.repository_settings

    model_name = _coalesce(
        request.model_name,
        repo.embedding_model_name if repo else None,
        proc_cfg.get_str("embedding.model_name", settings.model_name) if proc_cfg else None,
        settings.model_name,
    )
    if not model_name:
        raise ValueError(
            "Embedding model is not configured. Provide model_name on the request, "
            "repository_settings.embedding_model_name, or set MODEL_NAME in the processor environment."
        )
    model_dir = _coalesce(
        request.model_dir,
        repo.embedding_model_dir if repo else None,
        settings.model_dir,
    )
    chunk_size = _first(
        request.chunk_size,
        repo.chunk_size if repo else None,
        proc_cfg.get_int("chunking.chunk_size", settings.chunk_size) if proc_cfg else None,
        settings.chunk_size,
    )
    chunk_overlap_sentences = _first(
        request.chunk_overlap_sentences,
        repo.chunk_overlap_sentences if repo else None,
        proc_cfg.get_int("chunking.chunk_overlap_sentences", settings.chunk_overlap_sentences)
        if proc_cfg
        else None,
        settings.chunk_overlap_sentences,
    )

    chunking_config: dict[str, Any] = {}
    if repo and repo.chunking_config:
        chunking_config.update(repo.chunking_config)
    if request.chunking_config:
        chunking_config.update(request.chunking_config)

    min_content_words = _first(
        request.min_content_words,
        chunking_config.get("min_content_words"),
        proc_cfg.get_int("chunking.min_content_words", settings.min_content_words) if proc_cfg else None,
        settings.min_content_words,
    )
    if not isinstance(min_content_words, int):
        min_content_words = int(min_content_words)

    citation_retainment = request.citation_retainment
    if citation_retainment is None and repo is not None and repo.citation_retainment is not None:
        citation_retainment = repo.citation_retainment
    if citation_retainment is None:
        citation_retainment = True

    chunking_strategy = resolve_chunking_strategy(
        _first(
            request.chunking_strategy,
            repo.chunking_strategy if repo else None,
            DEFAULT_CHUNKING_STRATEGY,
        )
    )

    max_chunks = proc_cfg.get_int("limits.max_chunks_per_document", 10000) if proc_cfg else 10000

    return ResolvedChunkingSettings(
        model_name=str(model_name),
        model_dir=str(model_dir).strip() if model_dir else None,
        chunk_size=int(chunk_size),
        chunk_overlap_sentences=int(chunk_overlap_sentences),
        min_content_words=int(min_content_words),
        citation_retainment=bool(citation_retainment),
        chunking_strategy=chunking_strategy,
        chunking_config=chunking_config,
        max_chunks=int(max_chunks),
    )
