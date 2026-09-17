"""Bridge pdf_reader / template referenced_documents into Neo4jReferenceStore shape."""

from __future__ import annotations

import logging
from typing import Any

from src.features.references.extraction.reference_extractor import normalize_document_id

logger = logging.getLogger(__name__)

_FORMAL_SOURCES = frozenset(
    {"references_section", "annexure_section", "attachment_section", "header"}
)


def adapt_template_referenced_documents(referenced_documents: list | None) -> list[dict[str, Any]]:
    """Convert pdf_reader-style refs into save_document_reference_graph entries."""
    adapted: list[dict[str, Any]] = []
    seen: set[str] = set()

    for entry in referenced_documents or []:
        if not isinstance(entry, dict):
            continue
        raw_id = str(
            entry.get("document_id")
            or entry.get("target_document_id")
            or entry.get("ref_number")
            or ""
        ).strip()
        if not raw_id:
            continue

        normalized_target_id = normalize_document_id(
            entry.get("normalized_target_id") or raw_id
        )
        if not normalized_target_id or normalized_target_id in seen:
            continue
        seen.add(normalized_target_id)

        contexts = entry.get("contexts") or entry.get("evidence_sentences") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = [str(c).strip() for c in contexts if str(c or "").strip()]

        title = entry.get("title")
        title_text = str(title).strip() if title else ""
        source_sentence = (
            contexts[0]
            if contexts
            else str(entry.get("context") or entry.get("source_sentence") or title_text or raw_id)
        )
        reference_text = title_text or str(entry.get("reference_text") or raw_id)

        sources = entry.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        source = entry.get("source")
        if source and source not in sources:
            sources = [*sources, source]

        formal = bool(_FORMAL_SOURCES.intersection({str(s) for s in sources if s}))
        confidence = float(entry.get("confidence") or (0.9 if formal else 0.75))
        extraction_method = str(
            entry.get("extraction_method")
            or ("pdf_template_references_section" if formal else "pdf_template_reference")
        )

        adapted.append(
            {
                "target_document_id": normalized_target_id,
                "normalized_target_id": normalized_target_id,
                "normalized_reference": normalized_target_id,
                "extracted_target_id": raw_id.replace(" ", ""),
                "reference_text": reference_text,
                "source_sentence": source_sentence,
                "confidence": confidence,
                "extraction_method": extraction_method,
                "reference_type": str(entry.get("reference_type") or "document"),
                "source_scope": "document",
                "evidence_sentences": contexts,
                "trigger_phrase": entry.get("trigger_phrase"),
                "pages": list(entry.get("pages") or []),
                "title": title_text or None,
            }
        )
    return adapted


def _template_text_corpus(template_data: dict[str, Any] | None) -> str:
    """Flatten template text fields for canonical reference regex extraction."""
    if not isinstance(template_data, dict):
        return ""
    parts: list[str] = []
    for block in template_data.get("content_blocks") or []:
        if isinstance(block, dict):
            text = str(block.get("text") or "").strip()
            if text:
                parts.append(text)
    for node in template_data.get("document_tree") or []:
        stack = [node]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue
            text = str(current.get("text") or "").strip()
            if text:
                parts.append(text)
            stack.extend(current.get("children") or [])
    for header in template_data.get("headers_footers") or []:
        if isinstance(header, dict):
            text = str(header.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _references_from_template_text(template_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    text = _template_text_corpus(template_data)
    if not text.strip():
        return []
    from src.features.references.extraction.reference_extractor import (
        extract_document_references,
        _looks_like_document_id,
    )

    collected: list[dict[str, Any]] = []
    for item in extract_document_references(text):
        reference_id = str(item.get("reference_id") or "").strip()
        if not reference_id:
            continue
        normalized = normalize_document_id(reference_id)
        # Reject bare dictionary words (e.g. "REFERENCES") mistaken for IDs.
        if not _looks_like_document_id(normalized):
            continue
        collected.append(
            {
                "document_id": reference_id,
                "normalized_target_id": normalized,
                "title": item.get("matched_text") or reference_id,
                "reference_text": item.get("matched_text") or reference_id,
                "source_sentence": item.get("matched_text") or reference_id,
                "confidence": float(item.get("confidence") or 0.85),
                "extraction_method": "template_text_regex",
                "reference_type": "document",
                "contexts": [item.get("matched_text")] if item.get("matched_text") else [],
            }
        )
    return collected


def persist_template_references(
    *,
    document_id: str,
    repository_id: str | None,
    tenant_id: str | None,
    display_name: str | None,
    referenced_documents: list | None,
    template_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Adapt and persist template refs; soft-fail when store is disabled or empty."""
    merged_source = list(referenced_documents or [])
    merged_source.extend(_references_from_template_text(template_data))
    adapted = adapt_template_referenced_documents(merged_source)
    if not adapted:
        return None

    try:
        from src.features.references.infrastructure.neo4j_reference_store import get_reference_store

        store = get_reference_store()
        if not store.enabled:
            logger.info(
                "Reference store disabled; skipping %s template references for document_id=%s",
                len(adapted),
                document_id,
            )
            return None

        store.register_document(
            document_id=document_id,
            document_name=display_name or document_id,
            source_title=display_name,
            tenant_id=tenant_id,
            repository_id=repository_id,
        )
        stats = store.save_document_reference_graph(
            source_document_id=document_id,
            source_title=display_name,
            tenant_id=tenant_id,
            repository_id=repository_id,
            references=adapted,
        )
        logger.info(
            "Persisted template references document_id=%s saved=%s resolved=%s unresolved=%s",
            document_id,
            stats.get("saved", 0),
            stats.get("resolved", 0),
            stats.get("unresolved", 0),
        )
        return stats
    except Exception:
        logger.exception(
            "Failed to persist template references for document_id=%s",
            document_id,
        )
        return None
