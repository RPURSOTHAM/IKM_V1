"""Build grounded prompts without exposing internal metadata."""

from __future__ import annotations

from typing import Any

from src.features.generation.configuration.generation_config import GenerationConfig
from src.features.generation.domain.interfaces import GenerationPromptBuilder
from src.features.generation.domain.models import (
    INTERNAL_METADATA_KEYS,
    Citation,
    ConversationTurn,
    GenerationRequest,
)


def sanitize_public_metadata(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only user-safe metadata for prompt construction."""
    if not raw:
        return {}
    allowed = {
        "document_name",
        "original_file_name",
        "section",
        "section_name",
        "heading",
        "page",
        "topic",
        "category",
        "repository_name",
    }
    public: dict[str, Any] = {}
    for key, value in raw.items():
        if key in INTERNAL_METADATA_KEYS:
            continue
        if key not in allowed:
            continue
        if value in (None, "", [], {}):
            continue
        public[key] = value
    return public


def _as_optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def chunk_to_public_fields(chunk: Any) -> dict[str, Any]:
    raw_meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    metadata = sanitize_public_metadata(raw_meta)
    # Prefer retrieval-time resolved citation when present (precise supporting block).
    resolved = raw_meta.get("citation") if isinstance(raw_meta.get("citation"), dict) else {}
    doc_name = (
        resolved.get("document_name")
        or resolved.get("document")
        or metadata.get("original_file_name")
        or metadata.get("document_name")
        or getattr(chunk, "document_name", None)
        or getattr(chunk, "doc_name", None)
        or getattr(chunk, "source", None)
        or "document"
    )
    section = (
        resolved.get("section")
        or metadata.get("heading")
        or metadata.get("section")
        or metadata.get("section_name")
        or getattr(chunk, "section_name", None)
        or ""
    )
    page = resolved.get("page_start")
    if page is None:
        page = resolved.get("page")
    if page is None:
        page = metadata.get("page")
    if page is None:
        page = getattr(chunk, "page", None)
    text = str(getattr(chunk, "text", "") or "").strip()
    # document_id is not in the public allowlist (prompt-safe), but citations need it.
    cite_anchor = raw_meta.get("citation_anchor") if isinstance(raw_meta.get("citation_anchor"), dict) else None
    if cite_anchor is None:
        cite_anchor = getattr(chunk, "citation_anchor", None)
        if not isinstance(cite_anchor, dict):
            cite_anchor = None
    document_id = (
        resolved.get("document_id")
        or raw_meta.get("document_id")
        or getattr(chunk, "document_id", None)
        or (cite_anchor or {}).get("document_id")
    )
    page_end = resolved.get("page_end")
    if page_end is None:
        page_end = raw_meta.get("page_end")
    if page_end is None:
        page_end = getattr(chunk, "page_end", None)
    if page_end is None and cite_anchor:
        page_end = cite_anchor.get("end_page")
    line_start = resolved.get("line_start")
    if line_start is None:
        line_start = raw_meta.get("line_start")
    if line_start is None:
        line_start = getattr(chunk, "line_start", None)
    line_end = resolved.get("line_end")
    if line_end is None:
        line_end = raw_meta.get("line_end")
    if line_end is None:
        line_end = getattr(chunk, "line_end", None)
    section_path = (
        resolved.get("section_path")
        or raw_meta.get("section_path")
        or getattr(chunk, "section_path", None)
        or ""
    )
    source_blocks = resolved.get("supporting_source_blocks") or resolved.get("source_blocks")
    if not isinstance(source_blocks, list):
        source_blocks = raw_meta.get("source_blocks")
    if not isinstance(source_blocks, list):
        source_blocks = getattr(chunk, "source_blocks", None)
    if not isinstance(source_blocks, list) and cite_anchor:
        source_blocks = cite_anchor.get("source_blocks")
    if not isinstance(source_blocks, list):
        source_blocks = None
    chunk_id = (
        resolved.get("chunk_id")
        or getattr(chunk, "chunk_id", None)
        or getattr(chunk, "id", None)
        or raw_meta.get("chunk_id")
        or (cite_anchor or {}).get("chunk_id")
    )
    if section and str(section).strip().lower() in {"document", "unknown", ""}:
        # Prefer stored non-generic labels from resolution / path.
        section = resolved.get("section") or section_path or section
    page_i = _as_optional_int(page)
    page_end_i = _as_optional_int(page_end)
    line_start_i = _as_optional_int(line_start)
    line_end_i = _as_optional_int(line_end)
    score = float(getattr(chunk, "score", 0.0) or 0.0)
    return {
        "document_name": str(doc_name),
        "section": str(section) if section else None,
        "page": page_i,
        "page_end": page_end_i,
        "line_start": line_start_i,
        "line_end": line_end_i,
        "section_path": str(section_path) if section_path else None,
        "source_blocks": source_blocks,
        "chunk_id": str(chunk_id) if chunk_id else None,
        "citation_anchor": cite_anchor,
        "document_id": str(document_id).strip() if document_id else None,
        "text": text,
        "score": score,
        "public_metadata": {
            k: v
            for k, v in {
                "document_name": doc_name,
                "section": section or None,
                "page": page_i,
                "topic": metadata.get("topic"),
                "category": metadata.get("category"),
                "repository_name": metadata.get("repository_name"),
            }.items()
            if v not in (None, "")
        },
    }


def build_citations(
    chunks: list[Any],
    *,
    max_chunks: int,
    max_snippet_chars: int,
    query_text: str | None = None,
    answer_text: str | None = None,
) -> list[Citation]:
    citations: list[Citation] = []
    for index, chunk in enumerate(chunks[:max_chunks], start=1):
        fields = chunk_to_public_fields(chunk)
        # Re-resolve with question/answer terms when source_blocks allow precise narrowing.
        if fields.get("source_blocks") and (query_text or answer_text):
            try:
                from src.features.citations.resolution.citation_resolver import resolve_citation

                refined = resolve_citation(
                    {
                        "document_name": fields["document_name"],
                        "document_id": fields.get("document_id"),
                        "section_name": fields.get("section"),
                        "section_path": fields.get("section_path"),
                        "page": fields.get("page"),
                        "page_end": fields.get("page_end"),
                        "line_start": fields.get("line_start"),
                        "line_end": fields.get("line_end"),
                        "chunk_id": fields.get("chunk_id"),
                        "source_blocks": fields.get("source_blocks"),
                        "citation_anchor": fields.get("citation_anchor"),
                        "text": fields.get("text"),
                    },
                    document_display_name=fields["document_name"],
                    query_text=query_text,
                    answer_text=answer_text,
                )
                if refined.get("line_start") is not None:
                    fields["line_start"] = refined.get("line_start")
                if refined.get("line_end") is not None:
                    fields["line_end"] = refined.get("line_end")
                if refined.get("page_start") is not None or refined.get("page") is not None:
                    fields["page"] = refined.get("page_start") or refined.get("page")
                if refined.get("page_end") is not None:
                    fields["page_end"] = refined.get("page_end")
                if refined.get("section"):
                    fields["section"] = refined.get("section")
                if refined.get("section_path"):
                    fields["section_path"] = refined.get("section_path")
                if refined.get("supporting_source_blocks"):
                    fields["source_blocks"] = refined.get("supporting_source_blocks")
                elif refined.get("source_blocks"):
                    fields["source_blocks"] = refined.get("source_blocks")
            except Exception:
                pass
        snippet = fields["text"][:max_snippet_chars].strip()
        citations.append(
            Citation(
                index=index,
                document_name=fields["document_name"],
                page=fields["page"],
                section=fields["section"],
                document_id=fields["document_id"],
                snippet=snippet,
                page_end=fields.get("page_end"),
                line_start=fields.get("line_start"),
                line_end=fields.get("line_end"),
                chunk_id=fields.get("chunk_id"),
                section_path=fields.get("section_path"),
                source_blocks=fields.get("source_blocks"),
                citation_anchor=fields.get("citation_anchor"),
            )
        )
    return citations


def evidence_is_sufficient(
    chunks: list[Any],
    config: GenerationConfig,
) -> bool:
    usable = []
    for chunk in chunks:
        fields = chunk_to_public_fields(chunk)
        if not fields["text"]:
            continue
        if float(fields["score"]) < float(config.min_evidence_score):
            continue
        usable.append(fields)
    return len(usable) >= int(config.min_evidence_chunks)


def format_public_context(citations: list[Citation]) -> str:
    lines: list[str] = []
    for citation in citations:
        location = []
        if citation.section:
            location.append(f"section={citation.section}")
        if citation.page is not None:
            if citation.page_end is not None and citation.page_end != citation.page:
                location.append(f"pages={citation.page}-{citation.page_end}")
            else:
                location.append(f"page={citation.page}")
        if citation.line_start is not None and citation.line_end is not None:
            location.append(f"lines={citation.line_start}-{citation.line_end}")
        loc = f" ({', '.join(location)})" if location else ""
        lines.append(f"[{citation.index}] {citation.document_name}{loc}:\n{citation.snippet}")
    return "\n\n".join(lines)


def format_sources(citations: list[Citation]) -> str:
    lines = []
    for citation in citations:
        page_label = "n/a"
        if citation.page is not None:
            if citation.page_end is not None and citation.page_end != citation.page:
                page_label = f"{citation.page}-{citation.page_end}"
            else:
                page_label = str(citation.page)
        lines.append(
            f"[{citation.index}] {citation.document_name}"
            f" | section={citation.section or 'n/a'}"
            f" | page={page_label}"
        )
    return "\n".join(lines)


def ensure_citations_in_answer(answer: str, citations: list[Citation]) -> str:
    """Append citation markers when the model omits them."""
    text = (answer or "").strip()
    if not citations:
        return text
    if any(f"[{citation.index}]" in text for citation in citations):
        return text
    markers = " ".join(f"[{citation.index}]" for citation in citations)
    return f"{text}\n\nSources: {markers}".strip()


class GroundedPromptBuilder(GenerationPromptBuilder):
    """Compose System + Conversation + Retrieved Context + Metadata + User Question."""

    def build_messages(
        self,
        request: GenerationRequest,
        config: GenerationConfig,
        *,
        citations: list[Citation],
        public_context: str,
        public_metadata: dict[str, Any],
    ) -> list[dict[str, str]]:
        metadata_lines = [
            f"- {key}: {value}" for key, value in sorted(public_metadata.items())
        ]
        metadata_block = "\n".join(metadata_lines) if metadata_lines else "None"
        sources = format_sources(citations)
        user_payload = (
            f"Retrieved Context:\n{public_context or 'None'}\n\n"
            f"Sources:\n{sources or 'None'}\n\n"
            f"Metadata:\n{metadata_block}\n\n"
            f"User Question:\n{request.question.strip()}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": config.system_prompt},
        ]
        for turn in request.conversation:
            role = turn.role if isinstance(turn, ConversationTurn) else str(turn.get("role", "user"))
            content = turn.content if isinstance(turn, ConversationTurn) else str(turn.get("content", ""))
            role = role.strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            # Never inject prior system turns that could override grounding policy.
            if role == "system":
                continue
            if content.strip():
                messages.append({"role": role, "content": content.strip()})
        messages.append({"role": "user", "content": user_payload})
        return messages
