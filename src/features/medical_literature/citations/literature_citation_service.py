"""Strict [REPO:DOCUMENT:PAGE] citation helpers for the PubMed AI Agent."""

from __future__ import annotations

import re
from typing import Any, Iterable

from src.features.generation.domain.models import Citation

# Required citation contract: [REPO:DOCUMENT:PAGE] with UUID-safe tokens
CITATION_RE = re.compile(
    r"\[(?P<repo>[A-Za-z0-9_-]+):(?P<document>[A-Za-z0-9_-]+):(?P<page>[A-Za-z0-9_-]+)\]"
)

PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find sufficient evidence in this repository to answer that question."
)

PUBMED_SYSTEM_PROMPT = (
    "You are a PubMed AI agent. Answer ONLY using the "
    "retrieved repository evidence. Do not use model memory or outside knowledge. "
    "Every factual claim MUST include an inline citation in the exact format "
    "[REPO:DOCUMENT:PAGE] where REPO is the repository id, DOCUMENT is the document id, "
    "and PAGE is the page number (or '1' when page metadata is unavailable). "
    "Only cite documents that appear in the Sources list. "
    "If evidence is insufficient, reply exactly: "
    f"{PUBMED_INSUFFICIENT_EVIDENCE_MESSAGE}"
)


def citation_marker(*, repository_id: str, document_id: str, page: int | str | None) -> str:
    repo = _token(repository_id, fallback="REPO")
    doc = _token(document_id, fallback="DOCUMENT")
    page_token = _page_token(page)
    return f"[{repo}:{doc}:{page_token}]"


def format_pubmed_sources(citations: list[Citation], *, repository_id: str) -> str:
    lines: list[str] = []
    for citation in citations:
        doc_id = citation.document_id or citation.document_name
        marker = citation_marker(
            repository_id=repository_id,
            document_id=str(doc_id),
            page=citation.page,
        )
        lines.append(
            f"{marker} document_name={citation.document_name}"
            f" | section={citation.section or 'n/a'}"
            f" | page={citation.page if citation.page is not None else 'n/a'}"
        )
    return "\n".join(lines)


def format_pubmed_context(citations: list[Citation], *, repository_id: str) -> str:
    blocks: list[str] = []
    for citation in citations:
        doc_id = citation.document_id or citation.document_name
        marker = citation_marker(
            repository_id=repository_id,
            document_id=str(doc_id),
            page=citation.page,
        )
        blocks.append(f"{marker} {citation.document_name}:\n{citation.snippet}")
    return "\n\n".join(blocks)


def allowed_citation_markers(
    citations: Iterable[Citation],
    *,
    repository_id: str,
) -> set[str]:
    """Exact markers permitted for this retrieval set (UUID identity)."""
    allowed: set[str] = set()
    for citation in citations:
        doc_id = citation.document_id
        if not doc_id:
            continue
        allowed.add(
            citation_marker(
                repository_id=repository_id,
                document_id=str(doc_id),
                page=citation.page,
            )
        )
        # When page metadata is missing, PAGE token is normalized to "1".
        if citation.page is None:
            allowed.add(
                citation_marker(
                    repository_id=repository_id,
                    document_id=str(doc_id),
                    page=1,
                )
            )
    return allowed


def extract_citations(answer: str) -> list[str]:
    return [m.group(0) for m in CITATION_RE.finditer(answer or "")]


def validate_pubmed_citations(
    answer: str,
    *,
    repository_id: str,
    citations: list[Citation],
) -> tuple[bool, list[str], list[str]]:
    """Return (ok, found_markers, invalid_markers).

    Strict rules:
    - At least one inline citation is required.
    - Every citation must exactly match a retrieved (repo, document_id, page) marker.
    - Invented documents / wrong repositories are rejected.
    """
    found = extract_citations(answer)
    if not found:
        return False, [], ["missing_citation"]
    allowed = allowed_citation_markers(citations, repository_id=repository_id)
    if not allowed:
        return False, found, ["no_retrieved_document_identity"]
    invalid = [marker for marker in found if marker not in allowed]
    return (len(invalid) == 0), found, invalid


def ensure_pubmed_citations_in_answer(
    answer: str,
    citations: list[Citation],
    *,
    repository_id: str,
) -> str:
    text = (answer or "").strip()
    if not citations:
        return text
    if extract_citations(text):
        return text
    markers = " ".join(
        citation_marker(
            repository_id=repository_id,
            document_id=str(c.document_id),
            page=c.page,
        )
        for c in citations
        if c.document_id
    )
    if not markers:
        return text
    return f"{text}\n\nSources: {markers}".strip()


def citations_to_public_dicts(
    citations: list[Citation],
    *,
    repository_id: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for citation in citations:
        if not citation.document_id:
            continue
        marker = citation_marker(
            repository_id=repository_id,
            document_id=str(citation.document_id),
            page=citation.page,
        )
        payload = citation.to_public_dict()
        payload["citation"] = marker
        payload["repository_id"] = repository_id
        out.append(payload)
    return out


def _token(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "", str(value or "").strip())
    return text or fallback


def _page_token(page: int | str | None) -> str:
    if page is None or page == "":
        return "1"
    if isinstance(page, int):
        return str(page)
    text = str(page).strip()
    if not text or text.lower() in {"n/a", "na", "none"}:
        return "1"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", text)
    return cleaned or "1"
