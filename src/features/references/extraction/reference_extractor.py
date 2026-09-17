from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.features.document_processing.core.logger import log

if TYPE_CHECKING:
    from src.features.chunking.application.chunking_service import Chunk

REFERENCE_LABELS: dict[str, str] = {
    "Document No": "document",
    "Document Number": "document",
    "Document ID": "document",
    "Document Reference": "document",
    "Related Document": "document",
    "SOP No": "sop",
    "WI No": "work_instruction",
    "Form No": "form",
    "Ref": "document",
    "Protocol No": "protocol",
    "Protocol": "protocol",
    "Report No": "report",
    "Report": "report",
}

_LABEL_PATTERN = "|".join(re.escape(label).replace(r"\ ", r"\s+") for label in REFERENCE_LABELS)
_REFERENCE_ID_PATTERN = r"[A-Z0-9][A-Z0-9/_\-.]{1,63}[A-Z0-9]"
_LABELED_REFERENCE_PATTERN = re.compile(
    rf"\b(?P<label>{_LABEL_PATTERN})\.?\s*(?:[:#-]|\bis\b)?\s*(?P<reference_id>{_REFERENCE_ID_PATTERN})\b",
    re.IGNORECASE,
)

_SECTION_REFERENCE_PATTERN = re.compile(
    r"\b(?:Section|Clause|§)\s+(?P<section_ref>\d+(?:\.\d+){0,4}[A-Za-z]?)\b",
    re.IGNORECASE,
)
_ANNEXURE_REFERENCE_PATTERN = re.compile(
    r"\bAnnex(?:ure)?\s+(?P<annex_ref>[A-Z0-9]{1,4})\b",
    re.IGNORECASE,
)
_APPENDIX_REFERENCE_PATTERN = re.compile(
    r"\bAppendix\s+(?P<appendix_ref>[A-Z0-9]{1,4})\b",
    re.IGNORECASE,
)
_CITATION_REFERENCE_PATTERN = re.compile(
    r"(?:\[(?P<bracket_ref>\d{1,3})\]|"
    r"\((?P<paren_ref>[A-Z][A-Za-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\))",
)


def normalize_document_id(value: str | None) -> str:
    """Canonical document ID normalization used for extraction and Neo4j resolution."""
    if not value:
        return ""
    normalized = str(value).upper().strip()
    for extension in (".PDF", ".DOCX", ".DOC", ".TXT"):
        if normalized.endswith(extension):
            normalized = normalized[: -len(extension)]
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def normalize_reference_id(reference_id: str) -> str:
    return normalize_document_id(reference_id)


def normalize_document_key(document_id: str | None) -> str:
    return normalize_document_id(document_id)


def is_same_document_id(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return normalize_document_id(left) == normalize_document_id(right)


def _source_sentence(text: str, start: int, end: int) -> str:
    if not text:
        return ""
    window_start = max(0, start - 400)
    window_end = min(len(text), end + 400)
    window = text[window_start:window_end]
    relative_start = start - window_start
    sentence_start = 0
    for delimiter in (". ", ".\n", "! ", "? ", "\n"):
        idx = window.rfind(delimiter, 0, relative_start)
        if idx != -1:
            sentence_start = max(sentence_start, idx + len(delimiter))
    sentence_end = len(window)
    for delimiter in (". ", ".\n", "! ", "? ", "\n"):
        idx = window.find(delimiter, end - window_start)
        if idx != -1:
            sentence_end = min(sentence_end, idx + 1)
            break
    sentence = window[sentence_start:sentence_end].strip()
    return sentence or text[max(0, start - 80) : min(len(text), end + 80)].strip()


def _append_reference(
    references: list[dict[str, Any]],
    seen: set[str],
    *,
    reference_text: str,
    reference_type: str,
    source_sentence: str,
    source_section: str,
    target_document_id: str | None,
    target_document_name: str | None,
    target_section: str | None,
    confidence: float,
    reason: str,
    start_index: int,
    end_index: int,
    chunk_id: str | None = None,
) -> None:
    dedupe_key = f"{chunk_id or ''}:{reference_type}:{reference_text.strip().lower()}"
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    references.append(
        {
            "reference_text": reference_text,
            "reference_type": reference_type,
            "source_sentence": source_sentence,
            "source_section": source_section,
            "target_document_id": target_document_id,
            "target_document_name": target_document_name,
            "target_section": target_section,
            "confidence": confidence,
            "reason": reason,
            "start_index": start_index,
            "end_index": end_index,
            "chunk_id": chunk_id,
            # Backward-compatible fields for document-level Neo4j storage.
            "reference_id": target_document_id,
            "matched_text": reference_text,
            "extraction_method": "regex",
        }
    )


_IMPLICIT_REFERENCE_PATTERN = re.compile(
    rf"\b(?:refer(?:\s+to)?|see(?:\s+also)?|as\s+per|according\s+to|based\s+on)\s+(?P<reference_id>{_REFERENCE_ID_PATTERN})\b",
    re.IGNORECASE,
)
_BARE_DOCUMENT_ID_PATTERN = re.compile(
    rf"\b(?P<reference_id>{_REFERENCE_ID_PATTERN})\b",
    re.IGNORECASE,
)
_COLON_SUFFIX_DOCUMENT_ID_PATTERN = re.compile(
    rf":\s*(?P<reference_id>{_REFERENCE_ID_PATTERN})\.?\s*$",
    re.IGNORECASE,
)
_OUTLINE_LINE_PATTERN = re.compile(r"^\d+(?:\.\d+)+\.?\s+")
_SECTION_NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")


def extract_references_from_text(
    text: str,
    *,
    source_section: str = "",
    chunk_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract structured references from chunk text."""
    if not text or not text.strip():
        return []

    references: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in _LABELED_REFERENCE_PATTERN.finditer(text):
        label = match.group("label")
        reference_id = normalize_reference_id(match.group("reference_id"))
        if not reference_id:
            continue
        reference_type = next(
            (rtype for key, rtype in REFERENCE_LABELS.items() if key.lower() == label.lower()),
            "document",
        )
        _append_reference(
            references,
            seen,
            reference_text=match.group(0),
            reference_type=reference_type,
            source_sentence=_source_sentence(text, match.start(), match.end()),
            source_section=source_section,
            target_document_id=reference_id,
            target_document_name=None,
            target_section=None,
            confidence=0.92,
            reason=f"Labeled {reference_type} reference matched by regex.",
            start_index=match.start(),
            end_index=match.end(),
            chunk_id=chunk_id,
        )

    for match in _IMPLICIT_REFERENCE_PATTERN.finditer(text):
        reference_id = normalize_reference_id(match.group("reference_id"))
        if not reference_id or not re.search(r"[\-/0-9]", reference_id):
            continue
        _append_reference(
            references,
            seen,
            reference_text=match.group(0),
            reference_type="document",
            source_sentence=_source_sentence(text, match.start(), match.end()),
            source_section=source_section,
            target_document_id=reference_id,
            target_document_name=None,
            target_section=None,
            confidence=0.88,
            reason="Implicit document reference matched after trigger phrase.",
            start_index=match.start(),
            end_index=match.end(),
            chunk_id=chunk_id,
        )

    for match in _SECTION_REFERENCE_PATTERN.finditer(text):
        section_ref = match.group("section_ref")
        _append_reference(
            references,
            seen,
            reference_text=match.group(0),
            reference_type="section",
            source_sentence=_source_sentence(text, match.start(), match.end()),
            source_section=source_section,
            target_document_id=None,
            target_document_name=None,
            target_section=section_ref,
            confidence=0.84,
            reason="Section or clause reference matched by regex.",
            start_index=match.start(),
            end_index=match.end(),
            chunk_id=chunk_id,
        )

    for match in _ANNEXURE_REFERENCE_PATTERN.finditer(text):
        annex_ref = match.group("annex_ref")
        _append_reference(
            references,
            seen,
            reference_text=match.group(0),
            reference_type="annexure",
            source_sentence=_source_sentence(text, match.start(), match.end()),
            source_section=source_section,
            target_document_id=None,
            target_document_name=f"Annexure {annex_ref}",
            target_section=None,
            confidence=0.86,
            reason="Annexure reference matched by regex.",
            start_index=match.start(),
            end_index=match.end(),
            chunk_id=chunk_id,
        )

    for match in _APPENDIX_REFERENCE_PATTERN.finditer(text):
        appendix_ref = match.group("appendix_ref")
        _append_reference(
            references,
            seen,
            reference_text=match.group(0),
            reference_type="appendix",
            source_sentence=_source_sentence(text, match.start(), match.end()),
            source_section=source_section,
            target_document_id=None,
            target_document_name=f"Appendix {appendix_ref}",
            target_section=None,
            confidence=0.86,
            reason="Appendix reference matched by regex.",
            start_index=match.start(),
            end_index=match.end(),
            chunk_id=chunk_id,
        )

    for match in _CITATION_REFERENCE_PATTERN.finditer(text):
        citation = match.group("bracket_ref") or match.group("paren_ref")
        if not citation:
            continue
        _append_reference(
            references,
            seen,
            reference_text=match.group(0),
            reference_type="citation",
            source_sentence=_source_sentence(text, match.start(), match.end()),
            source_section=source_section,
            target_document_id=None,
            target_document_name=None,
            target_section=None,
            confidence=0.78,
            reason="Inline citation marker matched by regex.",
            start_index=match.start(),
            end_index=match.end(),
            chunk_id=chunk_id,
        )

    return references


def extract_references_from_chunk(chunk: Chunk) -> list[dict[str, Any]]:
    """Extract references from a single RAG chunk using ``chunk.text``."""
    source_section = (chunk.section_path or chunk.section_name or "").strip()
    return extract_references_from_text(
        chunk.text,
        source_section=source_section,
        chunk_id=chunk.id,
    )


def extract_references_from_chunks(
    chunks: list[Chunk],
    *,
    source_document_id: str,
) -> list[dict[str, Any]]:
    """Run reference extraction for every chunk produced by the RAG chunker."""
    references: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_refs = extract_references_from_chunk(chunk)
        for reference in chunk_refs:
            enriched = dict(reference)
            enriched["source_document_id"] = source_document_id
            enriched["chunk_id"] = chunk.id
            references.append(enriched)
    return references


def aggregate_document_references(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse chunk-level references to unique document-level targets."""
    aggregated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reference in references:
        target_id = reference.get("target_document_id")
        if not target_id:
            continue
        normalized = normalize_reference_id(str(target_id))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        aggregated.append(
            {
                "reference_id": normalized,
                "matched_text": reference.get("reference_text") or reference.get("matched_text"),
                "confidence": float(reference.get("confidence", 0.0)),
                "extraction_method": reference.get("extraction_method", "regex"),
            }
        )
    return aggregated


def extract_document_references(text: str) -> list[dict[str, Any]]:
    """Extract structured document references from raw text using regex patterns."""
    references = extract_references_from_text(text)
    seen: set[str] = set()
    document_references: list[dict[str, Any]] = []
    for reference in references:
        target_id = reference.get("target_document_id")
        if not target_id:
            continue
        normalized = normalize_reference_id(str(target_id))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        document_references.append(
            {
                "reference_id": normalized,
                "matched_text": reference.get("reference_text") or reference.get("matched_text"),
                "start_index": reference.get("start_index", 0),
                "end_index": reference.get("end_index", 0),
                "confidence": float(reference.get("confidence", 0.0)),
                "extraction_method": "regex",
            }
        )
    return document_references


DEFAULT_TRIGGER_PHRASES: tuple[str, ...] = (
    "reference",
    "references",
    "refer",
    "refer to",
    "referred to",
    "ref.",
    "see",
    "see also",
    "according to",
    "as per",
    "in accordance with",
    "follows",
    "follow",
    "following",
    "comply with",
    "complies with",
    "defined in",
    "described in",
    "specified in",
    "documented in",
    "mentioned in",
    "listed in",
    "outlined in",
    "governed by",
    "based on",
    "derived from",
    "related document",
    "related documents",
    "associated document",
    "supporting document",
    "parent document",
    "child document",
    "linked document",
    "annexure",
    "appendix",
    "attachment",
    "work instruction",
    "instructions",
    "gop for",
    "sop for",
    "list of annexures",
    "procedure",
    "policy",
    "guideline",
    "manual",
    "standard",
    "sop",
    "gop",
    "wi",
    "form",
    "doc",
    "ann",
)


def load_trigger_phrases() -> list[str]:
    import os

    raw = (os.getenv("REFERENCE_TRIGGER_PHRASES") or "").strip()
    if not raw:
        return list(DEFAULT_TRIGGER_PHRASES)
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def split_document_text_into_sentences(text: str) -> list[str]:
    """Split document text into extraction units, preserving PDF list/reference lines."""
    if not text or not text.strip():
        return []
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _should_split_line_into_sentences(line):
            parts = re.split(r"(?<=[.!?])\s+(?!\d)", line)
            units.extend(part.strip() for part in parts if part.strip())
        else:
            units.append(line)
    return units


def _should_split_line_into_sentences(line: str) -> bool:
    if _COLON_SUFFIX_DOCUMENT_ID_PATTERN.search(line):
        return False
    if _OUTLINE_LINE_PATTERN.match(line):
        return False
    if len(line) < 200:
        return False
    return bool(re.search(r"[.!?]\s+", line))


def _looks_like_document_id(reference_id: str) -> bool:
    return _looks_like_document_id_impl(reference_id, enforce_hyphen=True)


def _looks_like_document_id_labeled(reference_id: str) -> bool:
    return _looks_like_document_id_impl(reference_id, enforce_hyphen=False)


def _looks_like_document_id_impl(reference_id: str, enforce_hyphen: bool = True) -> bool:
    if not reference_id:
        return False
    normalized = normalize_document_id(reference_id)
    if not normalized or _SECTION_NUMBER_PATTERN.fullmatch(normalized):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)+", normalized):
        return False
    if not re.search(r"[A-Z]", normalized):
        return False
    if not re.search(r"\d", normalized):
        return False
    if enforce_hyphen:
        if normalized.count("-") + normalized.count("/") < 1:
            return False
    return True


def _reference_from_document_id(
    *,
    raw_id: str,
    source_sentence: str,
    reason: str,
    confidence: float,
    extraction_method: str = "pre_chunk_keyword_regex",
) -> dict[str, Any]:
    normalized = normalize_document_id(raw_id)
    extracted_target_id = re.sub(r"\s+", "", raw_id.strip())
    return {
        "reference_text": extracted_target_id,
        "reference_type": "document",
        "source_sentence": source_sentence,
        "target_document_id": normalized,
        "normalized_reference": normalized,
        "extracted_target_id": extracted_target_id,
        "normalized_target_id": normalized,
        "confidence": confidence,
        "reason": reason,
        "extraction_method": extraction_method,
    }


def extract_colon_suffixed_document_references(sentence: str) -> list[dict[str, Any]]:
    """Extract document IDs that appear after a colon at the end of a line."""
    line = sentence.strip()
    if not line:
        return []
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _COLON_SUFFIX_DOCUMENT_ID_PATTERN.finditer(line):
        raw_id = match.group("reference_id")
        normalized = normalize_document_id(raw_id)
        if not normalized or not _looks_like_document_id(normalized) or normalized in seen:
            continue
        seen.add(normalized)
        collected.append(
            _reference_from_document_id(
                raw_id=raw_id,
                source_sentence=line,
                reason="Document ID matched after colon suffix.",
                confidence=0.9,
            )
        )
    return collected


def blocks_to_document_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        text = str(getattr(block, "text", "") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _trigger_phrase_in_sentence(sentence: str, phrase: str) -> bool:
    if not phrase:
        return False
    if len(phrase) <= 4:
        # Avoid matching short tokens embedded in hyphenated document IDs (e.g. GOP in GL-CQA-GOP-0030).
        return bool(
            re.search(
                rf"(?:^|[^\w-]){re.escape(phrase)}(?:[^\w-]|$)",
                sentence,
                re.IGNORECASE,
            )
        )
    return phrase.lower() in sentence.lower()


def _find_trigger_phrase(sentence: str, trigger_phrases: list[str]) -> str | None:
    matches = [phrase for phrase in trigger_phrases if _trigger_phrase_in_sentence(sentence, phrase)]
    if not matches:
        return None
    return max(matches, key=len)


def extract_document_target_ids_from_sentence(sentence: str) -> list[dict[str, Any]]:
    """Collect every document ID mentioned in a trigger sentence."""
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    sentence_refs = extract_references_from_text(sentence, source_section="Document")
    for reference in sentence_refs:
        target_id = reference.get("target_document_id")
        if not target_id:
            continue
        raw_id = str(target_id).strip()
        normalized = normalize_document_id(raw_id)
        if not normalized or not _looks_like_document_id(normalized) or normalized in seen:
            continue
        seen.add(normalized)
        extracted_target_id = re.sub(r"\s+", "", raw_id)
        collected.append(
            {
                **reference,
                "normalized_reference": normalized,
                "extracted_target_id": extracted_target_id,
                "normalized_target_id": normalized,
            }
        )

    for match in _BARE_DOCUMENT_ID_PATTERN.finditer(sentence):
        raw_id = match.group("reference_id")
        normalized = normalize_document_id(raw_id)
        if not normalized or not _looks_like_document_id(normalized) or normalized in seen:
            continue
        seen.add(normalized)
        extracted_target_id = re.sub(r"\s+", "", raw_id)
        collected.append(
            {
                "reference_text": match.group(0),
                "reference_type": "document",
                "source_sentence": sentence,
                "target_document_id": normalized,
                "normalized_reference": normalized,
                "extracted_target_id": extracted_target_id,
                "normalized_target_id": normalized,
                "confidence": 0.85,
                "reason": "Bare document ID matched in trigger sentence.",
                "extraction_method": "pre_chunk_keyword_regex",
            }
        )
    return collected


def extract_pre_chunk_references(
    text: str,
    *,
    source_document_id: str,
    source_document_name: str | None = None,
    tenant_id: str | None = None,
    repository_id: str | None = None,
    trigger_phrases: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract document-level references from sentences containing trigger phrases or explicit labels."""
    phrases = trigger_phrases or load_trigger_phrases()
    references_by_key: dict[str, dict[str, Any]] = {}

    for sentence in split_document_text_into_sentences(text):
        sentence_refs: list[dict[str, Any]] = []
        seen: set[str] = set()

        # 1. Extract explicitly labeled references or implicit references from extract_references_from_text
        # These do NOT require trigger phrases since they are already explicitly matched via patterns.
        raw_refs = extract_references_from_text(sentence, source_section="Document")
        for ref in raw_refs:
            ref_type = ref.get("reference_type")
            if ref_type not in {"document", "sop", "work_instruction", "form", "protocol", "report"}:
                continue
            target_id = ref.get("target_document_id")
            if not target_id:
                continue
            raw_id = str(target_id).strip()
            normalized = normalize_document_id(raw_id)
            if not normalized or normalized in seen:
                continue
            
            # Use labeled variant (no hyphen requirement)
            if _looks_like_document_id_labeled(normalized):
                seen.add(normalized)
                extracted_target_id = re.sub(r"\s+", "", raw_id)
                sentence_refs.append(
                    {
                        **ref,
                        "normalized_reference": normalized,
                        "extracted_target_id": extracted_target_id,
                        "normalized_target_id": normalized,
                    }
                )

        # 2. Extract bare document IDs only if a trigger phrase is present in the sentence
        trigger_phrase = _find_trigger_phrase(sentence, phrases)
        if trigger_phrase:
            for match in _BARE_DOCUMENT_ID_PATTERN.finditer(sentence):
                raw_id = match.group("reference_id")
                normalized = normalize_document_id(raw_id)
                if not normalized or not _looks_like_document_id(normalized) or normalized in seen:
                    continue
                seen.add(normalized)
                extracted_target_id = re.sub(r"\s+", "", raw_id)
                sentence_refs.append(
                    {
                        "reference_text": match.group(0),
                        "reference_type": "document",
                        "source_sentence": sentence,
                        "target_document_id": normalized,
                        "normalized_reference": normalized,
                        "extracted_target_id": extracted_target_id,
                        "normalized_target_id": normalized,
                        "confidence": 0.85,
                        "reason": "Bare document ID matched in trigger sentence.",
                        "extraction_method": "pre_chunk_keyword_regex",
                    }
                )

        # 3. Extract colon-suffixed document IDs (which are also strong indicators at end-of-line)
        for colon_ref in extract_colon_suffixed_document_references(sentence):
            normalized = colon_ref.get("normalized_target_id")
            if normalized and normalized not in seen:
                seen.add(normalized)
                sentence_refs.append(colon_ref)

        if not sentence_refs:
            continue

        effective_trigger = trigger_phrase or "explicit_label"
        regex_matches = [item.get("extracted_target_id") for item in sentence_refs]
        log.info(
            "reference_extraction_sentence source_document_id=%s source_document_name=%s "
            "source_sentence=%r trigger_phrase=%s regex_matches=%s",
            source_document_id,
            source_document_name,
            sentence,
            effective_trigger,
            regex_matches,
        )

        for reference in sentence_refs:
            extracted_target_id = reference.get("extracted_target_id") or reference.get("normalized_reference")
            if not extracted_target_id:
                continue
            normalized_target_id = normalize_document_id(
                reference.get("normalized_target_id") or extracted_target_id
            )
            if not normalized_target_id:
                continue

            # Correct self-reference check
            if is_same_document_id(normalized_target_id, source_document_id) or is_same_document_id(normalized_target_id, source_document_name):
                log.info(
                    "self_reference_skipped source_document_id=%s extracted_target_id=%s source_sentence=%r",
                    source_document_id,
                    extracted_target_id,
                    sentence,
                )
                continue

            dedupe_key = f"{source_document_id}::{normalized_target_id}"
            if dedupe_key in references_by_key:
                log.info(
                    "duplicate_reference_key reference_key=%s source_document_id=%s extracted_target_id=%s",
                    dedupe_key,
                    source_document_id,
                    extracted_target_id,
                )

            existing = references_by_key.get(dedupe_key)
            ref_trigger = trigger_phrase if trigger_phrase else reference.get("trigger_phrase", effective_trigger)
            enriched = {
                **reference,
                "source_document_id": source_document_id,
                "normalized_reference": normalized_target_id,
                "extracted_target_id": extracted_target_id,
                "normalized_target_id": normalized_target_id,
                "trigger_phrase": ref_trigger,
                "source_scope": "document",
                "extraction_method": "pre_chunk_keyword_regex",
                "tenant_id": tenant_id,
                "repository_id": repository_id,
                "document_id": source_document_id,
                "target_document_id": normalized_target_id,
            }
            log.info(
                "reference_extracted source_document_id=%s extracted_target_id=%s normalized_target_id=%s trigger_phrase=%s",
                source_document_id,
                extracted_target_id,
                normalized_target_id,
                ref_trigger,
            )
            if existing is None:
                enriched["evidence_sentences"] = [sentence]
                references_by_key[dedupe_key] = enriched
            else:
                evidence = list(existing.get("evidence_sentences") or [])
                if sentence not in evidence:
                    evidence.append(sentence)
                existing["evidence_sentences"] = evidence
                references_by_key[dedupe_key] = existing
    return list(references_by_key.values())
