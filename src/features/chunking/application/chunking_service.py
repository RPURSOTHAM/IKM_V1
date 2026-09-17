"""
chunking.py
===========
Section-aware document chunking with context preservation.

PREVIOUS CHUNKING METHOD
-------------------------
The original approach in build_index_and_similarity_report.py used:
1. Pure word-count accumulation: sentences were concatenated until reaching
   a 150-word limit, then flushed as a chunk.
2. No overlap: consecutive chunks shared zero context. A concept split across
   the boundary was lost to both chunks.
3. No context metadata: chunks carried only doc_name, page, and section_name.
   Readers of a chunk had no idea WHERE in the document hierarchy it came from.
4. Paragraph boundaries were respected (double-newline splits) but within a
   paragraph, the chunker simply packed sentences by word count.

IMPROVED CHUNKING METHOD
--------------------------
This module implements section-wise chunking with context preservation:

1. **Section Boundary Respect**: Each section is chunked independently.
   A chunk never crosses a section boundary, preserving coherent topical units.

2. **Sentence-Level Overlap**: Consecutive chunks share `overlap_sentences`
   (default: 2) trailing sentences from the previous chunk.

3. **Context Metadata**: Every chunk carries a `section_path` string
   (e.g., "PROCEDURE > Cleaning Steps > Step 3") giving the full heading
   hierarchy.

4. **Improved Sentence Splitting**: Abbreviations (e.g., "i.e.", "Dr.")
   are protected from false sentence splits.

5. **Configurable Parameters**:
   - chunk_size: max words per chunk (default 150)
   - overlap_sentences: number of sentences to repeat (default 2)
   - min_content_words: minimum words for a chunk to be kept
"""

from __future__ import annotations

import os
import re
import uuid
import json
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.features.document_processing.core.logger import log
from src.features.document_processing.loaders.docxloader import DocxLoader
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdfloader import PdfLoader
from src.features.document_processing.loaders.component_classification import (
    COMPONENT_FOOTER,
    COMPONENT_HEADER,
    COMPONENT_IMAGE,
    COMPONENT_SUBTITLE,
    COMPONENT_TABLE,
    COMPONENT_TITLE,
    block_component_type,
    blocks_for_chunking,
    is_image_block,
)
from src.features.document_processing.utilities.postprocessing import PostProcessor
from src.features.document_processing.utilities.text_utils import (
    is_body_line,
    split_into_sentences,
    split_into_paragraphs,
    normalize_text_for_embedding,
    auto_detect_sections,
)
from src.shared.semantic.models import SemanticChunk


_MIN_CHUNK_WORD_FLOOR = 1

_SIGNATURE_METADATA_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bprepared\s+by\b",
        r"\bprepared\s*by",
        r"\breviewed\s+by\b",
        r"\breviewed\s*by",
        r"\bapproved\s+by\b",
        r"\bapproved\s*by",
        r"\bdate\s*&\s*time\b",
        r"\bprint\s*id\b",
        r"\bprinted\s+by\b",
        r"\bprinted\s+on\b",
        r"\bcopy\s+no\b",
        r"\bcopy\s+to\b",
        r"\bprinter\s+name\b",
        r"\bdriver\s+name\b",
        r"\bprint\s+type\b",
        r"\belectronically\s+generated\s+document\b",
        r"\bdoes\s+not\s+require\s+a\s+signature\b",
    )
]

_APPROVAL_ROW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*name\b",
        r"^\s*role\b",
        r"^\s*role\s*initiator",
        r"^\s*date\s*&\s*time\b",
    )
]

_TOC_TITLE_PATTERN = re.compile(
    r"^\s*(?:table\s+of\s+contents|contents|table\s+of\s+tables|list\s+of\s+tables)\s*$",
    re.IGNORECASE,
)
_TOC_TITLE_SEARCH_PATTERN = re.compile(
    r"\b(?:table\s+of\s+contents|table\s+of\s+tables|list\s+of\s+tables)\b",
    re.IGNORECASE,
)

_TOC_ENTRY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "1. PURPOSE..........................3"
        r"^\s*\d+(?:\.\d+)*\.?\s+\S.{1,160}?\.{2,}\s*\d+\s*$",
        # "PURPOSE............................3"
        r"^\s*[A-Z][A-Z0-9/&(),'\- ]{1,120}\.{2,}\s*\d+\s*$",
        # "Table 1: Solvent quantities........5"
        r"^\s*table\s+\d+[A-Za-z]?\s*[:\-.]\s+\S.{1,160}?\.{2,}\s*\d+\s*$",
        # Extractors sometimes collapse dotted leaders to whitespace.
        # Keep this constrained to TOC pages only via _looks_like_toc_page.
        r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9/&(),'\- ]{2,160}\s+\d+\s*$",
        r"^\s*table\s+\d+[A-Za-z]?\s*[:\-.]\s+[A-Z][A-Za-z0-9/&(),'\- ]{2,160}\s+\d+\s*$",
    )
]

_IMAGE_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:"
    r"(?:in\s+)?document\s+image\s+(?P<image_index_a>\d+)"
    r"|image\s+(?P<image_index_b>\d+)\s+in\s+document"
    r")\b",
    re.IGNORECASE,
)

_COVER_METADATA_LABELS = (
    "title",
    "document no",
    "document no.",
    "doc no",
    "sop no",
    "version no",
    "version no.",
    "effective date",
    "review date",
)

_BUSINESS_METADATA_LABELS = (
    "sub department",
    "sub-department",
    "department",
    "facility",
    "site",
    "unit",
    "unit name",
)


def _looks_like_cover_page_metadata(text: str) -> bool:
    """Detect cover/header metadata tables without removing real body sections."""
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    label_hits = sum(1 for label in _COVER_METADATA_LABELS if re.search(rf"\b{re.escape(label)}\b", normalized))
    if label_hits >= 2:
        return True
    if "global operating procedure" in normalized and label_hits >= 1:
        return True
    if re.match(r"^title\s*:\s*\S", normalized) and len(normalized.split()) <= 8:
        return True
    return False


def _extract_business_metadata_text(text: str) -> str:
    """Keep useful business scope metadata while dropping noisy cover metadata."""
    raw = " ".join((text or "").replace("|", " | ").split())
    if not raw:
        return ""
    raw = re.sub(r"AssuraSnuceb\s+Department", "Assurance Sub Department", raw, flags=re.IGNORECASE)
    raw = re.sub(r"AssuraT\*?nce", "Assurance", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+\*+\s*", " ", raw)

    label_display = {
        "facility": "Facility",
        "site": "Site",
        "unit": "Unit",
        "unit name": "Unit Name",
        "department": "Department",
        "sub department": "Sub Department",
        "sub-department": "Sub Department",
    }
    labels = sorted(_BUSINESS_METADATA_LABELS, key=len, reverse=True)
    label_pattern = "|".join(re.escape(label) for label in labels)

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    pipe_parts = [part.strip() for part in raw.split("|") if part.strip()]
    index = 0
    while index < len(pipe_parts):
        key = pipe_parts[index].lower().rstrip(":")
        if key in label_display and index + 1 < len(pipe_parts):
            canonical = label_display[key]
            value = pipe_parts[index + 1].strip()
            if value and canonical not in seen:
                pairs.append((canonical, value))
                seen.add(canonical)
            index += 2
            continue
        index += 1

    for match in re.finditer(
        rf"\b(?P<label>{label_pattern})\b\s*:?\s*(?P<value>.*?)(?=\s+\b(?:{label_pattern})\b\s*:?|$)",
        raw,
        re.IGNORECASE,
    ):
        label_key = match.group("label").lower().rstrip(":")
        canonical = label_display.get(label_key)
        value = " ".join(match.group("value").strip(" :|-").split())
        if canonical and value and canonical not in seen:
            pairs.append((canonical, value))
            seen.add(canonical)

    labels_found = {label for label, _ in pairs}
    if not ({"Facility", "Sub Department"} & labels_found) and len(pairs) < 2:
        return ""
    values_by_label = {label: value for label, value in pairs}
    if "Department" in values_by_label and len(values_by_label["Department"].split()) < 2:
        return ""
    if "Sub Department" in values_by_label and len(values_by_label["Sub Department"].split()) < 2:
        return ""
    return "\n".join(f"{label}: {value}" for label, value in pairs)


def _looks_like_running_header_footer(text: str) -> bool:
    """Detect repeated PDF running headers/footers and broken title fragments."""
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return False
    lower = normalized.lower()
    if normalized in {")", "):7"}:
        return True
    if lower == "jar":
        return True
    if lower == "release pending" or lower.startswith("release pending title:"):
        return True
    if lower.startswith("confidential work product"):
        return True
    if "grab your reader" in lower or "place this text box" in lower:
        return True
    if lower in {"great quote from the document or", "use this space to emphasize a key", "anywhere on the page, just drag it."}:
        return True
    if lower.startswith("global operating procedure"):
        return True
    if "jar" in lower and "gis" in lower and len(normalized.split()) <= 6:
        return True
    if re.fullmatch(r"\d+\s+of\s+trend\)?(?::\d+)?", lower):
        return True
    if re.search(r"\bof\s+trend\)?(?::\d+)?\b", lower) and len(normalized.split()) <= 6:
        return True
    if re.fullmatch(r"(?:u\s+\*|illiP\s+ortc|/u\s+en)", normalized, flags=re.IGNORECASE):
        return True
    return False


def _strip_image_placeholders(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove OCR image placeholder labels while preserving image counts as metadata."""
    image_refs: list[dict[str, Any]] = []

    def _replace(match: re.Match[str]) -> str:
        raw_index = match.group("image_index_a") or match.group("image_index_b")
        try:
            image_index: int | str = int(raw_index)
        except (TypeError, ValueError):
            image_index = raw_index
        image_refs.append({"image_index": image_index, "source": "text_placeholder"})
        return " "

    cleaned = _IMAGE_PLACEHOLDER_PATTERN.sub(_replace, text)
    cleaned_lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in cleaned.splitlines()]
    cleaned = "\n".join(line for line in cleaned_lines if line).strip()
    return cleaned, image_refs


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────


Chunk = SemanticChunk


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _emit_long_sentence(
    sent: str,
    chunk_size: int,
    chunks: list[str],
) -> None:
    """
    ✅ FIX 1: Handle sentences longer than chunk_size by splitting them into
    multiple word-boundary sub-chunks.

    The original code did `words[:chunk_size]` and discarded everything after
    the first chunk_size words — silently losing the tail of any long sentence.
    """
    words = sent.split()
    for w_start in range(0, len(words), chunk_size):
        sub = " ".join(words[w_start : w_start + chunk_size])
        if sub:
            chunks.append(sub)


def _merge_undersized_chunks(chunks: list[str], min_words: int) -> list[str]:
    """Merge undersized chunks into adjacent chunks from the same section."""
    if min_words <= 1 or len(chunks) <= 1:
        return [chunk for chunk in chunks if chunk.strip()]

    merged: list[str] = []
    carry = ""

    for chunk in chunks:
        current = " ".join(part for part in (carry, chunk) if part).strip()
        if not current:
            continue

        if len(current.split()) < min_words:
            carry = current
            continue

        merged.append(current)
        carry = ""

    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}".strip()
        else:
            merged.append(carry)

    return merged


def _word_count(text: str) -> int:
    return len((text or "").split())


def _looks_like_extraction_artifact_line(text: str) -> bool:
    normalized = " ".join((text or "").strip().split())
    if not normalized:
        return True
    if normalized.upper() in {"OOS", "OOT", "QMS", "CAPA", "OOAC"}:
        return False
    if _looks_like_running_header_footer(normalized):
        return True
    if _looks_like_cover_page_metadata(normalized):
        return True
    artifact_tokens = normalized.split()
    if (
        len(artifact_tokens) >= 2
        and all(re.fullmatch(r"(?:[A-Za-z]|\d|/\d+|[:/\\]+)", token) for token in artifact_tokens)
    ):
        return True
    if re.search(r"\b(?:title|document\s+no|reference|effective\s+date|review\s+date)\b", normalized, re.IGNORECASE):
        # Keep genuine body references, but drop short cover/header metadata fragments.
        return len(normalized.split()) <= 8
    if re.fullmatch(r"\d", normalized):
        return True
    if re.fullmatch(r"[A-Za-z]", normalized):
        return True
    if re.fullmatch(r"/\d+", normalized):
        return True
    alpha_tokens = re.findall(r"[A-Za-z]+", normalized)
    if alpha_tokens and all(token.upper() in {"OOS", "OOT", "QMS", "CAPA", "OOAC"} for token in alpha_tokens):
        return False
    if 2 <= len(alpha_tokens) <= 3 and all(len(token) <= 3 for token in alpha_tokens):
        return True
    if 2 <= len(alpha_tokens) <= 5 and all(len(token) <= 4 for token in alpha_tokens) and not re.search(r"\d", normalized):
        return True
    if len(alpha_tokens) >= 4:
        short_ratio = sum(1 for token in alpha_tokens if len(token) <= 3) / len(alpha_tokens)
        has_long_token = any(len(token) >= 6 for token in alpha_tokens)
        if short_ratio >= 0.7 and not has_long_token:
            return True
    if re.fullmatch(r"(?:[A-Z]\s+){2,}[A-Z]?", normalized):
        return True
    return False


def _clean_extracted_text_line(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    cleaned = re.sub(
        r"\bRelease\s+Pending\s+Title:\s+MANAGEMENT\s+OF\s+OOS\s*\(OUT\s+OF\s+SPECIFICATION\)\s+AND\s+OOT\s*\(OUT(?::\d+)?\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bGLOBAL\s+OPERATING\s+PROCEDURE\s+Title:\s+MANAGEMENT\s+OF\s+OOS\s*\(OUT\s+OF\s+SPECIFICATION\)\s+AND\s+OOT\s*\(OUT(?::\d+)?\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bConfidential\s+work\s+product(?:\s+Page\s+\d+\s+of\s+\d+)?\s+Title:\s+MANAGEMENT\s+OF\s+OOS\s*\(OUT\s+OF\s+SPECIFICATION\)\s+AND\s+OOT\s*\(OUT(?::\d+)?\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b\d+\s+OF\s+TREND\)?\s*:?\s*\d*\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\):\d+\b", " ", cleaned)
    cleaned = re.sub(r"\bjar\s+(?=[a-z]\.)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+gis\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^/\d+\s+", "", cleaned)
    cleaned = re.sub(r"^[:/\\]?\d+\s+(?=(?:title|document\s+no|reference)\b)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[:/\\]?\d+\s*(?=(?:title|document\s+no|reference)\b)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[:/\\]+\d*\s*(?=(?:title|document\s+no|reference)\b)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\d+\s+(?=[a-z]\.)", "", cleaned)
    cleaned = re.sub(r"^[a-z]{1,2}\s+(?=[a-z]\.)", "", cleaned)
    return " ".join(cleaned.split()).strip()


def _looks_like_instruction_line(text: str) -> bool:
    cleaned = (text or "").strip()
    return bool(
        re.match(r"^\d+\.\s+\S", cleaned)
        or re.match(r"^[a-z]\.\s+\S", cleaned, re.IGNORECASE)
        or re.match(r"^(?:i{1,3}|iv|v|vi{0,3}|ix|x)\.\s+\S", cleaned, re.IGNORECASE)
    )


def _optional_block_page(block: Any) -> int | None:
    """Read extracted page provenance; return None when absent (never invent 1)."""
    raw = getattr(block, "page", None)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _with_section_heading(raw: str, section_name: str) -> str:
    heading = " ".join((section_name or "").split()).strip()
    text = (raw or "").strip()
    if not heading or heading.lower() == "document" or _looks_like_extraction_artifact_line(heading):
        return text
    if heading.lower() in text[: max(len(heading) + 20, 80)].lower():
        return text
    return f"{heading}\n{text}".strip()


def _safe_section_label(section_name: str) -> str:
    label = " ".join((section_name or "").split()).strip()
    if not label or label.lower() == "document":
        return ""
    if label in {"*", ")", "):7"}:
        return ""
    if re.fullmatch(r"(?:Assurance|Department|Sub Department|Corporate Quality|\*)[\w\s*]*", label, re.IGNORECASE):
        return ""
    if re.search(r"\b(?:jar|gis|illiP|ortc|llac|nta)\b", label, re.IGNORECASE):
        return ""
    if _looks_like_cover_page_metadata(label) or _looks_like_extraction_artifact_line(label):
        return ""
    return label


def _prepare_blocks_for_section_detection(blocks: list[Any]) -> list[Any]:
    prepared: list[Any] = []
    for block in blocks:
        block_type = str(getattr(block, "block_type", "text") or "text").lower()
        component_type = block_component_type(block)
        block_text = str(getattr(block, "text", "") or "")
        block_page = _optional_block_page(block)
        metadata = dict(getattr(block, "metadata", {}) or {})
        region = str(metadata.get("region") or "").strip().lower()
        business_metadata = _extract_business_metadata_text(block_text) if block_page == 1 else ""
        if business_metadata:
            try:
                block.text = business_metadata
                block.component_type = COMPONENT_TABLE
                block.block_type = "table"
                block.heading_level = None
                metadata = dict(getattr(block, "metadata", {}) or {})
                metadata["component_type"] = COMPONENT_TABLE
                metadata["section_name"] = "Document Metadata"
                metadata["preserved_business_metadata"] = True
                metadata.pop("heading_level", None)
                block.metadata = metadata
            except Exception:
                pass
            prepared.append(block)
            continue
        # Do not let PDF margin content update section context.  A repeated
        # header/footer may be visually bold or centre-aligned and therefore
        # arrive pre-classified as a title by an extractor.  Region provenance
        # takes precedence over that heuristic.
        if component_type in {COMPONENT_HEADER, COMPONENT_FOOTER} or region in {"header", "footer"}:
            continue
        if block_type == "table" or component_type in {COMPONENT_TABLE, COMPONENT_IMAGE}:
            if component_type == COMPONENT_TABLE and (
                _looks_like_cover_page_metadata(block_text) or _looks_like_running_header_footer(block_text)
            ):
                business_metadata = _extract_business_metadata_text(block_text) if block_page == 1 else ""
                if business_metadata:
                    try:
                        block.text = business_metadata
                        block.component_type = COMPONENT_TABLE
                        block.block_type = "table"
                        block.heading_level = None
                        metadata = dict(getattr(block, "metadata", {}) or {})
                        metadata["component_type"] = COMPONENT_TABLE
                        metadata["section_name"] = "Document Metadata"
                        metadata["preserved_business_metadata"] = True
                        metadata.pop("heading_level", None)
                        block.metadata = metadata
                    except Exception:
                        pass
                    prepared.append(block)
                continue
            prepared.append(block)
            continue
        cleaned = _clean_extracted_text_line(str(getattr(block, "text", "") or ""))
        if _looks_like_cover_page_metadata(cleaned) or _looks_like_running_header_footer(cleaned):
            continue
        if _looks_like_extraction_artifact_line(cleaned):
            continue
        if re.match(r"^title\s*:\s+\S", cleaned, re.IGNORECASE):
            try:
                block.component_type = "paragraph"
                block.heading_level = None
                metadata = dict(getattr(block, "metadata", {}) or {})
                metadata["component_type"] = "paragraph"
                metadata.pop("heading_level", None)
                block.metadata = metadata
            except Exception:
                pass
        try:
            block.text = cleaned
        except Exception:
            pass
        prepared.append(block)
    return prepared


def _merge_metadata_label(left: str, right: str) -> str:
    """Keep all source section labels when chunks are merged across sections."""
    labels: list[str] = []
    for value in (left, right):
        for part in re.split(r"\s*/\s*", value or ""):
            part = part.strip()
            if part and part not in labels:
                labels.append(part)
    return " / ".join(labels)


def _section_leaf_label(section_path: str) -> str:
    """Return the visible section label from a hierarchical section path."""
    parts = [part.strip() for part in re.split(r"\s*>\s*", section_path or "") if part.strip()]
    if not parts:
        return ""
    for part in reversed(parts):
        safe = _safe_section_label(part)
        if safe:
            return safe
    return ""


def _looks_like_primary_section_label(section_name: str) -> bool:
    label = (section_name or "").strip()
    if not label or label.lower() == "document":
        return False

    if " / " in label:
        return False

    numbered = re.match(r"^\s*(?P<number>\d+(?:\.\d+)*\.?)\s*(?P<title>[A-Za-z].*)$", label)
    if numbered:
        number = numbered.group("number").rstrip(".")
        parts = [part for part in number.split(".") if part]
        if not parts or (len(parts) > 1 and parts[-1] != "0"):
            return False
        title = numbered.group("title").strip().rstrip(":")
        title_words = title.split()
        return 1 <= len(title_words) <= 10 and (title.isupper() or title.istitle())

    words = label.split()
    return 1 <= len(words) <= 10 and (label.isupper() or label.istitle())


def _should_preserve_short_section(chunk: Chunk, word_count: int) -> bool:
    """Keep valid small sections as their own chunks instead of merging them away."""
    if not _looks_like_primary_section_label(chunk.section_name):
        return False
    if word_count < 8:
        return False

    text = (chunk.text or "").strip()
    if not text:
        return False
    if re.fullmatch(r"[\W_]+", text):
        return False
    return bool(re.search(r"[A-Za-z]{3,}", text))


def _same_section(left: Chunk, right: Chunk) -> bool:
    return (
        (left.section_name or "").strip() == (right.section_name or "").strip()
        and (left.section_path or "").strip() == (right.section_path or "").strip()
    )


def _combine_chunk_objects(left: Chunk, right: Chunk) -> Chunk:
    """Append a neighboring chunk to another while preserving the left chunk id."""
    right_text = (right.text or "").strip()
    if not right_text:
        return left

    left.text = " ".join(part for part in ((left.text or "").strip(), right_text) if part).strip()
    left.raw_text = " ".join(
        part for part in ((left.raw_text or "").strip(), (right.raw_text or right_text).strip()) if part
    ).strip()

    left_start_raw = left.page_start if left.page_start is not None else left.page
    right_start_raw = right.page_start if right.page_start is not None else right.page
    left_end_raw = left.end_page if left.end_page is not None else left.page_end
    if left_end_raw is None:
        left_end_raw = left.page
    right_end_raw = right.end_page if right.end_page is not None else right.page_end
    if right_end_raw is None:
        right_end_raw = right.page

    page_values = [
        int(value)
        for value in (left_start_raw, right_start_raw, left_end_raw, right_end_raw)
        if value is not None and value != ""
    ]
    if page_values:
        merged_start = min(page_values)
        merged_end = max(page_values)
        left.page = merged_start
        left.page_start = merged_start
        left.end_page = merged_end
        left.page_end = merged_end
        left_start = int(left_start_raw) if left_start_raw is not None else merged_start
        right_start = int(right_start_raw) if right_start_raw is not None else merged_start
        if left_start == right_start == merged_end:
            starts = [value for value in (left.line_start, right.line_start) if value is not None]
            ends = [value for value in (left.line_end, right.line_end) if value is not None]
            left.line_start = min(starts) if starts else left.line_start
            left.line_end = max(ends) if ends else left.line_end
        elif left_start <= right_start:
            if left.line_start is None:
                left.line_start = right.line_start
            if left.line_end is None and left_start == right_start:
                left.line_end = right.line_end

    left_blocks = list(getattr(left, "source_blocks", None) or [])
    right_blocks = list(getattr(right, "source_blocks", None) or [])
    if right_blocks:
        left.source_blocks = left_blocks + right_blocks

    left.section_name = _merge_metadata_label(left.section_name, right.section_name)
    left.section_path = _merge_metadata_label(left.section_path or left.section_name, right.section_path or right.section_name)
    left.content_types = sorted(set(left.content_types or []) | set(right.content_types or []))
    left.table_count = int(left.table_count or 0) + int(right.table_count or 0)
    left.image_count = int(left.image_count or 0) + int(right.image_count or 0)
    left.extraction_metadata = {
        **(left.extraction_metadata or {}),
        "tables": [
            *((left.extraction_metadata or {}).get("tables") or []),
            *((right.extraction_metadata or {}).get("tables") or []),
        ],
        "images": [
            *((left.extraction_metadata or {}).get("images") or []),
            *((right.extraction_metadata or {}).get("images") or []),
        ],
    }

    return left


def _merge_undersized_chunk_objects(
    chunks: list[Chunk],
    min_words: int,
    chunk_size: int,
) -> list[Chunk]:
    """Merge undersized chunks only within the same section.

    Section boundaries are hard chunk boundaries. Even a short section must stay
    separate so retrieval and UI metadata do not mix neighboring sections.
    """
    if not chunks:
        return []

    effective_min_words = max(_MIN_CHUNK_WORD_FLOOR, min_words)
    effective_max_words = max(chunk_size, effective_min_words)
    soft_max_words = int(effective_max_words * 1.25)

    merged: list[Chunk] = []

    for chunk in chunks:
        if not (chunk.text or "").strip():
            continue

        current_words = _word_count(chunk.text)
        if (
            current_words < effective_min_words
            and merged
            and _same_section(merged[-1], chunk)
            and _word_count(merged[-1].text) + current_words <= soft_max_words
        ):
            _combine_chunk_objects(merged[-1], chunk)
            continue

        merged.append(chunk)

    return merged


def _font_matches(block: Any, font_cfg: dict[str, Any]) -> bool:
    """Return True when a block satisfies configured font constraints."""
    if not font_cfg:
        return True

    size_min = font_cfg.get("size_min")
    size_max = font_cfg.get("size_max")
    req_bold = font_cfg.get("bold")

    if size_min and block.font_size > 0 and block.font_size < size_min:
        return False
    if size_max and block.font_size > 0 and block.font_size > size_max:
        return False
    if req_bold is not None and block.bold != req_bold:
        return False
    return True


def _build_heading_matchers(section_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile heading-tree rules into matchers reusable by the chunker."""
    heading_tree = section_cfg.get("output", {}).get("heading_tree", {})
    levels_cfg = heading_tree.get("levels", [])
    matchers: list[dict[str, Any]] = []

    for level_cfg in sorted(levels_cfg, key=lambda item: item.get("level", 99)):
        detection = level_cfg.get("detection", {})
        pattern_text = detection.get("text_pattern", "")
        try:
            pattern = re.compile(pattern_text, re.IGNORECASE) if pattern_text else None
        except re.error:
            pattern = None

        matchers.append(
            {
                "level": level_cfg.get("level"),
                "styles": [style.lower() for style in detection.get("styles", [])],
                "font": detection.get("font", {}),
                "alignment": detection.get("alignment", ""),
                "pattern": pattern,
            }
        )

    return matchers


def _classify_heading_level(block: Any, matchers: list[dict[str, Any]]) -> int | None:
    """Return the configured heading level for a block, if it matches one."""
    for matcher in matchers:
        if matcher["styles"] and block.style.lower() not in matcher["styles"]:
            continue
        if not _font_matches(block, matcher["font"]):
            continue
        if matcher["alignment"] and block.alignment != matcher["alignment"]:
            continue
        if matcher["pattern"] and not matcher["pattern"].match(block.text.strip()):
            continue
        return matcher["level"]
    return None


def _split_blocks_by_heading_rules(
    sec_name: str,
    sec_cfg: dict[str, Any],
    sec_blocks: list[Any],
) -> list[tuple[str, list[Any]]]:
    """
    Split a template section into heading-aware segments using the same
    rule-based heading definitions as metadata extraction.
    """
    matchers = _build_heading_matchers(sec_cfg)
    if not matchers or not sec_blocks:
        return []

    segments: list[tuple[str, list[Any]]] = []
    heading_stack: list[tuple[int, str]] = []
    current_blocks: list[Any] = []

    def current_path() -> str:
        labels = [label for _, label in heading_stack if label]
        return " > ".join(labels) if labels else sec_name

    def flush_segment() -> None:
        nonlocal current_blocks
        if not current_blocks:
            return
        segments.append((current_path(), current_blocks))
        current_blocks = []

    for block in sec_blocks:
        level = _classify_heading_level(block, matchers)
        if level is not None:
            flush_segment()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.text.strip()))
            continue
        current_blocks.append(block)

    flush_segment()
    return segments


def _normalize_with_positions(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while keeping each normalized char mapped to original text."""
    normalized: list[str] = []
    positions: list[int] = []
    in_space = False
    for idx, ch in enumerate(text):
        if ch.isspace():
            if normalized and not in_space:
                normalized.append(" ")
                positions.append(idx)
            in_space = True
            continue
        normalized.append(ch)
        positions.append(idx)
        in_space = False
    if normalized and normalized[-1] == " ":
        normalized.pop()
        positions.pop()
    return "".join(normalized), positions


def _page_range_for_blocks(blocks: list[Any]) -> tuple[int | None, int | None]:
    pages = sorted(
        {
            int(getattr(block, "page"))
            for block in blocks
            if getattr(block, "page", None) is not None and getattr(block, "page", "") != ""
        }
    )
    if not pages:
        return None, None
    return min(pages), max(pages)


def _sort_lines_by_offset(lines_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve document order for line-to-chunk mapping."""
    return sorted(lines_info, key=lambda line: int(line.get("start", 0)))


def _table_batch_line_range(
    line_start_base: int,
    line_end_base: int,
    batch_start_row: int,
    batch_end_row: int,
    total_rows: int,
) -> tuple[int, int]:
    """Distribute a table's source-line range proportionally across row batches."""
    span = line_end_base - line_start_base + 1
    start = line_start_base + int(batch_start_row / total_rows * span)
    end = line_start_base + int((batch_end_row + 1) / total_rows * span) - 1
    return start, min(end, line_end_base)


def _page_range_from_lines(lines_info: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Return start/end pages from matched source lines (never invent page 1)."""
    pages: list[int] = []
    for line in lines_info:
        raw = line.get("page")
        if raw is None or raw == "":
            continue
        try:
            pages.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not pages:
        return None, None
    return min(pages), max(pages)


def _lines_overlapping_span(lines_info: list[dict[str, Any]], start_pos: int, end_pos: int) -> list[dict[str, Any]]:
    """Return source lines whose character offsets overlap a span in section text."""
    matched = [
        line
        for line in lines_info
        if int(line.get("start", 0)) < end_pos and int(line.get("end", 0)) > start_pos
    ]
    return _sort_lines_by_offset(matched)


def _line_is_substantial_for_match(line_text: str) -> bool:
    """Reject tiny/noisy lines that falsely substring-match into chunk text (e.g. '6')."""
    normalized = " ".join((line_text or "").strip().lower().split())
    if not normalized:
        return False
    if re.fullmatch(r"[\d\W_]+", normalized):
        return False
    words = normalized.split()
    if len(normalized) < 12 and len(words) < 3:
        return False
    return True


def _lines_matching_chunk_text(
    chunk_text: str,
    lines_info: list[dict[str, Any]],
    *,
    prefer_page: int | None = None,
) -> list[dict[str, Any]]:
    """Match chunk text to source lines by substring overlap (fallback only)."""
    needle = " ".join((chunk_text or "").strip().lower().split())
    if not needle:
        return []
    matched: list[dict[str, Any]] = []
    for line in lines_info:
        line_text = " ".join(str(line.get("text") or "").strip().lower().split())
        if not line_text or not _line_is_substantial_for_match(line_text):
            continue
        if line_text in needle or (len(line_text) >= 40 and line_text[:40] in needle):
            matched.append(line)
    # Prefer page only for exact duplicate lines across pages (same text), never to
    # discard multi-page content or keep a weak page-1 fragment.
    if prefer_page is not None and matched:
        texts = {
            " ".join(str(line.get("text") or "").strip().lower().split())
            for line in matched
        }
        if len(texts) == 1:
            on_page = [
                line
                for line in matched
                if line.get("page") is not None and int(line.get("page")) == prefer_page
            ]
            if on_page:
                matched = on_page
    return _sort_lines_by_offset(matched)


def _resolve_chunk_line_range(
    overlapping_lines: list[dict[str, Any]],
    *,
    chunk_page: int | None,
) -> tuple[int | None, int | None]:
    if chunk_page is None:
        numbered = [line for line in overlapping_lines if line.get("line_number") is not None]
        if not numbered:
            return None, None
        substantial = [
            line
            for line in numbered
            if _line_is_substantial_for_match(str(line.get("text") or ""))
        ]
        use_lines = substantial or numbered
        line_start = min(int(line["line_number"]) for line in use_lines)
        line_end = max(int(line.get("line_end") or line["line_number"]) for line in use_lines)
        return line_start, line_end

    same_page_lines = [
        line
        for line in overlapping_lines
        if line.get("page") is not None
        and int(line.get("page")) == chunk_page
        and line.get("line_number") is not None
    ]
    if not same_page_lines:
        return None, None
    substantial = [
        line
        for line in same_page_lines
        if _line_is_substantial_for_match(str(line.get("text") or ""))
    ]
    use_lines = substantial or same_page_lines
    line_start = min(int(line["line_number"]) for line in use_lines)
    line_end = max(int(line.get("line_end") or line["line_number"]) for line in use_lines)
    return line_start, line_end


def _locate_text_span(source_text: str, chunk_text: str, start_from: int = 0) -> tuple[int, int] | None:
    """Locate a chunk in source text, tolerating whitespace changes from sentence chunking."""
    needle = chunk_text.strip()
    if not needle:
        return None

    start_from = max(0, min(start_from, len(source_text)))
    direct = source_text.find(needle[: min(100, len(needle))] if len(needle) > 30 else needle, start_from)
    if direct == -1 and len(needle) > 30:
        direct = source_text.find(needle[:30], start_from)
    if direct == -1 and start_from:
        direct = source_text.find(needle[: min(100, len(needle))] if len(needle) > 30 else needle)
        if direct == -1 and len(needle) > 30:
            direct = source_text.find(needle[:30])
    if direct != -1:
        return direct, direct + len(needle)

    normalized_source, source_positions = _normalize_with_positions(source_text)
    normalized_needle, _ = _normalize_with_positions(needle)
    if not normalized_source or not normalized_needle:
        return None

    normalized_prefix, _ = _normalize_with_positions(source_text[:start_from])
    norm_search_from = min(len(normalized_prefix), len(normalized_source))

    norm_start = normalized_source.find(normalized_needle, norm_search_from)
    if norm_start == -1 and len(normalized_needle) > 100:
        norm_start = normalized_source.find(normalized_needle[:100], norm_search_from)
    if norm_start == -1 and len(normalized_needle) > 30:
        norm_start = normalized_source.find(normalized_needle[:30], norm_search_from)
    if norm_start == -1 and start_from:
        norm_start = normalized_source.find(normalized_needle)
        if norm_start == -1 and len(normalized_needle) > 100:
            norm_start = normalized_source.find(normalized_needle[:100])
        if norm_start == -1 and len(normalized_needle) > 30:
            norm_start = normalized_source.find(normalized_needle[:30])
    if norm_start == -1:
        return None

    norm_end = min(norm_start + len(normalized_needle), len(source_positions)) - 1
    return source_positions[norm_start], source_positions[norm_end] + 1


def _looks_like_signature_metadata(text: str) -> bool:
    """Detect approval/signature tables and generated-print metadata."""
    normalized = " ".join(text.split())
    if not normalized:
        return False

    metadata_hits = sum(1 for pattern in _SIGNATURE_METADATA_PATTERNS if pattern.search(normalized))
    if metadata_hits:
        return True

    return False


def _looks_like_toc_entry(block: Any) -> bool:
    if bool(getattr(block, "is_toc_entry", False)):
        return True

    raw_text = getattr(block, "text", "") or ""
    text = " ".join(raw_text.split())
    if not text or _TOC_TITLE_PATTERN.match(text):
        return False
    if any(pattern.match(text) for pattern in _TOC_ENTRY_PATTERNS):
        return True

    # Some PDF extractors preserve line breaks inside one block. Check each row
    # before falling back to page-level detection.
    for line in re.split(r"[\r\n]+", raw_text):
        normalized_line = " ".join(line.split())
        if normalized_line and any(pattern.match(normalized_line) for pattern in _TOC_ENTRY_PATTERNS):
            return True
    return False


def _toc_entry_count_from_text(text: str) -> int:
    count = 0
    for line in re.split(r"[\r\n]+", text or ""):
        normalized_line = " ".join(line.split())
        if normalized_line and any(pattern.match(normalized_line) for pattern in _TOC_ENTRY_PATTERNS):
            count += 1
    return count


def _combined_text_looks_like_toc(text: str) -> bool:
    """Detect TOC/TOT pages extracted as one large text block."""
    normalized = " ".join((text or "").split())
    if not normalized or not _TOC_TITLE_SEARCH_PATTERN.search(normalized[:1000]):
        return False

    line_entry_count = _toc_entry_count_from_text(text)
    dotted_leader_count = len(re.findall(r"\.{2,}\s*\d+\b", text or ""))
    numbered_heading_refs = len(
        re.findall(r"\b\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9/&(),'\- ]{2,120}\s+\.{2,}\s*\d+\b", text or "")
    )
    table_refs = len(
        re.findall(r"\btable\s+\d+[A-Za-z]?\s*[:\-.]\s+\S.{1,160}?\.{2,}\s*\d+\b", text or "", re.IGNORECASE)
    )
    return max(line_entry_count, dotted_leader_count, numbered_heading_refs + table_refs) >= 2


def _looks_like_toc_page(page_blocks: list[tuple[int, Any]], previous_page_was_toc: bool = False) -> bool:
    text_blocks = [
        block
        for _, block in page_blocks
        if (getattr(block, "block_type", "text") or "text") == "text"
        and (getattr(block, "text", "") or "").strip()
    ]
    if not text_blocks:
        return False

    page_text = "\n".join((getattr(block, "text", "") or "") for block in text_blocks)
    if _combined_text_looks_like_toc(page_text):
        return True

    title_indexes = [
        index
        for index, block in enumerate(text_blocks)
        if _TOC_TITLE_PATTERN.match(" ".join((getattr(block, "text", "") or "").split()))
    ]
    toc_entry_count = sum(1 for block in text_blocks if _looks_like_toc_entry(block))
    toc_entry_count = max(toc_entry_count, _toc_entry_count_from_text(page_text))
    toc_style_count = sum(1 for block in text_blocks if bool(getattr(block, "is_toc_entry", False)))
    entry_ratio = toc_entry_count / max(len(text_blocks), 1)

    if toc_style_count >= 3:
        return True
    if title_indexes and toc_entry_count >= 2:
        return True
    if title_indexes and len(text_blocks) <= 3:
        return True
    if previous_page_was_toc and toc_entry_count >= 3 and entry_ratio >= 0.4:
        return True
    return False


def _filter_signature_metadata_blocks(blocks: list[Any]) -> list[Any]:
    """Exclude approval/signature metadata before chunking.

    Removes document metadata/signature tables containing fields such as Name,
    Role, Date & Time, Prepared by, Reviewed by, Approved by, Print ID, Printed
    By, Copy No, and electronically generated document notes.
    """
    if not blocks:
        return blocks

    skip_indexes: set[int] = set()
    by_page: dict[int, list[tuple[int, Any]]] = {}
    for idx, block in enumerate(blocks):
        by_page.setdefault(getattr(block, "page", 0) or 0, []).append((idx, block))

    toc_skip_indexes: set[int] = set()
    previous_page_was_toc = False
    for _, page_blocks in sorted(by_page.items(), key=lambda item: item[0]):
        page_is_toc = _looks_like_toc_page(page_blocks, previous_page_was_toc)
        if page_is_toc:
            toc_skip_indexes.update(
                global_idx
                for global_idx, block in page_blocks
                if (getattr(block, "block_type", "text") or "text") == "text"
            )
        else:
            toc_skip_indexes.update(
                global_idx
                for global_idx, block in page_blocks
                if _looks_like_toc_entry(block)
            )
        previous_page_was_toc = page_is_toc

    skip_indexes.update(toc_skip_indexes)

    for page_blocks in by_page.values():
        for local_idx, (global_idx, block) in enumerate(page_blocks):
            if global_idx in skip_indexes:
                original_text = block.text.strip()
                block_page = _optional_block_page(block)
                business_metadata = _extract_business_metadata_text(original_text) if block_page == 1 else ""
                if not business_metadata:
                    continue
                try:
                    block.text = business_metadata
                    block.component_type = COMPONENT_TABLE
                    block.block_type = "table"
                    block.heading_level = None
                    metadata = dict(getattr(block, "metadata", {}) or {})
                    metadata["component_type"] = COMPONENT_TABLE
                    metadata["section_name"] = "Document Metadata"
                    metadata["preserved_business_metadata"] = True
                    metadata.pop("heading_level", None)
                    block.metadata = metadata
                    skip_indexes.discard(global_idx)
                except Exception:
                    continue
            original_text = block.text.strip()
            text = _clean_extracted_text_line(original_text)
            component = block_component_type(block)
            block_page = _optional_block_page(block)
            business_metadata = _extract_business_metadata_text(original_text) if block_page == 1 else ""
            if business_metadata:
                try:
                    block.text = business_metadata
                    block.component_type = COMPONENT_TABLE
                    block.block_type = "table"
                    block.heading_level = None
                    metadata = dict(getattr(block, "metadata", {}) or {})
                    metadata["component_type"] = COMPONENT_TABLE
                    metadata["section_name"] = "Document Metadata"
                    metadata["preserved_business_metadata"] = True
                    metadata.pop("heading_level", None)
                    block.metadata = metadata
                except Exception:
                    pass
                continue
            if component in {COMPONENT_HEADER, COMPONENT_FOOTER}:
                business_metadata = _extract_business_metadata_text(original_text) if block_page == 1 else ""
                if business_metadata:
                    try:
                        block.text = business_metadata
                        block.component_type = COMPONENT_TABLE
                        block.block_type = "table"
                        block.heading_level = None
                        metadata = dict(getattr(block, "metadata", {}) or {})
                        metadata["component_type"] = COMPONENT_TABLE
                        metadata["section_name"] = "Document Metadata"
                        metadata["preserved_business_metadata"] = True
                        metadata.pop("heading_level", None)
                        block.metadata = metadata
                    except Exception:
                        pass
                continue
            if _looks_like_cover_page_metadata(text) or _looks_like_running_header_footer(text):
                business_metadata = _extract_business_metadata_text(original_text) if block_page == 1 else ""
                if business_metadata:
                    try:
                        block.text = business_metadata
                        block.component_type = COMPONENT_TABLE
                        block.block_type = "table"
                        block.heading_level = None
                        metadata = dict(getattr(block, "metadata", {}) or {})
                        metadata["component_type"] = COMPONENT_TABLE
                        metadata["section_name"] = "Document Metadata"
                        metadata["preserved_business_metadata"] = True
                        metadata.pop("heading_level", None)
                        block.metadata = metadata
                    except Exception:
                        pass
                    continue
                skip_indexes.add(global_idx)
                continue
            if _looks_like_signature_metadata(text):
                skip_indexes.add(global_idx)

                # Approval tables often extract as separate rows:
                # "Prepared by Reviewed by Approved by", then Name/Role/Date rows.
                for next_global_idx, next_block in page_blocks[local_idx + 1 : local_idx + 5]:
                    next_text = " ".join(next_block.text.split())
                    if any(pattern.search(next_text) for pattern in _APPROVAL_ROW_PATTERNS):
                        skip_indexes.add(next_global_idx)
                    elif _looks_like_signature_metadata(next_text):
                        skip_indexes.add(next_global_idx)
                    else:
                        break

            elif any(pattern.search(text) for pattern in _APPROVAL_ROW_PATTERNS):
                window = page_blocks[max(0, local_idx - 2) : min(len(page_blocks), local_idx + 4)]
                window_text = " ".join(item_block.text for _, item_block in window)
                if any(pattern.search(window_text) for pattern in _SIGNATURE_METADATA_PATTERNS):
                    skip_indexes.add(global_idx)

    if not skip_indexes:
        return blocks

    if toc_skip_indexes:
        log.info("    Excluded %d table-of-contents block(s) before chunking", len(toc_skip_indexes))
    log.info("    Excluded %d signature/TOC metadata block(s) before chunking", len(skip_indexes))
    return [block for idx, block in enumerate(blocks) if idx not in skip_indexes]


def _table_row_looks_like_header(row_text: str) -> bool:
    joined = " ".join(part.strip() for part in row_text.split("|")).lower()
    header_signals = (
        "sl. no",
        "parameter",
        "requirement",
        "specification",
        "description",
        "department",
        "designation",
        "version number",
        "effective date",
        "change control",
        "summary of changes",
    )
    return sum(1 for signal in header_signals if signal in joined) >= 2


def _clean_table_row_for_chunk(row_text: str) -> str:
    parts = [part.strip() for part in str(row_text or "").split("|")]
    if len(parts) > 1:
        parts = [part for part in parts if part]
        row_text = " | ".join(parts)
    cleaned = _clean_extracted_text_line(row_text)
    if _looks_like_extraction_artifact_line(cleaned) or _looks_like_running_header_footer(cleaned):
        return ""
    return cleaned


def _append_table_block_chunks(
    chunks: list[Chunk],
    doc_name: str,
    section_name: str,
    section_path: str,
    block: Any,
    chunk_size: int,
    min_content_words: int,
    extra_image_refs: list[dict[str, Any]] | None = None,
) -> None:
    """Chunk table content on row boundaries without table labels or header rows."""
    text = str(getattr(block, "text", "") or "").strip()
    if not text:
        return

    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows:
        return

    if rows[0].lower().startswith("table"):
        if len(rows) == 1:
            first_row = re.sub(r"^\s*table\s*\d*[A-Za-z]?\s*[:.\-]?\s*", "", rows[0], flags=re.IGNORECASE).strip()
            data_rows = [first_row] if first_row else []
        else:
            data_rows = rows[1:]
    else:
        data_rows = rows
    header = data_rows[0] if len(data_rows) > 1 and _table_row_looks_like_header(data_rows[0]) else None
    if header:
        data_rows = data_rows[1:]
    data_rows = [cleaned for row in data_rows if (cleaned := _clean_table_row_for_chunk(row))]
    if not data_rows:
        return

    table_min_words = max(12, min(min_content_words // 3, 30))
    batches: list[list[str]] = []
    current: list[str] = []
    current_words = sum(len(line.split()) for line in current)

    for row in data_rows:
        row_words = len(row.split())
        if current_words + row_words > chunk_size and current:
            batches.append(list(current))
            current = []
            current_words = 0
        current.append(row)
        current_words += row_words

    if current:
        batches.append(current)

    if not batches:
        batches = [data_rows]

    metadata = dict(getattr(block, "metadata", {}) or {})
    block_section_name = str(metadata.get("section_name") or section_name).strip() or section_name
    block_section_path = str(metadata.get("section_path") or section_path or block_section_name).strip()
    page = _optional_block_page(block)
    line_number = getattr(block, "line_number", None)

    emitted = False
    for batch in batches:
        raw, image_refs = _strip_image_placeholders("\n".join(batch).strip())
        if len(raw.split()) < table_min_words and len(data_rows) > 1:
            continue
        if extra_image_refs:
            image_refs = [*extra_image_refs, *image_refs]
            extra_image_refs = None
        emitted = True
        content_types = [COMPONENT_TABLE]
        if image_refs:
            content_types.append(COMPONENT_IMAGE)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                doc_name=doc_name,
                page=page,
                section_name=block_section_name,
                section_path=block_section_path,
                text=raw,
                line_start=int(line_number) if line_number is not None else None,
                line_end=int(metadata.get("source_line_end") or line_number or 0) or None,
                content_types=content_types,
                table_count=1,
                image_count=len(image_refs),
                category=COMPONENT_TABLE,
                category_confidence=1.0,
                chunk_type="table",
                extraction_metadata={
                    "tables": [metadata],
                    "images": image_refs,
                    "component_types": content_types,
                },
            )
        )

    if not emitted and batches:
        raw, image_refs = _strip_image_placeholders("\n".join(batches[0]).strip())
        if extra_image_refs:
            image_refs = [*extra_image_refs, *image_refs]
        if raw:
            content_types = [COMPONENT_TABLE]
            if image_refs:
                content_types.append(COMPONENT_IMAGE)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_name=doc_name,
                    page=page,
                    section_name=block_section_name,
                    section_path=block_section_path,
                    text=raw,
                    line_start=int(line_number) if line_number is not None else None,
                    line_end=int(metadata.get("source_line_end") or line_number or 0) or None,
                    content_types=content_types,
                    table_count=1,
                    image_count=len(image_refs),
                    category=COMPONENT_TABLE,
                    category_confidence=1.0,
                    chunk_type="table",
                    extraction_metadata={
                        "tables": [metadata],
                        "images": image_refs,
                        "component_types": content_types,
                    },
                )
            )


def _append_image_block_chunk(
    chunks: list[Chunk],
    doc_name: str,
    section_name: str,
    section_path: str,
    block: Any,
) -> None:
    """Image blocks are counted in metadata only; they do not create text chunks."""
    return


def _append_chunks_from_blocks(
    chunks: list[Chunk],
    doc_name: str,
    section_name: str,
    section_path: str,
    sec_blocks: list[Any],
    chunk_size: int,
    min_content_words: int,
    overlap_sentences: int,
    post_procs: list[dict[str, Any]] | None = None,
) -> None:
    """Convert a block slice into chunk objects and append them to the output."""
    if not sec_blocks:
        return

    page_groups: list[list[Any]] = []
    current_group: list[Any] = []
    current_page: int | None = None
    for block in sec_blocks:
        page = _optional_block_page(block)
        if current_group and current_page is not None and page != current_page:
            page_groups.append(current_group)
            current_group = []
        current_group.append(block)
        current_page = page
    if current_group:
        page_groups.append(current_group)
    if len(page_groups) > 1:
        for page_group in page_groups:
            _append_chunks_from_blocks(
                chunks=chunks,
                doc_name=doc_name,
                section_name=section_name,
                section_path=section_path,
                sec_blocks=page_group,
                chunk_size=chunk_size,
                min_content_words=min_content_words,
                overlap_sentences=overlap_sentences,
                post_procs=post_procs,
            )
        return

    pp = PostProcessor()
    lines_info = []
    pending_image_refs: list[dict[str, Any]] = []
    cursor = 0
    
    for block in sec_blocks:
        block_type = getattr(block, "block_type", "text") or "text"
        component_type = block_component_type(block)

        if is_image_block(block):
            pending_image_refs.append(dict(getattr(block, "metadata", {}) or {}))
            continue
        if component_type in {COMPONENT_HEADER, COMPONENT_FOOTER}:
            continue

        if component_type == COMPONENT_TABLE or block_type == "table":
            _append_table_block_chunks(
                chunks=chunks,
                doc_name=doc_name,
                section_name=section_name,
                section_path=section_path,
                block=block,
                chunk_size=chunk_size,
                min_content_words=min_content_words,
                extra_image_refs=pending_image_refs,
            )
            pending_image_refs = []
            continue

        processed = pp.run(block.text, post_procs or [])
        processed, image_refs = _strip_image_placeholders(processed)
        processed = _clean_extracted_text_line(processed)
        if _looks_like_extraction_artifact_line(processed):
            continue
        if pending_image_refs:
            image_refs = [*pending_image_refs, *image_refs]
            pending_image_refs = []
        include_line = (
            component_type in {COMPONENT_HEADER, COMPONENT_FOOTER, COMPONENT_TABLE, COMPONENT_TITLE, COMPONENT_SUBTITLE}
            or is_body_line(processed)
            or _looks_like_instruction_line(processed)
            or block_type == "table"
        )
        if include_line and processed.strip():
            start = cursor
            end = start + len(processed)
            lines_info.append(
                {
                    "text": processed,
                    "page": block.page,
                    "line_number": getattr(block, "line_number", 0) or None,
                    "line_end": (getattr(block, "metadata", {}) or {}).get("source_line_end")
                    or getattr(block, "line_number", 0)
                    or None,
                    "start": start,
                    "end": end,
                    "block_type": block_type,
                    "component_type": component_type,
                    "metadata": {
                        **(getattr(block, "metadata", {}) or {}),
                        "image_placeholders": image_refs,
                    },
                }
            )
            cursor = end + 1

    if pending_image_refs and lines_info:
        metadata = lines_info[-1].setdefault("metadata", {})
        existing = metadata.get("image_placeholders") or []
        metadata["image_placeholders"] = [*existing, *pending_image_refs]

    section_text = "\n".join([line["text"] for line in lines_info]).strip()
    if not section_text:
        return

    span_search_from = 0
    emitted_section_heading = False
    for raw in chunk_section_text(
        section_text,
        chunk_size=chunk_size,
        min_words=min_content_words,
        overlap_sentences=overlap_sentences,
    ):
        if (
            _looks_like_extraction_artifact_line(raw)
            or _looks_like_cover_page_metadata(raw)
            or _looks_like_running_header_footer(raw)
        ):
            continue
        if normalize_text_for_embedding(raw):
            raw_for_mapping = raw
            if not emitted_section_heading:
                raw = _with_section_heading(raw, section_name)
                emitted_section_heading = True
            # Map chunk back to actual page number
            chunk_page = sec_blocks[0].page
            span = _locate_text_span(section_text, raw_for_mapping, span_search_from)

            if span is not None:
                start_pos, chunk_end = span
                span_search_from = max(start_pos + 1, span_search_from)
                overlapping_lines = [
                    line
                    for line in lines_info
                    if line["start"] < chunk_end and line["end"] > start_pos
                ]
                if overlapping_lines:
                    chunk_page = overlapping_lines[0]["page"]
                    content_types = sorted(
                        {line.get("component_type") or line.get("block_type") or "paragraph" for line in overlapping_lines}
                    )
                    tables = [
                        line.get("metadata") or {}
                        for line in overlapping_lines
                        if (line.get("component_type") or line.get("block_type")) in {COMPONENT_TABLE, "table"}
                    ]
                    images = [
                        image
                        for line in overlapping_lines
                        for image in ((line.get("metadata") or {}).get("image_placeholders") or [])
                    ]
                    if images and COMPONENT_IMAGE not in content_types:
                        content_types.append(COMPONENT_IMAGE)
                        content_types = sorted(content_types)
                    same_page_lines = [
                        line
                        for line in overlapping_lines
                        if line["page"] == chunk_page and line["line_number"] is not None
                    ]
                    if same_page_lines:
                        line_start = min(int(line["line_number"]) for line in same_page_lines)
                        line_end = max(int(line.get("line_end") or line["line_number"]) for line in same_page_lines)
                    else:
                        line_start = None
                        line_end = None
                else:
                    line_start = None
                    line_end = None
                    content_types = ["paragraph"]
                    tables = []
                    images = []
            else:
                line_start = None
                line_end = None
                content_types = ["paragraph"]
                tables = []
                images = []
                        
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_name=doc_name,
                    page=chunk_page,
                    section_name=section_name,
                    section_path=section_path,
                    text=raw,
                    line_start=line_start,
                    line_end=line_end,
                    content_types=content_types,
                    table_count=len(tables),
                    image_count=len(images),
                    extraction_metadata={
                        "tables": tables,
                        "images": images,
                        "component_types": content_types,
                    },
                )
            )


def chunk_document_blocks(
    blocks: list[Any],
    doc_name: str,
    chunk_size: int = 150,
    min_content_words: int = 60,
    overlap_sentences: int = 2,
    template: dict | None = None,
) -> list[Chunk]:
    """Chunk a single already-loaded document into section-aware chunks."""
    chunks: list[Chunk] = []

    if not blocks:
        log.warning("No blocks extracted from %s", doc_name)
        return chunks

    if template:
        log.info(
            "Template support is deprecated for %s; using auto section detection instead.",
            doc_name,
        )

    blocks = _filter_signature_metadata_blocks(blocks)
    if not blocks:
        log.warning("No chunkable blocks left in %s after metadata/signature filtering", doc_name)
        return chunks

    blocks = blocks_for_chunking(blocks)
    if not blocks:
        log.warning("No chunkable blocks left in %s after excluding image blocks", doc_name)
        return chunks
    blocks = _prepare_blocks_for_section_detection(blocks)
    if not blocks:
        log.warning("No chunkable blocks left in %s after cleaning PDF extraction artifacts", doc_name)
        return chunks

    sections = auto_detect_sections(blocks)
    log.info("    Auto-detected %d section(s) in %s", len(sections), doc_name)

    for sec_path, _, sec_blocks in sections:
        sec_name = _section_leaf_label(sec_path)
        _append_chunks_from_blocks(
            chunks=chunks,
            doc_name=doc_name,
            section_name=sec_name,
            section_path=sec_path,
            sec_blocks=sec_blocks,
            chunk_size=chunk_size,
            min_content_words=min_content_words,
            overlap_sentences=overlap_sentences,
        )

    before_merge = len(chunks)
    chunks = _merge_undersized_chunk_objects(chunks, min_content_words, chunk_size)
    if len(chunks) != before_merge:
        log.info(
            "    Merged %d undersized section fragment(s) in %s",
            before_merge - len(chunks),
            doc_name,
        )

    log.info("    -> %d chunks (auto sections) from %s", len(chunks), doc_name)
    return chunks


def _load_txt_blocks(fpath: str) -> list[Block]:
    blocks: list[Block] = []
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
            line_number = 0
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                line_number += 1
                blocks.append(Block(text=text, page=None, line_number=line_number))
    except Exception as exc:
        log.error("Failed to read text file %s: %s", fpath, exc)
    return blocks


def chunk_document_file(
    fpath: str,
    input_dir: str,
    chunk_size: int,
    min_content_words: int,
    overlap_sentences: int,
    template: dict | None = None,
) -> list[Chunk]:
    """Load and chunk a single document file."""
    doc_name = os.path.relpath(fpath, input_dir)

    try:
        if fpath.lower().endswith(".pdf"):
            blocks = PdfLoader().load(fpath)
        elif fpath.lower().endswith(".docx"):
            blocks = DocxLoader().load(fpath)
        elif fpath.lower().endswith(".txt"):
            blocks = _load_txt_blocks(fpath)
        else:
            log.error("Unsupported document type for %s", doc_name)
            return []
    except Exception as exc:
        log.exception("Failed to load %s: %s", doc_name, exc)
        raise

    return chunk_document_blocks(
        blocks=blocks,
        doc_name=doc_name,
        chunk_size=chunk_size,
        min_content_words=min_content_words,
        overlap_sentences=overlap_sentences,
        template=template,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION-AWARE CHUNKING
# ─────────────────────────────────────────────────────────────────────────────


def chunk_section_text(
    text: str,
    chunk_size: int = 150,
    min_words: int = 60,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Split section text into overlapping chunks that respect sentence and
    paragraph boundaries.

    Parameters
    ----------
    text : str
        The section text to chunk.
    chunk_size : int
        Maximum number of words per chunk.
    min_words : int
        Minimum words required for a chunk to be kept.
    overlap_sentences : int
        Number of sentences from the end of the previous chunk to carry
        over into the next chunk for context continuity.

    Returns
    -------
    List of chunk strings.
    """
    effective_min_words = max(_MIN_CHUNK_WORD_FLOOR, min_words)
    effective_chunk_size = max(chunk_size, effective_min_words)

    paragraphs = split_into_paragraphs(text)
    all_sentences: List[str] = []
    for para in paragraphs:
        sents = split_into_sentences(para)
        all_sentences.extend(sents)

    if not all_sentences:
        return []

    chunks: List[str] = []
    start_idx = 0

    while start_idx < len(all_sentences):
        current_sents: List[str] = []
        current_words = 0

        idx = start_idx
        while idx < len(all_sentences):
            sent = all_sentences[idx]
            sent_words = len(sent.split())

            if sent_words == 0:
                idx += 1
                continue

            # ✅ FIX 1 applied: oversized sentence → emit multiple sub-chunks,
            # then advance past it. The original discarded everything after
            # the first chunk_size words.
            if sent_words > effective_chunk_size and not current_sents:
                _emit_long_sentence(sent, effective_chunk_size, chunks)
                idx += 1
                start_idx = idx
                break

            # Would adding this sentence exceed the limit?
            if current_sents and current_words + sent_words > effective_chunk_size:
                break

            current_sents.append(sent)
            current_words += sent_words
            idx += 1

        if current_sents:
            candidate = " ".join(current_sents)
            if candidate:
                chunks.append(candidate)

            sents_consumed = len(current_sents)
            if idx >= len(all_sentences):
                break

            # Step back by overlap_sentences for the next chunk.
            # ✅ FIX 2: Guard overlap so we always advance by at least 1,
            # preventing an infinite loop when sents_consumed == overlap.
            # Original used `sents_consumed - 1` as the cap which was correct
            # but only when sents_consumed > 1; the guard is made explicit here.
            overlap = min(overlap_sentences, max(sents_consumed - 1, 0))
            advance = sents_consumed - overlap
            # Defensive: ensure we always make forward progress.
            start_idx = start_idx + max(advance, 1)
        else:
            # No sentences were collected (all were zero-length or we broke
            # early). Always advance to prevent an infinite loop.
            start_idx = max(start_idx + 1, idx)

    return _merge_undersized_chunks(chunks, effective_min_words)


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE LOADER
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────


def _process_one_doc(
    fpath: str,
    input_dir: str,
    chunk_size: int,
    min_content_words: int,
    overlap_sentences: int,
    template: dict,
) -> List[Chunk]:
    """Process a single document into chunks."""
    return chunk_document_file(
        fpath=fpath,
        input_dir=input_dir,
        chunk_size=chunk_size,
        min_content_words=min_content_words,
        overlap_sentences=overlap_sentences,
        template=template,
    )


def collect_chunks(
    input_dir: str,
    chunk_size: int = 150,
    min_content_words: int = 60,
    overlap_sentences: int = 2,
    template: dict | None = None,
    max_workers: int = 8,
) -> List[Chunk]:
    """
    Walk input_dir, process all PDF/DOCX files, and return a flat list of
    Chunk objects ready for embedding.
    """
    doc_paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            lower = fname.lower()
            if lower.endswith(".pdf") or lower.endswith(".docx"):
                doc_paths.append(os.path.join(root, fname))

    if not doc_paths:
        log.warning("No PDF/DOCX files found in %s", input_dir)
        return []

    log.info("Found %d document(s) in %s", len(doc_paths), input_dir)

    # ✅ FIX 4: Collect results into a dict keyed by path so we can
    # re-sort by original document order after concurrent execution.
    # The original used `as_completed()` which returns futures in
    # non-deterministic completion order, making chunk ordering
    # non-reproducible across runs.
    results: Dict[str, List[Chunk]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_one_doc,
                fp,
                input_dir,
                chunk_size,
                min_content_words,
                overlap_sentences,
                template or {},
            ): fp
            for fp in doc_paths
        }
        for future in as_completed(futures):
            fp = futures[future]
            try:
                results[fp] = future.result()
            except Exception as exc:
                log.error("Unexpected error processing %s: %s", fp, exc)
                results[fp] = []

    # Reassemble in deterministic document order.
    all_chunks: List[Chunk] = []
    for fp in doc_paths:
        all_chunks.extend(results.get(fp, []))

    log.info("Total chunks collected: %d", len(all_chunks))
    return all_chunks
