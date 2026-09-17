from __future__ import annotations

import copy
import re
from typing import Any

from src.features.document_processing.utilities.extraction_filters import clean_extracted_line, should_skip_content_line

try:
    from src.features.document_processing.loaders.component_classification import (
        COMPONENT_FOOTER,
        COMPONENT_HEADER,
        block_component_type,
    )
except ImportError:
    COMPONENT_HEADER = "header"
    COMPONENT_FOOTER = "footer"

    def block_component_type(block: Any) -> str:
        return str(getattr(block, "component_type", "") or "paragraph").lower() or "paragraph"

_ABBREVIATIONS = (
    r"\b(e\.g|i\.e|vs|dr|mr|mrs|ms|prof|st|no|fig|sec|approx"
    r"|dept|govt|inc|ltd|etc)\."
)


def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []

    protected = re.sub(
        _ABBREVIATIONS,
        lambda m: m.group().replace(".", "<!DOT!>"),
        text,
        flags=re.IGNORECASE,
    )
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\"])', protected)
    sentences = [s.replace("<!DOT!>", ".").strip() for s in raw if s.strip()]
    return sentences or [text.strip()]


def normalize_text_for_embedding(text: str) -> str:
    if not text:
        return ""

    text = clean_extracted_line(text)
    text = re.sub(r"\b(\d+(?:\.\d+)*)(?:kg|g|mg|mcg|ml|l|bar|mbar|psi|%)\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(v\d+(?:\.\d+)*)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(rev\s*\d+)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_display(text: str) -> str:
    if not text:
        return ""
    lines = re.split(r"[\r\n]+", text)
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^table\s*[:\.]", line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return " ".join(cleaned)


_SKIP_HEADING_PATTERNS = (
    r"^table\s+of\s+contents$",
    r"^contents$",
    r"^(?:table|list)\s+of\s+tables$",
    r"^(?:table|list)\s+of\s+figures$",
    r"^(?:list\s+of\s+)?(?:figures|tables)$",
)

_COMMON_SECTION_TITLE_WORDS = {
    "abbreviations",
    "accountability",
    "annexure",
    "attachments",
    "background",
    "definitions",
    "distribution",
    "documents",
    "equipment",
    "frequency",
    "glossary",
    "history",
    "instructions",
    "materials",
    "note",
    "objective",
    "overview",
    "policy",
    "purpose",
    "references",
    "responsibilities",
    "responsibility",
    "revision",
    "scope",
    "summary",
    "procedure",
}


def split_into_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|[\r\n]+", text) if part.strip()]
    return paragraphs or [text.strip()]


def is_skip_heading(text: str) -> bool:
    value = clean_extracted_line(text).lower()
    if not value:
        return True
    return any(re.match(pattern, value, flags=re.IGNORECASE) for pattern in _SKIP_HEADING_PATTERNS)


def is_body_line(text: str) -> bool:
    value = clean_extracted_line(text)
    if not value:
        return False
    if is_skip_heading(value) or should_skip_content_line(value):
        return False
    if re.fullmatch(r"[\W_]+", value):
        return False
    return True


def is_body_section_name(text: str) -> bool:
    return is_body_line(text) and not is_skip_heading(text)


def _heading_level_from_number(number: str) -> int:
    parts = [part for part in number.rstrip(".").split(".") if part]
    if not parts:
        return 1
    if len(parts) == 1 or parts[-1] == "0":
        return 1
    return min(len(parts), 6)


def _strip_leading_outline_number(text: str) -> str:
    """Remove a leading outline/form number so heading cues can match the title body."""
    match = re.match(r"^\s*\d+(?:\.\d+)*\.?\s+(?P<body>\S.*)$", text or "")
    if not match:
        return (text or "").strip()
    return match.group("body").strip()


def _looks_like_instructional_heading(text: str) -> bool:
    """Detect instructional/procedure headings, including '6 Instructions for...'."""
    body = _strip_leading_outline_number(text).rstrip(":")
    if not body or is_skip_heading(body):
        return False
    if not re.match(
        r"^(?:instructions\s+for|procedure\s+for|note|revision\s+history|"
        r"deviation\s+recurrence\s+check\s+instructions)\b",
        body,
        re.IGNORECASE,
    ):
        return False
    return len(body.split()) <= 24


def _is_numbered_heading_candidate(number: str) -> bool:
    parts = [part for part in number.rstrip(".").split(".") if part]
    return bool(parts) and len(parts) <= 6


def _heading_body_from_numbered_text(text: str) -> tuple[str, str] | None:
    """Split compact PDF text such as ``1.0OBJECTIVE This...`` generically.

    PDF extraction often joins a section number, heading, and the first body
    sentence into one line. This keeps the leading numbered heading as section
    metadata while preserving the remaining body text for chunking.
    """
    match = re.match(r"^\s*(?P<number>\d+(?:\.\d+)*\.?)\s*(?P<rest>[A-Za-z].*)$", text)
    if not match:
        return None

    number = match.group("number").strip()
    rest = match.group("rest").strip()
    if not _is_numbered_heading_candidate(number):
        return None

    # Long instructional titles (often prefixed by a form/row number in PDF text).
    if _looks_like_instructional_heading(rest):
        return rest.rstrip(":"), ""

    body_match = re.match(
        r"^(?P<title>[A-Z][A-Z0-9/&(),'\- ]{1,80}?)(?:\s{1,}|[:\-–])(?P<body>[A-Z][a-z].*)$",
        rest,
    )
    if body_match:
        title = body_match.group("title").strip(" :-–")
        body = body_match.group("body").strip()
        if 1 <= len(title.split()) <= 8 and len(body.split()) >= 3:
            return f"{number} {title}", body

    suffix_words = rest.split()
    if len(suffix_words) <= 8 and (rest.isupper() or rest.endswith(":") or rest.istitle()):
        return f"{number} {rest.rstrip(':')}", ""

    return None


def _looks_like_spaced_numbered_heading(text: str) -> bool:
    match = re.match(r"^\s*(?P<number>\d+(?:\.\d+)*\.?)\s+(?P<title>\S.*)$", text)
    if not match:
        return False

    number = match.group("number").strip()
    title = match.group("title").strip().rstrip(":")
    if not _is_numbered_heading_candidate(number):
        return False
    if _looks_like_instructional_heading(title):
        return True

    title_words = title.split()
    if not 1 <= len(title_words) <= 8:
        return False
    return title.isupper() or title.istitle()


def _looks_like_common_section_heading(text: str) -> bool:
    value = clean_extracted_line(text).strip().rstrip(":")
    if not value or is_skip_heading(value):
        return False
    if _looks_like_instructional_heading(value):
        return True
    if re.search(r"[.!?]$", value):
        return False
    words = value.split()
    if not 1 <= len(words) <= 8:
        return False
    normalized_words = {re.sub(r"[^a-z]", "", word.lower()) for word in words}
    if normalized_words & _COMMON_SECTION_TITLE_WORDS:
        return value.isupper() or value.istitle() or len(words) <= 3
    return False


def _clone_block_with_text(block: Any, text: str) -> Any:
    cloned = copy.copy(block)
    try:
        cloned.text = text
    except Exception:
        return block
    return cloned


def _is_plausible_section_heading(text: str) -> bool:
    """Reject watermark fragments and single-glyph false headings from PDF noise."""
    value = clean_extracted_line(text).strip().rstrip(":")
    if not value or is_skip_heading(value):
        return False
    if len(value) <= 2:
        return False
    if re.fullmatch(r"[A-Za-z0-9/*]+", value) and len(value) <= 3:
        return False
    if re.search(r"\b(?:jar|gis|illip|ortc|llac|nta)\b", value, re.IGNORECASE):
        return False
    if re.match(
        r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:document\s+no\.?|effective(?:\s+date)?|reference|version\s+no\.?|annexure)\b",
        value,
        re.IGNORECASE,
    ):
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?title\s*:", value, re.IGNORECASE):
        return False
    if re.match(r"^confidential\s+work\s+product\b", value, re.IGNORECASE):
        return False
    if re.match(r"^reference\s+copy\b", value, re.IGNORECASE):
        return False
    # Facility / department running headers.
    if re.match(r"^facility\b", value, re.IGNORECASE) and "department" in value.lower():
        return False
    # Document-number style tokens (GL-CQA-GOP-0007) are not section titles.
    if re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){2,}\b", value) and len(value.split()) <= 5:
        return False
    # OCR noise before lettered list items: "ta b. On discrepancies..."
    if re.match(r"^[a-z]{1,4}\s+[a-z]\.\s+\S", value, re.IGNORECASE):
        return False
    # Sentence-like body text is not a heading.
    if re.search(r"[.!?]$", value) and not value.isupper() and len(value.split()) >= 6:
        return False
    if "*" in value:
        return False
    if re.fullmatch(r"[A-Za-z]\s+[A-Za-z]", value):
        return False
    if "/" in value and len(value.split()) <= 3:
        return False
    if re.match(r"^[A-Za-z]\s+revision\s+history\b", value, re.IGNORECASE):
        return True
    if re.match(r"^revision\s+history\b", value, re.IGNORECASE):
        return True
    words = value.split()
    if len(words) == 1 and len(value) < 4 and value.lower() not in {"note", "scope"}:
        return False
    # Two short tokens like "U *" already handled; "E e" handled above.
    if len(words) == 2 and all(len(w) <= 2 for w in words):
        return False
    return True


def _looks_like_heading(block: Any) -> bool:
    component = block_component_type(block)
    if component in {COMPONENT_HEADER, COMPONENT_FOOTER, "table", "image"}:
        return False

    heading_level = getattr(block, "heading_level", None)
    if heading_level is not None:
        text = (getattr(block, "text", "") or "").strip()
        return _is_plausible_section_heading(text) if text else True
    if component in {"title", "subtitle"}:
        text = (getattr(block, "text", "") or "").strip()
        return _is_plausible_section_heading(text)

    text = (getattr(block, "text", "") or "").strip()
    if not text or is_skip_heading(text):
        return False
    if not _looks_like_instructional_heading(text) and not _is_plausible_section_heading(text):
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?title\s*:\s+\S", text, re.IGNORECASE):
        return False
    if re.match(r"^[a-z]\.\s+\S", text, re.IGNORECASE):
        return False
    if re.match(r"^(?:i{1,3}|iv|v|vi{0,3}|ix|x)\.\s+\S", text, re.IGNORECASE):
        return False
    if _heading_body_from_numbered_text(text):
        return True
    compact_numbered = re.match(r"^\d+(?:\.\d+)*\.?([A-Za-z].*)$", text)
    if compact_numbered:
        suffix = compact_numbered.group(1).strip()
        suffix_words = suffix.split()
        if len(suffix_words) > 8:
            return False
        if suffix.isupper() or suffix.endswith(":"):
            return True
        if not suffix.istitle():
            return False
    style = (getattr(block, "style", "") or "").lower()
    if "heading" in style or "title" in style:
        return _is_plausible_section_heading(text)
    if compact_numbered:
        return False
    if getattr(block, "bold", False):
        word_count = len(text.split())
        if _looks_like_instructional_heading(text):
            return True
        if 2 <= word_count <= 12 and _is_plausible_section_heading(text):
            return True
    if _looks_like_spaced_numbered_heading(text):
        return True
    if _looks_like_common_section_heading(text):
        return True
    return text.isupper() and 2 <= len(text.split()) <= 12 and _is_plausible_section_heading(text)


def auto_detect_sections(blocks: list[Any]) -> list[tuple[str, int, list[Any]]]:
    sections: list[tuple[str, int, list[Any]]] = []
    current_name = ""
    current_start = 0
    current_blocks: list[Any] = []
    heading_stack: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal current_blocks
        if current_blocks:
            section_path = " > ".join(label for _, label in heading_stack) or current_name
            sections.append((section_path, current_start, current_blocks))
            current_blocks = []

    def set_current_heading(name: str, level: int | None, index: int) -> None:
        nonlocal current_name, current_start
        clean_name = clean_extracted_line(name).strip().rstrip(":")
        if not clean_name:
            return
        if not _looks_like_instructional_heading(clean_name) and not _is_plausible_section_heading(clean_name):
            return
        heading_level = int(level or 1)
        while heading_stack and heading_stack[-1][0] >= heading_level:
            heading_stack.pop()
        heading_stack.append((heading_level, clean_name))
        current_name = clean_name
        current_start = index

    for index, block in enumerate(blocks):
        if block_component_type(block) in {COMPONENT_HEADER, COMPONENT_FOOTER}:
            continue

        heading_body = _heading_body_from_numbered_text((getattr(block, "text", "") or "").strip())
        if heading_body:
            flush()
            heading_name, body_text = heading_body
            number_match = re.match(r"^\s*(?P<number>\d+(?:\.\d+)*\.?)", heading_name)
            detected_level = _heading_level_from_number(number_match.group("number")) if number_match else 1
            set_current_heading(heading_name, getattr(block, "heading_level", None) or detected_level, index)
            if body_text:
                current_blocks.append(_clone_block_with_text(block, body_text))
            continue

        if _looks_like_heading(block):
            flush()
            set_current_heading(
                str(getattr(block, "text", "")).strip(),
                getattr(block, "heading_level", None),
                index,
            )
            continue
        current_blocks.append(block)

    flush()
    # Empty section path means "no heading detected" — never invent "Document".
    return sections or [("", 0, blocks)]
