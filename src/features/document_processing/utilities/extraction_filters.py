from __future__ import annotations

import re

_PAGE_MARKER_RE = re.compile(r"^(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?$", re.IGNORECASE)
_TOC_DOTS_RE = re.compile(r"\.{3,}\s*\d+\s*$")
_TOC_ENTRY_RE = re.compile(
    r"(?i)^\s*(?:"
    r"(?:\d+(?:\.\d+)*|[IVXLCDM]+|appendix\s+[A-Z0-9]+|table\s+[A-Z0-9.\-]+|fig(?:ure)?\.?\s+[A-Z0-9.\-]+)"
    r"(?:[.)-]|\s+)\s*)?"
    r"[A-Z][A-Za-z0-9,()/%&:'\- ]{2,90}\s+\.{0,}\s*\d{1,4}\s*$"
)
_FORM_LABEL_RE = re.compile(
    r"(?i)\b(?:sign(?:ature)?|date|checked\s+by|approved\s+by|reviewed\s+by|prepared\s+by|yes/?no)\b"
)
_TABLE_HEADER_RE = re.compile(
    r"(?i)\b(?:si\.?\s*no\.?|s\.?\s*no\.?|serial\s+no\.?|tag\s*no\.?|"
    r"description|remarks|date\s*/\s*time|carried\s+out\s+by|parameter|specification|"
    r"acceptance\s+criteria|observation|observed|result|status|frequency|responsibility|"
    r"recorded\s+by|checked\s+by|approved\s+by|minimum|maximum|limit|unit|value)\b"
)
_TABLE_CAPTION_RE = re.compile(r"(?i)^(?:table|tbl)\s+[A-Z0-9][A-Z0-9.\-]*\s*[:.\-]?\s*\S+")
_FIGURE_CAPTION_RE = re.compile(
    r"(?i)^(?:figure|fig\.?|image|photo|diagram|chart|graph)\s+[A-Z0-9][A-Z0-9.\-]*\s*[:.\-]?\s*\S+"
)
_TABLE_DELIMITER_RE = re.compile(r"(?:\s{2,}|\t|\|)")
_TABLE_CELL_CODE_RE = re.compile(
    r"(?i)^(?:[-+]?\d+(?:\.\d+)?%?|n/?a|yes|no|ok|pass|fail|[A-Z0-9]{1,8}(?:[-/][A-Z0-9]{1,12})+)$"
)
_CODE_HEAVY_TOKEN_RE = re.compile(
    r"(?i)^(?:[A-Z]{1,6}(?:[-/][A-Z0-9]{1,10})+|[A-Z0-9]{2,}|\d+(?:\.\d+)?|(?:kg|g|mg|ml|l|mm|cm|rpm|psi|bar))$"
)
_DOCUMENT_METADATA_RE = re.compile(
    r"(?i)\b(?:document\s+name|document\s+number|document\s+no|doc(?:ument)?\s+no|"
    r"sop\s+number|department|site\s*/\s*division|site|division|sub\s+organization|"
    r"sub-?department|organization|effective\s+date|review\s+date|version|revision\s+no|"
    r"prepared\s+by|reviewed\s+by|approved\s+by|document\s+type|export\s+date)\b"
)
_DOCUMENT_METADATA_LINE_RE = re.compile(
    r"(?i)^\s*(?:document\s+name|document\s+number|document\s+no|doc(?:ument)?\s+no|"
    r"sop\s+number|department|site\s*/\s*division|site|division|sub\s+organization|"
    r"sub-?department|organization|effective\s+date|review\s+date|version|revision\s+no|"
    r"prepared\s+by|reviewed\s+by|approved\s+by|document\s+type|export\s+date)\s*[:\-]?\s+\S+"
)
_SIGNATURE_ROLE_LINE_RE = re.compile(
    r"(?i)^\s*role\s+.+\b(?:initiator|approver|hod|reviewer|checker|qa|cqa)\b"
)
_ELECTRONIC_SIGNATURE_RE = re.compile(
    r"(?i)\b(?:electronically\s+generated\s+document|does\s+not\s+require\s+a\s+signature)\b"
)
_PRINT_METADATA_RE = re.compile(
    r"(?i)\b(?:print\s*id|printed\s+by|printed\s+on|copy\s+no|copy\s+to|printer\s+name|"
    r"driver\s+name|print\s+type|uncontrolled\s+copy)\b"
)
_APPROVAL_NOTE_RE = re.compile(
    r"(?i)approval\s+signatures?.*electronically.*(?:veeva\s+vault|vault)|"
    r"refer\s+to\s+the\s+last\s+page"
)


def clean_extracted_line(text: str) -> str:
    value = (text or "").replace("\u00a0", " ").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def looks_like_table_or_image_caption(text: str) -> bool:
    value = clean_extracted_line(text)
    return bool(value and (_TABLE_CAPTION_RE.match(value) or _FIGURE_CAPTION_RE.match(value)))


def looks_like_toc_entry(text: str) -> bool:
    value = clean_extracted_line(text)
    if not value:
        return False
    if _TOC_DOTS_RE.search(value):
        return True
    if not _TOC_ENTRY_RE.match(value):
        return False
    if re.search(r"[.!?]\s+\S+", value):
        return False
    words = value.split()
    return 2 <= len(words) <= 14


def looks_like_tabular_line(text: str) -> bool:
    raw = str(text or "")
    value = clean_extracted_line(raw)
    if not value:
        return False
    if looks_like_table_or_image_caption(value):
        return True
    header_hits = len(_TABLE_HEADER_RE.findall(value))
    has_sentence_punct = bool(re.search(r"[.!?]$", value))
    if header_hits >= 2 and len(value.split()) <= 16 and (not has_sentence_punct or _TABLE_DELIMITER_RE.search(raw)):
        return True
    tokens = [token.strip("()[]{}:;,.") for token in value.split() if token.strip("()[]{}:;,.")]
    if len(tokens) < 3:
        return False
    code_like = sum(
        1
        for token in tokens
        if _CODE_HEAVY_TOKEN_RE.fullmatch(token)
        and (not re.fullmatch(r"[A-Za-z]+", token) or token.isupper())
    )
    table_cell = sum(1 for token in tokens if _TABLE_CELL_CODE_RE.fullmatch(token))
    long_alpha = sum(1 for token in tokens if re.fullmatch(r"[A-Za-z]{4,}", token))
    if _TABLE_DELIMITER_RE.search(raw) and table_cell >= 2 and long_alpha <= 4 and not has_sentence_punct:
        return True
    if tokens[0].isdigit() and code_like >= 2 and not has_sentence_punct:
        return True
    if table_cell >= max(3, len(tokens) // 2) and long_alpha <= 3 and not has_sentence_punct:
        return True
    return code_like >= max(3, len(tokens) // 2) and long_alpha <= 2 and not has_sentence_punct


def is_document_metadata_line(text: str) -> bool:
    value = clean_extracted_line(text)
    if not value:
        return False
    if _SIGNATURE_ROLE_LINE_RE.search(value):
        return True
    if _ELECTRONIC_SIGNATURE_RE.search(value):
        return True
    if len(_PRINT_METADATA_RE.findall(value)) >= 2:
        return True
    if _APPROVAL_NOTE_RE.search(value):
        return True
    if _DOCUMENT_METADATA_LINE_RE.match(value):
        return True
    return len(_DOCUMENT_METADATA_RE.findall(value)) >= 2


def should_skip_content_line(text: str) -> bool:
    """Return True when a line should not contribute to chunk body text."""
    value = clean_extracted_line(text)
    if not value:
        return True
    if is_document_metadata_line(value):
        return True
    if _PAGE_MARKER_RE.fullmatch(value) or looks_like_toc_entry(value):
        return True
    if looks_like_table_or_image_caption(value) or looks_like_tabular_line(value):
        return True
    if _FORM_LABEL_RE.search(value) and len(value.split()) <= 6:
        return True
    return False
