from __future__ import annotations

import re

_MESH_TAG_RE = re.compile(r"\[mesh(?:\s+terms)?\]", re.IGNORECASE)
_FIELD_TAG_RE = re.compile(r"\[[^\]]+\]", re.IGNORECASE)


def normalize_mesh_terms(raw_terms: str | list[str] | None) -> list[str]:
    """Split user input into search terms."""
    if isinstance(raw_terms, list):
        terms: list[str] = []
        for item in raw_terms:
            cleaned = str(item or "").strip()
            if cleaned and cleaned not in terms:
                terms.append(cleaned)
        return terms
    text = str(raw_terms or "").strip()
    if not text:
        return []
    parts = re.split(r"[;\n,]+", text)
    terms = []
    for part in parts:
        term = part.strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def _quote_if_needed(term: str) -> str:
    cleaned = term.strip().strip('"').strip()
    if not cleaned:
        return ""
    if " " in cleaned or "-" in cleaned:
        return f'"{cleaned}"'
    return cleaned


def format_pubmed_title_query(term: str, *, field: str = "Title") -> str:
    """Restrict a user term to a PubMed field (Title or Title/Abstract)."""
    cleaned = term.strip()
    if not cleaned:
        return ""
    if _MESH_TAG_RE.search(cleaned) or _FIELD_TAG_RE.search(cleaned):
        return cleaned
    field_tag = field if field.startswith("[") else f"[{field}]"
    quoted = _quote_if_needed(cleaned)
    return f"{quoted}{field_tag}" if quoted else ""


def build_pubmed_search_query(
    terms: list[str],
    *,
    mesh_mode: bool = False,
    title_only: bool = True,
) -> str:
    """Build a PubMed esearch query.

    Plain mode defaults to Title-field clauses so results track the user's query in the
    title, not incidental body mentions. Multi-word free-text terms are expanded to an
    AND of per-word Title clauses (so ``metformin type 2 diabetes`` finds titles that
    contain those words, not only the exact contiguous phrase). MeSH mode uses PubMed
    MeSH tags only.
    """
    parts: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if not cleaned:
            continue
        if mesh_mode:
            formatted = format_pubmed_mesh_query(cleaned)
        elif _MESH_TAG_RE.search(cleaned) or _FIELD_TAG_RE.search(cleaned):
            formatted = cleaned
        elif title_only:
            field = "Title"
            # Expand multi-word free text into AND of title words (not one exact phrase).
            words = [w.strip().strip('"') for w in cleaned.split() if w.strip().strip('"')]
            if len(words) >= 2:
                word_clauses = [
                    c for w in words if (c := format_pubmed_title_query(w, field=field))
                ]
                formatted = " AND ".join(word_clauses) if word_clauses else ""
            else:
                formatted = format_pubmed_title_query(cleaned, field=field)
        else:
            formatted = format_pubmed_title_query(cleaned, field="Title/Abstract")
        if formatted and formatted not in parts:
            parts.append(formatted)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({part})" for part in parts)


def format_pubmed_mesh_query(term: str) -> str:
    """Format a single term for PubMed esearch (MeSH when not already tagged)."""
    cleaned = term.strip()
    if not cleaned:
        return ""
    if _MESH_TAG_RE.search(cleaned):
        return cleaned
    if cleaned.startswith('"') and cleaned.endswith('"'):
        return f"{cleaned}[MeSH Terms]"
    return f'"{cleaned}"[MeSH Terms]'


def format_clinicaltrials_query(term: str) -> str:
    """ClinicalTrials.gov plain term; strip MeSH tags if present."""
    cleaned = term.strip()
    cleaned = _MESH_TAG_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip('"').strip()
    return cleaned


def build_clinicaltrials_title_query(terms: list[str]) -> str:
    """Build a ClinicalTrials.gov title query from the user's terms only."""
    parts: list[str] = []
    for term in terms:
        cleaned = format_clinicaltrials_query(term)
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(parts)


def title_matches_query(title: str, terms: list[str]) -> bool:
    """True when the title matches any user query term.

    - Exact phrase match (case-insensitive), or
    - For multi-word terms: all significant tokens appear in the title
      (so ``metformin type 2 diabetes`` matches titles that contain those words
      even if not as one contiguous phrase).
    """
    haystack = str(title or "").lower()
    if not haystack:
        return False
    for term in terms:
        needle = format_clinicaltrials_query(term).lower()
        if not needle:
            continue
        if needle in haystack:
            return True
        tokens = [tok for tok in re.split(r"\s+", needle) if len(tok) > 1]
        if len(tokens) >= 2 and all(tok in haystack for tok in tokens):
            return True
    return False


def build_ikm_pubmed_query(
    *,
    query: str,
    mesh_terms: list[str] | None = None,
    title: str | None = None,
) -> str:
    """Build a PubMed query from DMS PubMedSearchRequest fields."""
    parts: list[str] = []
    mesh_list = normalize_mesh_terms(mesh_terms)
    title_q = (title or "").strip()
    base = (query or "").strip()

    if mesh_list and not base and not title_q:
        result = build_pubmed_search_query(mesh_list, mesh_mode=True)
        if not result:
            raise ValueError("PubMed search requires query, title, or MeSH terms")
        return result

    if title_q:
        formatted_title = format_pubmed_title_query(title_q, field="Title")
        if formatted_title:
            parts.append(formatted_title)

    if mesh_list:
        mesh_q = build_pubmed_search_query(mesh_list, mesh_mode=True)
        if mesh_q:
            parts.append(mesh_q)

    if base:
        if _MESH_TAG_RE.search(base) or _FIELD_TAG_RE.search(base):
            parts.insert(0, base)
        else:
            formatted = format_pubmed_title_query(base, field="Title")
            if formatted and formatted not in parts:
                parts.insert(0, formatted)

    if not parts:
        raise ValueError("PubMed search requires query, title, or MeSH terms")
    if len(parts) == 1:
        return parts[0]
    return " AND ".join(f"({part})" for part in parts)


def query_title_accuracy(title: str, query: str) -> int:
    """Score 0-100 for title relevance to the user query.

    Technique 2 + 5:
    - Exact title match → 100
    - Full phrase anywhere in the title → high band (90-96), adjusted by coverage
      (late position is still treated as a valid strong match)
    - Most query tokens present → 70-85
    - Few query tokens present → 35-65
    - Weak/no meaningful overlap → 0-20

    Uses only the raw query text; no synonym expansion.
    """
    title_l = str(title or "").lower().strip()
    query_l = str(query or "").lower().strip()
    if not title_l or not query_l:
        return 0

    phrases = [p.strip() for p in re.split(r"[;\n,]+", query_l) if p.strip()]
    primary = phrases[0] if phrases else query_l
    if not primary:
        return 0

    # Exact title match (ignore trailing punctuation).
    title_core = re.sub(r"[.!?:;,\s]+$", "", title_l)
    if title_core == primary:
        return 100

    title_len = max(len(title_l), 1)
    phrase_len = max(len(primary), 1)
    coverage = phrase_len / title_len

    # Full phrase present anywhere → strong match (do not punish late position).
    if re.search(rf"(?<![a-z0-9]){re.escape(primary)}(?![a-z0-9])", title_l):
        # Coverage gently separates short topic-focused titles from long mentions.
        return max(90, min(96, 90 + int(round(coverage * 20))))

    # No full phrase: score by token completeness (Technique 5).
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", primary) if len(tok) > 1]
    if not tokens:
        return 0

    title_token_set = set(re.findall(r"[a-z0-9]+", title_l))
    hits = sum(1 for tok in tokens if tok in title_token_set)
    if hits == 0:
        return 0

    ratio = hits / len(tokens)
    if ratio >= 0.75:
        # Most/all query words present.
        return max(70, min(85, int(round(70 + ratio * 15))))
    if ratio >= 0.4:
        # Partial overlap.
        return max(35, min(65, int(round(35 + ratio * 40))))
    # Weak overlap only.
    return max(5, min(20, int(round(ratio * 40))))
