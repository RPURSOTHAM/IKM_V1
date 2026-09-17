"""
Page-1 metadata extraction via colon separators only.

Rules:
  - Use text from page 1 body only (caller supplies lines; not header/footer).
  - Key = text before ':'; value = text after ':' (exact, whitespace-trimmed).
  - Metadata only keeps mapped control slots (+ optional document_type).
  - Do not invent values; no dumping of body Key: Value noise into metadata.
"""

from __future__ import annotations

from typing import Any


def normalize_meta_key(key: str) -> str:
    """Lowercase alphanumerics only — for slot matching, not for discovery."""
    out: list[str] = []
    for ch in str(key or "").casefold():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != " ":
            out.append(" ")
    return "".join(out).strip()


def is_valid_meta_key(key: str) -> bool:
    text = " ".join(str(key or "").split()).strip().rstrip(".")
    if not text or len(text) > 60:
        return False
    if not text[0].isalpha():
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    if letters < 4:
        return False
    if len(text.split()) > 6:
        return False
    return True


def metadata_slot_for_key(key: str) -> str | None:
    """Map a discovered key into fixed schema slots by simple word presence."""
    norm = normalize_meta_key(key)
    if not norm:
        return None
    words = set(norm.split())
    if "title" in words and "table" not in words:
        return "title"
    if ("document" in words or "doc" in words) and (
        "no" in words or "number" in words or "id" in words
    ):
        return "document_no"
    if "version" in words:
        return "version_no"
    if "effective" in words and "date" in words:
        return "effective_date"
    if "review" in words and "date" in words:
        return "review_date"
    return None


def _line_has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def _line_has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _slot_value_plausible(slot: str, value: str) -> bool:
    """Reject body prose wrongly paired into control slots."""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return False
    words = text.split()
    if slot == "title":
        return 2 <= len(words) <= 40 and len(text) <= 240
    if slot == "document_no":
        # IDs are short tokens with digits/hyphens; reject plain phrases.
        return (
            len(text) <= 64
            and len(words) <= 6
            and (_line_has_digit(text) or "-" in text or "/" in text)
        )
    if slot == "version_no":
        return len(text) <= 24 and len(words) <= 3 and _line_has_digit(text)
    if slot in {"effective_date", "review_date"}:
        return len(text) <= 40 and len(words) <= 6 and _line_has_digit(text)
    return True


def _parse_colon_segments(line: str) -> tuple[list[str], str]:
    """
    Walk ':' separators with string split only.

    Returns (keys, trailing_value).
    - "Title: Foo" -> (['Title'], 'Foo')
    - "Document No.: Effective Date:" -> (['Document No.', 'Effective Date'], '')
    - "Version No: 4.0" -> (['Version No'], '4.0')
    """
    text = " ".join(str(line or "").split()).strip()
    if ":" not in text:
        return [], ""

    keys: list[str] = []
    rest = text
    while ":" in rest:
        before, after = rest.split(":", 1)
        before = before.strip()
        after = after.strip()
        if before and is_valid_meta_key(before):
            keys.append(before)
        elif before and not is_valid_meta_key(before):
            # Not a metadata key (e.g. timestamps "12:50:04") — ignore line.
            return [], ""
        rest = after
        if rest and ":" not in rest:
            return keys, rest
        if not rest:
            return keys, ""
    return keys, rest


def colon_fields_from_page_lines(lines: list[str]) -> list[dict[str, str]]:
    """
    Extract {name, value} pairs from page-1 lines using ':' only.

    Value is preserved exactly as appearing in the source line (after strip).
    """
    fields: list[dict[str, str]] = []
    pending: list[str] = []
    digit_buf: list[str] = []

    def _emit(name: str, value: str) -> None:
        name = " ".join(str(name or "").split()).strip()
        value = str(value or "").strip()
        if name and value:
            fields.append({"name": name, "value": value})

    def _clear_pending() -> None:
        nonlocal pending, digit_buf
        pending = []
        digit_buf = []

    for raw in lines:
        line = " ".join(str(raw or "").split()).strip()
        if not line:
            continue

        if ":" not in line:
            if not pending:
                continue
            tokens = line.split()
            if len(pending) == 1:
                _emit(pending[0], line)
                _clear_pending()
                continue
            # Multiple empty keys waiting.
            if len(tokens) >= len(pending):
                # Require a digit somewhere so prose like "Signature and Date"
                # is not treated as Document No / Effective Date values.
                if not _line_has_digit(line):
                    continue
                for key, token in zip(pending, tokens):
                    _emit(key, token)
                _clear_pending()
                continue
            # Digit-only short lines (e.g. "4.0" then "29/04/2028") fill FIFO.
            if _line_has_digit(line) and not _line_has_letter(line):
                digit_buf.append(line)
                if len(digit_buf) >= len(pending):
                    for key, val in zip(pending, digit_buf):
                        _emit(key, val)
                    _clear_pending()
                continue
            # Alphabetic / mixed noise (Signatures, 14-Apr-2026) — skip, keep pending.
            continue

        keys, value = _parse_colon_segments(line)
        if not keys:
            continue

        _clear_pending()
        if value:
            if len(keys) == 1:
                _emit(keys[0], value)
            else:
                # First keys empty; only last segment is a value for the last key.
                for key in keys[:-1]:
                    pending.append(key)
                _emit(keys[-1], value)
                digit_buf = []
        else:
            pending = list(keys)
            digit_buf = []

    return _unique_named_fields(fields)


def metadata_from_colon_fields(fields: list[dict[str, str]]) -> dict[str, str]:
    """
    Fill only fixed control slots from discovered Key/Value pairs.

    Does not dump arbitrary keys into metadata (avoids body/definitions pollution).
    """
    metadata: dict[str, str] = {
        "title": "",
        "document_no": "",
        "version_no": "",
        "effective_date": "",
        "review_date": "",
    }
    for item in fields:
        name = str(item.get("name") or item.get("label") or "").strip()
        value = str(item.get("value") or "").strip()
        if not name or not value:
            continue
        slot = metadata_slot_for_key(name)
        if not slot:
            continue
        if not _slot_value_plausible(slot, value):
            continue
        existing = metadata[slot]
        if not existing:
            metadata[slot] = value
        elif slot == "title" and len(value) > len(existing):
            metadata[slot] = value
    return metadata


def document_control_from_fields(fields: list[dict[str, str]]) -> dict[str, str]:
    """Deprecated helper retained for callers; prefer metadata slots only."""
    meta = metadata_from_colon_fields(fields)
    return {
        "document_no": meta.get("document_no") or "",
        "version_no": meta.get("version_no") or "",
        "effective_date": meta.get("effective_date") or "",
        "review_date": meta.get("review_date") or "",
    }


def column_key(label: str) -> str:
    """Dynamic row-key from a table header cell (no hardcoded column names)."""
    norm = normalize_meta_key(label)
    if not norm:
        return ""
    return "_".join(norm.split())


def extract_page1_metadata(lines: list[str]) -> dict[str, Any]:
    """Convenience: fields + slotted metadata from page-1 body lines."""
    fields = colon_fields_from_page_lines(lines)
    return {
        "fields": fields,
        "metadata": metadata_from_colon_fields(fields),
    }


def _unique_named_fields(fields: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep last non-empty value per normalized key (page-1 may repeat labels)."""
    best: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for field in fields:
        name = str(field.get("name") or "").strip()
        value = str(field.get("value") or "").strip()
        if not name or not value:
            continue
        key = normalize_meta_key(name)
        if not key:
            continue
        if key not in best:
            order.append(key)
        best[key] = {"name": name, "value": value}
    return [best[key] for key in order]
