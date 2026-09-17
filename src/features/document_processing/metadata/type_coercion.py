"""Coerce and validate extracted metadata values by declared data type."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$"),
)

_DATETIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$"),
    re.compile(r"^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}(:\d{2})?$"),
)

_TRUTHY = frozenset({"true", "yes", "y", "1", "on"})
_FALSY = frozenset({"false", "no", "n", "0", "off"})


def coerce_metadata_value(raw_value: str | None, data_type: str, *, enum_values: tuple[str, ...] = ()) -> Any | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None

    normalized_type = str(data_type or "string").strip().lower()
    if normalized_type == "boolean":
        token = value.lower()
        if token in _TRUTHY:
            return True
        if token in _FALSY:
            return False
        return None
    if normalized_type == "number":
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
        if not match:
            return None
        number_text = match.group(0)
        return float(number_text) if "." in number_text else int(number_text)
    if normalized_type == "date":
        return value if any(pattern.match(value) for pattern in _DATE_PATTERNS) else None
    if normalized_type == "datetime":
        if any(pattern.match(value) for pattern in _DATETIME_PATTERNS):
            return value
        if any(pattern.match(value) for pattern in _DATE_PATTERNS):
            return value
        return None
    if normalized_type == "enum":
        if not enum_values:
            return value
        lowered = value.lower()
        for option in enum_values:
            if option.lower() == lowered:
                return option
        return None
    return value
