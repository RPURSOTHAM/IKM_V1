"""Phase 4 document validation engine — pure rules over effective fields + extraction results."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

DEFAULT_CONFIDENCE_THRESHOLD = 0.85
WARNING_CONFIDENCE_FLOOR = 0.60

_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{4}/\d{2}/\d{2}$"),
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),
)


def _field_name(field: dict[str, Any]) -> str:
    return str(field.get("field_name") or field.get("name") or "").strip()


def _field_type(field: dict[str, Any]) -> str:
    raw = str(field.get("type") or field.get("field_type") or field.get("data_type") or "string").strip().lower()
    aliases = {
        "text": "string",
        "number": "float",
        "datetime": "date",
        "int": "integer",
        "double": "float",
        "bool": "boolean",
    }
    return aliases.get(raw, raw)


def _is_required(field: dict[str, Any]) -> bool:
    return bool(field.get("required", False))


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _as_float_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_value_type(value: Any, field_type: str) -> str | None:
    """Return an error reason code when value does not match field_type; else None."""
    if _is_empty(value):
        return None  # emptiness handled by required-field rules
    text = str(value).strip()
    ft = (field_type or "string").lower()

    if ft == "string":
        return None
    if ft == "integer":
        try:
            # Reject floats like "1.5"
            if isinstance(value, bool):
                return "invalid_integer"
            if isinstance(value, int):
                return None
            if isinstance(value, float) and value.is_integer():
                return None
            if re.fullmatch(r"[+-]?\d+", text):
                return None
            return "invalid_integer"
        except Exception:
            return "invalid_integer"
    if ft == "float":
        try:
            if isinstance(value, bool):
                return "invalid_float"
            float(text)
            return None
        except Exception:
            return "invalid_float"
    if ft == "boolean":
        if isinstance(value, bool):
            return None
        if text.lower() in {"true", "false", "1", "0", "yes", "no"}:
            return None
        return "invalid_boolean"
    if ft == "date":
        if any(p.match(text) for p in _DATE_PATTERNS):
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
                try:
                    datetime.strptime(text, fmt)
                    return None
                except ValueError:
                    continue
        try:
            # ISO datetime prefix
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            return None
        except Exception:
            return "invalid_date"
        return "invalid_date"
    return None


def normalize_extracted_map(
    extracted_fields: list[dict[str, Any]] | dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize extraction results to {field_name: {value, confidence}}."""
    out: dict[str, dict[str, Any]] = {}
    if isinstance(extracted_fields, dict):
        for name, payload in extracted_fields.items():
            key = str(name).strip()
            if not key:
                continue
            if isinstance(payload, dict):
                out[key] = {
                    "value": payload.get("value"),
                    "confidence": _as_float_confidence(payload.get("confidence")),
                }
            else:
                out[key] = {"value": payload, "confidence": None}
        return out
    if isinstance(extracted_fields, list):
        for item in extracted_fields:
            if not isinstance(item, dict):
                continue
            key = _field_name(item)
            if not key:
                continue
            out[key] = {
                "value": item.get("value"),
                "confidence": _as_float_confidence(item.get("confidence")),
            }
    return out


def map_validation_status_to_document_status(status: str) -> str:
    mapping = {"PASS": "VALID", "WARNING": "WARNING", "FAIL": "INVALID"}
    return mapping.get(str(status or "").upper(), "INVALID")


def run_document_validation(
    *,
    effective_fields: list[dict[str, Any]],
    extracted_fields: list[dict[str, Any]] | dict[str, Any] | None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Validate extracted metadata against effective document-type fields."""
    threshold = float(confidence_threshold) if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
    extracted = normalize_extracted_map(extracted_fields)

    missing_required_fields: list[str] = []
    invalid_fields: list[dict[str, Any]] = []
    low_confidence_fields: list[dict[str, Any]] = []

    for field in effective_fields or []:
        if not isinstance(field, dict):
            continue
        name = _field_name(field)
        if not name:
            continue
        entry = extracted.get(name) or {}
        value = entry.get("value")
        confidence = entry.get("confidence")

        if _is_required(field) and _is_empty(value):
            missing_required_fields.append(name)

        if not _is_empty(value):
            reason = validate_value_type(value, _field_type(field))
            if reason:
                invalid_fields.append({"field": name, "reason": reason, "value": value})

        if confidence is not None and confidence < threshold:
            low_confidence_fields.append({"field": name, "confidence": confidence})

    missing_required_fields = sorted(set(missing_required_fields))
    # Deduplicate invalid/low-confidence by field name (keep first)
    seen_invalid: set[str] = set()
    dedup_invalid: list[dict[str, Any]] = []
    for item in invalid_fields:
        key = str(item.get("field") or "")
        if key in seen_invalid:
            continue
        seen_invalid.add(key)
        dedup_invalid.append(item)
    invalid_fields = dedup_invalid

    seen_low: set[str] = set()
    dedup_low: list[dict[str, Any]] = []
    for item in low_confidence_fields:
        key = str(item.get("field") or "")
        if key in seen_low:
            continue
        seen_low.add(key)
        dedup_low.append(item)
    low_confidence_fields = sorted(dedup_low, key=lambda x: str(x.get("field") or ""))

    if missing_required_fields or invalid_fields:
        status = "FAIL"
    elif low_confidence_fields:
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "status": status,
        "document_status": map_validation_status_to_document_status(status),
        "missing_required_fields": missing_required_fields,
        "invalid_fields": invalid_fields,
        "low_confidence_fields": low_confidence_fields,
        "confidence_threshold": threshold,
        "validated_field_count": len([f for f in (effective_fields or []) if _field_name(f)]),
        "extracted_field_count": len(extracted),
    }
