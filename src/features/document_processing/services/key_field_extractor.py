from __future__ import annotations

from typing import Any

from src.features.document_processing.metadata.field_extractor import extract_metadata_fields


def fields_list_to_map(extracted: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for item in extracted:
        field_name = str(item.get("field_name") or "").strip()
        if not field_name:
            continue
        payload[field_name] = {
            "value": item.get("value"),
            "confidence": item.get("confidence"),
        }
    return payload


def extract_key_fields(
    blocks: list[Any],
    fields: list[dict[str, Any]],
    *,
    extraction_model: dict[str, Any] | None = None,
    embed_fn: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    extracted = extract_metadata_fields(
        blocks,
        fields,
        extraction_model=extraction_model,
        embed_fn=embed_fn,
    )
    return extracted, fields_list_to_map(extracted)
