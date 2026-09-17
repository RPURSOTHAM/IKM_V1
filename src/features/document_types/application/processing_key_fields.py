"""Convert Phase 1 key field definitions into processor extraction payloads."""

from __future__ import annotations

from typing import Any

_FIELD_TYPE_MAP = {
    "string": "string",
    "text": "string",
    "number": "number",
    "integer": "number",
    "float": "number",
    "date": "date",
    "datetime": "date",
    "boolean": "boolean",
}


def _map_field_type(field_type: str | None) -> str:
    normalized = str(field_type or "string").strip().lower()
    return _FIELD_TYPE_MAP.get(normalized, "string")


def key_field_to_processing_dict(field: dict[str, Any]) -> dict[str, Any]:
    description = str(field.get("description") or "").strip()
    field_name = str(field.get("field_name") or "").strip()
    aliases: list[str] = []
    if description:
        aliases.append(description)
    return {
        "field_name": field_name,
        "display_label": description or field_name.replace("_", " "),
        "data_type": _map_field_type(field.get("field_type")),
        "required": bool(field.get("required", False)),
        "aliases": aliases,
        "key_field_id": field.get("id"),
    }


def resolve_key_fields_for_type(document_type_id: str) -> list[dict[str, Any]]:
    from src.features.document_types.application.document_type_service import DocumentTypeService

    bundle = DocumentTypeService().resolve_effective_fields(document_type_id)
    fields = bundle.get("fields") if isinstance(bundle, dict) else []
    if not isinstance(fields, list):
        return []
    return [key_field_to_processing_dict(item) for item in fields if isinstance(item, dict)]
