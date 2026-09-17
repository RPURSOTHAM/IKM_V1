"""Resolve effective metadata fields for processor queue payloads."""

from __future__ import annotations

from typing import Any

from src.features.document_types.domain.models import MetadataFieldRecord


def metadata_field_to_processing_dict(field: MetadataFieldRecord) -> dict[str, Any]:
    return {
        "metadata_field_id": field.metadata_field_id,
        "field_name": field.field_name,
        "display_label": field.display_label,
        "data_type": field.data_type,
        "required": field.required,
        "enum_values": field.enum_values,
        "max_length": field.max_length,
        "field_group": field.field_group,
        "is_active": field.is_active,
    }


def resolve_metadata_fields_for_type(document_type_id: str) -> list[dict[str, Any]]:
    from src.features.document_types.infrastructure.document_type_repository import get_document_type_store

    store = get_document_type_store()
    if not store or not document_type_id:
        return []
    bundle = store.resolve_metadata_bundle(document_type_id)
    return [
        metadata_field_to_processing_dict(field)
        for field in bundle.effective
        if field.is_active
    ]
