from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.features.document_types.application.field_group_service import (
    DEFAULT_FIELD_GROUP,
    DEFAULT_FIELD_SEQUENCE,
    DEFAULT_GROUP_LABEL,
    DEFAULT_GROUP_SEQUENCE,
)


class MetadataDataType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ENUM = "enum"
    TEXT = "text"

    @classmethod
    def values(cls) -> frozenset[str]:
        return frozenset(m.value for m in cls)


@dataclass
class DocumentTypeRecord:
    document_type_id: str
    repository_id: str | None
    name: str
    code: str | None
    description: str | None
    parent_document_type_id: str | None
    depth_level: int
    is_system: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: str | None = None
    updated_by: str | None = None


class KeyFieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    # Backward-compatible aliases retained for Phase 2 payloads / migrations.
    NUMBER = "number"
    DATETIME = "datetime"
    TEXT = "text"

    @classmethod
    def values(cls) -> frozenset[str]:
        return frozenset(m.value for m in cls)

    @classmethod
    def canonical(cls, value: str | None) -> str:
        """Normalize aliases to Phase 3 canonical field types."""
        normalized = str(value or cls.STRING.value).strip().lower()
        alias_map = {
            "text": cls.STRING.value,
            "number": cls.FLOAT.value,
            "datetime": cls.DATE.value,
        }
        mapped = alias_map.get(normalized, normalized)
        allowed = {
            cls.STRING.value,
            cls.INTEGER.value,
            cls.FLOAT.value,
            cls.BOOLEAN.value,
            cls.DATE.value,
        }
        return mapped if mapped in allowed else cls.STRING.value


@dataclass
class KeyFieldDefinitionRecord:
    id: str
    document_type_id: str
    field_name: str
    field_type: str
    required: bool
    default_value: Any | None
    description: str | None
    created_at: datetime
    updated_at: datetime
    inherited_from_document_type_id: str | None = None


@dataclass
class MetadataFieldRecord:
    metadata_field_id: str
    document_type_id: str
    field_name: str
    display_label: str
    data_type: str
    required: bool
    default_value: Any | None
    enum_values: list[str] | None
    max_length: int | None
    is_system: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    field_group: str = DEFAULT_FIELD_GROUP
    group_label: str = DEFAULT_GROUP_LABEL
    group_sequence: int = DEFAULT_GROUP_SEQUENCE
    field_sequence: int = DEFAULT_FIELD_SEQUENCE
    inherited_from_document_type_id: str | None = None


@dataclass
class MetadataFieldGroup:
    field_group: str
    group_label: str
    group_sequence: int
    fields: list[MetadataFieldRecord] = field(default_factory=list)


@dataclass
class MetadataFieldsBundle:
    own: list[MetadataFieldRecord] = field(default_factory=list)
    inherited: list[MetadataFieldRecord] = field(default_factory=list)
    effective: list[MetadataFieldRecord] = field(default_factory=list)

    def to_dicts(self) -> dict[str, Any]:
        return {
            "own": {"groups": [group_to_dict(g) for g in group_fields(self.own)]},
            "inherited": {"groups": [group_to_dict(g) for g in group_fields(self.inherited)]},
            "effective": {"groups": [group_to_dict(g) for g in group_fields(self.effective)]},
        }


def field_sort_key(field: MetadataFieldRecord) -> tuple[int, int, str]:
    return (field.group_sequence, field.field_sequence, field.field_name)


def group_fields(fields: list[MetadataFieldRecord]) -> list[MetadataFieldGroup]:
    if not fields:
        return []
    ordered = sorted(fields, key=field_sort_key)
    groups: list[MetadataFieldGroup] = []
    current_key: str | None = None
    current_group: MetadataFieldGroup | None = None
    for item in ordered:
        if current_key != item.field_group or current_group is None:
            current_group = MetadataFieldGroup(
                field_group=item.field_group,
                group_label=item.group_label,
                group_sequence=item.group_sequence,
                fields=[],
            )
            groups.append(current_group)
            current_key = item.field_group
        current_group.fields.append(item)
    return groups


def group_to_dict(group: MetadataFieldGroup) -> dict[str, Any]:
    return {
        "field_group": group.field_group,
        "group_label": group.group_label,
        "group_sequence": group.group_sequence,
        "fields": [_field_to_dict(f) for f in group.fields],
    }


def key_field_to_dict(field: KeyFieldDefinitionRecord) -> dict[str, Any]:
    type_alias = {
        "string": "string",
        "text": "string",
        "number": "float",
        "integer": "integer",
        "float": "float",
        "date": "date",
        "datetime": "date",
        "boolean": "boolean",
    }
    normalized_type = type_alias.get(str(field.field_type).strip().lower(), "string")
    payload: dict[str, Any] = {
        "id": field.id,
        "field_id": field.id,
        "document_type_id": field.document_type_id,
        "field_name": field.field_name,
        "name": field.field_name,
        "field_type": field.field_type,
        "type": normalized_type,
        "required": field.required,
        "default_value": field.default_value,
        "description": field.description,
        "inherited": field.inherited_from_document_type_id is not None,
        "created_at": field.created_at.isoformat() + "Z",
        "updated_at": field.updated_at.isoformat() + "Z",
    }
    if field.inherited_from_document_type_id is not None:
        payload["inherited_from_document_type_id"] = field.inherited_from_document_type_id
    return payload


def document_type_to_dict(record: DocumentTypeRecord, *, metadata_fields: MetadataFieldsBundle | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "document_type_id": record.document_type_id,
        "repository_id": record.repository_id,
        "name": record.name,
        "code": record.code,
        "description": record.description,
        "parent_document_type_id": record.parent_document_type_id,
        "depth_level": record.depth_level,
        "is_system": record.is_system,
        "is_active": record.is_active,
        "created_at": record.created_at.isoformat() + "Z",
        "updated_at": record.updated_at.isoformat() + "Z",
    }
    if record.created_by is not None:
        payload["created_by"] = record.created_by
    if record.updated_by is not None:
        payload["updated_by"] = record.updated_by
    if metadata_fields is not None:
        payload["metadata_fields"] = metadata_fields.to_dicts()
    return payload


def _field_to_dict(field: MetadataFieldRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata_field_id": field.metadata_field_id,
        "document_type_id": field.document_type_id,
        "field_name": field.field_name,
        "display_label": field.display_label,
        "data_type": field.data_type,
        "required": field.required,
        "default_value": field.default_value,
        "enum_values": field.enum_values,
        "max_length": field.max_length,
        "field_group": field.field_group,
        "group_label": field.group_label,
        "group_sequence": field.group_sequence,
        "field_sequence": field.field_sequence,
        "is_system": field.is_system,
        "is_editable": not field.is_system,
        "is_active": field.is_active,
        "created_at": field.created_at.isoformat() + "Z",
        "updated_at": field.updated_at.isoformat() + "Z",
    }
    if field.inherited_from_document_type_id is not None:
        payload["inherited_from_document_type_id"] = field.inherited_from_document_type_id
    return payload


def field_to_dict(field: MetadataFieldRecord) -> dict[str, Any]:
    return _field_to_dict(field)


def flatten_grouped_fields(grouped: dict[str, Any], scope: str = "effective") -> list[dict[str, Any]]:
    """Return a priority-sorted flat field list from a grouped metadata_fields payload."""
    scope_payload = grouped.get(scope) or {}
    flat: list[dict[str, Any]] = []
    for group in scope_payload.get("groups") or []:
        flat.extend(group.get("fields") or [])
    return flat
