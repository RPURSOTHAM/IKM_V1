"""Normalize document-type metadata field definitions for processors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FIELD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "document_version": ("version", "revision", "rev", "revision no", "rev no", "rev."),
    "document_number": ("document no", "doc no", "document number", "sop number", "procedure number"),
    "document_name": ("document name", "doc name", "file name"),
    "document_title": ("title", "subject", "document title", "sop title", "gop title"),
    "title": ("title", "subject", "document title"),
    "description": ("description", "summary", "abstract"),
    "document_type": ("document type", "doc type", "type"),
    "document_classification": ("classification", "document classification", "security classification"),
    "language": ("language", "lang"),
    "status": ("status", "document status", "state"),
    "effective_date": ("effective date", "effective from", "date effective"),
    "review_date": ("review date", "next review", "review due"),
    "equipment_name": ("equipment name", "equipment", "asset name"),
}

SKIP_CONTENT_EXTRACTION: frozenset[str] = frozenset(
    {
        "id",
        "document_uuid",
        "checksum_value",
        "uploaded_date",
        "uploaded_by",
    }
)


@dataclass(frozen=True)
class MetadataFieldDefinition:
    field_name: str
    display_label: str
    data_type: str = "string"
    required: bool = False
    enum_values: tuple[str, ...] = field(default_factory=tuple)
    max_length: int | None = None
    metadata_field_id: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def searchable_labels(self) -> tuple[str, ...]:
        labels = [self.field_name, self.display_label, self.field_name.replace("_", " "), *self.aliases]
        labels.extend(FIELD_SYNONYMS.get(self.field_name.lower(), ()))
        seen: set[str] = set()
        ordered: list[str] = []
        for label in labels:
            key = _normalize_label(label)
            if key and key not in seen:
                seen.add(key)
                ordered.append(label.strip())
        return tuple(ordered)


def _normalize_label(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def normalize_field_definition(raw: dict[str, Any]) -> MetadataFieldDefinition | None:
    field_name = str(raw.get("field_name") or raw.get("name") or "").strip()
    if not field_name:
        return None
    display_label = str(raw.get("display_label") or raw.get("field_label") or field_name.replace("_", " ")).strip()
    enum_values = raw.get("enum_values") or ()
    if isinstance(enum_values, list):
        enum_values = tuple(str(item).strip() for item in enum_values if str(item).strip())
    else:
        enum_values = ()
    max_length = raw.get("max_length")
    aliases = raw.get("aliases") or ()
    if isinstance(aliases, list):
        aliases = tuple(str(item).strip() for item in aliases if str(item).strip())
    else:
        aliases = ()
    return MetadataFieldDefinition(
        field_name=field_name,
        display_label=display_label,
        data_type=str(raw.get("data_type") or "string").strip().lower(),
        required=bool(raw.get("required", False)),
        enum_values=enum_values,
        max_length=int(max_length) if max_length is not None else None,
        metadata_field_id=str(raw.get("metadata_field_id") or "") or None,
        aliases=aliases,
    )


def normalize_field_definitions(fields: list[dict[str, Any]]) -> list[MetadataFieldDefinition]:
    normalized: list[MetadataFieldDefinition] = []
    for item in fields:
        field_def = normalize_field_definition(item)
        if field_def is not None:
            normalized.append(field_def)
    return normalized
