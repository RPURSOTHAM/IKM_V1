"""Map configured key-field IDs onto extracted extraction results."""

from __future__ import annotations

import re
from typing import Any

from src.features.logical_folders.domain.exceptions import FolderHierarchyConfigurationError

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_FIELD_TOKENS = re.compile(r"[a-z0-9]+")
_SENTENCE_PUNCT = re.compile(r"[.?!;]")
_MAX_AUTO_FOLDER_DEPTH = 4
_MAX_FOLDER_VALUE_WORDS = 6

# Linguistic field roles — not business labels (no Department/SOP/Clinical allowlist).
_INSTANCE_FIELD_TOKENS = frozenset(
    {
        "title",
        "filename",
        "name",
        "number",
        "no",
        "id",
        "identifier",
        "version",
        "revision",
        "date",
        "description",
        "narrative",
        "comment",
        "comments",
        "remark",
        "remarks",
        "owner",
        "author",
        "page",
        "pages",
    }
)
_GROUPING_FIELD_TOKENS = frozenset(
    {
        "type",
        "category",
        "class",
        "classification",
        "kind",
        "group",
        "department",
        "unit",
        "area",
        "domain",
        "topic",
        "status",
    }
)


def normalize_folder_name(value: Any) -> str | None:
    """Generic storage-safe normalization. Does not map onto business categories."""
    if value is None:
        return None
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\\", " ").replace("/", " ").replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    return text[:256]


def _id_variants(raw: Any) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return set()
    return {text, text.lower()}


def catalog_key_fields(
    *,
    settings: dict[str, Any] | None,
    document_type_fields: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Index key-field definitions by id (and lowercase id) plus field name."""
    by_id: dict[str, dict[str, Any]] = {}

    def _register(field_id: str | None, field_name: str | None, extra: dict[str, Any] | None = None) -> None:
        fid = str(field_id or "").strip()
        name = str(field_name or "").strip()
        if not fid and not name:
            return
        record = {"field_id": fid or None, "field_name": name or None, **(extra or {})}
        for key in _id_variants(fid):
            by_id[key] = record
        if name:
            by_id[f"name:{name.lower()}"] = record

    for item in list((settings or {}).get("key_fields") or []):
        if not isinstance(item, dict):
            continue
        _register(
            item.get("field_id") or item.get("id") or item.get("key_field_id"),
            item.get("name") or item.get("field_name"),
        )
    for item in document_type_fields or []:
        if not isinstance(item, dict):
            continue
        _register(
            item.get("id") or item.get("key_field_id") or item.get("field_id"),
            item.get("field_name") or item.get("name"),
        )
    return by_id


def parse_folder_hierarchy(settings: dict[str, Any] | None) -> list[str]:
    raw = (settings or {}).get("folder_hierarchy")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise FolderHierarchyConfigurationError(
            "folder_hierarchy must be an array of key-field identifiers.",
            details={"folder_hierarchy": raw},
        )
    ids: list[str] = []
    seen: set[str] = set()
    for item in raw:
        field_id = str(item or "").strip()
        if not field_id:
            raise FolderHierarchyConfigurationError(
                "folder_hierarchy entries must be non-empty key-field identifiers.",
            )
        key = field_id.lower()
        if key in seen:
            raise FolderHierarchyConfigurationError(
                "folder_hierarchy contains duplicate key-field identifiers.",
                details={"field_id": field_id},
            )
        seen.add(key)
        ids.append(field_id)
    return ids


def validate_folder_hierarchy_config(
    settings: dict[str, Any] | None,
    *,
    document_type_fields: list[dict[str, Any]] | None = None,
) -> list[str]:
    ids = parse_folder_hierarchy(settings)
    if not ids:
        return []
    catalog = catalog_key_fields(settings=settings, document_type_fields=document_type_fields)
    for field_id in ids:
        if field_id not in catalog and field_id.lower() not in catalog:
            raise FolderHierarchyConfigurationError(
                "folder_hierarchy references a key field that is not configured on this repository.",
                details={"field_id": field_id},
            )
    return ids


def _extracted_items(extracted_fields: Any) -> list[dict[str, Any]]:
    if isinstance(extracted_fields, list):
        return [item for item in extracted_fields if isinstance(item, dict)]
    if isinstance(extracted_fields, dict):
        items: list[dict[str, Any]] = []
        for key, value in extracted_fields.items():
            if isinstance(value, dict):
                items.append(
                    {
                        "field_name": value.get("field_name") or key,
                        "value": value.get("value"),
                        "key_field_id": value.get("key_field_id") or value.get("metadata_field_id"),
                        "metadata_field_id": value.get("metadata_field_id"),
                    }
                )
            else:
                items.append({"field_name": key, "value": value})
        return items
    return []


def resolve_extracted_value(
    *,
    field_id: str,
    field_name: str | None,
    extracted_fields: Any,
) -> Any:
    items = _extracted_items(extracted_fields)
    id_keys = _id_variants(field_id)
    for item in items:
        for candidate in (
            item.get("key_field_id"),
            item.get("metadata_field_id"),
            item.get("field_id"),
            item.get("id"),
        ):
            if str(candidate or "").strip() in id_keys or str(candidate or "").strip().lower() in id_keys:
                return item.get("value")
    if field_name:
        wanted = field_name.strip().lower()
        for item in items:
            name = str(item.get("field_name") or item.get("name") or "").strip().lower()
            if name and name == wanted:
                return item.get("value")
    return None


def hierarchy_values_from_extraction(
    *,
    hierarchy_ids: list[str],
    catalog: dict[str, dict[str, Any]],
    extracted_fields: Any,
) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for field_id in hierarchy_ids:
        definition = catalog.get(field_id) or catalog.get(field_id.lower())
        if definition is None:
            raise FolderHierarchyConfigurationError(
                "folder_hierarchy references a key field that is not configured on this repository.",
                details={"field_id": field_id},
            )
        field_name = definition.get("field_name")
        raw_value = resolve_extracted_value(
            field_id=field_id,
            field_name=field_name,
            extracted_fields=extracted_fields,
        )
        folder_name = normalize_folder_name(raw_value)
        if folder_name and _UUID_RE.match(folder_name) and folder_name.lower() == field_id.lower():
            raise FolderHierarchyConfigurationError(
                "Refusing to use a key-field identifier as a logical folder name.",
                details={"field_id": field_id},
            )
        levels.append(
            {
                "key_field_id": definition.get("field_id") or field_id,
                "key_field_name": field_name,
                "extracted_value": raw_value,
                "folder_name": folder_name,
            }
        )
    return levels


def _field_tokens(field_name: str) -> set[str]:
    return set(_FIELD_TOKENS.findall(str(field_name or "").lower()))


def _is_instance_specific_field(field_name: str) -> bool:
    tokens = _field_tokens(field_name)
    if "document" in tokens and "title" in tokens:
        return True
    if "file" in tokens and "name" in tokens:
        return True
    return bool(tokens & _INSTANCE_FIELD_TOKENS)


def _is_grouping_field(field_name: str) -> bool:
    return bool(_field_tokens(field_name) & _GROUPING_FIELD_TOKENS)


def _looks_like_unique_document_code(value: str) -> bool:
    compact = str(value or "").strip()
    if _UUID_RE.match(compact):
        return True
    if re.fullmatch(r"\d{6,}", compact):
        return True
    parts = [p for p in re.split(r"[-_/]", compact) if p]
    if len(parts) >= 3 and any(any(ch.isdigit() for ch in part) for part in parts):
        return True
    return False


def is_shared_folder_value(value: Any) -> bool:
    """True when a value can group multiple documents (not a unique title/id)."""
    folder_name = normalize_folder_name(value)
    if not folder_name:
        return False
    words = folder_name.split()
    if len(words) > _MAX_FOLDER_VALUE_WORDS:
        return False
    if _SENTENCE_PUNCT.search(folder_name) and len(words) > 3:
        return False
    if _looks_like_unique_document_code(folder_name):
        return False
    return True


def hierarchy_levels_from_extracted_fields(extracted_fields: Any) -> list[dict[str, Any]]:
    """Build folder levels from shared extracted values in document order.

    Used when ``folder_hierarchy`` is empty. Instance-specific fields (titles,
    numbers, dates) are skipped so documents of the same type/category reuse
    one folder. Empty values are skipped; nothing is invented.
    """
    grouped: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_folder_values: set[str] = set()
    for item in _extracted_items(extracted_fields):
        field_name = str(item.get("field_name") or item.get("name") or "").strip()
        if not field_name:
            continue
        key = field_name.lower()
        if key in seen_names:
            continue
        if _is_instance_specific_field(field_name):
            continue
        folder_name = normalize_folder_name(item.get("value"))
        if not folder_name or not is_shared_folder_value(folder_name):
            continue
        if _UUID_RE.match(folder_name) and folder_name.lower() == str(
            item.get("key_field_id") or item.get("field_id") or ""
        ).strip().lower():
            continue
        value_key = folder_name.lower()
        if value_key in seen_folder_values:
            continue
        seen_names.add(key)
        seen_folder_values.add(value_key)
        level = {
            "key_field_id": item.get("key_field_id") or item.get("field_id") or item.get("metadata_field_id"),
            "key_field_name": field_name,
            "extracted_value": item.get("value"),
            "folder_name": folder_name,
        }
        if _is_grouping_field(field_name):
            grouped.append(level)
        else:
            other.append(level)
    return (grouped + other)[:_MAX_AUTO_FOLDER_DEPTH]


def extracted_field_names(extracted_fields: Any) -> list[str]:
    """Unique field names (document order) that have a usable folder value."""
    names: list[str] = []
    seen: set[str] = set()
    for level in hierarchy_levels_from_extracted_fields(extracted_fields):
        name = str(level.get("key_field_name") or "").strip()
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names
