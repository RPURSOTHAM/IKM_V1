"""Field-level change tracking for audit events."""

from __future__ import annotations

import json
from typing import Any

SETTINGS_FIELD_LABELS: dict[str, str] = {
    "embedding_model": "Embedding Model",
    "embedding_model.model_id": "Embedding Model",
    "embedding_model.local_model_dir": "Embedding Model",
    "embedding_model.provider": "Embedding Provider",
    "chunk_size": "Chunk Size",
    "chunk_overlap": "Chunk Overlap",
    "chunking_strategy": "Chunking Strategy",
    "chunking_config": "Chunking Config",
    "chunking_config.min_content_words": "Minimum Words",
    "indexing_strategy": "Indexing Strategy",
    "metadata_extraction": "Metadata Extraction",
    "citation_retainment": "Citation Retention",
    "template_extraction": "Template Extraction",
    "reference_document_extraction": "Reference Document Extraction",
    "conversion_for_rendering": "Conversion For Rendering",
    "intelligent_extraction": "Intelligent Extraction",
    "reranking": "Reranking",
    "retrieval_search_mode": "Search Mode",
    "lexical_composition": "Lexical Composition",
    "document_type_id": "Document Type",
    "extraction_model": "Extraction Model",
    "status": "Status",
}

SEARCH_MODE_LABELS = {
    "vector": "Vector",
    "hybrid": "Hybrid",
    "keyword": "Keyword",
    "bm25": "Keyword",
}


def _normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    try:
        return json.loads(json.dumps(value, default=str, sort_keys=True))
    except (TypeError, ValueError):
        return str(value)


def _flatten_mapping(obj: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {prefix: obj} if prefix else {}
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            child = _flatten_mapping(value, path)
            if child:
                flat.update(child)
            else:
                flat[path] = value
        else:
            flat[path] = value
    return flat


def _display_value(field: str, value: Any) -> str:
    if value is None:
        return "-"
    if field.endswith("retrieval_search_mode") or field == "retrieval_search_mode":
        key = str(value).strip().lower()
        return SEARCH_MODE_LABELS.get(key, str(value))
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, dict):
        if "model_id" in value:
            return str(value.get("model_id"))
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _field_label(field: str) -> str:
    return SETTINGS_FIELD_LABELS.get(field, field.replace("_", " ").replace(".", " / ").title())


def get_changes(old: Any, new: Any) -> list[dict[str, Any]]:
    """Return only modified top-level fields between old and new mappings."""
    old_map = dict(old) if isinstance(old, dict) else {}
    new_map = dict(new) if isinstance(new, dict) else {}
    keys = sorted(set(old_map.keys()) | set(new_map.keys()))
    changes: list[dict[str, Any]] = []
    for field in keys:
        old_val = _normalize(old_map.get(field))
        new_val = _normalize(new_map.get(field))
        if old_val != new_val:
            changes.append({"field": field, "old": old_map.get(field), "new": new_map.get(field)})
    return changes


def get_deep_changes(old: Any, new: Any) -> list[dict[str, Any]]:
    """Return modified fields using dotted paths for nested settings objects."""
    old_flat = _flatten_mapping(old if isinstance(old, dict) else {})
    new_flat = _flatten_mapping(new if isinstance(new, dict) else {})
    keys = sorted(set(old_flat.keys()) | set(new_flat.keys()))
    changes: list[dict[str, Any]] = []
    for field in keys:
        old_val = _normalize(old_flat.get(field))
        new_val = _normalize(new_flat.get(field))
        if old_val != new_val:
            changes.append(
                {
                    "field": field,
                    "label": _field_label(field),
                    "old": old_flat.get(field),
                    "new": new_flat.get(field),
                    "old_display": _display_value(field, old_flat.get(field)),
                    "new_display": _display_value(field, new_flat.get(field)),
                }
            )
    return changes


def build_change_metadata(changes: list[dict[str, Any]]) -> dict[str, Any]:
    """Structured change summary for repository and settings audits."""
    if not changes:
        return {}

    changed_fields = [str(c.get("field")) for c in changes if c.get("field")]
    previous_values = {str(c["field"]): c.get("old") for c in changes if c.get("field")}
    new_values = {str(c["field"]): c.get("new") for c in changes if c.get("field")}
    lines = []
    for change in changes:
        label = change.get("label") or _field_label(str(change.get("field") or ""))
        before = change.get("old_display")
        if before is None:
            before = _display_value(str(change.get("field") or ""), change.get("old"))
        after = change.get("new_display")
        if after is None:
            after = _display_value(str(change.get("field") or ""), change.get("new"))
        lines.append(f"{label}: {before} -> {after}")

    return {
        "changed_fields": changed_fields,
        "previous_values": previous_values,
        "new_values": new_values,
        "number_of_fields_changed": len(changed_fields),
        "change_summary": "; ".join(lines),
    }
