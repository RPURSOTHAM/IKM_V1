"""Idempotent Phase 3 migration helpers for repository document types."""

from __future__ import annotations

from typing import Any

from src.features.document_types.domain.document_type_exceptions import DocumentTypeError
from src.features.document_types.domain.models import KeyFieldType
from src.features.document_types.application.document_type_service import DocumentTypeService, get_document_type_service


def _normalize_field_name(raw: dict[str, Any]) -> str:
    return str(raw.get("name") or raw.get("field_name") or "").strip()


def _normalize_field_type(raw: dict[str, Any]) -> str:
    return KeyFieldType.canonical(str(raw.get("type") or raw.get("field_type") or "string"))


def migrate_repository_document_types(
    repository_id: str,
    *,
    dry_run: bool = False,
    settings: dict[str, Any] | None = None,
    dt_service: DocumentTypeService | None = None,
    update_settings: Any | None = None,
) -> dict[str, Any]:
    """Ensure Basic exists and migrate repository settings.key_fields into Basic.

    ``settings`` should be the repository settings dict (may include key_fields /
    document_type_id). When ``update_settings`` is provided it is called as
    ``update_settings(repository_id, patch)`` to persist document_type_id.
    """
    service = dt_service or get_document_type_service()
    service.initialize()

    result: dict[str, Any] = {
        "repository_id": repository_id,
        "basic_created": False,
        "document_type_id_set": False,
        "fields_migrated": 0,
        "fields_skipped": 0,
    }

    listing_before = service.list_repository_document_types(repository_id)
    had_basic = any(
        item.get("name") == "Basic" and item.get("is_system")
        for item in listing_before.get("document_types") or []
    )

    settings_dict = dict(settings or {})
    if dry_run and not had_basic:
        result["basic_created"] = True
        basic_id = str(settings_dict.get("document_type_id") or "").strip() or None
    else:
        basic_id = service.bootstrap_repository(repository_id)
        result["basic_created"] = not had_basic

    if not settings_dict.get("document_type_id") and basic_id:
        result["document_type_id_set"] = True
        if not dry_run and update_settings is not None:
            update_settings(repository_id, {"document_type_id": basic_id})

    existing_names: set[str] = set()
    if basic_id:
        fields = service.list_key_fields(basic_id)
        existing_names = {
            str(item.get("field_name") or "").strip()
            for item in fields.get("fields") or []
            if str(item.get("field_name") or "").strip()
        }

    for raw in settings_dict.get("key_fields") or []:
        if not isinstance(raw, dict):
            continue
        field_name = _normalize_field_name(raw)
        if not field_name:
            continue
        if field_name in existing_names:
            result["fields_skipped"] += 1
            continue
        if dry_run or not basic_id:
            result["fields_migrated"] += 1
            existing_names.add(field_name)
            continue
        try:
            # Insert via store so Basic system-type field immutability does not block migration.
            store = service._require_store()
            if store.key_field_name_on_type(basic_id, field_name):
                result["fields_skipped"] += 1
                continue
            store.insert_key_field(
                document_type_id=basic_id,
                field_name=field_name,
                field_type=_normalize_field_type(raw),
                required=bool(raw.get("required", False)),
                default_value=raw.get("default_value"),
                description=raw.get("description"),
            )
            result["fields_migrated"] += 1
            existing_names.add(field_name)
        except DocumentTypeError as exc:
            message = str(exc.message if hasattr(exc, "message") else exc)
            if "already exists" in message or "Basic document type" in message:
                result["fields_skipped"] += 1
                continue
            raise

    return result
