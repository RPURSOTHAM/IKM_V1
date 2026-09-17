from __future__ import annotations

from src.features.document_types.application.basic_fields import BASIC_TYPE_NAME, DEFAULT_BASIC_KEY_FIELDS
from src.features.document_types.domain.models import DocumentTypeRecord
from src.features.document_types.infrastructure.document_type_repository import DocumentTypeStore


def ensure_basic_document_type(
    store: DocumentTypeStore,
    repository_id: str,
    *,
    created_by: str | None = None,
) -> DocumentTypeRecord:
    """Create the repository-scoped Basic root type and default key fields."""
    existing = store.get_basic_type_for_repository(repository_id)
    if existing is not None:
        _ensure_default_key_fields(store, existing.document_type_id)
        return existing

    return store.bootstrap_repository_document_types(
        repository_id,
        created_by=created_by,
    )


def _ensure_default_key_fields(store: DocumentTypeStore, basic_type_id: str) -> None:
    existing_names = {f.field_name for f in store.list_key_fields_for_type(basic_type_id)}
    for field_name, field_type, required, default_value, description in DEFAULT_BASIC_KEY_FIELDS:
        if field_name in existing_names:
            continue
        store.insert_key_field(
            document_type_id=basic_type_id,
            field_name=field_name,
            field_type=field_type,
            required=required,
            default_value=default_value,
            description=description,
        )
