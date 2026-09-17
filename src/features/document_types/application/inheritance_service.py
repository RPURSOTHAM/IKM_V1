from __future__ import annotations

from src.features.document_types.domain.models import KeyFieldDefinitionRecord
from src.features.document_types.infrastructure.document_type_repository import DocumentTypeStore


class KeyFieldInheritance:
    """Resolve effective key fields by walking the document type hierarchy."""

    def __init__(self, store: DocumentTypeStore) -> None:
        self._store = store

    def resolve_effective_fields(self, document_type_id: str) -> list[KeyFieldDefinitionRecord]:
        chain = self._store.ancestor_chain(document_type_id)
        effective_by_name: dict[str, KeyFieldDefinitionRecord] = {}

        for ancestor_id in reversed(chain):
            fields = self._store.list_key_fields_for_type(ancestor_id)
            for field in fields:
                if ancestor_id != document_type_id:
                    effective_by_name[field.field_name] = KeyFieldDefinitionRecord(
                        id=field.id,
                        document_type_id=field.document_type_id,
                        field_name=field.field_name,
                        field_type=field.field_type,
                        required=field.required,
                        default_value=field.default_value,
                        description=field.description,
                        created_at=field.created_at,
                        updated_at=field.updated_at,
                        inherited_from_document_type_id=ancestor_id,
                    )
                else:
                    effective_by_name[field.field_name] = field

        return sorted(effective_by_name.values(), key=lambda f: f.field_name)
