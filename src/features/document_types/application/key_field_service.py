from __future__ import annotations

from typing import Any

from src.features.document_types.domain.document_type_exceptions import (
    DuplicateFieldError,
    NotFoundError,
    SystemTypeImmutableError,
    ValidationError,
)
from src.features.document_types.domain.models import DocumentTypeRecord, KeyFieldDefinitionRecord, KeyFieldType
from src.features.document_types.infrastructure.document_type_repository import DocumentTypeStore


class KeyFieldDomain:
    """CRUD and validation for repository-scoped key field definitions."""

    def __init__(self, store: DocumentTypeStore) -> None:
        self._store = store

    def assert_basic_type_mutable(self, record: DocumentTypeRecord, field_name: str) -> None:
        if record.is_system and record.parent_document_type_id is None:
            raise SystemTypeImmutableError(
                "Default key fields on the Basic document type cannot be modified or deleted.",
                details={"document_type_id": record.document_type_id, "field_name": field_name},
            )

    def validate_field_name(self, field_name: str) -> None:
        try:
            self._store.validate_field_name(field_name)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def validate_field_type(self, field_type: str) -> None:
        if field_type not in KeyFieldType.values():
            raise ValidationError(
                "field_type must be one of: string, integer, float, boolean, date "
                "(aliases: number, datetime, text)",
            )

    def assert_field_name_unique_in_hierarchy(self, document_type_id: str, field_name: str) -> None:
        """Reject duplicate field names anywhere in the ancestor chain (including self)."""
        if self._store.key_field_name_on_type(document_type_id, field_name):
            raise DuplicateFieldError(
                "A key field with this name already exists on the document type.",
                details={"document_type_id": document_type_id, "field_name": field_name},
            )
        chain = self._store.ancestor_chain(document_type_id)
        for ancestor_id in chain:
            if ancestor_id == document_type_id:
                continue
            if self._store.key_field_name_on_type(ancestor_id, field_name):
                raise DuplicateFieldError(
                    "A key field with this name already exists in the document type hierarchy.",
                    details={
                        "document_type_id": document_type_id,
                        "field_name": field_name,
                        "inherited_from_document_type_id": ancestor_id,
                    },
                )

    def create_field(
        self,
        document_type_id: str,
        payload: dict[str, Any],
    ) -> KeyFieldDefinitionRecord:
        doc_type = self._store.get_type_by_id(document_type_id)
        if doc_type is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": document_type_id})

        field_name = str(payload.get("field_name", "")).strip()
        self.validate_field_name(field_name)
        self.assert_field_name_unique_in_hierarchy(document_type_id, field_name)

        field_type = KeyFieldType.canonical(str(payload.get("field_type", KeyFieldType.STRING.value)))
        self.validate_field_type(field_type)

        return self._store.insert_key_field(
            document_type_id=document_type_id,
            field_name=field_name,
            field_type=field_type,
            required=bool(payload.get("required", False)),
            default_value=payload.get("default_value"),
            description=payload.get("description"),
        )

    def update_field(self, field_id: str, payload: dict[str, Any]) -> KeyFieldDefinitionRecord:
        field = self._store.get_key_field_by_id(field_id)
        if field is None:
            raise NotFoundError("Key field not found.", details={"field_id": field_id})

        doc_type = self._store.get_type_by_id(field.document_type_id)
        if doc_type is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": field.document_type_id})

        self.assert_basic_type_mutable(doc_type, field.field_name)

        field_type = payload.get("field_type")
        if field_type is not None:
            field_type = KeyFieldType.canonical(str(field_type))
            self.validate_field_type(str(field_type))

        return self._store.update_key_field(
            field_id,
            field_type=field_type,
            required=payload.get("required"),
            default_value=payload.get("default_value"),
            description=payload.get("description"),
            unset_default="default_value" in payload and payload["default_value"] is None,
        )

    def delete_field(self, field_id: str) -> None:
        field = self._store.get_key_field_by_id(field_id)
        if field is None:
            raise NotFoundError("Key field not found.", details={"field_id": field_id})

        doc_type = self._store.get_type_by_id(field.document_type_id)
        if doc_type is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": field.document_type_id})

        self.assert_basic_type_mutable(doc_type, field.field_name)
        self._store.delete_key_field(field_id)
