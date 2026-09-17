from __future__ import annotations

from typing import Any

from src.features.document_types.domain.document_type_exceptions import (
    CircularReferenceError,
    DuplicateNameError,
    NotFoundError,
    ParentInactiveError,
    RepositoryMismatchError,
    SystemTypeImmutableError,
    TypeInUseError,
    ValidationError,
)
from src.features.document_types.domain.models import DocumentTypeRecord
from src.features.document_types.infrastructure.document_type_repository import DocumentTypeStore


class DocumentTypeDomain:
    """Lifecycle and validation for repository-scoped document type entities."""

    def __init__(self, store: DocumentTypeStore) -> None:
        self._store = store

    def assert_repository_match(self, record: DocumentTypeRecord, repository_id: str) -> None:
        if record.repository_id != repository_id:
            raise RepositoryMismatchError(
                "Document type does not belong to this repository.",
                details={
                    "repository_id": repository_id,
                    "document_type_id": record.document_type_id,
                    "type_repository_id": record.repository_id,
                },
            )

    def assert_not_system_mutation(self, record: DocumentTypeRecord) -> None:
        if record.is_system:
            raise SystemTypeImmutableError(
                "The Basic document type cannot be modified or deleted.",
                details={"document_type_id": record.document_type_id},
            )

    def assert_not_in_use(self, document_type_id: str) -> None:
        doc_count = self._store.count_documents_for_type(document_type_id)
        if doc_count > 0:
            raise TypeInUseError(
                f"Document type cannot be deleted because {doc_count} documents reference it.",
                details={"document_type_id": document_type_id, "document_count": doc_count},
            )

    def assert_no_children(self, document_type_id: str) -> None:
        child_count = self._store.count_children(document_type_id)
        if child_count > 0:
            raise TypeInUseError(
                "Document type cannot be deleted because it has child document types.",
                details={"document_type_id": document_type_id, "child_count": child_count},
            )

    def validate_parent_active(self, parent: DocumentTypeRecord) -> None:
        if not parent.is_active:
            raise ParentInactiveError(
                "Cannot create a child under an inactive document type.",
                details={"parent_document_type_id": parent.document_type_id},
            )

    def validate_name(self, name: str) -> None:
        trimmed = (name or "").strip()
        if not trimmed:
            raise ValidationError("Document type name is required.")
        if len(trimmed) > 256:
            raise ValidationError("Document type name must be at most 256 characters.")

    def validate_parent_chain(self, parent_id: str) -> None:
        if self._store.has_circular_ancestor_chain(parent_id):
            raise CircularReferenceError(
                "Document type hierarchy contains a circular reference.",
                details={"document_type_id": parent_id},
            )

    def create_child(
        self,
        parent_id: str,
        payload: dict[str, Any],
        *,
        created_by: str | None = None,
    ) -> DocumentTypeRecord:
        parent = self._store.get_type_by_id(parent_id)
        if parent is None:
            raise NotFoundError(
                "Parent document type not found.",
                details={"document_type_id": parent_id},
            )
        if parent.repository_id is None:
            raise ValidationError(
                "Cannot create child types under legacy global document types.",
                details={"document_type_id": parent_id},
            )
        merged = dict(payload)
        merged["parent_document_type_id"] = parent_id
        return self.create_type(parent.repository_id, merged, created_by=created_by)

    def create_type(
        self,
        repository_id: str,
        payload: dict[str, Any],
        *,
        created_by: str | None = None,
    ) -> DocumentTypeRecord:
        parent_id = payload.get("parent_document_type_id")
        parent: DocumentTypeRecord | None = None
        depth_level = 0

        if parent_id is not None:
            parent_id = str(parent_id).strip() or None

        if parent_id:
            parent = self._store.get_type_by_id(parent_id)
            if parent is None:
                raise NotFoundError(
                    "Parent document type not found.",
                    details={"document_type_id": parent_id},
                )
            self.assert_repository_match(parent, repository_id)
            self.validate_parent_active(parent)
            self.validate_parent_chain(parent_id)
            depth_level = parent.depth_level + 1
        else:
            # Repository-scoped mode: when parent is omitted, auto-parent to Basic.
            # Legacy repositories without a Basic type may still create a root.
            basic = self._store.get_basic_type_for_repository(repository_id)
            if basic is not None:
                parent = basic
                parent_id = basic.document_type_id
                self.validate_parent_active(basic)
                self.validate_parent_chain(parent_id)
                depth_level = basic.depth_level + 1

        name = str(payload.get("name", "")).strip()
        self.validate_name(name)
        if self._store.sibling_name_exists(repository_id, parent_id, name):
            raise DuplicateNameError(
                "A sibling document type with this name already exists.",
                details={"repository_id": repository_id, "parent_document_type_id": parent_id, "name": name},
            )

        child = self._store.insert_type(
            repository_id=repository_id,
            name=name,
            code=payload.get("code"),
            description=payload.get("description"),
            parent_document_type_id=parent_id,
            depth_level=depth_level,
            is_system=False,
            is_active=True,
            created_by=created_by,
        )
        if parent_id:
            self._store.insert_relationship(parent_id, child.document_type_id, child.depth_level)
        return child

    def update(
        self,
        type_id: str,
        payload: dict[str, Any],
        *,
        updated_by: str | None = None,
    ) -> DocumentTypeRecord:
        record = self._store.get_type_by_id(type_id)
        if record is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": type_id})
        self.assert_not_system_mutation(record)
        name = payload.get("name")
        if name is not None:
            name = str(name).strip()
            self.validate_name(name)
            parent_key = record.parent_document_type_id
            if self._store.sibling_name_exists(record.repository_id or "", parent_key, name, exclude_id=type_id):
                raise DuplicateNameError(
                    "A sibling document type with this name already exists.",
                    details={"document_type_id": type_id, "name": name},
                )
        return self._store.update_type(
            type_id,
            name=name,
            code=payload.get("code"),
            description=payload.get("description"),
            is_active=payload.get("is_active"),
            updated_by=updated_by,
        )

    def delete(self, type_id: str) -> None:
        record = self._store.get_type_by_id(type_id)
        if record is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": type_id})
        self.assert_not_system_mutation(record)
        self.assert_no_children(type_id)
        self.assert_not_in_use(type_id)
        self._store.delete_type(type_id)
