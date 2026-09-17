from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.features.document_types.domain.document_type import DocumentTypeDomain
from src.features.document_types.domain.document_type_exceptions import NotFoundError, RepositoryMismatchError, ServiceUnavailableError
from src.features.document_types.application.inheritance_service import KeyFieldInheritance
from src.features.document_types.application.key_field_service import KeyFieldDomain
from src.features.document_types.application.document_field_service import MetadataDomain
from src.features.document_types.domain.models import (
    DocumentTypeRecord,
    MetadataFieldsBundle,
    document_type_to_dict,
    key_field_to_dict,
)
from src.features.document_types.application.relationship_service import DocumentTypeRelationship
from src.features.document_types.seed.document_type_seeder import ensure_basic_document_type
from src.features.document_types.infrastructure.document_type_repository import DocumentTypeStore, get_document_type_store
from src.features.observability.audit.application.business_audit import audit_business_change


class DocumentTypeService:
    """Facade for repository-scoped document type taxonomy and key field management."""

    def __init__(self, store: DocumentTypeStore | None = None) -> None:
        self._store = store
        self._types: DocumentTypeDomain | None = None
        self._relationship: DocumentTypeRelationship | None = None
        self._metadata: MetadataDomain | None = None
        self._key_fields: KeyFieldDomain | None = None
        self._inheritance: KeyFieldInheritance | None = None

    @staticmethod
    def _audit_type_change(
        event_type: str,
        *,
        entity_type: str,
        entity_id: str,
        old: dict[str, Any],
        new: dict[str, Any],
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        skip = {"created_at", "updated_at"}
        old_cmp = {k: v for k, v in (old or {}).items() if k not in skip}
        new_cmp = {k: v for k, v in (new or {}).items() if k not in skip}
        audit_business_change(
            event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            old=old_cmp,
            new=new_cmp,
            category="document",
            action=action,
            source="API",
            metadata=metadata,
        )

    def _require_store(self) -> DocumentTypeStore:
        try:
            store = self._store or get_document_type_store()
        except Exception as exc:
            raise ServiceUnavailableError(
                "Document Type Service persistence layer is unavailable.",
                details={
                    "hint": "Configure DOCUMENT_TYPE_POSTGRES_* / DOCUMENT_JOBS_POSTGRES_* / POSTGRES_* environment variables.",
                    "error": str(exc),
                },
            ) from exc
        if store is None:
            raise ServiceUnavailableError(
                "Document Type Service persistence layer is unavailable.",
                details={
                    "hint": "Configure DOCUMENT_TYPE_POSTGRES_* / DOCUMENT_JOBS_POSTGRES_* / POSTGRES_* environment variables."
                },
            )
        if not store.ping():
            raise ServiceUnavailableError(
                "Document Type Service persistence layer is unavailable.",
                details={
                    "hint": "PostgreSQL is not reachable for document types; verify DOCUMENT_TYPE_POSTGRES_* settings."
                },
            )
        return store

    @property
    def types(self) -> DocumentTypeDomain:
        store = self._require_store()
        if self._types is None or self._store is not store:
            self._types = DocumentTypeDomain(store)
        return self._types

    @property
    def relationship(self) -> DocumentTypeRelationship:
        store = self._require_store()
        if self._relationship is None or self._store is not store:
            self._relationship = DocumentTypeRelationship(store)
        return self._relationship

    @property
    def metadata(self) -> MetadataDomain:
        store = self._require_store()
        if self._metadata is None or self._store is not store:
            self._metadata = MetadataDomain(store)
        return self._metadata

    @property
    def key_fields(self) -> KeyFieldDomain:
        store = self._require_store()
        if self._key_fields is None or self._store is not store:
            self._key_fields = KeyFieldDomain(store)
        return self._key_fields

    @property
    def inheritance(self) -> KeyFieldInheritance:
        store = self._require_store()
        if self._inheritance is None or self._store is not store:
            self._inheritance = KeyFieldInheritance(store)
        return self._inheritance

    def initialize(self) -> None:
        store = self._require_store()
        store.ensure_schema()

    def bootstrap_repository(self, repository_id: str, *, created_by: str | None = None) -> str:
        """Create Basic document type and default key fields for a repository. Returns Basic type id."""
        store = self._require_store()
        basic = ensure_basic_document_type(store, repository_id, created_by=created_by)
        return basic.document_type_id

    def delete_repository_document_types(self, repository_id: str) -> None:
        store = self._require_store()
        for doc_type in store.list_types_for_repository(repository_id):
            if doc_type.is_system:
                continue
            store.delete_type(doc_type.document_type_id)
        basic = store.get_basic_type_for_repository(repository_id)
        if basic is not None:
            store.delete_type(basic.document_type_id)

    def resolve_effective_fields(self, document_type_id: str) -> dict[str, Any]:
        store = self._require_store()
        if store.get_type_by_id(document_type_id) is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": document_type_id})
        fields = self.inheritance.resolve_effective_fields(document_type_id)
        return {
            "document_type_id": document_type_id,
            "fields": [key_field_to_dict(field) for field in fields],
            "count": len(fields),
        }

    def create_repository_document_type(
        self,
        repository_id: str,
        payload: dict[str, Any],
        *,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        record = self.types.create_type(repository_id, payload, created_by=created_by)
        created = document_type_to_dict(record)
        self._audit_type_change(
            "DOCUMENT_TYPE_CREATED",
            entity_type="document_type",
            entity_id=record.document_type_id,
            old={},
            new=created,
            action="create",
            metadata={"repository_id": repository_id},
        )
        return created

    def list_repository_document_types(self, repository_id: str) -> dict[str, Any]:
        store = self._require_store()
        records = store.list_types_for_repository(repository_id)
        items = [document_type_to_dict(record) for record in records]
        return {"repository_id": repository_id, "document_types": items, "count": len(items)}

    def get_repository_document_type_tree(self, repository_id: str) -> dict[str, Any]:
        store = self._require_store()
        records = store.list_types_for_repository(repository_id)
        tree = self._build_tree(records)
        return {"repository_id": repository_id, "tree": tree, "count": len(records)}

    @staticmethod
    def _build_tree(records: list[DocumentTypeRecord]) -> list[dict[str, Any]]:
        children_map: dict[str | None, list[DocumentTypeRecord]] = defaultdict(list)
        for record in records:
            children_map[record.parent_document_type_id].append(record)

        def build_node(record: DocumentTypeRecord) -> dict[str, Any]:
            payload = document_type_to_dict(record)
            children = sorted(children_map.get(record.document_type_id, []), key=lambda item: item.name)
            payload["children"] = [build_node(child) for child in children]
            return payload

        roots = sorted(children_map.get(None, []), key=lambda item: item.name)
        return [build_node(root) for root in roots]

    def create_child_type(
        self,
        parent_id: str,
        payload: dict[str, Any],
        *,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        record = self.types.create_child(parent_id, payload, created_by=created_by)
        created = document_type_to_dict(record)
        self._audit_type_change(
            "DOCUMENT_TYPE_CREATED",
            entity_type="document_type",
            entity_id=record.document_type_id,
            old={},
            new=created,
            action="create",
            metadata={"repository_id": record.repository_id, "parent_document_type_id": parent_id},
        )
        return created

    def update_type(
        self,
        type_id: str,
        payload: dict[str, Any],
        *,
        updated_by: str | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        existing = store.get_type_by_id(type_id)
        before = document_type_to_dict(existing) if existing is not None else {}
        record = self.types.update(type_id, payload, updated_by=updated_by)
        after = document_type_to_dict(record)
        self._audit_type_change(
            "DOCUMENT_TYPE_UPDATED",
            entity_type="document_type",
            entity_id=type_id,
            old=before,
            new=after,
            action="update",
            metadata={"repository_id": record.repository_id},
        )
        return after

    def delete_type(self, type_id: str) -> None:
        store = self._require_store()
        existing = store.get_type_by_id(type_id)
        before = document_type_to_dict(existing) if existing is not None else {}
        repository_id = existing.repository_id if existing is not None else None
        self.types.delete(type_id)
        self._audit_type_change(
            "DOCUMENT_TYPE_DELETED",
            entity_type="document_type",
            entity_id=type_id,
            old=before,
            new={},
            action="delete",
            metadata={"repository_id": repository_id} if repository_id else None,
        )

    def create_key_field(self, document_type_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        field = self.key_fields.create_field(document_type_id, payload)
        created = key_field_to_dict(field)
        self._audit_type_change(
            "DOCUMENT_KEY_FIELD_CREATED",
            entity_type="key_field",
            entity_id=field.id,
            old={},
            new=created,
            action="create",
            metadata={"document_type_id": document_type_id},
        )
        return created

    def update_key_field(self, field_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        store = self._require_store()
        existing = store.get_key_field_by_id(field_id)
        before = key_field_to_dict(existing) if existing is not None else {}
        field = self.key_fields.update_field(field_id, payload)
        after = key_field_to_dict(field)
        self._audit_type_change(
            "DOCUMENT_KEY_FIELD_UPDATED",
            entity_type="key_field",
            entity_id=field_id,
            old=before,
            new=after,
            action="update",
            metadata={"document_type_id": field.document_type_id},
        )
        return after

    def delete_key_field(self, field_id: str) -> None:
        store = self._require_store()
        existing = store.get_key_field_by_id(field_id)
        before = key_field_to_dict(existing) if existing is not None else {}
        document_type_id = existing.document_type_id if existing is not None else None
        self.key_fields.delete_field(field_id)
        self._audit_type_change(
            "DOCUMENT_KEY_FIELD_DELETED",
            entity_type="key_field",
            entity_id=field_id,
            old=before,
            new={},
            action="delete",
            metadata={"document_type_id": document_type_id} if document_type_id else None,
        )

    def list_key_fields(self, document_type_id: str) -> dict[str, Any]:
        store = self._require_store()
        if store.get_type_by_id(document_type_id) is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": document_type_id})
        fields = store.list_key_fields_for_type(document_type_id)
        return {
            "document_type_id": document_type_id,
            "fields": [key_field_to_dict(field) for field in fields],
            "count": len(fields),
        }

    def get_type(self, type_id: str, *, repository_id: str | None = None) -> dict[str, Any]:
        store = self._require_store()
        record = store.get_type_by_id(type_id)
        if record is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": type_id})
        if repository_id is not None and record.repository_id != repository_id:
            raise RepositoryMismatchError(
                "Document type does not belong to this repository.",
                details={"repository_id": repository_id, "document_type_id": type_id},
            )
        bundle = self.relationship.metadata_for_type(type_id)
        return document_type_to_dict(record, metadata_fields=bundle)

    def create_metadata_field(self, document_type_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        field = self.metadata.create_field(document_type_id, payload)
        from src.features.document_types.domain.models import field_to_dict

        created = field_to_dict(field)
        self._audit_type_change(
            "DOCUMENT_METADATA_FIELD_CREATED",
            entity_type="metadata_field",
            entity_id=field.metadata_field_id,
            old={},
            new=created,
            action="create",
            metadata={"document_type_id": document_type_id},
        )
        return created

    def update_metadata_field(
        self,
        document_type_id: str,
        metadata_field_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        store = self._require_store()
        existing = store.get_field_by_id(metadata_field_id)
        from src.features.document_types.domain.models import field_to_dict

        before = field_to_dict(existing) if existing is not None else {}
        field = self.metadata.update_field(document_type_id, metadata_field_id, payload)
        after = field_to_dict(field)
        self._audit_type_change(
            "DOCUMENT_METADATA_FIELD_UPDATED",
            entity_type="metadata_field",
            entity_id=metadata_field_id,
            old=before,
            new=after,
            action="update",
            metadata={"document_type_id": document_type_id},
        )
        return after

    def delete_metadata_field(self, document_type_id: str, metadata_field_id: str) -> None:
        store = self._require_store()
        existing = store.get_field_by_id(metadata_field_id)
        from src.features.document_types.domain.models import field_to_dict

        before = field_to_dict(existing) if existing is not None else {}
        self.metadata.delete_field(document_type_id, metadata_field_id)
        self._audit_type_change(
            "DOCUMENT_METADATA_FIELD_DELETED",
            entity_type="metadata_field",
            entity_id=metadata_field_id,
            old=before,
            new={},
            action="delete",
            metadata={"document_type_id": document_type_id},
        )

    def list_types(
        self,
        *,
        include_metadata: bool = False,
        is_active: bool | None = None,
        depth_level: int | None = None,
    ) -> dict[str, Any]:
        store = self._require_store()
        records = store.list_types(is_active=is_active, depth_level=depth_level)
        items: list[dict[str, Any]] = []
        for record in records:
            bundle: MetadataFieldsBundle | None = None
            if include_metadata:
                bundle = self.relationship.metadata_for_type(record.document_type_id)
            items.append(document_type_to_dict(record, metadata_fields=bundle))
        return {"document_types": items, "count": len(items)}

    def list_children(
        self,
        parent_id: str,
        *,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        store = self._require_store()
        if store.get_type_by_id(parent_id) is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": parent_id})
        children = self.relationship.get_children(parent_id)
        items: list[dict[str, Any]] = []
        for record in children:
            bundle: MetadataFieldsBundle | None = None
            if include_metadata:
                bundle = self.relationship.metadata_for_type(record.document_type_id)
            items.append(document_type_to_dict(record, metadata_fields=bundle))
        return {"children": items, "count": len(items)}

    def get_effective_metadata(self, type_id: str) -> dict[str, Any]:
        store = self._require_store()
        if store.get_type_by_id(type_id) is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": type_id})
        bundle = self.relationship.metadata_for_type(type_id)
        return bundle.to_dicts()


_default_service: DocumentTypeService | None = None


def get_document_type_service() -> DocumentTypeService:
    global _default_service
    if _default_service is None:
        _default_service = DocumentTypeService()
    return _default_service
