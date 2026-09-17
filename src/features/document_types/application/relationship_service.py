from __future__ import annotations

from src.features.document_types.domain.models import DocumentTypeRecord, MetadataFieldsBundle
from src.features.document_types.infrastructure.document_type_repository import DocumentTypeStore


class DocumentTypeRelationship:
    """Query helpers for parent/child links and depth."""

    def __init__(self, store: DocumentTypeStore) -> None:
        self._store = store

    def get_children(self, parent_id: str) -> list[DocumentTypeRecord]:
        return self._store.list_children(parent_id)

    def get_depth(self, type_id: str) -> int:
        record = self._store.get_type_by_id(type_id)
        if record is None:
            raise KeyError(type_id)
        return record.depth_level

    def metadata_for_type(self, type_id: str) -> MetadataFieldsBundle:
        return self._store.resolve_metadata_bundle(type_id)
