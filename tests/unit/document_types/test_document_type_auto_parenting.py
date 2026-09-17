from __future__ import annotations

from datetime import datetime

from src.features.document_types.domain.document_type import DocumentTypeDomain
from src.features.document_types.domain.models import DocumentTypeRecord


def _record(
    *,
    document_type_id: str,
    repository_id: str,
    name: str,
    parent_document_type_id: str | None,
    depth_level: int,
    is_system: bool = False,
    is_active: bool = True,
) -> DocumentTypeRecord:
    now = datetime.utcnow()
    return DocumentTypeRecord(
        document_type_id=document_type_id,
        repository_id=repository_id,
        name=name,
        code=None,
        description=None,
        parent_document_type_id=parent_document_type_id,
        depth_level=depth_level,
        is_system=is_system,
        is_active=is_active,
        created_at=now,
        updated_at=now,
        created_by="test",
        updated_by="test",
    )


class _FakeStore:
    def __init__(self) -> None:
        self.records: dict[str, DocumentTypeRecord] = {}
        self.relationships: list[tuple[str, str, int]] = []
        self._next = 1

    def get_type_by_id(self, type_id: str) -> DocumentTypeRecord | None:
        return self.records.get(type_id)

    def get_basic_type_for_repository(self, repository_id: str) -> DocumentTypeRecord | None:
        for record in self.records.values():
            if record.repository_id == repository_id and record.is_system and record.parent_document_type_id is None:
                return record
        return None

    def has_circular_ancestor_chain(self, parent_id: str) -> bool:
        return False

    def sibling_name_exists(
        self,
        repository_id: str,
        parent_document_type_id: str | None,
        name: str,
        exclude_id: str | None = None,
    ) -> bool:
        for record in self.records.values():
            if exclude_id and record.document_type_id == exclude_id:
                continue
            if record.repository_id != repository_id:
                continue
            if record.parent_document_type_id != parent_document_type_id:
                continue
            if record.name == name:
                return True
        return False

    def insert_type(
        self,
        *,
        repository_id: str,
        name: str,
        code: str | None,
        description: str | None,
        parent_document_type_id: str | None,
        depth_level: int,
        is_system: bool,
        is_active: bool,
        created_by: str | None,
    ) -> DocumentTypeRecord:
        type_id = f"type-{self._next}"
        self._next += 1
        record = _record(
            document_type_id=type_id,
            repository_id=repository_id,
            name=name,
            parent_document_type_id=parent_document_type_id,
            depth_level=depth_level,
            is_system=is_system,
            is_active=is_active,
        )
        self.records[type_id] = record
        return record

    def insert_relationship(self, parent_id: str, child_id: str, depth_level: int) -> None:
        self.relationships.append((parent_id, child_id, depth_level))


def test_create_without_parent_auto_parents_to_basic() -> None:
    store = _FakeStore()
    basic = _record(
        document_type_id="basic-1",
        repository_id="repo-1",
        name="Basic",
        parent_document_type_id=None,
        depth_level=0,
        is_system=True,
    )
    store.records[basic.document_type_id] = basic
    domain = DocumentTypeDomain(store)  # type: ignore[arg-type]

    sop = domain.create_type("repo-1", {"name": "SOP"})
    assert sop.parent_document_type_id == basic.document_type_id
    assert sop.depth_level == 1


def test_create_with_explicit_parent_preserves_parent() -> None:
    store = _FakeStore()
    basic = _record(
        document_type_id="basic-1",
        repository_id="repo-1",
        name="Basic",
        parent_document_type_id=None,
        depth_level=0,
        is_system=True,
    )
    parent = _record(
        document_type_id="sop-1",
        repository_id="repo-1",
        name="SOP",
        parent_document_type_id="basic-1",
        depth_level=1,
    )
    store.records[basic.document_type_id] = basic
    store.records[parent.document_type_id] = parent
    domain = DocumentTypeDomain(store)  # type: ignore[arg-type]

    child = domain.create_type("repo-1", {"name": "Work Instruction", "parent_document_type_id": "sop-1"})
    assert child.parent_document_type_id == "sop-1"
    assert child.depth_level == 2


def test_omitted_parent_never_creates_second_root() -> None:
    store = _FakeStore()
    basic = _record(
        document_type_id="basic-1",
        repository_id="repo-1",
        name="Basic",
        parent_document_type_id=None,
        depth_level=0,
        is_system=True,
    )
    store.records[basic.document_type_id] = basic
    domain = DocumentTypeDomain(store)  # type: ignore[arg-type]

    first = domain.create_type("repo-1", {"name": "SOP"})
    second = domain.create_type("repo-1", {"name": "Policy"})

    assert first.parent_document_type_id == "basic-1"
    assert second.parent_document_type_id == "basic-1"
    roots = [record for record in store.records.values() if record.repository_id == "repo-1" and record.parent_document_type_id is None]
    assert len(roots) == 1
    assert roots[0].document_type_id == "basic-1"
