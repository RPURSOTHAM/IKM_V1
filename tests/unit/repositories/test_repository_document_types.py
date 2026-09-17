"""Unit tests for repository-scoped document type hierarchy and key field inheritance."""

from __future__ import annotations

import uuid

import pytest

from src.features.document_types.domain.document_type_exceptions import (
    CircularReferenceError,
    DuplicateFieldError,
    DuplicateNameError,
    RepositoryMismatchError,
    SystemTypeImmutableError,
)
from src.features.document_types.application.document_type_service import DocumentTypeService
from src.features.repositories.application.repository_service import RepositoryService


@pytest.fixture
def repo_service() -> RepositoryService:
    service = RepositoryService()
    service._require_store()
    return service


@pytest.fixture
def doc_type_service() -> DocumentTypeService:
    service = DocumentTypeService()
    service._require_store()
    return service


def _unique_name() -> str:
    return f"unit-doctype-{uuid.uuid4().hex[:10]}"


def _create_repository(repo_service: RepositoryService) -> dict:
    return repo_service.create_repository({"name": _unique_name(), "owner_user_id": "admin"})


def test_repository_creation_creates_basic_type(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]

    listing = doc_type_service.list_repository_document_types(repository_id)
    assert listing["count"] == 1
    basic = listing["document_types"][0]
    assert basic["name"] == "Basic"
    assert basic["repository_id"] == repository_id
    assert basic["parent_document_type_id"] is None
    assert basic["is_system"] is True
    assert created["settings"]["document_type_id"] == basic["document_type_id"]


def test_default_key_fields_are_created(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    basic_id = created["settings"]["document_type_id"]

    fields = doc_type_service.list_key_fields(basic_id)
    assert fields["count"] == 4
    field_names = {field["field_name"] for field in fields["fields"]}
    assert field_names == {"document_title", "document_number", "version", "effective_date"}


def test_parent_child_hierarchy_creation(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    assert sop["parent_document_type_id"] == basic_id
    assert sop["repository_id"] == repository_id

    tree = doc_type_service.get_repository_document_type_tree(repository_id)
    assert len(tree["tree"]) == 1
    assert tree["tree"][0]["name"] == "Basic"
    assert len(tree["tree"][0]["children"]) == 1
    assert tree["tree"][0]["children"][0]["name"] == "SOP"


def test_omitted_parent_auto_attaches_to_basic(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    sop = doc_type_service.create_repository_document_type(repository_id, {"name": "SOP"})
    assert sop["parent_document_type_id"] == basic_id
    assert sop["depth_level"] == 1


def test_explicit_parent_is_preserved_when_provided(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    work_instruction = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "Work Instruction", "parent_document_type_id": sop["document_type_id"]},
    )
    assert work_instruction["parent_document_type_id"] == sop["document_type_id"]
    assert work_instruction["parent_document_type_id"] != basic_id
    assert work_instruction["depth_level"] == 2


def test_omitted_parent_does_not_create_second_root(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]

    doc_type_service.create_repository_document_type(repository_id, {"name": "SOP"})
    doc_type_service.create_repository_document_type(repository_id, {"name": "Policy"})

    tree = doc_type_service.get_repository_document_type_tree(repository_id)
    assert len(tree["tree"]) == 1
    root = tree["tree"][0]
    assert root["name"] == "Basic"
    child_names = {child["name"] for child in root["children"]}
    assert child_names == {"SOP", "Policy"}


def test_multi_level_hierarchy_creation(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    manufacturing = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "Manufacturing SOP", "parent_document_type_id": sop["document_type_id"]},
    )
    validation = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "Validation SOP", "parent_document_type_id": sop["document_type_id"]},
    )

    tree = doc_type_service.get_repository_document_type_tree(repository_id)
    basic_node = tree["tree"][0]
    sop_node = basic_node["children"][0]
    child_names = {child["name"] for child in sop_node["children"]}
    assert child_names == {"Manufacturing SOP", "Validation SOP"}
    assert manufacturing["depth_level"] == 2
    assert validation["depth_level"] == 2


def test_circular_reference_prevention(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]
    store = doc_type_service._require_store()

    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    child = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "Manufacturing SOP", "parent_document_type_id": sop["document_type_id"]},
    )

    with store._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE document_type SET parent_document_type_id = %s WHERE document_type_id = %s",
                (child["document_type_id"], sop["document_type_id"]),
            )

    with pytest.raises(CircularReferenceError):
        doc_type_service.create_repository_document_type(
            repository_id,
            {"name": "Blocked", "parent_document_type_id": sop["document_type_id"]},
        )


def test_effective_field_inheritance_resolution(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    doc_type_service.create_key_field(
        sop["document_type_id"],
        {"field_name": "owner", "field_type": "string", "description": "Owner"},
    )
    manufacturing = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "Manufacturing SOP", "parent_document_type_id": sop["document_type_id"]},
    )
    doc_type_service.create_key_field(
        manufacturing["document_type_id"],
        {"field_name": "production_line", "field_type": "string"},
    )

    effective = doc_type_service.resolve_effective_fields(manufacturing["document_type_id"])
    field_names = {field["field_name"] for field in effective["fields"]}
    assert field_names == {
        "document_title",
        "document_number",
        "version",
        "effective_date",
        "owner",
        "production_line",
    }


def test_child_field_override_behavior(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    """Duplicate field names in the hierarchy are rejected (Phase 3 rule)."""
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    with pytest.raises(DuplicateFieldError):
        doc_type_service.create_key_field(
            sop["document_type_id"],
            {
                "field_name": "version",
                "field_type": "text",
                "required": True,
                "description": "SOP-specific version",
            },
        )


def test_integer_and_float_field_types(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]
    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    count_field = doc_type_service.create_key_field(
        sop["document_type_id"],
        {"field_name": "batch_count", "field_type": "integer", "required": True},
    )
    amount_field = doc_type_service.create_key_field(
        sop["document_type_id"],
        {"field_name": "batch_amount", "field_type": "float"},
    )
    assert count_field["field_type"] == "integer"
    assert count_field["type"] == "integer"
    assert amount_field["field_type"] == "float"
    assert amount_field["type"] == "float"


def test_nested_get_enforces_repository_scope(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    repo_a = _create_repository(repo_service)
    repo_b = _create_repository(repo_service)
    basic_a = repo_a["settings"]["document_type_id"]
    with pytest.raises(RepositoryMismatchError):
        doc_type_service.get_type(basic_a, repository_id=repo_b["repository_id"])


def test_repository_isolation(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    repo_a = _create_repository(repo_service)
    repo_b = _create_repository(repo_service)

    listing_a = doc_type_service.list_repository_document_types(repo_a["repository_id"])
    listing_b = doc_type_service.list_repository_document_types(repo_b["repository_id"])
    assert listing_a["count"] == 1
    assert listing_b["count"] == 1
    assert listing_a["document_types"][0]["document_type_id"] != listing_b["document_types"][0]["document_type_id"]

    with pytest.raises(RepositoryMismatchError):
        doc_type_service.get_type(
            listing_a["document_types"][0]["document_type_id"],
            repository_id=repo_b["repository_id"],
        )


def test_type_in_use_cannot_be_deleted(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    from src.features.document_types.domain.document_type_exceptions import TypeInUseError

    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]
    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    store = doc_type_service._require_store()
    store.upsert_document_instance(
        document_id=f"doc-{uuid.uuid4().hex[:12]}",
        document_type_id=sop["document_type_id"],
        repository_id=repository_id,
        tenant_id=None,
    )
    with pytest.raises(TypeInUseError):
        doc_type_service.delete_type(sop["document_type_id"])


def test_duplicate_name_validation(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    with pytest.raises(DuplicateNameError):
        doc_type_service.create_repository_document_type(
            repository_id,
            {"name": "SOP", "parent_document_type_id": basic_id},
        )


def test_duplicate_field_name_validation(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    repository_id = created["repository_id"]
    basic_id = created["settings"]["document_type_id"]

    sop = doc_type_service.create_repository_document_type(
        repository_id,
        {"name": "SOP", "parent_document_type_id": basic_id},
    )
    doc_type_service.create_key_field(
        sop["document_type_id"],
        {"field_name": "owner", "field_type": "string"},
    )
    with pytest.raises(DuplicateFieldError):
        doc_type_service.create_key_field(
            sop["document_type_id"],
            {"field_name": "owner", "field_type": "string"},
        )


def test_basic_type_cannot_be_deleted(
    repo_service: RepositoryService,
    doc_type_service: DocumentTypeService,
) -> None:
    created = _create_repository(repo_service)
    basic_id = created["settings"]["document_type_id"]
    with pytest.raises(SystemTypeImmutableError):
        doc_type_service.delete_type(basic_id)
