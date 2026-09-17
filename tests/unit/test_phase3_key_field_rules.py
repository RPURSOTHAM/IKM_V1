"""Pure unit tests for Phase 3 key-field type canonicalization and hierarchy rules."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.features.document_types.domain.document_type_exceptions import DuplicateFieldError
from src.features.document_types.application.key_field_service import KeyFieldDomain
from src.features.document_types.domain.models import DocumentTypeRecord, KeyFieldType


def test_key_field_type_canonical_aliases() -> None:
    assert KeyFieldType.canonical("string") == "string"
    assert KeyFieldType.canonical("integer") == "integer"
    assert KeyFieldType.canonical("float") == "float"
    assert KeyFieldType.canonical("boolean") == "boolean"
    assert KeyFieldType.canonical("date") == "date"
    assert KeyFieldType.canonical("number") == "float"
    assert KeyFieldType.canonical("text") == "string"
    assert KeyFieldType.canonical("datetime") == "date"


def test_hierarchy_duplicate_field_names_rejected() -> None:
    store = MagicMock()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    store.get_type_by_id.return_value = DocumentTypeRecord(
        document_type_id="child",
        repository_id="repo",
        name="SOP",
        code=None,
        description=None,
        parent_document_type_id="basic",
        depth_level=1,
        is_system=False,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    store.key_field_name_on_type.side_effect = lambda type_id, name: type_id == "basic" and name == "version"
    store.ancestor_chain.return_value = ["child", "basic"]
    store.validate_field_name.return_value = None

    domain = KeyFieldDomain(store)
    with pytest.raises(DuplicateFieldError):
        domain.create_field("child", {"field_name": "version", "field_type": "string"})
