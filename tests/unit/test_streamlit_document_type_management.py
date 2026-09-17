from __future__ import annotations

from src.simulators.streamlit_document_uploader import (
    DOCUMENT_TYPE_QUICK_TEMPLATES,
    _basic_document_type,
    _deletable_document_types,
    _document_types_missing_for_templates,
    _find_document_type_by_name,
    _flatten_document_type_tree,
    _resolve_template_parent_id,
)


def test_find_document_type_by_name_is_case_insensitive() -> None:
    rows = [{"document_type_id": "1", "name": "SOP"}]
    found = _find_document_type_by_name(rows, "sop")
    assert found is not None
    assert found["document_type_id"] == "1"


def test_basic_document_type_prefers_system_flag() -> None:
    rows = [
        {"document_type_id": "a", "name": "Basic", "is_system": False},
        {"document_type_id": "b", "name": "Basic", "is_system": True},
    ]
    basic = _basic_document_type(rows)
    assert basic is not None
    assert basic["document_type_id"] == "b"


def test_flatten_document_type_tree_preserves_hierarchy() -> None:
    tree = [
        {
            "name": "Basic",
            "document_type_id": "basic",
            "depth_level": 0,
            "is_system": True,
            "is_active": True,
            "children": [
                {
                    "name": "SOP",
                    "document_type_id": "sop",
                    "depth_level": 1,
                    "is_system": False,
                    "is_active": True,
                    "children": [],
                }
            ],
        }
    ]
    rows = _flatten_document_type_tree(tree)
    assert [row["display_name"] for row in rows] == ["Basic", "SOP"]
    assert rows[1]["parent"] == "Basic"
    assert rows[1]["name"].startswith("  ")


def test_deletable_document_types_excludes_system_and_parents() -> None:
    rows = [
        {"document_type_id": "basic", "name": "Basic", "is_system": True},
        {"document_type_id": "sop", "name": "SOP", "parent_document_type_id": "basic"},
        {
            "document_type_id": "msop",
            "name": "Manufacturing SOP",
            "parent_document_type_id": "sop",
        },
    ]
    deletable = _deletable_document_types(rows)
    assert [row["document_type_id"] for row in deletable] == ["msop"]


def test_document_types_missing_for_templates() -> None:
    rows = [{"document_type_id": "basic", "name": "Basic", "is_system": True}]
    missing = _document_types_missing_for_templates(rows)
    assert [item["key"] for item in missing] == ["sop", "manufacturing_sop"]

    rows_with_sop = rows + [{"document_type_id": "sop", "name": "SOP"}]
    missing_after_sop = _document_types_missing_for_templates(rows_with_sop)
    assert [item["key"] for item in missing_after_sop] == ["manufacturing_sop"]


def test_resolve_template_parent_id_uses_basic_type() -> None:
    rows = [{"document_type_id": "basic-id", "name": "Basic", "is_system": True}]
    assert _resolve_template_parent_id(rows, "Basic") == "basic-id"


def test_quick_templates_define_sop_hierarchy() -> None:
    keys = [item["key"] for item in DOCUMENT_TYPE_QUICK_TEMPLATES]
    assert keys == ["sop", "manufacturing_sop"]
    assert DOCUMENT_TYPE_QUICK_TEMPLATES[1]["parent_name"] == "SOP"
