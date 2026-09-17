from __future__ import annotations

from src.simulators.streamlit_document_uploader import (
    _dedupe_document_types_by_id,
    _document_type_option_label,
)


def test_dedupe_document_types_by_id() -> None:
    rows = [
        {"document_type_id": "a1", "name": "Basic"},
        {"document_type_id": "a1", "name": "Basic"},
        {"document_type_id": "b2", "name": "SOP"},
    ]
    assert _dedupe_document_types_by_id(rows) == [
        {"document_type_id": "a1", "name": "Basic"},
        {"document_type_id": "b2", "name": "SOP"},
    ]


def test_document_type_option_label_uses_repository_name() -> None:
    label = _document_type_option_label(
        {"document_type_id": "abc", "name": "Basic", "depth_level": 1},
        repository_name="ph1-repo",
    )
    assert label == "Basic (ph1-repo, L1)"


def test_document_type_option_label_name_only_for_root_type() -> None:
    label = _document_type_option_label(
        {"document_type_id": "abc", "name": "Basic", "depth_level": 0},
        repository_name="ph1-repo",
    )
    assert label == "Basic (ph1-repo)"
