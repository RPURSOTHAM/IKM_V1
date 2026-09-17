from __future__ import annotations

from src.simulators.streamlit_document_uploader import (
    DEFAULT_KEY_FIELDS_TEXT,
    _format_key_fields_text,
    _normalize_key_field_row,
    _parse_key_fields,
)


def test_parse_key_fields_line_format() -> None:
    rows = _parse_key_fields(
        "document_number | string | true | SOP No\neffective_date | date | false | Effective Date"
    )
    assert rows == [
        {
            "name": "document_number",
            "type": "string",
            "required": True,
            "description": "SOP No",
        },
        {
            "name": "effective_date",
            "type": "date",
            "required": False,
            "description": "Effective Date",
        },
    ]


def test_parse_key_fields_json_accepts_metadata_aliases() -> None:
    rows = _parse_key_fields(
        '[{"field_name": "title", "data_type": "string", "required": true, "display_label": "SOP Title"}]'
    )
    assert rows[0]["name"] == "title"
    assert rows[0]["type"] == "string"
    assert rows[0]["required"] is True
    assert rows[0]["description"] == "SOP Title"


def test_format_key_fields_text_round_trip() -> None:
    parsed = _parse_key_fields(DEFAULT_KEY_FIELDS_TEXT)
    text = _format_key_fields_text(parsed)
    assert "document_number | string | true" in text
    assert len(_parse_key_fields(text)) == len(parsed)


def test_normalize_key_field_row_requires_name() -> None:
    try:
        _normalize_key_field_row({"type": "string"})
    except ValueError as exc:
        assert "name" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
