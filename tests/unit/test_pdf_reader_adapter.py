import pytest
pytest.skip("legacy docx_template/pdf_template removed; replaced by template_compliance", allow_module_level=True)

"""
Unit tests for pdf_reader_adapter.py
"""

from __future__ import annotations

import pytest
from src.features.document_processing.extractors.pdf_template.pdf_reader_adapter import adapt_pdf_reader_output_to_blocks


def test_adapt_pdf_reader_output_to_blocks():
    # Arrange: Mock a native pdf_reader output dictionary
    mock_document = {
        "source": "dummy.pdf",
        "document_name": "dummy.pdf",
        "summary": {
            "page_count": 1,
            "document_title": "Test Title",
            "metadata": {"Document No": "DOC-123", "Version": "1.0"},
        },
        "pages": [
            {
                "page_number": 1,
                "width": 600,
                "height": 800,
                "items": [
                    {
                        "type": "text",
                        "bbox": [50.0, 100.0, 300.0, 120.0],
                        "text": "Header Text",
                        "attributes": {
                            "font_size": 10.0,
                            "layout_role": "page_header",
                            "spans": [{"font_name": "Helvetica-Bold", "bold": True, "italic": False}],
                        },
                    },
                    {
                        "type": "text",
                        "bbox": [50.0, 200.0, 500.0, 240.0],
                        "text": "Main body paragraph text.",
                        "attributes": {
                            "font_size": 12.0,
                            "spans": [{"font_name": "Helvetica", "bold": False, "italic": False}],
                        },
                    },
                    {
                        "type": "table",
                        "bbox": [50.0, 300.0, 550.0, 450.0],
                        "attributes": {
                            "table_index": 1,
                            "row_count": 2,
                            "column_count": 2,
                        },
                        "rows": [
                            ["Col 1", "Col 2"],
                            ["Val 1", "Val 2"],
                        ],
                    },
                    {
                        "type": "image",
                        "bbox": [100.0, 500.0, 400.0, 700.0],
                        "attributes": {
                            "image_index": 5,
                            "width": 300,
                            "height": 200,
                        },
                    },
                ],
            }
        ],
    }

    # Act
    blocks = adapt_pdf_reader_output_to_blocks(mock_document)

    # Assert: We expect 4 blocks (2 text, 1 table, 1 image)
    assert len(blocks) == 4

    # Assert block 1: Header Text
    b1 = blocks[0]
    assert b1.text == "Header Text"
    assert b1.page == 1
    assert b1.line_number == 1
    assert b1.bold is True
    assert b1.italic is False
    assert b1.font_name == "Helvetica-Bold"
    assert b1.font_size == 10.0
    assert b1.block_type == "text"
    assert b1.metadata.get("region") == "header"

    # Assert block 2: Main body text
    b2 = blocks[1]
    assert b2.text == "Main body paragraph text."
    assert b2.page == 1
    assert b2.line_number == 2
    assert b2.bold is False
    assert b2.italic is False
    assert b2.font_name == "Helvetica"
    assert b2.font_size == 12.0
    assert b2.block_type == "text"
    assert b2.metadata.get("region") is None

    # Assert block 3: Table block
    b3 = blocks[2]
    assert "Table 1:" in b3.text
    assert "Col 1 | Col 2" in b3.text
    assert "Val 1 | Val 2" in b3.text
    assert b3.page == 1
    assert b3.line_number == 3
    assert b3.block_type == "table"
    assert b3.metadata["table_index"] == 1
    assert b3.metadata["row_count"] == 2
    assert b3.metadata["column_count"] == 2
    assert b3.metadata["source_line_end"] == 4  # 3 + max(1, 2) - 1 = 4

    # Assert block 4: Image block
    b4 = blocks[3]
    assert b4.text == ""
    assert b4.page == 1
    assert b4.line_number == 5  # previous line_number 3 + span 2 = 5
    assert b4.block_type == "image"
    assert b4.metadata["image_index"] == 5
    assert b4.metadata["width"] == 300
    assert b4.metadata["height"] == 200
    assert b4.metadata["x0"] == 100.0
    assert b4.metadata["top"] == 500.0

