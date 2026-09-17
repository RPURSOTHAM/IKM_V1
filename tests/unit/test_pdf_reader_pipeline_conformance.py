import pytest
pytest.skip("legacy docx_template/pdf_template removed; replaced by template_compliance", allow_module_level=True)

"""Pipeline conformance: pdf_reader contract must survive adapter â†’ template nodes.

Uses the shared pdf_reader field definitions as the source of truth. Does NOT
upload or parse the pdf_reader package itself as a document.
"""

from __future__ import annotations

from src.features.document_processing.extractors.pdf_template.pdf_reader.utils import span_attributes
from src.features.document_processing.extractors.pdf_template.pdf_reader_adapter import (
    adapt_pdf_reader_output_to_blocks,
)
from src.features.document_processing.extractors.pdf_template.extractor import (
    _build_document_tree,
    _heading_node_from_block,
)


def test_pdf_reader_span_attributes_contract() -> None:
    """pdf_reader span schema: font_name, font_size, font_color, bold, italic, bbox."""
    attrs = span_attributes(
        {
            "text": "Purpose",
            "font": "Helvetica-Bold",
            "size": 14.4,
            "color": 0x112233,
            "flags": 16,  # TEXT_FONT_BOLD in PyMuPDF when available; style also checks name
            "bbox": (10.0, 20.0, 100.0, 40.0),
        }
    )
    assert attrs["font_name"] == "Helvetica-Bold"
    assert attrs["font_size"] == 14
    assert attrs["font_color"].startswith("#")
    assert "bold" in attrs and "italic" in attrs
    assert attrs["bbox"] == [10.0, 20.0, 100.0, 40.0]


def test_adapter_preserves_font_and_full_bbox_from_pdf_reader() -> None:
    raw = {
        "pages": [
            {
                "page_number": 1,
                "width": 612.0,
                "height": 792.0,
                "items": [
                    {
                        "type": "text",
                        "text": "1.0 PURPOSE",
                        "bbox": [72.0, 100.0, 300.0, 118.0],
                        "attributes": {
                            "font_size": 14,
                            "page_zone": None,
                            "text_role": "header_1",
                            "spans": [
                                {
                                    "text": "1.0 PURPOSE",
                                    "font_name": "TimesNewRomanPS-BoldMT",
                                    "font_size": 14,
                                    "font_color": "#000000",
                                    "bold": True,
                                    "italic": False,
                                    "bbox": [72.0, 100.0, 300.0, 118.0],
                                }
                            ],
                        },
                    },
                    {
                        "type": "table",
                        "bbox": [72.0, 200.0, 540.0, 280.0],
                        "attributes": {"table_index": 1, "row_count": 2, "column_count": 2},
                        "rows": [["A", "B"], ["1", "2"]],
                    },
                ],
            }
        ],
        "summary": {
            "normal_font_size": 11,
            "font_size_categories": {"11": "body", "14": "header_1"},
        },
    }

    blocks = adapt_pdf_reader_output_to_blocks(raw)
    text_blocks = [b for b in blocks if b.block_type == "text"]
    table_blocks = [b for b in blocks if b.block_type == "table"]
    assert text_blocks, "expected text block from pdf_reader item"
    block = text_blocks[0]
    assert block.font_name == "TimesNewRomanPS-BoldMT"
    assert float(block.font_size) == 14.0
    assert block.bold is True
    assert block.metadata.get("font_color") == "#000000"
    assert block.metadata.get("x0") == 72.0
    assert block.metadata.get("top") == 100.0
    assert block.metadata.get("x1") == 300.0
    assert block.metadata.get("bottom") == 118.0
    assert block.metadata.get("bbox") == [72.0, 100.0, 300.0, 118.0]
    assert block.metadata.get("spans")
    assert block.metadata.get("text_role") == "header_1"

    assert table_blocks
    assert table_blocks[0].metadata.get("x0") == 72.0
    assert table_blocks[0].metadata.get("bottom") == 280.0
    assert table_blocks[0].metadata.get("rows") == [["A", "B"], ["1", "2"]]


def test_heading_node_receives_pdf_reader_typography_and_bbox() -> None:
    raw = {
        "pages": [
            {
                "page_number": 1,
                "items": [
                    {
                        "type": "text",
                        "text": "Document Title",
                        "bbox": [50.0, 60.0, 400.0, 90.0],
                        "attributes": {
                            "font_size": 18,
                            "spans": [
                                {
                                    "text": "Document Title",
                                    "font_name": "Arial",
                                    "font_size": 18,
                                    "font_color": "#010203",
                                    "bold": True,
                                    "italic": False,
                                    "bbox": [50.0, 60.0, 400.0, 90.0],
                                }
                            ],
                        },
                    }
                ],
            }
        ]
    }
    blocks = adapt_pdf_reader_output_to_blocks(raw)
    assert blocks
    # Force title component so document_tree emits a heading node.
    blocks[0].component_type = "title"
    blocks[0].heading_level = 1
    node = _heading_node_from_block(blocks[0])
    assert node["font_name"] == "Arial"
    assert node["font_size"] == 18.0
    assert node["font_color"] == "#010203"
    assert node["bold"] is True
    assert node["position"]["x0"] == 50.0
    assert node["position"]["bottom"] == 90.0

    tree = _build_document_tree(blocks, {})
    assert any(n.get("type") == "heading" and n.get("font_name") == "Arial" for n in tree)

