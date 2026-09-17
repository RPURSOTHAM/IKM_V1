"""Unit tests for document component classification."""

from __future__ import annotations

from src.features.document_processing.loaders.component_classification import (
    COMPONENT_FOOTER,
    COMPONENT_HEADER,
    COMPONENT_IMAGE,
    COMPONENT_PARAGRAPH,
    COMPONENT_SUBTITLE,
    COMPONENT_TABLE,
    COMPONENT_TITLE,
    apply_component_metadata,
    block_component_type,
    classify_text_block,
    heading_level_from_style,
)
from src.features.document_processing.loaders.loader import Block


def test_heading_level_from_word_styles() -> None:
    assert heading_level_from_style("Heading 1") == 1
    assert heading_level_from_style("Heading 2") == 2
    assert heading_level_from_style("Title") == 1
    assert heading_level_from_style("Subtitle") == 2
    assert heading_level_from_style("Normal") is None


def test_classify_title_from_style() -> None:
    block = Block(text="Purpose", page=1, line_number=1, style="Heading 1", bold=True)
    component, level = classify_text_block(block)
    assert component == COMPONENT_TITLE
    assert level == 1


def test_classify_subtitle_from_numbered_text() -> None:
    block = Block(text="2.1 Scope Of Application", page=1, line_number=2, bold=True)
    component, level = classify_text_block(block)
    assert component == COMPONENT_SUBTITLE
    assert level == 2



def test_classify_header_region() -> None:
    block = Block(text="SOP No: ABC-001", page=1, line_number=1)
    component, _ = classify_text_block(block, region="header")
    assert component == COMPONENT_HEADER


def test_top_title_is_not_forced_to_header_when_prominent() -> None:
    block = Block(
        text="Equipment Specification",
        page=1,
        line_number=1,
        font_size=18.0,
        bold=True,
    )
    component, level = classify_text_block(block, region="header", median_font_size=11.0)
    assert component == COMPONENT_TITLE
    assert level == 1


def test_classify_footer_region() -> None:
    block = Block(text="Page 1 of 10", page=1, line_number=99)
    component, _ = classify_text_block(block, region="footer")
    assert component == COMPONENT_FOOTER


def test_classify_table_and_image_block_types() -> None:
    table = Block(text="Table:\na | b", page=1, line_number=3, block_type="table")
    image = Block(text="Image 1 on page 1", page=1, line_number=4, block_type="image")
    assert classify_text_block(table)[0] == COMPONENT_TABLE
    assert classify_text_block(image)[0] == COMPONENT_IMAGE


def test_apply_component_metadata_sets_fields() -> None:
    block = Block(text="1.0 OBJECTIVE", page=1, line_number=5, style="Heading 1")
    apply_component_metadata(block)
    assert block.component_type == COMPONENT_TITLE
    assert block.heading_level == 1
    assert block.metadata.get("component_type") == COMPONENT_TITLE


def test_block_component_type_prefers_explicit_value() -> None:
    block = Block(
        text="Body text",
        page=1,
        line_number=6,
        component_type=COMPONENT_PARAGRAPH,
    )
    assert block_component_type(block) == COMPONENT_PARAGRAPH
