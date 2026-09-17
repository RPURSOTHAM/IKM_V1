from __future__ import annotations

from unittest.mock import patch

from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdf_visual_content import (
    attach_visual_extraction,
    image_block_extraction_metadata,
    infer_visual_content_type,
    looks_like_tabular_ocr_text,
    merge_bboxes,
    should_request_vision_description,
)


def test_merge_bboxes_combines_regions() -> None:
    merged = merge_bboxes([(0, 0, 100, 100), (80, 50, 200, 150)])
    assert merged == (0, 0, 200, 150)


def test_tabular_ocr_heuristic_detects_grid_like_text() -> None:
    text = "Col A | Col B\nRow1 | Val1\nRow2 | Val2\nRow3 | Val3"
    assert looks_like_tabular_ocr_text(text) is True


def test_infer_visual_content_type() -> None:
    assert infer_visual_content_type("label one", "") == "image_ocr"
    assert infer_visual_content_type("", "scatter plot with legend") == "image_vision"
    assert infer_visual_content_type("axis label", "legend entry") == "image_multimodal"


def test_attach_visual_extraction_sets_metadata() -> None:
    block = Block(text="", page=3, line_number=1, block_type="image", component_type="image")
    attach_visual_extraction(
        block,
        ocr_text="BMab1200 45 mg PFS",
        vision_text="Scatter plot comparing samples",
        source="tesseract_and_vision",
    )
    assert "BMab1200" in block.text
    assert block.metadata["embedded_image_ocr"] is True
    assert block.metadata["content_type"] == "image_multimodal"


def test_image_block_extraction_metadata_includes_bbox() -> None:
    block = Block(
        text="chart labels",
        page=58,
        line_number=2,
        block_type="image",
        component_type="image",
        metadata={
            "image_index": 1,
            "x0": 40.0,
            "top": 120.0,
            "x1": 560.0,
            "bottom": 420.0,
            "ocr_text": "SC CE SDS",
            "visual_description": "Four scatter plots",
            "embedded_image_ocr": True,
            "image_text_source": "tesseract_and_vision",
        },
    )
    metadata = image_block_extraction_metadata(block)
    assert metadata["image_reference"]["page_number"] == 58
    assert metadata["image_reference"]["bbox"] == [40.0, 120.0, 560.0, 420.0]
    assert metadata["content_source"] == "tesseract_and_vision"


def test_should_request_vision_when_ocr_is_sparse() -> None:
    with patch.dict(
        "os.environ",
        {"PDF_VISION_FOR_EMBEDDED_IMAGES": "true", "PDF_VISION_WHEN_OCR_CHARS_BELOW": "50"},
        clear=False,
    ):
        assert should_request_vision_description("short") is True
        assert should_request_vision_description("x" * 80) is False
