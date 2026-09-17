from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdf_embedded_image_ocr import (
    _select_ocr_candidates,
    apply_ocr_to_embedded_pdf_images,
    embedded_image_ocr_enabled,
)

REPO_TMP = Path(__file__).resolve().parents[3] / ".pytest-ocr-tmp"


def _scratch() -> Path:
    REPO_TMP.mkdir(parents=True, exist_ok=True)
    return REPO_TMP


def test_embedded_image_ocr_attaches_text_to_image_block() -> None:
    pdf_path = _scratch() / "chart.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    blocks = [
        Block(
            text="",
            page=2,
            line_number=3,
            block_type="image",
            component_type="image",
            metadata={
                "image_index": 1,
                "x0": 50.0,
                "top": 100.0,
                "x1": 500.0,
                "bottom": 400.0,
                "width": 450.0,
                "height": 300.0,
            },
        )
    ]
    ocr_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="148281 G0F/G0F\nIJS1801-US-130mg-Vial\n",
        stderr="",
    )

    with (
        patch.dict("os.environ", {"PDF_EMBEDDED_IMAGE_OCR_ENABLED": "true"}, clear=False),
        patch(
            "src.features.document_processing.loaders.pdf_embedded_image_ocr.embedded_image_ocr_enabled",
            return_value=True,
        ),
        patch(
            "src.features.document_processing.loaders.pdf_embedded_image_ocr.render_pdf_image_crop",
        ),
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.shutil.which",
            return_value="/usr/bin/tesseract",
        ),
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.subprocess.run",
            return_value=ocr_result,
        ),
    ):
        updated = apply_ocr_to_embedded_pdf_images(pdf_path, blocks)

    image = updated[0]
    assert "148281" in image.text
    assert image.metadata["image_caption"] == image.text
    assert image.metadata["embedded_image_ocr"] is True
    assert image.metadata["ocr_used"] is True


def test_embedded_image_ocr_skips_when_disabled() -> None:
    pdf_path = _scratch() / "disabled.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    blocks = [
        Block(
            text="",
            page=1,
            line_number=1,
            block_type="image",
            component_type="image",
            metadata={"x0": 0, "top": 0, "x1": 100, "bottom": 100, "width": 100, "height": 100},
        )
    ]

    with patch.dict("os.environ", {"PDF_EMBEDDED_IMAGE_OCR_ENABLED": "false"}, clear=False):
        assert embedded_image_ocr_enabled() is False
        updated = apply_ocr_to_embedded_pdf_images(pdf_path, blocks)

    assert updated[0].text == ""


def test_embedded_image_ocr_respects_max_image_limit() -> None:
    pdf_path = _scratch() / "many-images.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    blocks = [
        Block(
            text="",
            page=1,
            line_number=index,
            block_type="image",
            component_type="image",
            metadata={
                "image_index": index,
                "x0": 0.0,
                "top": float(index * 120),
                "x1": float(100 + index),
                "bottom": float(100 + index * 120),
                "width": float(100 + index),
                "height": 100.0,
            },
        )
        for index in range(1, 5)
    ]
    ocr_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="peak 148281\n", stderr=""
    )

    with (
        patch(
            "src.features.document_processing.loaders.pdf_embedded_image_ocr.embedded_image_ocr_enabled",
            return_value=True,
        ),
        patch.dict(
            "os.environ",
            {"PDF_MAX_EMBEDDED_IMAGES_OCR": "2"},
            clear=False,
        ),
        patch(
            "src.features.document_processing.loaders.pdf_embedded_image_ocr.render_pdf_image_crop",
        ),
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.shutil.which",
            return_value="/usr/bin/tesseract",
        ),
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.subprocess.run",
            return_value=ocr_result,
        ),
    ):
        updated = apply_ocr_to_embedded_pdf_images(pdf_path, blocks)

    ocr_count = sum(1 for block in updated if block.text.strip())
    assert ocr_count == 2


def test_select_ocr_candidates_applies_per_page_limit() -> None:
    blocks = [
        Block(
            text="",
            page=1,
            line_number=index,
            block_type="image",
            component_type="image",
            metadata={
                "x0": 0.0,
                "top": 0.0,
                "x1": float(100 + index * 10),
                "bottom": float(100 + index * 10),
                "width": float(100 + index * 10),
                "height": float(100 + index * 10),
            },
        )
        for index in range(1, 4)
    ] + [
        Block(
            text="",
            page=2,
            line_number=index,
            block_type="image",
            component_type="image",
            metadata={
                "x0": 0.0,
                "top": 0.0,
                "x1": float(200 + index * 10),
                "bottom": float(200 + index * 10),
                "width": float(200 + index * 10),
                "height": float(200 + index * 10),
            },
        )
        for index in range(1, 3)
    ]

    selected = _select_ocr_candidates(blocks, max_total=10, max_per_page=1)
    pages = {int(block.page or 0) for block in selected}

    assert len(selected) == 2
    assert pages == {1, 2}


def test_embedded_image_ocr_adds_fitz_discovered_blocks() -> None:
    pdf_path = _scratch() / "fitz-discovery.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    blocks = [
        Block(text="Figure caption", page=58, line_number=1, block_type="text", component_type="paragraph"),
    ]
    discovered = Block(
        text="",
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
            "width": 520.0,
            "height": 300.0,
        },
    )
    ocr_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="SC CE SDS NgHC BMab1200 45 mg PFS\n",
        stderr="",
    )

    with (
        patch(
            "src.features.document_processing.loaders.pdf_embedded_image_ocr.embedded_image_ocr_enabled",
            return_value=True,
        ),
        patch(
            "src.features.document_processing.loaders.pdf_embedded_image_ocr._discover_fitz_image_blocks",
            return_value=[discovered],
        ),
        patch(
            "src.features.document_processing.loaders.pdf_embedded_image_ocr.render_pdf_image_crop",
        ),
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.shutil.which",
            return_value="/usr/bin/tesseract",
        ),
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.subprocess.run",
            return_value=ocr_result,
        ),
    ):
        updated = apply_ocr_to_embedded_pdf_images(pdf_path, blocks)

    image_blocks = [block for block in updated if block.block_type == "image"]
    assert len(image_blocks) == 1
    assert "BMab1200" in image_blocks[0].text
    assert image_blocks[0].metadata["embedded_image_ocr"] is True
