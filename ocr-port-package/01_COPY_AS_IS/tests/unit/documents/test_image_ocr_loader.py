from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from src.features.document_processing.loaders.image_ocr_loader import ImageOcrLoader

REPO_TMP = Path(__file__).resolve().parents[3] / ".pytest-ocr-tmp"


def _scratch() -> Path:
    REPO_TMP.mkdir(parents=True, exist_ok=True)
    return REPO_TMP


def test_image_ocr_loader_returns_tesseract_text_blocks() -> None:
    image_path = _scratch() / "scan.png"
    image_path.write_bytes(b"not-needed-when-tesseract-is-mocked")
    result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Patient name\nJane Doe\n", stderr=""
    )

    with (
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.shutil.which",
            return_value="/usr/bin/tesseract",
        ),
        patch(
            "src.features.document_processing.loaders.image_ocr_loader.subprocess.run",
            return_value=result,
        ) as run,
    ):
        blocks = ImageOcrLoader().load(image_path)

    assert [block.text for block in blocks] == ["Patient name", "Jane Doe"]
    assert [block.line_number for block in blocks] == [1, 2]
    assert all(block.metadata["source"] == "tesseract_ocr" for block in blocks)
    assert run.call_args.args[0][0] == "/usr/bin/tesseract"
    assert "--psm" in run.call_args.args[0]


def test_garbage_pdf_text_layer_is_ocrd() -> None:
    from src.features.document_processing.loaders.loader import Block
    from src.features.document_processing.loaders.scanned_pdf_ocr import pdf_pages_needing_ocr

    garbage = [
        Block(text="(cid:12)(cid:18) .", page=1, line_number=1, block_type="text"),
    ]
    assert pdf_pages_needing_ocr(garbage, 1) == [1]


def test_scanned_pdf_ocr_adds_text_when_native_layer_empty() -> None:
    from src.features.document_processing.loaders.loader import Block
    from src.features.document_processing.loaders.scanned_pdf_ocr import apply_ocr_to_scanned_pdf

    pdf_path = _scratch() / "receipt.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    image_only = [
        Block(
            text="",
            page=1,
            line_number=1,
            block_type="image",
            component_type="image",
            metadata={"image_index": 1},
        )
    ]
    ocr_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="STORE 12\nTOTAL 4.50\n", stderr=""
    )

    with (
        patch(
            "src.features.document_processing.loaders.scanned_pdf_ocr._ocr_processor_enabled",
            return_value=True,
        ),
        patch(
            "src.features.document_processing.loaders.scanned_pdf_ocr.render_pdf_page_png",
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
        blocks = apply_ocr_to_scanned_pdf(pdf_path, image_only)

    texts = [block.text for block in blocks if block.text]
    assert "STORE 12" in texts
    assert "TOTAL 4.50" in texts
    assert any(block.metadata.get("ocr_used") for block in blocks)


def test_scanned_pdf_ocr_skips_pages_with_native_text() -> None:
    from src.features.document_processing.loaders.loader import Block
    from src.features.document_processing.loaders.scanned_pdf_ocr import apply_ocr_to_scanned_pdf

    pdf_path = _scratch() / "native.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    native = [
        Block(
            text="This SOP already has a selectable text layer on the page.",
            page=1,
            line_number=1,
            block_type="text",
            component_type="paragraph",
        )
    ]

    with (
        patch(
            "src.features.document_processing.loaders.scanned_pdf_ocr._ocr_processor_enabled",
            return_value=True,
        ),
        patch(
            "src.features.document_processing.loaders.scanned_pdf_ocr.ocr_scanned_pdf_pages",
        ) as ocr_pages,
    ):
        blocks = apply_ocr_to_scanned_pdf(pdf_path, native)

    ocr_pages.assert_not_called()
    assert blocks == native
