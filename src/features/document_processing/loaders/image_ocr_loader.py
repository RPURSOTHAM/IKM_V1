"""Standalone image loader backed by the Tesseract command-line OCR engine."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Union

from src.features.document_processing.loaders.loader import Block, DocumentLoader

_DEFAULT_PSM = "6"
_PSM_CANDIDATES = ("6", "4", "11", "3")


def tesseract_binary() -> str:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise RuntimeError(
            "OCR requires Tesseract. Install tesseract-ocr in the service image."
        )
    return tesseract


def ocr_text_score(text: str) -> int:
    """Prefer results with more letters/digits over sparse or noisy output."""
    cleaned = " ".join((text or "").split())
    return sum(ch.isalnum() for ch in cleaned)


def ocr_image_file(image_path: Path, *, psm: str = _DEFAULT_PSM, timeout: int = 120) -> str:
    """Run Tesseract on an image file and return raw stdout text."""
    tesseract = tesseract_binary()
    try:
        result = subprocess.run(
            [tesseract, str(image_path), "stdout", "--psm", str(psm), "-l", "eng"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Image OCR failed for {image_path.name}: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown Tesseract error").strip()
        raise RuntimeError(f"Image OCR failed for {image_path.name}: {detail[:500]}")
    return result.stdout or ""


def prepare_ocr_variants(image_path: Path, work_dir: Path) -> list[Path]:
    """Build contrast/upscale/inverted copies so scanned receipts OCR more cleanly."""
    variants = [image_path]
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except Exception:
        return variants

    try:
        image = Image.open(image_path)
        image.load()
    except Exception:
        return variants

    gray = ImageOps.exif_transpose(image).convert("L")
    width, height = gray.size
    longest = max(width, height)
    if longest < 1600:
        scale = 1600 / max(longest, 1)
        gray = gray.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.6)
    gray = gray.filter(ImageFilter.SHARPEN)

    prepared = work_dir / "prepared.png"
    gray.save(prepared, format="PNG")
    variants.append(prepared)

    inverted = work_dir / "inverted.png"
    ImageOps.invert(gray).save(inverted, format="PNG")
    variants.append(inverted)
    return variants


def ocr_image_best(image_path: Path, *, page_segmentation_modes: Iterable[str] | None = None) -> str:
    """Try preprocessed variants and several Tesseract layouts; keep the richest text."""
    modes = tuple(page_segmentation_modes or _PSM_CANDIDATES)
    best_text = ""
    best_score = -1
    with tempfile.TemporaryDirectory(prefix="ocr-prep-") as tmp:
        for candidate in prepare_ocr_variants(image_path, Path(tmp)):
            for psm in modes:
                try:
                    text = ocr_image_file(candidate, psm=psm)
                except RuntimeError:
                    continue
                score = ocr_text_score(text)
                if score > best_score:
                    best_score = score
                    best_text = text
    return best_text


def blocks_from_ocr_text(text: str, *, page: int = 1) -> list[Block]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return [
        Block(
            text=line,
            page=page,
            line_number=index,
            block_type="text",
            component_type="paragraph",
            metadata={"source": "tesseract_ocr", "ocr_used": True},
        )
        for index, line in enumerate(lines, start=1)
    ]


class ImageOcrLoader(DocumentLoader):
    """Extract readable text from a single uploaded image using Tesseract."""

    def load(self, path: Union[str, Path]) -> list[Block]:
        document_path = Path(path)
        return blocks_from_ocr_text(ocr_image_best(document_path), page=1)
