"""Public API for template extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import (
    ExtractionError,
    TemplateFileNotFoundError,
    UnsupportedFormatError,
)


def extract_template(path: str | Path) -> dict[str, Any]:
    """
    Extract a template document into the common JSON structure.

    DOCX is fully supported. PDF is stubbed (same schema later).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise TemplateFileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        from .docx_extract import extract_docx_template

        try:
            return extract_docx_template(file_path)
        except TemplateFileNotFoundError:
            raise
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise ExtractionError(f"DOCX extraction failed for {file_path}: {exc}") from exc

    if suffix == ".pdf":
        from .pdf_extract import extract_pdf_template

        return extract_pdf_template(file_path)

    raise UnsupportedFormatError(
        f"Unsupported format '{suffix}'. Supported today: .docx (PDF stubbed)."
    )


def extract_template_to_json(
    path: str | Path,
    output_path: str | Path,
    *,
    indent: int = 2,
) -> dict[str, Any]:
    """Extract and write JSON to ``output_path``."""
    document = extract_template(path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=indent), encoding="utf-8")
    return document
