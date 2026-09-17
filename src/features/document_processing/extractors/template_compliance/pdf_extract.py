"""PDF template extraction — same page-1 colon metadata rules as DOCX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import UnsupportedFormatError
from .page_meta import extract_page1_metadata  # noqa: F401 — shared metadata API


def extract_pdf_template(path: Path) -> dict[str, Any]:
    """
    PDF extraction is stubbed while DOCX is completed.

    When implemented, page-1 text lines must be passed to
    ``extract_page1_metadata(lines)`` so metadata uses the same
    Key: Value colon rules (body page 1 only — not headers/footers).
    """
    raise UnsupportedFormatError(
        f"PDF extraction is not implemented yet (focused on DOCX). Got: {path}"
    )
