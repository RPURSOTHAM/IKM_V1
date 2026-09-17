"""High-fidelity DOCX to HTML conversion for read-only preview."""

from __future__ import annotations

from pathlib import Path

from src.features.documents.infrastructure.content.render_html import (
    blocks_to_html,
    wrap_article_in_page_shell,
    wrap_render_document_html,
)


def docx_to_preview_html(
    document_path: Path,
    *,
    title: str,
    blocks: list | None = None,
    page_count: int | None = None,
) -> str:
    """Convert DOCX to styled HTML using the same paginated shell as PDF/TXT previews."""
    normalized_page_count = int(page_count or 0) or None
    if blocks and normalized_page_count and normalized_page_count > 1:
        return wrap_render_document_html(
            blocks_to_html(blocks, page_count=normalized_page_count),
            title=title,
        )

    try:
        import mammoth
    except ImportError:
        mammoth = None  # type: ignore[assignment]

    if mammoth is not None:
        try:
            with document_path.open("rb") as handle:
                result = mammoth.convert_to_html(handle)
            body = result.value.strip()
            if body:
                styled = wrap_article_in_page_shell(body, extra_class="mammoth-preview")
                return wrap_render_document_html(styled, title=title)
        except Exception:
            pass

    if blocks is not None:
        return wrap_render_document_html(
            blocks_to_html(blocks, page_count=normalized_page_count),
            title=title,
        )

    raise RuntimeError("DOCX preview requires mammoth or pre-loaded document blocks.")
