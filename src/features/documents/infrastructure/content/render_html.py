from __future__ import annotations

import html
import re
from typing import Any

from src.features.documents.infrastructure.content.blocks import Block

_RENDER_STYLES = """
:root {
  color-scheme: light;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  background: #525659;
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  color: #1f2937;
}
.dms-rendered-document {
  max-width: 816px;
  margin: 0 auto;
}
.dms-rendered-document section[data-page] {
  background: #fff;
  min-height: 1056px;
  padding: 72px 84px;
  margin: 0 auto 24px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
  page-break-after: always;
}
.dms-page-label {
  display: block;
  margin: 0 0 16px;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #64748b;
}
.dms-rendered-document h1,
.dms-rendered-document h2,
.dms-rendered-document h3,
.dms-rendered-document h4,
.dms-rendered-document h5,
.dms-rendered-document h6,
.dms-rendered-document p,
.dms-rendered-document li {
  margin: 0 0 0.75rem;
  line-height: 1.55;
}
.dms-rendered-document h1 { font-size: 1.75rem; font-weight: 700; }
.dms-rendered-document h2 { font-size: 1.35rem; font-weight: 700; }
.dms-rendered-document h3 { font-size: 1.15rem; font-weight: 600; }
.dms-rendered-document h4 { font-size: 1.05rem; font-weight: 600; }
.dms-rendered-document p { font-size: 1rem; }
.dms-rendered-document .align-center { text-align: center; }
.dms-rendered-document .align-right { text-align: right; }
.dms-rendered-document .align-justified { text-align: justify; }
.dms-rendered-document table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 1rem;
  font-size: 0.95rem;
}
.dms-rendered-document th,
.dms-rendered-document td {
  border: 1px solid #cbd5e1;
  padding: 0.45rem 0.55rem;
  vertical-align: top;
}
.dms-rendered-document th {
  background: #f8fafc;
  font-weight: 600;
}
.dms-rendered-document .table-caption {
  font-weight: 600;
  margin-bottom: 0.35rem;
}
.dms-rendered-document .image-placeholder {
  border: 1px dashed #94a3b8;
  color: #64748b;
  padding: 1rem;
  margin: 0 0 1rem;
  text-align: center;
  background: #f8fafc;
}
.dms-rendered-document.mammoth-preview table {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 1rem;
}
.dms-rendered-document.mammoth-preview th,
.dms-rendered-document.mammoth-preview td {
  border: 1px solid #cbd5e1;
  padding: 0.45rem 0.55rem;
  vertical-align: top;
}
.dms-rendered-document.mammoth-preview ul,
.dms-rendered-document.mammoth-preview ol {
  margin: 0 0 0.75rem 1.25rem;
}
.dms-rendered-document.mammoth-preview strong,
.dms-rendered-document.mammoth-preview b {
  font-weight: 700;
}
"""


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    return getattr(block, name, default)


def _heading_tag(block: Any) -> str:
    style = str(_block_attr(block, "style", "") or "").strip().lower()
    if "heading 1" in style or style in {"title", "document title"}:
        return "h1"
    if "heading 2" in style:
        return "h2"
    if "heading 3" in style:
        return "h3"
    if "heading 4" in style:
        return "h4"
    if "heading 5" in style:
        return "h5"
    if "heading 6" in style:
        return "h6"
    font_size = float(_block_attr(block, "font_size", 0) or 0)
    if _block_attr(block, "bold", False) and font_size >= 16:
        return "h1"
    if _block_attr(block, "bold", False) and font_size >= 14:
        return "h2"
    if _block_attr(block, "bold", False) and font_size >= 12:
        return "h3"
    return "p"


def _alignment_class(block: Any) -> str:
    alignment = str(_block_attr(block, "alignment", "left") or "left").lower()
    if alignment in {"center", "right", "justified"}:
        return f"align-{alignment}"
    return ""


def _table_html(block: Any) -> str:
    text = str(_block_attr(block, "text", "") or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("table:"):
        lines = lines[1:]
    if not lines:
        return ""

    rows = [[cell.strip() for cell in line.split("|")] for line in lines]
    parts = ['<div class="table-caption">Table</div>', "<table>"]
    for index, row in enumerate(rows):
        tag = "th" if index == 0 else "td"
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<{tag}>{html.escape(cell)}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def _block_html(block: Block | Any) -> str:
    block_type = str(_block_attr(block, "block_type", "text") or "text").lower()
    if block_type == "table":
        return _table_html(block)
    if block_type == "image":
        return f'<div class="image-placeholder">{html.escape(str(_block_attr(block, "text", "") or "Image"))}</div>'

    text = str(_block_attr(block, "text", "") or "").strip()
    if not text:
        return ""

    tag = _heading_tag(block)
    css = _alignment_class(block)
    class_attr = f' class="{css}"' if css else ""
    return f"<{tag}{class_attr}>{html.escape(text)}</{tag}>"


def _distribute_blocks_across_pages(blocks: list[Any], page_count: int | None) -> list[Any]:
    """Spread single-page DOCX blocks across known page count for paginated HTML preview."""
    if not blocks or not page_count or page_count <= 1:
        return blocks
    observed = {int(_block_attr(block, "page", 1) or 1) for block in blocks}
    if len(observed) > 1 or max(observed) >= page_count:
        return blocks

    per_page = max(1, len(blocks) // page_count)
    distributed: list[Any] = []
    for index, block in enumerate(blocks):
        page = min(page_count, (index // per_page) + 1)
        if hasattr(block, "__dataclass_fields__"):
            from dataclasses import replace

            distributed.append(replace(block, page=page))
        else:
            block.page = page  # type: ignore[attr-defined]
            distributed.append(block)
    return distributed


def blocks_to_html(blocks: list[Block | Any], *, page_count: int | None = None) -> str:
    """Convert structured blocks into a paginated HTML article."""
    normalized = _distribute_blocks_across_pages(blocks, page_count)
    parts = ['<article class="dms-rendered-document">']
    current_page = None
    for block in normalized:
        page = int(_block_attr(block, "page", 1) or 1)
        if page != current_page:
            if current_page is not None:
                parts.append("</section>")
            current_page = page
            parts.append(f'<section data-page="{current_page}">')
            parts.append(f'<span class="dms-page-label">Page {current_page}</span>')
        rendered = _block_html(block)
        if rendered:
            parts.append(rendered)
    if current_page is not None:
        parts.append("</section>")
    parts.append("</article>")
    return "\n".join(parts)


def wrap_article_in_page_shell(
    body_html: str,
    *,
    page: int = 1,
    css_class: str = "dms-rendered-document",
    extra_class: str = "",
) -> str:
    """Wrap rendered body HTML in the standard paginated preview page card."""
    classes = css_class.strip()
    if extra_class:
        classes = f"{classes} {extra_class}".strip()
    return (
        f'<article class="{classes}">'
        f'<section data-page="{page}">'
        f'<span class="dms-page-label">Page {page}</span>'
        f"{body_html}"
        "</section></article>"
    )


def wrap_render_document_html(body_html: str, *, title: str = "Document preview") -> str:
    """Wrap rendered article HTML in a full document suitable for iframe preview."""
    inner = body_html.strip()
    if re.search(r"<!doctype|<html[\s>]", inner, flags=re.IGNORECASE):
        return inner
    safe_title = html.escape(title)
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        f"<title>{safe_title}</title>"
        f"<style>{_RENDER_STYLES}</style></head>"
        f"<body>{inner}</body></html>"
    )
