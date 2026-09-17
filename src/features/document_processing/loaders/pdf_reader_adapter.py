"""Adapt vendored pdf_reader_engine output into pipeline ``Block`` objects."""

from __future__ import annotations

import statistics
from typing import Any

from src.features.document_processing.loaders.component_classification import (
    COMPONENT_IMAGE,
    COMPONENT_TABLE,
    apply_component_metadata,
)
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdf_reader_engine.document import (
    ContentKind,
    DocumentContent,
    DocumentItem,
    ImageRef,
    Table,
    TextBlock,
)
from src.features.document_processing.loaders.table_heuristics import (
    looks_like_form_table,
    table_matrix_stats,
)

_HEADER_BAND_RATIO = 0.08
_FOOTER_BAND_RATIO = 0.92


def adapt_pdf_reader_content_to_blocks(content: DocumentContent) -> list[Block]:
    page_heights = _page_heights(content)
    blocks: list[Block] = []
    image_counters: dict[int, int] = {}
    table_counters: dict[int, int] = {}

    for item in sorted(
        content.items,
        key=lambda entry: (int(entry.page or 1), float((_item_bbox(entry) or (0, 0, 0, 0))[1]), int(entry.line_number or 0)),
    ):
        page = int(item.page or 1)
        if item.kind == ContentKind.TEXT:
            block = _text_item_to_block(item, page_heights.get(page))
            if block is not None:
                blocks.append(block)
        elif item.kind == ContentKind.TABLE:
            table_counters[page] = table_counters.get(page, 0) + 1
            block = _table_item_to_block(
                item,
                table_index=table_counters[page],
                page_height=page_heights.get(page),
            )
            if block is not None:
                blocks.append(block)
        elif item.kind == ContentKind.IMAGE:
            image_counters[page] = image_counters.get(page, 0) + 1
            blocks.append(
                _image_item_to_block(
                    item,
                    image_index=image_counters[page],
                    page_height=page_heights.get(page),
                )
            )

    blocks = _renumber_blocks(blocks)
    font_sizes = [
        float(block.font_size)
        for block in blocks
        if (getattr(block, "block_type", "text") or "text") == "text" and float(block.font_size or 0) > 0
    ]
    median_font = statistics.median(font_sizes) if font_sizes else None
    for block in blocks:
        region = (getattr(block, "metadata", {}) or {}).get("region")
        apply_component_metadata(block, region=region, median_font_size=median_font)

    return blocks


def _renumber_blocks(blocks: list[Block]) -> list[Block]:
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        by_page.setdefault(int(block.page or 1), []).append(block)

    ordered: list[Block] = []
    for page in sorted(by_page):
        page_blocks = sorted(
            by_page[page],
            key=lambda block: (
                float((getattr(block, "metadata", {}) or {}).get("source_top") or 0),
                block.block_type != "text",
            ),
        )
        line_number = 1
        for block in page_blocks:
            block.line_number = line_number
            if block.block_type == "table":
                metadata = dict(getattr(block, "metadata", {}) or {})
                span = max(1, int(metadata.get("row_count") or 1))
                metadata["source_line_end"] = line_number + span - 1
                block.metadata = metadata
                line_number += span
            else:
                line_number += 1
            ordered.append(block)
    return ordered


def adapt_pdf_reader_output_to_blocks(document: DocumentContent | dict[str, Any]) -> list[Block]:
    """Backward-compatible entry for dict payloads used in older tests."""
    if isinstance(document, DocumentContent):
        return adapt_pdf_reader_content_to_blocks(document)
    return _adapt_legacy_dict(document)


def _page_heights(content: DocumentContent) -> dict[int, float]:
    heights: dict[int, float] = {}
    for item in content.items:
        bbox = _item_bbox(item)
        if bbox is None:
            continue
        page = int(item.page or 1)
        heights[page] = max(heights.get(page, 0.0), float(bbox[3]))
    return heights


def _item_bbox(item: DocumentItem) -> tuple[float, float, float, float] | None:
    if item.bbox is not None:
        return (item.bbox.x0, item.bbox.y0, item.bbox.x1, item.bbox.y1)
    content = item.content
    if isinstance(content, TextBlock) and content.bbox is not None:
        return (content.bbox.x0, content.bbox.y0, content.bbox.x1, content.bbox.y1)
    if isinstance(content, Table) and content.bbox is not None:
        return (content.bbox.x0, content.bbox.y0, content.bbox.x1, content.bbox.y1)
    if isinstance(content, ImageRef):
        return (content.bbox.x0, content.bbox.y0, content.bbox.x1, content.bbox.y1)
    return None


def _region_hint(
    text: str,
    *,
    line_top: float,
    page_height: float | None,
) -> str | None:
    if not page_height:
        return None
    header_threshold = page_height * _HEADER_BAND_RATIO
    footer_threshold = page_height * _FOOTER_BAND_RATIO
    normalized = " ".join((text or "").lower().split())
    word_count = len(normalized.split())
    if line_top <= header_threshold:
        if word_count <= 10:
            return "header"
        return None
    if line_top >= footer_threshold:
        if word_count <= 12:
            return "footer"
        return None
    return None


def _text_item_to_block(item: DocumentItem, page_height: float | None) -> Block | None:
    content = item.content
    if not isinstance(content, TextBlock):
        return None
    text = str(content.text or "").strip()
    if not text:
        return None
    bbox = _item_bbox(item)
    line_top = float(bbox[1]) if bbox else 0.0
    region = _region_hint(text, line_top=line_top, page_height=page_height)
    layout_role = str((item.meta or {}).get("layout_role") or "").strip().lower()
    if layout_role in {"page_header", "header"}:
        region = "header"
    elif layout_role in {"page_footer", "footer"}:
        region = "footer"

    bold = any(span.bold for line in content.lines for span in line.spans)
    italic = any(span.italic for line in content.lines for span in line.spans)
    font_name = str((item.meta or {}).get("font_name") or "")
    font_size = float((item.meta or {}).get("font_size") or 0.0)
    if font_size <= 0 and content.lines and content.lines[0].bbox is not None:
        font_size = max(1.0, float(content.lines[0].bbox.height))

    metadata: dict[str, Any] = {"source_top": line_top}
    if region:
        metadata["region"] = region
    if bbox is not None:
        metadata.update(
            {
                "x0": bbox[0],
                "top": bbox[1],
                "x1": bbox[2],
                "bottom": bbox[3],
                "bbox": list(bbox),
            }
        )

    return Block(
        text=text,
        page=int(item.page or 1),
        line_number=int(item.line_number or 0),
        font_name=font_name,
        font_size=font_size,
        bold=bold,
        italic=italic,
        metadata=metadata,
    )


def _table_item_to_block(
    item: DocumentItem,
    *,
    table_index: int,
    page_height: float | None,
) -> Block | None:
    content = item.content
    if not isinstance(content, Table):
        return None
    matrix = _table_matrix(content)
    table_text = _format_table(matrix)
    if not table_text:
        return None
    row_count, column_count, empty_ratio = table_matrix_stats(matrix)
    if looks_like_form_table(
        table_text,
        row_count=row_count,
        column_count=column_count,
        empty_ratio=empty_ratio,
    ):
        return None

    bbox = _item_bbox(item)
    source_top = float(bbox[1]) if bbox else float(table_index) * 1000.0
    region = None
    if bbox and page_height and source_top <= page_height * _HEADER_BAND_RATIO:
        region = "header"
    metadata: dict[str, Any] = {
        "table_index": table_index,
        "row_count": row_count,
        "column_count": column_count,
        "source_line_end": 0,
        "component_type": COMPONENT_TABLE,
        "region": region,
        "source_top": source_top,
        "rows": matrix,
    }
    if bbox is not None:
        metadata.update(
            {
                "x0": bbox[0],
                "top": bbox[1],
                "x1": bbox[2],
                "bottom": bbox[3],
                "bbox": list(bbox),
            }
        )
    return Block(
        text=f"Table {table_index}:\n{table_text}",
        page=int(item.page or 1),
        line_number=int(item.line_number or 0),
        block_type="table",
        component_type=COMPONENT_TABLE,
        metadata=metadata,
    )


def _image_item_to_block(
    item: DocumentItem,
    *,
    image_index: int,
    page_height: float | None,
) -> Block:
    content = item.content
    if not isinstance(content, ImageRef):
        raise TypeError(f"Expected ImageRef, got {type(content)!r}")
    bbox = _item_bbox(item) or (
        content.bbox.x0,
        content.bbox.y0,
        content.bbox.x1,
        content.bbox.y1,
    )
    labels = [str(label).strip() for label in (content.labels or []) if str(label).strip()]
    text = "\n".join(labels).strip()
    source_top = float(bbox[1])
    region = None
    if page_height:
        if source_top <= page_height * _HEADER_BAND_RATIO:
            region = "header"
        elif source_top >= page_height * _FOOTER_BAND_RATIO:
            region = "footer"
    resolved_index = int((item.meta or {}).get("image_index") or image_index)
    metadata: dict[str, Any] = {
        "image_index": resolved_index,
        "x0": bbox[0],
        "top": bbox[1],
        "x1": bbox[2],
        "bottom": bbox[3],
        "width": content.width,
        "height": content.height,
        "component_type": COMPONENT_IMAGE,
        "region": region,
        "source_top": source_top,
        "image_discovery_source": "pdf_reader_engine",
    }
    if labels:
        metadata["image_caption"] = text
        metadata["ocr_text"] = text
    return Block(
        text=text,
        page=int(item.page or 1),
        line_number=int(item.line_number or 0),
        block_type="image",
        component_type=COMPONENT_IMAGE,
        metadata=metadata,
    )


def _table_matrix(table: Table) -> list[list[str]]:
    rows = max(int(table.row_count or 0), 0)
    cols = max(int(table.column_count or 0), 0)
    if rows <= 0 or cols <= 0:
        return []
    matrix = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in table.cells:
        row = int(cell.row)
        col = int(cell.column)
        if 0 <= row < rows and 0 <= col < cols:
            matrix[row][col] = str(cell.text or "").strip()
    return matrix


def _format_table(table: list[list[str]]) -> str:
    rows: list[str] = []
    for row in table or []:
        cells = [str(cell or "").strip() for cell in row if str(cell or "").strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()


def _adapt_legacy_dict(document: dict[str, Any]) -> list[Block]:
    """Support the historical ``pages/items`` JSON shape in unit tests."""
    items: list[DocumentItem] = []
    line_number = 0
    for page in document.get("pages") or []:
        page_number = int(page.get("page_number") or 1)
        for raw in page.get("items") or []:
            line_number += 1
            item_type = str(raw.get("type") or "text").lower()
            bbox_raw = raw.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            from src.features.document_processing.loaders.pdf_reader_engine.document import BBox

            bbox = BBox(
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
            attrs = dict(raw.get("attributes") or {})
            if item_type == "text":
                from src.features.document_processing.loaders.pdf_reader_engine.document import (
                    TextLine,
                    TextSpan,
                )

                spans = []
                for span in attrs.get("spans") or []:
                    spans.append(
                        TextSpan(
                            text=str(span.get("text") or raw.get("text") or ""),
                            bold=bool(span.get("bold")),
                            italic=bool(span.get("italic")),
                        )
                    )
                if not spans:
                    spans = [TextSpan(text=str(raw.get("text") or ""), bold=bool(attrs.get("bold")))]
                block = TextBlock(
                    order=line_number,
                    lines=[
                        TextLine(
                            line_number=1,
                            text=str(raw.get("text") or ""),
                            indent=0.0,
                            indent_level=0,
                            is_bullet=False,
                            bullet=None,
                            bbox=bbox,
                            spans=spans,
                        )
                    ],
                    bbox=bbox,
                )
                meta = {}
                layout_role = str(attrs.get("layout_role") or "").strip().lower()
                if layout_role:
                    meta["layout_role"] = layout_role
                if attrs.get("font_size") is not None:
                    meta["font_size"] = float(attrs.get("font_size") or 0)
                for span in attrs.get("spans") or []:
                    font_name = str(span.get("font_name") or "").strip()
                    if font_name:
                        meta["font_name"] = font_name
                    if span.get("font_size") is not None and "font_size" not in meta:
                        meta["font_size"] = float(span.get("font_size") or 0)
                    if font_name:
                        break
                items.append(
                    DocumentItem(
                        page=page_number,
                        line_number=line_number,
                        kind=ContentKind.TEXT,
                        content=block,
                        bbox=bbox,
                        meta=meta,
                    )
                )
            elif item_type == "table":
                rows = raw.get("rows") or []
                row_count = int(attrs.get("row_count") or len(rows) or 0)
                column_count = int(attrs.get("column_count") or (len(rows[0]) if rows else 0))
                cells = []
                for row_index, row in enumerate(rows):
                    for col_index, value in enumerate(row or []):
                        from src.features.document_processing.loaders.pdf_reader_engine.document import TableCell

                        cells.append(
                            TableCell(
                                row=row_index,
                                column=col_index,
                                rowspan=1,
                                colspan=1,
                                align="left",
                                valign="top",
                                shading=None,
                                text=str(value or ""),
                            )
                        )
                table = Table(
                    columns=[str(value or "") for value in (rows[0] if rows else [])],
                    column_count=column_count,
                    row_count=row_count,
                    cells=cells,
                    bbox=bbox,
                )
                items.append(
                    DocumentItem(
                        page=page_number,
                        line_number=line_number,
                        kind=ContentKind.TABLE,
                        content=table,
                        bbox=bbox,
                    )
                )
            elif item_type == "image":
                image = ImageRef(
                    path="",
                    xref=0,
                    width=int(attrs.get("width") or bbox.width),
                    height=int(attrs.get("height") or bbox.height),
                    bbox=bbox,
                    ext="png",
                )
                image_meta = {}
                if attrs.get("image_index") is not None:
                    image_meta["image_index"] = int(attrs.get("image_index"))
                items.append(
                    DocumentItem(
                        page=page_number,
                        line_number=line_number,
                        kind=ContentKind.IMAGE,
                        content=image,
                        bbox=bbox,
                        meta=image_meta,
                    )
                )
    return adapt_pdf_reader_content_to_blocks(
        DocumentContent(
            name=str(document.get("document_name") or document.get("source") or "document.pdf"),
            page_count=len(document.get("pages") or []) or 1,
            items=items,
        )
    )
