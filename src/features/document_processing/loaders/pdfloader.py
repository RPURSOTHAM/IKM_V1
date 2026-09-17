"""
Module: pdfloader.py
Extract blocks from a PDF file using pdf_reader_engine (default) or pdfplumber.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Union

from src.features.document_processing.loaders.component_classification import (
    COMPONENT_IMAGE,
    COMPONENT_TABLE,
    apply_component_metadata,
)
from src.features.document_processing.loaders.loader import Block, DocumentLoader
from src.features.document_processing.loaders.pdf_reader_loader import (
    load_pdf_blocks_with_reader,
    pdf_reader_engine_enabled,
)
from src.features.document_processing.loaders.table_heuristics import (
    looks_like_form_table,
    table_matrix_stats,
)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]


class PdfLoader(DocumentLoader):
    """Load a PDF document and return the blocks."""

    _BOLD_HINTS = ("bold", "bd", "-b", "heavy", "black", "demi", "semibold")
    _HEADER_BAND_RATIO = 0.08
    _FOOTER_BAND_RATIO = 0.92

    def load(self, path: Union[str, Path]) -> list[Block]:
        resolved = Path(path)
        if pdf_reader_engine_enabled():
            return load_pdf_blocks_with_reader(resolved)
        return self._load_with_pdfplumber(resolved)

    def _load_with_pdfplumber(self, path: Path) -> list[Block]:
        if pdfplumber is None:
            raise RuntimeError(
                "pdfplumber is not installed. Run: pip install pdfplumber"
            )

        blocks: list[Block] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                blocks.extend(self._page_to_blocks(page, page_num))

        font_sizes = [
            float(block.font_size)
            for block in blocks
            if (getattr(block, "block_type", "text") or "text") == "text" and float(block.font_size or 0) > 0
        ]
        median_font = statistics.median(font_sizes) if font_sizes else None
        for block in blocks:
            region = (getattr(block, "metadata", {}) or {}).get("region")
            apply_component_metadata(block, region=region, median_font_size=median_font)
        return sorted(blocks, key=lambda block: (block.page, block.line_number or 0))

    def _page_to_blocks(self, page, page_num: int) -> list[Block]:
        page_height = float(page.height or 0)
        page_width = float(page.width or 0)
        header_threshold = page_height * self._HEADER_BAND_RATIO if page_height else 0.0
        footer_threshold = page_height * self._FOOTER_BAND_RATIO if page_height else float("inf")

        table_regions: list[tuple[float, float, float, float]] = []
        table_payloads: list[tuple[list[list[object]], tuple[float, float, float, float] | None]] = []
        try:
            found_tables = page.find_tables() or []
        except Exception:
            found_tables = []

        for table_obj in found_tables:
            bbox = getattr(table_obj, "bbox", None)
            extracted = table_obj.extract() if hasattr(table_obj, "extract") else None
            if bbox and len(bbox) == 4:
                table_regions.append(tuple(float(v) for v in bbox))
            if extracted:
                table_payloads.append((extracted, tuple(float(v) for v in bbox) if bbox else None))

        if not table_payloads:
            for index, extracted in enumerate(page.extract_tables() or [], start=1):
                if extracted:
                    table_payloads.append((extracted, None))

        candidates: list[tuple[float, Block]] = []
        candidates.extend(
            self._text_line_candidates(
                page,
                page_num,
                page_width=page_width,
                header_threshold=header_threshold,
                footer_threshold=footer_threshold,
                table_regions=table_regions,
            )
        )

        for table_index, (table, bbox) in enumerate(table_payloads, start=1):
            block = self._table_to_block(
                table,
                page_num=page_num,
                table_index=table_index,
                bbox=bbox,
                header_threshold=header_threshold,
            )
            if block is not None:
                top = float((block.metadata or {}).get("source_top") or 0)
                candidates.append((top, block))

        for image_index, image in enumerate(getattr(page, "images", []) or [], start=1):
            block = self._image_to_block(
                image,
                page_num=page_num,
                image_index=image_index,
                page_height=page_height,
                header_threshold=header_threshold,
                footer_threshold=footer_threshold,
            )
            top = float((block.metadata or {}).get("source_top") or 0)
            candidates.append((top, block))

        candidates.sort(key=lambda item: (item[0], item[1].block_type != "text"))
        blocks: list[Block] = []
        line_number = 1
        for _, block in candidates:
            block.line_number = line_number
            span = 1
            if block.block_type == "table":
                metadata = dict(block.metadata or {})
                span = max(1, int(metadata.get("row_count") or 1))
                metadata["source_line_end"] = line_number + span - 1
                block.metadata = metadata
            blocks.append(block)
            line_number += span
        return blocks

    def _text_line_candidates(
        self,
        page,
        page_num: int,
        *,
        page_width: float,
        header_threshold: float,
        footer_threshold: float,
        table_regions: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, Block]]:
        chars = [ch for ch in (page.chars or []) if not self._char_in_table(ch, table_regions)]
        if not chars:
            return []

        left_margin = page_width * 0.12
        chars_sorted = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))

        lines: list[list[dict]] = []
        current_line: list[dict] = []
        last_top: float | None = None

        for ch in chars_sorted:
            top = round(ch["top"], 1)
            if last_top is None or abs(top - last_top) <= 3:
                current_line.append(ch)
                last_top = top
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [ch]
                last_top = top
        if current_line:
            lines.append(current_line)

        candidates: list[tuple[float, Block]] = []
        for line_chars in lines:
            text = self._join_line_chars(line_chars).strip()
            if not text:
                continue

            rep = next((c for c in line_chars if c["text"].strip()), line_chars[0])
            fontname: str = rep.get("fontname", "") or ""
            fontsize: float = float(rep.get("size", 0) or 0)
            bold: bool = any(h in fontname.lower() for h in self._BOLD_HINTS)

            x0 = min(c["x0"] for c in line_chars)
            x1 = max(c["x1"] for c in line_chars)
            mid = (x0 + x1) / 2
            page_mid = page_width / 2
            line_top = min(float(c["top"]) for c in line_chars)

            if abs(mid - page_mid) < page_width * 0.08:
                alignment = "center"
            elif x0 > page_width * 0.45:
                alignment = "right"
            else:
                alignment = "left"

            region = self._region_hint(text, line_top=line_top, header_threshold=header_threshold, footer_threshold=footer_threshold)
            metadata: dict = {"source_top": line_top}
            if region:
                metadata["region"] = region

            candidates.append(
                (
                    line_top,
                    Block(
                        text=text,
                        page=page_num,
                        line_number=0,
                        font_name=fontname,
                        font_size=fontsize,
                        bold=bold,
                        alignment=alignment,
                        indent_left=max(0.0, x0 - left_margin),
                        metadata=metadata,
                    ),
                )
            )
        return candidates

    @staticmethod
    def _join_line_chars(line_chars: list[dict]) -> str:
        if not line_chars:
            return ""
        ordered = sorted(line_chars, key=lambda c: c["x0"])
        parts: list[str] = []
        prev_x1: float | None = None
        for ch in ordered:
            text = str(ch.get("text") or "")
            if not text:
                continue
            x0 = float(ch.get("x0") or 0)
            if prev_x1 is not None and x0 - prev_x1 > float(ch.get("size") or 8) * 0.2:
                parts.append(" ")
            parts.append(text)
            prev_x1 = float(ch.get("x1") or x0)
        return "".join(parts)

    @staticmethod
    def _char_in_table(char: dict, table_regions: list[tuple[float, float, float, float]]) -> bool:
        if not table_regions:
            return False
        x = (float(char.get("x0", 0)) + float(char.get("x1", 0))) / 2
        y = (float(char.get("top", 0)) + float(char.get("bottom", 0))) / 2
        for x0, top, x1, bottom in table_regions:
            if x0 - 1 <= x <= x1 + 1 and top - 1 <= y <= bottom + 1:
                return True
        return False

    @classmethod
    def _region_hint(
        cls,
        text: str,
        *,
        line_top: float,
        header_threshold: float,
        footer_threshold: float,
    ) -> str | None:
        normalized = " ".join((text or "").lower().split())
        word_count = len(normalized.split())
        if line_top <= header_threshold:
            if word_count <= 10 or re.search(r"\b(confidential|draft|copy|page)\b", normalized):
                return "header"
            return None
        if line_top >= footer_threshold:
            if word_count <= 12 or re.search(r"page\s+\d+", normalized):
                return "footer"
            return None
        return None

    def _table_to_block(
        self,
        table: list[list[object]],
        *,
        page_num: int,
        table_index: int,
        bbox: tuple[float, float, float, float] | None,
        header_threshold: float,
    ) -> Block | None:
        table_text = self._format_table(table)
        if not table_text:
            return None

        row_count, column_count, empty_ratio = table_matrix_stats(table)
        if looks_like_form_table(
            table_text,
            row_count=row_count,
            column_count=column_count,
            empty_ratio=empty_ratio,
        ):
            return None

        source_top = float(bbox[1]) if bbox else float(table_index) * 1000.0
        region: str | None = None
        if bbox and source_top <= header_threshold:
            region = "header"

        return Block(
            text=f"Table {table_index}:\n{table_text}",
            page=page_num,
            line_number=0,
            block_type="table",
            component_type=COMPONENT_TABLE,
            metadata={
                "table_index": table_index,
                "row_count": row_count,
                "column_count": column_count,
                "source_line_end": 0,
                "component_type": COMPONENT_TABLE,
                "region": region,
                "source_top": source_top,
            },
        )

    @staticmethod
    def _image_to_block(
        image: dict,
        *,
        page_num: int,
        image_index: int,
        page_height: float,
        header_threshold: float,
        footer_threshold: float,
    ) -> Block:
        x0 = image.get("x0")
        top = image.get("top")
        x1 = image.get("x1")
        bottom = image.get("bottom")
        width = image.get("width")
        height = image.get("height")
        source_top = float(top or 0)
        region: str | None = None
        if top is not None and page_height:
            if float(top) <= header_threshold:
                region = "header"
            elif float(top) >= footer_threshold:
                region = "footer"
        return Block(
            text="",
            page=page_num,
            line_number=0,
            block_type="image",
            component_type=COMPONENT_IMAGE,
            metadata={
                "image_index": image_index,
                "x0": x0,
                "top": top,
                "x1": x1,
                "bottom": bottom,
                "width": width,
                "height": height,
                "component_type": COMPONENT_IMAGE,
                "region": region,
                "source_top": source_top,
            },
        )

    @staticmethod
    def _format_table(table: list[list[object]]) -> str:
        rows: list[str] = []
        for row in table or []:
            cells = [str(cell or "").strip() for cell in (row or []) if str(cell or "").strip()]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows).strip()
