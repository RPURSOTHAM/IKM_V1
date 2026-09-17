"""
Module: docxloader.py
Extract blocks from a DOCX file using python-docx.
"""

# ✅ FIX 3: Guard the import with try/except so that DocxDocument is None
# when python-docx is missing. The original code imported DocxDocument
# directly (which raises ImportError immediately on missing package) but
# then guarded with `if DocxDocument is None` — a check that could NEVER
# be True given the unconditional import above it. The guard was dead code.
try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError:
    DocxDocument = None  # type: ignore[assignment]
    qn = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Paragraph = None  # type: ignore[assignment]

import re
import tempfile
from pathlib import Path
from typing import Union

# ✅ FIX 4: Correct module name. The base class lives in loader.py,
# not docloader.py. The original import would raise ModuleNotFoundError
# at startup.
from src.features.document_processing.loaders.loader import DocumentLoader, Block
from src.features.document_processing.loaders.pdfloader import PdfLoader
from src.features.document_processing.loaders.component_classification import (
    COMPONENT_FOOTER,
    COMPONENT_HEADER,
    COMPONENT_IMAGE,
    COMPONENT_TABLE,
    apply_component_metadata,
)
from src.features.document_processing.loaders.table_heuristics import (
    looks_like_form_table,
    looks_like_specification_table,
)
from src.features.document_processing.loaders.ooxml_table_parser import parse_and_clean_ooxml_table
from src.features.documents.infrastructure.content.convert_for_rendering import docx_to_pdf


class DocxLoader(DocumentLoader):
    """Extract blocks from a DOCX using python-docx."""

    # Split very large spec tables so each block vectorizes with header context.
    _TABLE_SPLIT_MIN_ROWS = 12
    _TABLE_ROWS_PER_BLOCK = 8

    _ALIGN_MAP = {
        "CENTER": "center",
        "RIGHT": "right",
        "JUSTIFY": "justified",
        None: "left",
    }

    def load(self, path: Union[str, Path], *, citation_retainment: bool = True) -> list[Block]:
        if DocxDocument is None:
            raise RuntimeError(
                "python-docx is not installed. Run: pip install python-docx"
            )

        source_path = Path(path)
        layout_attempted = False
        if citation_retainment:
            layout_attempted = True
            layout_blocks = self._load_layout_pdf_blocks(source_path)
            if layout_blocks:
                return layout_blocks

        doc = DocxDocument(str(source_path))
        blocks: list[Block] = []
        page_num = 1
        line_number = 0

        header_blocks = self._extract_header_footer_blocks(doc, region="header", page=1, start_line=0)
        line_number = max((block.line_number for block in header_blocks), default=0)

        for element in doc.element.body:
            if element.tag == qn("w:p"):
                para = Paragraph(element, doc)
                has_page_break = any(
                    br.get(qn("w:type"), "") == "page"
                    for run in para.runs
                    for br in run._r.findall(qn("w:br"))
                )
                block = self._paragraph_to_block(para, page_num, 0)
                if block is not None:
                    line_number += 1
                    block.line_number = line_number
                    blocks.append(block)

                if has_page_break:
                    page_num += 1
                    line_number = 0
            elif element.tag == qn("w:tbl"):
                table = Table(element, doc)
                table_blocks, line_number = self._emit_table_blocks(
                    table,
                    page=page_num,
                    line_number=line_number,
                )
                blocks.extend(table_blocks)

        for image_index, shape in enumerate(getattr(doc, "inline_shapes", []) or [], start=1):
            line_number += 1
            width = getattr(getattr(shape, "width", None), "pt", None)
            height = getattr(getattr(shape, "height", None), "pt", None)
            blocks.append(
                Block(
                    text="",
                    page=page_num,
                    line_number=line_number,
                    block_type="image",
                    component_type=COMPONENT_IMAGE,
                    metadata={
                        "image_index": image_index,
                        "width_pt": width,
                        "height_pt": height,
                        "component_type": COMPONENT_IMAGE,
                        "source_top": float(line_number),
                    },
                )
            )

        max_page = max((block.page for block in blocks), default=page_num)
        footer_start = max(
            (block.line_number for block in blocks if block.page == max_page),
            default=0,
        )
        footer_blocks = self._extract_header_footer_blocks(
            doc,
            region="footer",
            page=max_page,
            start_line=footer_start,
        )

        all_blocks = header_blocks + blocks + footer_blocks
        for block in all_blocks:
            if block.component_type in {COMPONENT_HEADER, COMPONENT_FOOTER, COMPONENT_TABLE, COMPONENT_IMAGE}:
                continue
            apply_component_metadata(block)
        normalized = self._normalize_line_numbers_by_page(all_blocks)
        if layout_attempted:
            for block in normalized:
                metadata = dict(block.metadata or {})
                metadata.setdefault("source_format", "docx")
                metadata["layout_source"] = "python_docx_fallback"
                metadata["citation_retainment_degraded"] = True
                block.metadata = metadata
        return normalized

    @staticmethod
    def _load_layout_pdf_blocks(path: Path) -> list[Block]:
        """Use LibreOffice layout output for real DOCX page and line references."""
        with tempfile.TemporaryDirectory(prefix="docx-layout-") as temp_dir:
            pdf_path = docx_to_pdf(path, output_dir=Path(temp_dir))
            if pdf_path is None:
                return []
            blocks = PdfLoader().load(pdf_path)
            for block in blocks:
                metadata = dict(block.metadata or {})
                metadata["source_format"] = "docx"
                metadata["layout_source"] = "libreoffice_pdf"
                block.metadata = metadata
            return blocks

    def _extract_header_footer_blocks(
        self,
        doc,
        *,
        region: str,
        page: int,
        start_line: int,
    ) -> list[Block]:
        """Extract Word header/footer parts as dedicated blocks."""
        component = COMPONENT_HEADER if region == "header" else COMPONENT_FOOTER
        blocks: list[Block] = []
        line_number = start_line
        seen_text: set[str] = set()

        for section_index, section in enumerate(doc.sections):
            part = section.header if region == "header" else section.footer
            if part.is_linked_to_previous and section_index > 0:
                continue

            for para in part.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                signature = " ".join(text.lower().split())
                if signature in seen_text:
                    continue
                seen_text.add(signature)
                line_number += 1
                style_name = para.style.name if para.style else ""
                blocks.append(
                    Block(
                        text=text,
                        page=page,
                        line_number=line_number,
                        style=style_name,
                        block_type="text",
                        component_type=component,
                        metadata={"component_type": component, "region": region},
                    )
                )

            for table in part.tables:
                table_blocks, line_number = self._emit_table_blocks(
                    table,
                    page=page,
                    line_number=line_number,
                    region=region,
                    seen_text=seen_text,
                )
                blocks.extend(table_blocks)

        return blocks

    def _paragraph_to_block(self, para, page_num: int, _line_number: int) -> Block | None:
        text = para.text.strip()
        if not text:
            return None

        style_name: str = para.style.name if para.style else ""

        # ✅ FIX 5: Collect font properties from the FIRST run that has
        # a font size, rather than accumulating across multiple runs.
        # The original loop set bold/italic from early runs, then
        # overwrote them from later runs (whichever run finally had a
        # font_size), mixing properties from different runs unpredictably.
        # Example bug: run-1 bold=True/size=None, run-2 bold=False/size=12
        # → original result bold=False (wrong); fixed result bold=True.
        font_size = 0.0
        font_name = ""
        bold = False
        italic = False

        primary_run = None
        for run in para.runs:
            if run.font.size:
                primary_run = run
                break
        # Fall back to the first run if none carries an explicit size.
        if primary_run is None and para.runs:
            primary_run = para.runs[0]

        if primary_run is not None:
            font_size = primary_run.font.size.pt if primary_run.font.size else 0.0
            font_name = primary_run.font.name or ""
            bold = bool(primary_run.bold)
            italic = bool(primary_run.italic)

        # Fallback: read font size from paragraph style.
        if font_size == 0.0 and para.style and para.style.font.size:
            font_size = para.style.font.size.pt

        # Paragraph-level bold override via pPr/rPr/w:b.
        ppr = para._p.find(qn("w:pPr"))
        if ppr is not None:
            rpr = ppr.find(qn("w:rPr"))
            if rpr is not None and rpr.find(qn("w:b")) is not None:
                bold = True

        # Left indent.
        indent_left = 0.0
        if para.paragraph_format.left_indent:
            try:
                indent_left = para.paragraph_format.left_indent.pt
            except Exception:
                pass

        # Alignment.
        align_raw = (
            str(para.alignment).split(".")[-1] if para.alignment else None
        )
        alignment = self._ALIGN_MAP.get(align_raw, "left")

        # TOC entry detection via style name.
        is_toc = "toc" in style_name.lower()

        return Block(
            text=text,
            page=page_num,
            line_number=0,
            style=style_name,
            font_name=font_name,
            font_size=font_size,
            bold=bold,
            italic=italic,
            alignment=alignment,
            indent_left=indent_left,
            is_toc_entry=is_toc,
        )

    @staticmethod
    def _cell_text(cell) -> str:
        """Extract paragraph and nested-table text from a table cell."""
        parts: list[str] = []
        for para in cell.paragraphs:
            text = " ".join(para.text.split())
            if text:
                parts.append(text)
        for nested in getattr(cell, "tables", []) or []:
            nested_text, _, _, _ = DocxLoader._table_to_text(nested)
            if nested_text:
                parts.append(f"[Nested table: {nested_text}]")
        return " ".join(parts).strip()

    @classmethod
    def _table_row_matrix(cls, table) -> tuple[list[list[str]], int, int, float]:
        """Return table rows as a matrix with merged-cell duplicates collapsed."""
        matrix: list[list[str]] = []
        max_columns = 0
        total_cells = 0
        filled_cells = 0
        physical_rows = len(table.rows)
        for row in table.rows:
            raw_cells = [cls._cell_text(cell) for cell in row.cells]
            collapsed: list[str] = []
            for cell_text in raw_cells:
                total_cells += 1
                if cell_text:
                    filled_cells += 1
                if collapsed and cell_text and cell_text == collapsed[-1]:
                    continue
                collapsed.append(cell_text)
            max_columns = max(max_columns, len(collapsed))
            matrix.append(collapsed)
        empty_ratio = 1 - (filled_cells / total_cells) if total_cells else 1.0
        return matrix, physical_rows, max_columns, empty_ratio

    @staticmethod
    def _row_looks_like_header(cells: list[str]) -> bool:
        joined = " ".join(cells).lower()
        if not joined.strip():
            return False
        header_signals = (
            "sl. no",
            "sl no",
            "s. no",
            "parameter",
            "requirement",
            "specification",
            "description",
            "department",
            "designation",
        )
        return sum(1 for signal in header_signals if signal in joined) >= 2

    @classmethod
    def _matrix_to_text(cls, matrix: list[list[str]], *, labeled: bool = False) -> str:
        if not matrix:
            return ""
        header = matrix[0]
        has_header = cls._row_looks_like_header(header) and len(matrix) > 1
        use_labels = labeled and has_header
        data_rows = matrix[1:] if has_header else matrix
        lines: list[str] = []
        for row in data_rows:
            if use_labels and len(row) == len(header):
                row_text = " | ".join(value.strip() for value in row if value.strip()).strip()
                if row_text:
                    lines.append(row_text)
                continue
            row_text = " | ".join(cell for cell in row if cell).strip()
            if row_text:
                lines.append(row_text)
        return "\n".join(lines).strip()

    @classmethod
    def _split_matrix_into_groups(cls, matrix: list[list[str]], max_data_rows: int) -> list[list[list[str]]]:
        if len(matrix) <= max_data_rows + 1:
            return [matrix]
        header = matrix[0]
        data_rows = matrix[1:]
        if not cls._row_looks_like_header(header):
            groups: list[list[list[str]]] = []
            for start in range(0, len(matrix), max_data_rows):
                groups.append(matrix[start : start + max_data_rows])
            return groups
        groups = []
        for start in range(0, len(data_rows), max_data_rows):
            groups.append(data_rows[start : start + max_data_rows])
        return groups

    @classmethod
    def _emit_table_blocks(
        cls,
        table,
        *,
        page: int,
        line_number: int,
        region: str | None = None,
        seen_text: set[str] | None = None,
    ) -> tuple[list[Block], int]:
        ooxml_meta: dict = {}
        tbl_element = getattr(table, "_element", None)
        if tbl_element is not None:
            ooxml_res = parse_and_clean_ooxml_table(tbl_element)
            if ooxml_res.get("matrix"):
                matrix = ooxml_res["matrix"]
                row_count = ooxml_res["row_count"]
                column_count = ooxml_res["column_count"]
                empty_ratio = ooxml_res["empty_ratio"]
                ooxml_meta = {
                    "quality_score": ooxml_res.get("quality_score", 0.0),
                    "table_type": ooxml_res.get("table_type", "GRID"),
                    "decision": ooxml_res.get("decision", "ACCEPT"),
                    "grid_spans": ooxml_res.get("grid_spans", []),
                    "v_merges": ooxml_res.get("v_merges", []),
                    "has_page_breaks": ooxml_res.get("has_page_breaks", False),
                    "has_images": ooxml_res.get("has_images", False),
                    "ooxml_extracted": True,
                    "ooxml_fallback": False,
                }
            else:
                matrix, row_count, column_count, empty_ratio = cls._table_row_matrix(table)
                ooxml_meta = {"ooxml_extracted": False, "ooxml_fallback": True}
        else:
            matrix, row_count, column_count, empty_ratio = cls._table_row_matrix(table)
            ooxml_meta = {"ooxml_extracted": False, "ooxml_fallback": True}

        table_text = cls._matrix_to_text(matrix)
        if not table_text:
            return [], line_number + max(1, row_count)

        if looks_like_form_table(
            table_text,
            row_count=row_count,
            column_count=column_count,
            empty_ratio=empty_ratio,
        ):
            return [], line_number + max(1, row_count)

        signature = " ".join(table_text.lower().split())
        if seen_text is not None:
            if signature in seen_text:
                return [], line_number + max(1, row_count)
            seen_text.add(signature)

        labeled = looks_like_specification_table(table_text, column_count)
        groups = (
            cls._split_matrix_into_groups(matrix, cls._TABLE_ROWS_PER_BLOCK)
            if row_count >= cls._TABLE_SPLIT_MIN_ROWS
            else [matrix]
        )

        blocks: list[Block] = []
        for group_index, group in enumerate(groups):
            group_text = cls._matrix_to_text(group, labeled=labeled)
            if not group_text:
                continue
            line_number += 1
            group_row_count = len(group)
            source_line_end = line_number + max(group_row_count - 1, 0)
            metadata = {
                "row_count": group_row_count,
                "column_count": column_count,
                "source_line_end": source_line_end,
                "component_type": COMPONENT_TABLE,
                "table_part": group_index + 1,
                "table_parts": len(groups),
                "labeled_rows": labeled,
                **ooxml_meta,
            }
            if region:
                metadata["region"] = region
            blocks.append(
                Block(
                    text=f"Table:\n{group_text}",
                    page=page,
                    line_number=line_number,
                    block_type="table",
                    component_type=COMPONENT_TABLE,
                    metadata=metadata,
                )
            )
            line_number = source_line_end
        return blocks, line_number

    @staticmethod
    def _table_to_text(table) -> tuple[str, int, int, float]:
        matrix, physical_rows, max_columns, empty_ratio = DocxLoader._table_row_matrix(table)
        return DocxLoader._matrix_to_text(matrix), physical_rows, max_columns, empty_ratio

    @staticmethod
    def _normalize_line_numbers_by_page(blocks: list[Block]) -> list[Block]:
        """Ensure extracted line numbers restart from 1 on every page."""
        ordered = sorted(blocks, key=lambda block: (block.page, block.line_number or 0))
        next_line_by_page: dict[int, int] = {}
        for block in ordered:
            page = int(block.page or 1)
            next_line = next_line_by_page.get(page, 1)
            original_start = int(block.line_number or next_line)
            original_end = int((block.metadata or {}).get("source_line_end") or original_start)
            span = max(1, original_end - original_start + 1)
            block.page = page
            block.line_number = next_line
            metadata = dict(block.metadata or {})
            metadata["source_line_end"] = next_line + span - 1
            block.metadata = metadata
            next_line_by_page[page] = next_line + span
        return ordered
