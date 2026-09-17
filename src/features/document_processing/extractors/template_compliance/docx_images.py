"""DOCX extract helper module (split from docx_extract)."""

from __future__ import annotations

import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from .docx_common import (
    A_NS,
    R_NS,
    V_NS,
    W_NS,
    _EMU_PER_INCH,
    _rels_map,
    _resolve_media_part,
)

def _image_refs(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for img in images or []:
        part = str(img.get("part_name") or "").strip()
        rid = str(img.get("id") or "").strip()
        key = part or rid
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "id": rid,
                "part_name": part,
                "content_type": str(img.get("content_type") or ""),
            }
        )
    return out


def _count_body_images_and_tables(path: Path) -> dict[str, int]:
    """
    Count unique images and tables in word/document.xml (body only).

    Images = unique relationship targets referenced by a:blip / v:imagedata.
    Tables = every w:tbl element in the body document part.
    Header/footer media are ignored (separate package parts).
    """
    images = 0
    tables = 0
    try:
        with zipfile.ZipFile(path) as archive:
            if "word/document.xml" not in archive.namelist():
                return {"images": 0, "tables": 0}
            root = ET.fromstring(archive.read("word/document.xml"))
            tables = sum(1 for _ in root.iter(f"{{{W_NS}}}tbl"))

            rid_to_target = _rels_map(archive, "word/_rels/document.xml.rels")
            embeds: set[str] = set()
            for blip in root.iter(f"{{{A_NS}}}blip"):
                rid = blip.get(qn("r:embed")) or blip.get(f"{{{R_NS}}}embed")
                if not rid:
                    for key, value in blip.attrib.items():
                        if str(key).endswith("embed"):
                            rid = value
                            break
                if rid:
                    embeds.add(str(rid))
            for imagedata in root.iter(f"{{{V_NS}}}imagedata"):
                rid = imagedata.get(qn("r:id")) or imagedata.get(f"{{{R_NS}}}id")
                if not rid:
                    for key, value in imagedata.attrib.items():
                        if str(key).endswith("id") and value:
                            rid = value
                            break
                if rid:
                    embeds.add(str(rid))

            targets: set[str] = set()
            for rid in embeds:
                target = str(rid_to_target.get(rid) or "").replace("\\", "/")
                if not target:
                    continue
                targets.add(_resolve_media_part(target))
            images = len(targets) if targets else len(embeds)
    except Exception:
        return {"images": 0, "tables": 0}
    return {"images": images, "tables": tables}


def _extract_body_images(doc: Document) -> list[dict[str, Any]]:
    images = _images_from_paragraphs(doc.paragraphs, location="body", section_index=None)
    # Non-table body paragraphs only; table images collected with tables.
    return images


def _images_from_paragraphs(
    paragraphs,
    *,
    location: str,
    section_index: int | None,
    table_index: int | None = None,
    row_index: int | None = None,
    col_index: int | None = None,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        try:
            element = paragraph._element
        except Exception:
            continue
        for blip in element.iter(f"{{{A_NS}}}blip"):
            embed = blip.get(qn("r:embed")) or blip.get(f"{{{R_NS}}}embed")
            if not embed:
                continue
            width_emu, height_emu = _nearby_extent(blip)
            images.append(
                {
                    "id": embed,
                    "part_name": "",
                    "content_type": "",
                    "width_emu": width_emu,
                    "height_emu": height_emu,
                    "width_inches": round(width_emu / _EMU_PER_INCH, 3) if width_emu else None,
                    "height_inches": round(height_emu / _EMU_PER_INCH, 3) if height_emu else None,
                    "location": location,
                    "section_index": section_index,
                    "in_table": table_index is not None,
                    "table_index": table_index,
                    "row": row_index,
                    "column": col_index,
                }
            )
        for imagedata in element.iter(f"{{{V_NS}}}imagedata"):
            rid = imagedata.get(qn("r:id")) or imagedata.get(f"{{{R_NS}}}id")
            if not rid:
                continue
            images.append(
                {
                    "id": rid,
                    "part_name": "",
                    "content_type": "",
                    "width_emu": None,
                    "height_emu": None,
                    "width_inches": None,
                    "height_inches": None,
                    "location": location,
                    "section_index": section_index,
                    "in_table": table_index is not None,
                    "table_index": table_index,
                    "row": row_index,
                    "column": col_index,
                }
            )
    return images


def _nearby_extent(blip) -> tuple[int | None, int | None]:
    node = blip
    for _ in range(8):
        parent = node.getparent() if hasattr(node, "getparent") else None
        if parent is None:
            break
        for child in parent.iter():
            if child.tag in {qn("wp:extent"), qn("a:ext")}:
                cx = child.get("cx")
                cy = child.get("cy")
                try:
                    return (int(cx) if cx else None, int(cy) if cy else None)
                except Exception:
                    return None, None
        node = parent
    return None, None


def _dedupe_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for image in images:
        key = "|".join(
            [
                str(image.get("id") or ""),
                str(image.get("part_name") or ""),
                str(image.get("location") or ""),
                str(image.get("table_index")),
                str(image.get("row")),
                str(image.get("column")),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(image)
    return out
