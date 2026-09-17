"""Document content model.

Stores page content in original reading order. Each item is tagged with a
page number, a line number, and a kind of text, table, or image.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Union


class ContentKind(str, Enum):
    """Kind of extracted page item: body text, table, or image."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned page box: ``(x0, y0)`` top-left, ``(x1, y1)`` bottom-right."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Horizontal span in PDF points."""
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        """Vertical span in PDF points."""
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        """Width times height, or ``0`` when the box is empty."""
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        """True when width or height is not positive."""
        return self.width <= 0 or self.height <= 0

    def intersection(self, other: "BBox") -> "BBox":
        """Return the overlapping rectangle, which may be empty."""
        return BBox(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )

    def intersection_area(self, other: "BBox") -> float:
        """Area of the overlap with ``other``, or ``0`` if they miss."""
        inter = self.intersection(other)
        return inter.area if not inter.is_empty else 0.0

    def overlap_ratio(self, other: "BBox") -> float:
        """Fraction of this box covered by ``other``."""
        if self.area == 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def contains_point(self, x: float, y: float) -> bool:
        """True when ``(x, y)`` lies inside or on the edge of this box."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(x0, y0, x1, y1)`` for PyMuPDF ``Rect`` constructors."""
        return (self.x0, self.y0, self.x1, self.y1)

    def union(self, other: "BBox") -> "BBox":
        """Smallest box that covers both this box and ``other``."""
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    @classmethod
    def combine(cls, left: "BBox | None", right: "BBox | None") -> "BBox | None":
        """Union of two optional boxes; ``None`` only when both are missing."""
        if left is None:
            return right
        if right is None:
            return left
        return left.union(right)

    @classmethod
    def union_all(cls, boxes: Sequence["BBox"]) -> "BBox | None":
        """Union of every box in ``boxes``, or ``None`` when the sequence is empty."""
        result: BBox | None = None
        for box in boxes:
            result = cls.combine(result, box)
        return result

    @classmethod
    def from_rect(cls, rect: Any) -> "BBox":
        """Build a box from a PyMuPDF ``Rect`` or any object with x0/y0/x1/y1."""
        return cls(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


@dataclass
class ImageRef:
    """Reference to an original image stored on disk (not the bytes)."""

    path: str
    xref: int
    width: int
    height: int
    bbox: BBox
    ext: str
    labels: list[str] = field(default_factory=list)


@dataclass
class TextSpan:
    """A run of text with the same emphasis."""

    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False

    def can_merge_with(
        self,
        other: "TextSpan",
        *,
        join_across_newlines: bool = True,
    ) -> bool:
        """True when ``other`` can be appended to this run."""
        if self.bold != other.bold:
            return False
        if self.italic != other.italic:
            return False
        if self.underline != other.underline:
            return False
        if join_across_newlines:
            return True
        return "\n" not in self.text and not other.text.startswith("\n")

    @staticmethod
    def merge_adjacent(
        spans: Sequence["TextSpan"],
        *,
        join_across_newlines: bool = True,
    ) -> list["TextSpan"]:
        """Merge neighboring runs that share emphasis."""
        merged: list[TextSpan] = []
        for span in spans:
            if not span.text:
                continue
            if merged and merged[-1].can_merge_with(
                span,
                join_across_newlines=join_across_newlines,
            ):
                merged[-1].text += span.text
            else:
                merged.append(
                    TextSpan(
                        text=span.text,
                        bold=span.bold,
                        italic=span.italic,
                        underline=span.underline,
                    )
                )
        return merged


@dataclass
class TextLine:
    """One visual line inside a text block, with indent and optional bullet."""

    line_number: int
    text: str
    indent: float
    indent_level: int
    is_bullet: bool
    bullet: str | None
    bbox: BBox
    spans: list[TextSpan] = field(default_factory=list)


@dataclass
class TextBlock:
    """A contiguous text block in document order."""

    order: int
    lines: list[TextLine] = field(default_factory=list)
    indent: float = 0.0
    bbox: BBox | None = None

    @property
    def text(self) -> str:
        """Plain text of the block, one extracted line per newline."""
        return "\n".join(line.text for line in self.lines)


@dataclass
class CellPart:
    """One piece of content inside a table cell, in reading order."""

    kind: str  # "text" or "image"
    text: str | None = None
    image: ImageRef | None = None
    spans: list[TextSpan] = field(default_factory=list)


@dataclass
class TableCell:
    """One table cell, including merged span, alignment, and inner parts."""

    row: int
    column: int
    rowspan: int
    colspan: int
    align: str
    valign: str
    shading: str | None
    text: str
    parts: list[CellPart] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    bbox: BBox | None = None
    is_header: bool = False


@dataclass
class Table:
    """A detected table: column names, grid size, and flattened cell list."""

    columns: list[str]
    column_count: int
    row_count: int
    cells: list[TableCell] = field(default_factory=list)
    bbox: BBox | None = None
    header_row_count: int = 0


ContentPayload = Union[TextBlock, Table, ImageRef]


@dataclass
class DocumentItem:
    """One extracted object on a page, in reading order.

    ``meta`` holds optional analysis fields such as ``word_count``,
    ``relation_group``, ``verbose``, ``grammar``, ``passive_voice``,
    ``empty_space``, ``resizing``, ``section_consistency``, and
    ``reconstruction``. Extractors
    leave it empty; grouping analysis fills it later.
    """

    page: int
    line_number: int
    kind: ContentKind
    content: ContentPayload
    bbox: BBox | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentContent:
    """Full extraction result: document path, page count, and ordered items."""

    name: str
    page_count: int
    items: list[DocumentItem] = field(default_factory=list)

    def items_on_page(self, page: int) -> list[DocumentItem]:
        """Return items whose ``page`` matches ``page`` (1-based)."""
        return [item for item in self.items if item.page == page]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready nested dict of this document."""
        return _to_plain(self)


def _to_plain(value: Any) -> Any:
    """Convert dataclasses, enums, and boxes into JSON-serializable values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BBox):
        return {
            "x0": value.x0,
            "y0": value.y0,
            "x1": value.x1,
            "y1": value.y1,
        }
    if is_dataclass(value) and not isinstance(value, type):
        data = {
            key: _to_plain(getattr(value, key))
            for key in value.__dataclass_fields__
        }
        if isinstance(value, TextBlock):
            data["text"] = value.text
        return data
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value
