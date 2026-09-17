"""
loader.py
=========
Base abstractions shared by all document loaders.

Provides:
    Block           — a single extracted line/paragraph from a document.
    DocumentLoader  — abstract base class every loader must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class Block:
    """
    A single unit of extracted content from a document.

    Attributes
    ----------
    text : str
        The visible text of the block.
    page : int
        1-based page number where the block appears.
    line_number : int
        1-based extracted line number within the page.
    style : str
        Paragraph / character style name (e.g. "Heading 1", "Normal").
        Empty string when not available (e.g. PDF blocks).
    font_name : str
        Name of the primary font (e.g. "Arial", "TimesNewRoman").
    font_size : float
        Font size in points; 0.0 when unknown.
    bold : bool
        True if the block's primary run is bold.
    italic : bool
        True if the block's primary run is italic.
    alignment : str
        One of "left", "center", "right", "justified".
    indent_left : float
        Left indentation in points relative to the page margin.
    is_toc_entry : bool
        True if the block was identified as a Table of Contents entry.
    block_type : str
        Type of extracted content: text, table, or image.
    component_type : str
        Structural role: header, footer, title, subtitle, paragraph, table, or image.
    heading_level : int | None
        1-based heading depth when the block is a title or subtitle.
    metadata : dict
        Extra source metadata such as table index, image bounds, or row counts.
    """

    text: str
    page: int
    line_number: int = 0
    style: str = ""
    font_name: str = ""
    font_size: float = 0.0
    bold: bool = False
    italic: bool = False
    alignment: str = "left"
    indent_left: float = 0.0
    is_toc_entry: bool = False
    block_type: str = "text"
    component_type: str = ""
    heading_level: int | None = None
    metadata: dict = field(default_factory=dict)


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    Subclasses implement :meth:`load` to parse a document file and return
    an ordered list of :class:`Block` objects representing its content.
    """

    @abstractmethod
    def load(self, path: Union[str, Path]) -> list[Block]:
        """
        Load a document and return its content as an ordered list of Blocks.

        Parameters
        ----------
        path : str or Path
            Absolute or relative path to the document file.

        Returns
        -------
        list[Block]
            Ordered sequence of blocks extracted from the document.
        """
