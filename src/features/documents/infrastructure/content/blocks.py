from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Block:
    text: str
    page: int
    bold: bool = False
    font_size: float = 0.0
