"""Map PDF Symbol/Wingdings PUA glyphs to Unicode browsers can display.

Those fonts store a check, square, or bullet as a Private Use code (for
example Wingdings ``U+F0FC``). The PDF looks correct; a web viewer shows
a missing-glyph box or, if we fold the whole PUA into ``•``, a dot.

``canonical_text`` replaces each mapped dingbat with the Unicode character
that matches the original glyph. Unmapped characters, including real
Unicode ticks (``√``, ``✓``) and ``X``, are left unchanged.
"""

from __future__ import annotations

STANDARD_BULLET = "•"
STANDARD_CHECK = "✓"

# PUA code point -> Unicode. Keys are the full U+F0xx values used by
# Symbol/Wingdings in these SOPs, plus nearby ballot marks so similar
# documents keep the same glyph instead of collapsing to a bullet.
PUA_TO_UNICODE = {
    0xF0B7: STANDARD_BULLET,  # Symbol filled bullet
    0xF050: "\U0001F3F1",  # Wingdings P: white pennant (shelf marker)
    0xF076: "\u2756",  # Wingdings v: black diamond minus white X
    0xF0A7: STANDARD_BULLET,  # Symbol/Wingdings small bullet
    0xF0B2: "\u25A0",  # filled square
    0xF0FB: "\u2717",  # Wingdings ballot X
    0xF0FC: STANDARD_CHECK,  # Wingdings check / tick
    0xF0FD: "\u2612",  # ballot box with X
    0xF0FE: "\u2611",  # ballot box with check
}


def canonical_symbol(char: str) -> str:
    """Return the Unicode stand-in for a Symbol/Wingdings PUA character."""

    if len(char) != 1:
        return char
    return PUA_TO_UNICODE.get(ord(char), char)


def canonical_text(text: str) -> str:
    """Replace dingbat PUA characters in ``text``; leave everything else as-is."""

    if not text:
        return text
    return "".join(canonical_symbol(char) for char in text)


LIST_BULLET_MARKERS = frozenset(
    "\u2022\u2023\u25e6\u2043\u2219\u00b7\u25cf\u25cb\u25aa\u25ab\u25b6\u25b8•"
)


def is_list_bullet_marker(text: str) -> bool:
    """True when ``text`` is a list bullet, not a tick or other mark."""

    if len(text) != 1:
        return False
    return canonical_symbol(text) in LIST_BULLET_MARKERS
