"""Exception types treated as recoverable in the viewer and analysis."""

from __future__ import annotations

RECOVERABLE = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    ImportError,
    LookupError,
    AttributeError,
    KeyError,
    IndexError,
)
