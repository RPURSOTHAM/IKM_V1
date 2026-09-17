"""Shared chunking strategy identifiers for repository settings and processors."""

from __future__ import annotations

DEFAULT_CHUNKING_STRATEGY = "section-based"

CHUNKING_STRATEGIES: frozenset[str] = frozenset(
    {
        "fixed-overlap-based",
        "sentence-based",
        "paragraph-based",
        "section-based",
        "hierarchical",
        "semantic",
        "semantic-hierarchy",
        "sliding-window",
    }
)

LEGACY_CHUNKING_STRATEGY_ALIASES: dict[str, str] = {
    "fixed": "fixed-overlap-based",
    "fixed-overlap": "fixed-overlap-based",
    "sentence": "sentence-based",
    "section": "section-based",
    "paragraph": "paragraph-based",
    "semantic-based": "semantic",
    "semantic-hierarchical": "semantic-hierarchy",
    "hierarchical-semantic": "semantic-hierarchy",
    "sliding-window-based": "sliding-window",
}


def coerce_optional_chunking_strategy(value: str | None) -> str | None:
    """Normalize a strategy id, or return None when the caller did not set one.

    ``normalize_chunking_strategy(None)`` historically returned section-based.
    That default must not be written onto ProcessRequest — it overrides a
    repository's paragraph-based (or other) setting.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return normalize_chunking_strategy(raw)


def normalize_chunking_strategy(value: str | None) -> str:
    """Map legacy aliases to canonical chunking strategy identifiers."""
    raw = str(value or DEFAULT_CHUNKING_STRATEGY).strip().lower()
    return LEGACY_CHUNKING_STRATEGY_ALIASES.get(raw, raw)


def is_supported_chunking_strategy(value: str | None) -> bool:
    return normalize_chunking_strategy(value) in CHUNKING_STRATEGIES


def resolve_chunking_strategy(value: str | None) -> str:
    """Normalize and validate a chunking strategy."""
    normalized = normalize_chunking_strategy(value)
    if normalized not in CHUNKING_STRATEGIES:
        aliases = ", ".join(f"{alias}->{target}" for alias, target in sorted(LEGACY_CHUNKING_STRATEGY_ALIASES.items()))
        raise ValueError(
            f"Unsupported chunking strategy: {value!r} (normalized: {normalized!r}). "
            f"Supported strategies: {', '.join(sorted(CHUNKING_STRATEGIES))}. "
            f"Known aliases: {aliases}."
        )
    return normalized
