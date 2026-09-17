"""Catalog of repository chunking strategies and per-strategy configuration fields."""

from __future__ import annotations

from typing import Any

from src.features.chunking.domain.chunking_strategy import (
    CHUNKING_STRATEGIES,
    DEFAULT_CHUNKING_STRATEGY,
    resolve_chunking_strategy,
)

# strategy_id, label, description, sequence
_STRATEGY_DEFINITIONS: tuple[tuple[str, str, str, int], ...] = (
    (
        "fixed-overlap-based",
        "Fixed overlap",
        "Splits text into fixed word windows with configurable overlap.",
        1,
    ),
    (
        "sentence-based",
        "Sentence based",
        "Groups complete sentences up to a maximum sentence count per chunk.",
        2,
    ),
    (
        "paragraph-based",
        "Paragraph based",
        "Merges whole paragraphs until reaching target size; never breaks mid-paragraph.",
        3,
    ),
    (
        "section-based",
        "Section based",
        "Detects document heading boundaries and chunks within each section.",
        4,
    ),
    (
        "hierarchical",
        "Hierarchical",
        "Preserves multi-level heading tree paths (Heading 1 > Heading 2) in chunk metadata.",
        5,
    ),
    (
        "semantic",
        "Semantic",
        "Merges adjacent sentences while embedding cosine similarity remains above a threshold.",
        6,
    ),
    (
        "semantic-hierarchy",
        "Semantic hierarchy",
        "Respects structural elements (Section > Subsection > Paragraph > Table > Image Caption).",
        7,
    ),
    (
        "sliding-window",
        "Sliding window",
        "Advances a fixed-size word window by a step size across document text.",
        8,
    ),
)

_CONFIG_FIELDS: dict[str, list[dict[str, Any]]] = {
    "fixed-overlap-based": [
        {
            "key": "chunk_size",
            "scope": "settings",
            "label": "Chunk size",
            "value_type": "integer",
            "required": True,
            "minimum": 200,
            "maximum": 2000,
            "default": 512,
            "description": "Maximum words per chunk.",
        },
        {
            "key": "chunk_overlap",
            "scope": "settings",
            "label": "Chunk overlap",
            "value_type": "integer",
            "required": True,
            "minimum": 0,
            "default": 1,
            "description": "Number of overlapping words carried into the next chunk.",
        },
        {
            "key": "min_content_words",
            "scope": "chunking_config",
            "label": "Minimum words",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "default": 60,
            "description": "Minimum words required to keep a chunk.",
        },
    ],
    "sentence-based": [
        {
            "key": "max_sentences_per_chunk",
            "scope": "chunking_config",
            "label": "Max sentences per chunk",
            "value_type": "integer",
            "required": True,
            "minimum": 1,
            "maximum": 50,
            "default": 8,
            "description": "Maximum sentences grouped into one chunk.",
        },
        {
            "key": "chunk_overlap",
            "scope": "settings",
            "label": "Sentence overlap",
            "value_type": "integer",
            "required": False,
            "minimum": 0,
            "default": 1,
            "description": "Sentences repeated at chunk boundaries.",
        },
        {
            "key": "min_content_words",
            "scope": "chunking_config",
            "label": "Minimum words",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "default": 20,
            "description": "Minimum words required to keep a chunk.",
        },
    ],
    "paragraph-based": [
        {
            "key": "max_paragraphs_per_chunk",
            "scope": "chunking_config",
            "label": "Max paragraphs per chunk",
            "value_type": "integer",
            "required": True,
            "minimum": 1,
            "maximum": 20,
            "default": 3,
            "description": "Maximum paragraphs merged into one chunk.",
        },
        {
            "key": "chunk_size",
            "scope": "settings",
            "label": "Word budget",
            "value_type": "integer",
            "required": False,
            "minimum": 200,
            "maximum": 2000,
            "default": 512,
            "description": "Soft maximum words before flushing a chunk.",
        },
        {
            "key": "min_content_words",
            "scope": "chunking_config",
            "label": "Minimum words",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "default": 40,
            "description": "Minimum words required to keep a chunk.",
        },
    ],
    "section-based": [
        {
            "key": "chunk_size",
            "scope": "settings",
            "label": "Chunk size",
            "value_type": "integer",
            "required": True,
            "minimum": 200,
            "maximum": 2000,
            "default": 512,
            "description": "Maximum words per chunk inside a section.",
        },
        {
            "key": "chunk_overlap",
            "scope": "settings",
            "label": "Sentence overlap",
            "value_type": "integer",
            "required": False,
            "minimum": 0,
            "default": 2,
            "description": "Sentences repeated across chunk boundaries within a section.",
        },
        {
            "key": "min_content_words",
            "scope": "chunking_config",
            "label": "Minimum words",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "default": 60,
            "description": "Minimum words required to keep a chunk.",
        },
    ],
    "hierarchical": [
        {
            "key": "chunk_size",
            "scope": "settings",
            "label": "Chunk size",
            "value_type": "integer",
            "required": True,
            "minimum": 200,
            "maximum": 2000,
            "default": 512,
            "description": "Maximum words per chunk inside a heading subtree.",
        },
        {
            "key": "chunk_overlap",
            "scope": "settings",
            "label": "Sentence overlap",
            "value_type": "integer",
            "required": False,
            "minimum": 0,
            "default": 2,
            "description": "Sentences repeated across chunk boundaries.",
        },
        {
            "key": "max_heading_depth",
            "scope": "chunking_config",
            "label": "Max heading depth",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "maximum": 6,
            "default": 4,
            "description": "Maximum heading levels included in hierarchical section paths.",
        },
        {
            "key": "min_content_words",
            "scope": "chunking_config",
            "label": "Minimum words",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "default": 60,
            "description": "Minimum words required to keep a chunk.",
        },
    ],
    "semantic": [
        {
            "key": "chunk_size",
            "scope": "settings",
            "label": "Target chunk size",
            "value_type": "integer",
            "required": True,
            "minimum": 200,
            "maximum": 2000,
            "default": 512,
            "description": "Soft maximum words per semantic chunk.",
        },
        {
            "key": "similarity_threshold",
            "scope": "chunking_config",
            "label": "Similarity threshold",
            "value_type": "number",
            "required": True,
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.72,
            "description": "Minimum cosine similarity to merge the next sentence into the current chunk.",
        },
        {
            "key": "max_sentences_per_chunk",
            "scope": "chunking_config",
            "label": "Max sentences per chunk",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "maximum": 100,
            "default": 20,
            "description": "Hard cap on sentences per chunk regardless of similarity.",
        },
        {
            "key": "min_content_words",
            "scope": "chunking_config",
            "label": "Minimum words",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "default": 40,
            "description": "Minimum words required to keep a chunk.",
        },
    ],
    "semantic-hierarchy": [
        {
            "key": "similarity_threshold",
            "scope": "chunking_config",
            "label": "Similarity threshold",
            "value_type": "number",
            "required": False,
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.72,
            "description": "Configured threshold for optional semantic grouping within an oversized structural element.",
        },
    ],
    "sliding-window": [
        {
            "key": "chunk_size",
            "scope": "settings",
            "label": "Window size",
            "value_type": "integer",
            "required": True,
            "minimum": 200,
            "maximum": 2000,
            "default": 512,
            "description": "Number of words in each sliding window.",
        },
        {
            "key": "chunk_overlap",
            "scope": "settings",
            "label": "Window step",
            "value_type": "integer",
            "required": True,
            "minimum": 1,
            "default": 128,
            "description": "Words advanced between consecutive windows (step size). Must be less than window size.",
        },
        {
            "key": "min_content_words",
            "scope": "chunking_config",
            "label": "Minimum words",
            "value_type": "integer",
            "required": False,
            "minimum": 1,
            "default": 20,
            "description": "Minimum words required to keep a chunk.",
        },
    ],
}


def build_chunking_strategy_catalog() -> dict[str, Any]:
    strategies: list[dict[str, Any]] = []
    for strategy_id, label, description, sequence in _STRATEGY_DEFINITIONS:
        if strategy_id not in CHUNKING_STRATEGIES:
            continue
        config_fields = list(_CONFIG_FIELDS.get(strategy_id, []))
        strategies.append(
            {
                "strategy_id": strategy_id,
                "label": label,
                "description": description,
                "sequence": sequence,
                "config_fields": config_fields,
                "example_settings": example_settings_for_strategy(strategy_id),
            }
        )
    strategies.sort(key=lambda item: int(item["sequence"]))
    return {
        "default": DEFAULT_CHUNKING_STRATEGY,
        "strategies": strategies,
        "strategy_ids": [item["strategy_id"] for item in strategies],
        "count": len(strategies),
    }


def get_chunking_strategy(strategy_id: str) -> dict[str, Any]:
    """Return one strategy catalog entry; raises ValueError when unknown."""
    normalized = resolve_chunking_strategy(strategy_id)
    catalog = build_chunking_strategy_catalog()
    for item in catalog["strategies"]:
        if item["strategy_id"] == normalized:
            return item
    raise ValueError(f"Unsupported chunking strategy: {strategy_id!r}")


def example_settings_for_strategy(strategy_id: str) -> dict[str, Any]:
    fields = _CONFIG_FIELDS.get(strategy_id, [])
    settings: dict[str, Any] = {"chunking_strategy": strategy_id}
    chunking_config: dict[str, Any] = {}
    for field in fields:
        if field.get("default") is None:
            continue
        if field["scope"] == "settings":
            settings[field["key"]] = field["default"]
        else:
            chunking_config[field["key"]] = field["default"]
    if chunking_config:
        settings["chunking_config"] = chunking_config
    return settings


def default_chunking_config(strategy_id: str) -> dict[str, Any]:
    fields = _CONFIG_FIELDS.get(strategy_id, [])
    config: dict[str, Any] = {}
    for field in fields:
        if field["scope"] != "chunking_config" or field.get("default") is None:
            continue
        config[field["key"]] = field["default"]
    return config


def default_settings_params(strategy_id: str) -> dict[str, Any]:
    """Return default top-level settings keys (chunk_size, chunk_overlap, …) for a strategy."""
    fields = _CONFIG_FIELDS.get(strategy_id, [])
    params: dict[str, Any] = {}
    for field in fields:
        if field["scope"] != "settings" or field.get("default") is None:
            continue
        params[field["key"]] = field["default"]
    return params
