from __future__ import annotations

from typing import List
from src.features.chunking.application.chunking_service import Chunk


def rewrite_chunks_for_embedding(
    chunks: List[Chunk],
    mode: str = "rule_based",
    max_workers: int = 1,
    max_tokens: int = 700,
    timeout_seconds: int = 60,
    min_chunk_words: int = 60,
) -> None:
    """Placeholder rewrite step for embedding preparation."""
    for chunk in chunks:
        chunk.rewritten_text = chunk.text
