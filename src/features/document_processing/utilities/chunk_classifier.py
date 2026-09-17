from __future__ import annotations

from src.features.document_processing.core.logger import log
from src.features.chunking.application.chunking_service import Chunk


def apply_document_classification(chunks: list[Chunk]) -> None:
    """Apply a default classification to chunks when no category is present."""
    total = len(chunks)
    if not total:
        return

    log.info("Classifying %d chunk(s).", total)
    progress_every = max(1, min(50, total // 10 or 1))

    for index, chunk in enumerate(chunks, start=1):
        if not chunk.category:
            chunk.category = "default"
            chunk.category_confidence = 1.0

        if index == total or index % progress_every == 0:
            log.info("Classified %d/%d chunk(s).", index, total)
