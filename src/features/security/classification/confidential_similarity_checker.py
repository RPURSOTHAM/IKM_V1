"""Backward-compatible facade over ConfidentialSimilarityEngine."""

from __future__ import annotations

from typing import Any

from src.features.security.classification.confidential_similarity_engine import (
    ConfidentialSimilarityEngine,
    add_sensitive_document,
    evaluate_similarity_risk,
    get_similarity_engine,
    search_similar_content,
)


class ConfidentialSimilarityChecker:
    """Upload-pipeline adapter — real embedding similarity, no mock corpus."""

    def __init__(self) -> None:
        self._engine = get_similarity_engine()

    def check_similarity(self, text: str, **kwargs: Any) -> dict[str, Any]:
        result = self._engine.search_similar_content(text, **kwargs)
        # Attach risk evaluation for DLP consumers.
        risk = evaluate_similarity_risk(float(result.get("max_similarity") or 0.0))
        result["risk"] = risk
        return result

    def add_sensitive_document(self, text: str, **kwargs: Any) -> str:
        return self._engine.add_sensitive_document(text, **kwargs)


__all__ = [
    "ConfidentialSimilarityChecker",
    "ConfidentialSimilarityEngine",
    "add_sensitive_document",
    "evaluate_similarity_risk",
    "get_similarity_engine",
    "search_similar_content",
]
