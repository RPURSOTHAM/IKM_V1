"""Retrieval strategy registry and selection."""

from __future__ import annotations

from src.features.retrieval.strategies.base import RetrievalStrategy
from src.features.retrieval.strategies.bm25_search import Bm25RetrievalStrategy
from src.features.retrieval.strategies.hybrid_search import HybridRetrievalStrategy
from src.features.retrieval.strategies.vector_search import VectorRetrievalStrategy

_VECTOR = VectorRetrievalStrategy()
_BM25 = Bm25RetrievalStrategy()
_HYBRID = HybridRetrievalStrategy()


def get_retrieval_strategy(search_mode: str) -> RetrievalStrategy:
    mode = (search_mode or "hybrid").strip().lower()
    if mode == "vector":
        return _VECTOR
    if mode == "keyword":
        return _BM25
    return _HYBRID
