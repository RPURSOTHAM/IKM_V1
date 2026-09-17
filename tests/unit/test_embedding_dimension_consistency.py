"""Unit tests for repository embedding dimension consistency checks."""

from __future__ import annotations

import pytest

from src.features.repositories.infrastructure.collection_naming import (
    EmbeddingDimensionMismatchError,
    assert_embedding_dimension_consistency,
    embedding_vector_dimensions,
    local_embedding_model_from_paths,
)


def test_embedding_vector_dimensions_supports_hub_style_model_dir() -> None:
    model = {
        "provider": "local",
        "model_id": "all-MiniLM-L6-v2",
        "local_model_dir": "models--sentence-transformers--all-MiniLM-L6-v2",
    }
    assert embedding_vector_dimensions(model) == 384


def test_assert_embedding_dimension_consistency_accepts_match() -> None:
    model = local_embedding_model_from_paths(
        "sentence-transformers/all-MiniLM-L6-v2",
        "models--sentence-transformers--all-MiniLM-L6-v2",
    )
    assert_embedding_dimension_consistency(model, 384, context="test")


def test_assert_embedding_dimension_consistency_rejects_mismatch() -> None:
    model = {
        "provider": "local",
        "model_id": "bge-base-en",
        "local_model_dir": "bge-base-en",
    }
    with pytest.raises(EmbeddingDimensionMismatchError):
        assert_embedding_dimension_consistency(model, 384, context="test")
