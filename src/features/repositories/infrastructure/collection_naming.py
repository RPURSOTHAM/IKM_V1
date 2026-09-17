from __future__ import annotations

import re
from typing import Any

# Vector dimensions for collection naming and dimension-consistency checks (FRD §4.1.1).
LOCAL_MODEL_VECTOR_DIMS: dict[str, int] = {
    "bge-m3": 1024,
    "bge-base-en": 768,
    "bge-small-en": 384,
    "bge-large-en": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "e5-base-v2": 768,
    "e5-large-v2": 1024,
    # Hugging Face hub-style directory names under MODELS_ROOT
    "models--BAAI--bge-base-en": 768,
    "models--BAAI--bge-small-en": 384,
    "models--BAAI--bge-large-en": 1024,
    "models--BAAI--bge-m3": 1024,
    "models--sentence-transformers--all-MiniLM-L6-v2": 384,
    "models--sentence-transformers--all-mpnet-base-v2": 768,
    "models--intfloat--e5-base-v2": 768,
    "models--intfloat--e5-large-v2": 1024,
}

CLOUD_MODEL_VECTOR_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2:0": 1024,
    "cohere.embed-english-v3": 1024,
}

LOCAL_MODEL_HUB_IDS: dict[str, str] = {
    "bge-m3": "BAAI/bge-m3",
    "bge-base-en": "BAAI/bge-base-en",
    "bge-small-en": "BAAI/bge-small-en",
    "bge-large-en": "BAAI/bge-large-en",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
}


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when an embedding vector width disagrees with repository configuration."""


def local_embedding_model_from_paths(model_name: str, model_dir: str | None = None) -> dict[str, Any]:
    """Build a minimal local embedding_model object from processor/retrieval paths."""
    local_dir = str(model_dir or "").strip()
    model_id = str(model_name or "").strip()
    if not local_dir:
        local_dir = model_id.split("/")[-1] if "/" in model_id else model_id
    if not model_id:
        model_id = local_dir
    elif "/" in model_id:
        model_id = model_id.split("/")[-1]
    return {"provider": "local", "model_id": model_id, "local_model_dir": local_dir}


def assert_embedding_dimension_consistency(
    embedding_model: dict[str, Any],
    actual_dim: int,
    *,
    context: str = "embedding",
) -> None:
    """Reject ingest/retrieve when runtime vector width differs from repository config."""
    expected = embedding_vector_dimensions(embedding_model)
    if expected is None:
        return
    if int(actual_dim) != int(expected):
        raise EmbeddingDimensionMismatchError(
            f"{context}: embedding dimension {actual_dim} does not match repository-configured "
            f"expected dimension {expected} (model_id={embedding_model.get('model_id')!r})"
        )


def embedding_vector_dimensions(embedding_model: dict[str, Any]) -> int | None:
    """Return expected vector width for a repository embedding_model object."""
    provider = str(embedding_model.get("provider", "")).strip().lower()
    model_id = str(embedding_model.get("model_id", "")).strip()
    if provider == "local":
        local_dir = str(embedding_model.get("local_model_dir") or model_id).strip()
        return LOCAL_MODEL_VECTOR_DIMS.get(local_dir) or LOCAL_MODEL_VECTOR_DIMS.get(model_id)
    override = embedding_model.get("dimensions")
    if isinstance(override, int) and override > 0:
        return override
    return CLOUD_MODEL_VECTOR_DIMS.get(model_id)


def resolve_processor_model_name(embedding_model: dict[str, Any]) -> str:
    """Map repository embedding_model to a processor/retrieval model identifier."""
    provider = str(embedding_model.get("provider", "")).strip().lower()
    model_id = str(embedding_model.get("model_id", "")).strip()
    if provider == "local":
        local_dir = str(embedding_model.get("local_model_dir") or model_id).strip()
        return LOCAL_MODEL_HUB_IDS.get(local_dir, model_id)
    return model_id


def collection_name_from_repository_name(name: str) -> str:
    """Derive a Weaviate-safe collection name from the repository name."""
    trimmed = (name or "").strip()
    if not trimmed:
        raise ValueError("Repository name is required.")
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", trimmed)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Repository name cannot produce a valid Weaviate collection name.")
    if not re.match(r"^[A-Za-z]", normalized):
        normalized = f"R_{normalized}"
    if not re.match(r"^[A-Z]", normalized):
        normalized = normalized[0].upper() + normalized[1:]
    return normalized[:255]



