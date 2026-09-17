"""Catalog of selectable embedding models for repository settings."""

from __future__ import annotations

from typing import Any

from src.features.repositories.infrastructure.collection_naming import (
    CLOUD_MODEL_VECTOR_DIMS,
    LOCAL_MODEL_HUB_IDS,
    LOCAL_MODEL_VECTOR_DIMS,
)
from src.features.repositories.application.repository_settings_service import (
    CLOUD_PROVIDERS,
    DEFAULT_EMBEDDING_MODEL,
    discover_local_embedding_models,
    models_root,
)

# Known cloud models per provider (FRD §4.1.1 / repository-service.md).
CLOUD_EMBEDDING_CATALOG: dict[str, list[dict[str, Any]]] = {
    "openai": [
        {
            "model_id": "text-embedding-3-small",
            "dimensions": 1536,
            "dimensions_configurable": True,
            "description": "Cost-effective OpenAI embedding model.",
        },
        {
            "model_id": "text-embedding-3-large",
            "dimensions": 3072,
            "dimensions_configurable": True,
            "description": "Higher-quality OpenAI embedding model.",
        },
        {
            "model_id": "text-embedding-ada-002",
            "dimensions": 1536,
            "dimensions_configurable": False,
            "description": "Legacy OpenAI embedding model.",
        },
    ],
    "azure_openai": [
        {
            "model_id": "text-embedding-3-small",
            "dimensions": 1536,
            "dimensions_configurable": True,
            "description": "Azure OpenAI deployment of text-embedding-3-small.",
        },
        {
            "model_id": "text-embedding-3-large",
            "dimensions": 3072,
            "dimensions_configurable": True,
            "description": "Azure OpenAI deployment of text-embedding-3-large.",
        },
        {
            "model_id": "text-embedding-ada-002",
            "dimensions": 1536,
            "dimensions_configurable": False,
            "description": "Azure OpenAI deployment of text-embedding-ada-002.",
        },
    ],
    "aws": [
        {
            "model_id": "amazon.titan-embed-text-v1",
            "dimensions": 1536,
            "dimensions_configurable": False,
            "description": "Amazon Bedrock Titan Embeddings v1.",
        },
        {
            "model_id": "amazon.titan-embed-text-v2:0",
            "dimensions": 1024,
            "dimensions_configurable": False,
            "description": "Amazon Bedrock Titan Embeddings v2.",
        },
        {
            "model_id": "cohere.embed-english-v3",
            "dimensions": 1024,
            "dimensions_configurable": False,
            "description": "Cohere English embedding model on Amazon Bedrock.",
        },
    ],
    "anthropic": [
        {
            "model_id": "voyage-3",
            "dimensions": 1024,
            "dimensions_configurable": False,
            "description": "Partner embedding model exposed via Anthropic/AWS integrations (deployment-specific).",
        },
    ],
}

PROVIDER_LABELS: dict[str, str] = {
    "local": "Local (src/models/)",
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "aws": "AWS Bedrock",
    "anthropic": "Anthropic / partner embeddings",
}

LOCAL_MODEL_DESCRIPTIONS: dict[str, str] = {
    "bge-m3": "Preferred multilingual, multi-function BGE-M3 embeddings (1024-dim); selectable when installed.",
    "bge-base-en": "Default; balanced quality and latency (768-dim).",
    "bge-small-en": "Fast English embeddings with lower memory use (384-dim).",
    "bge-large-en": "Higher-quality English embeddings (1024-dim).",
    "all-MiniLM-L6-v2": "Lightweight / fast embeddings (384-dim).",
    "all-mpnet-base-v2": "General-purpose MPNet embeddings (768-dim).",
    "e5-base-v2": "E5 base retrieval embeddings (768-dim).",
    "e5-large-v2": "E5 large retrieval embeddings (1024-dim).",
}


def _local_model_option(name: str, *, installed: bool) -> dict[str, Any]:
    return {
        "provider": "local",
        "model_id": name,
        "local_model_dir": name,
        "hub_id": LOCAL_MODEL_HUB_IDS.get(name),
        "dimensions": LOCAL_MODEL_VECTOR_DIMS.get(name),
        "available": installed,
        "description": LOCAL_MODEL_DESCRIPTIONS.get(name),
        "embedding_model": {
            "provider": "local",
            "model_id": name,
            "local_model_dir": name,
        },
    }


def _cloud_model_option(provider: str, entry: dict[str, Any]) -> dict[str, Any]:
    model_id = str(entry["model_id"])
    dimensions = entry.get("dimensions") or CLOUD_MODEL_VECTOR_DIMS.get(model_id)
    return {
        "provider": provider,
        "model_id": model_id,
        "dimensions": dimensions,
        "dimensions_configurable": bool(entry.get("dimensions_configurable")),
        "requires_credential_ref": True,
        "description": entry.get("description"),
        "embedding_model": {
            "provider": provider,
            "model_id": model_id,
            **({"dimensions": dimensions} if dimensions else {}),
        },
    }


def build_embedding_model_catalog() -> dict[str, Any]:
    """Return all selectable embedding models grouped by provider."""
    installed_local = set(discover_local_embedding_models())
    local_names = sorted(installed_local | set(LOCAL_MODEL_VECTOR_DIMS.keys()))

    local_models = [_local_model_option(name, installed=name in installed_local) for name in local_names]

    providers: list[dict[str, Any]] = [
        {
            "provider": "local",
            "label": PROVIDER_LABELS["local"],
            "requires_credential_ref": False,
            "models": local_models,
        }
    ]

    for provider in sorted(CLOUD_PROVIDERS):
        models = [
            _cloud_model_option(provider, entry)
            for entry in CLOUD_EMBEDDING_CATALOG.get(provider, [])
        ]
        providers.append(
            {
                "provider": provider,
                "label": PROVIDER_LABELS.get(provider, provider),
                "requires_credential_ref": True,
                "models": models,
            }
        )

    flat_options: list[dict[str, Any]] = []
    for group in providers:
        for model in group["models"]:
            flat_options.append(model)

    return {
        "default": dict(DEFAULT_EMBEDDING_MODEL),
        "local_models_root": str(models_root()),
        "providers": providers,
        "options": flat_options,
        "models": flat_options,
        "count": len(flat_options),
    }
