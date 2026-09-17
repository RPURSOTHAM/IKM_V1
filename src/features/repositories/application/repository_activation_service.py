from __future__ import annotations

from typing import Any

from src.features.repositories.application.repository_settings_service import CLOUD_PROVIDERS


def configuration_checklist(
    *,
    settings: dict[str, Any] | None,
    weaviate_collection: str | None,
) -> tuple[bool, list[str]]:
    """Validate persisted repository settings required before activation.

    Uses stored settings only — platform defaults are not applied here so
    repositories created without explicit configuration cannot be activated.
    """
    stored = dict(settings or {})
    missing: list[str] = []

    embedding = stored.get("embedding_model")
    if not isinstance(embedding, dict):
        missing.append("embedding_model")
    else:
        provider = str(embedding.get("provider", "")).strip().lower()
        if provider == "local":
            if not str(embedding.get("local_model_dir") or embedding.get("model_id") or "").strip():
                missing.append("embedding_model.local_model_dir")
        elif provider in CLOUD_PROVIDERS:
            if not str(embedding.get("credential_ref") or "").strip():
                missing.append("embedding_model.credential_ref")
        elif not provider:
            missing.append("embedding_model.provider")
        else:
            missing.append("embedding_model.provider")

    chunk_size = stored.get("chunk_size")
    if chunk_size is None:
        missing.append("chunk_size")
    else:
        try:
            if int(chunk_size) < 200:
                missing.append("chunk_size")
        except (TypeError, ValueError):
            missing.append("chunk_size")

    if stored.get("chunk_overlap") is None:
        missing.append("chunk_overlap")

    if not str(weaviate_collection or "").strip():
        missing.append("weaviate_collection")

    return len(missing) == 0, missing
