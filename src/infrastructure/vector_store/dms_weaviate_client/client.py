"""Shared Weaviate sync client factory for retrieval and repository admin."""

from __future__ import annotations

from typing import Any

from src.features.configuration.platform_settings import settings
from src.features.configuration.application.schema_registry import strip_wrapping_quotes


def connect_weaviate() -> Any:
    from urllib.parse import urlparse

    from weaviate import connect_to_custom
    from weaviate.auth import Auth

    parsed = urlparse(settings.weaviate_url)
    host = parsed.hostname or "localhost"
    secure = (parsed.scheme or "http").lower() == "https"
    if parsed.port is not None:
        port = parsed.port
    else:
        port = 443 if secure else 8080

    kwargs: dict[str, Any] = {
        "http_host": host,
        "http_port": port,
        "http_secure": secure,
        "grpc_host": host,
        "grpc_port": settings.weaviate_grpc_port,
        "grpc_secure": secure,
    }
    key = strip_wrapping_quotes((settings.weaviate_api_key or "").strip())
    if key:
        kwargs["auth_credentials"] = Auth.api_key(key)

    return connect_to_custom(**kwargs)
