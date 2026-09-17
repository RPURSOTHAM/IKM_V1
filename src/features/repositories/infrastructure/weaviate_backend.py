from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from src.features.configuration.platform_settings import settings
from src.infrastructure.vector_store.dms_weaviate_client.client import connect_weaviate

logger = logging.getLogger(__name__)


class WeaviateRepositoryBackend:
    def __init__(self) -> None:
        self._client: Any = None
        self._error: str | None = None

    @property
    def enabled(self) -> bool:
        return settings.enable_repository_admin

    @property
    def ready(self) -> bool:
        return self._client is not None and self._error is None

    @property
    def last_error(self) -> str | None:
        return self._error

    def _try_connect(self) -> None:
        if not self.enabled:
            self._error = None
            return
        try:
            client = connect_weaviate()
            if not client.is_ready():
                raise RuntimeError(f"Weaviate is not ready at {settings.weaviate_url}")
            if self._client is not None:
                close = getattr(self._client, "close", None)
                if callable(close):
                    close()
            self._client = client
            self._error = None
            logger.info("Weaviate repository admin connected to %s", settings.weaviate_url)
        except Exception as exc:
            if self._client is not None:
                close = getattr(self._client, "close", None)
                if callable(close):
                    close()
            self._client = None
            self._error = str(exc)
            logger.warning("Weaviate repository admin connection attempt failed: %s", exc)

    def require_client(self) -> Any:
        if not self.enabled:
            raise HTTPException(
                status_code=503,
                detail="Repository admin is disabled. Set RAG_API_ENABLE_REPOSITORY_ADMIN=true.",
            )
        if self._client is None:
            self._try_connect()
        if self._client is None:
            msg = self._error or "client not connected"
            no_key = not (settings.weaviate_api_key or "").strip()
            hint = (
                "Weaviate in src/dependencies/docker-compose.yml disables anonymous access and expects "
                "WEAVIATE_API_KEY to match AUTHENTICATION_APIKEY_ALLOWED_KEYS (default dev: weaviate_secret_key)."
                if no_key
                else "Verify WEAVIATE_URL, WEAVIATE_GRPC_PORT, and WEAVIATE_API_KEY match your Weaviate instance."
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "weaviate_repository_unavailable",
                    "message": msg,
                    "weaviate_url": settings.weaviate_url,
                    "weaviate_grpc_port": settings.weaviate_grpc_port,
                    "api_key_configured": not no_key,
                    "hint": hint,
                },
            )
        return self._client

    async def startup(self) -> None:
        if not self.enabled:
            self._error = None
            return
        self._try_connect()
        if self._client is None:
            logger.error(
                "Weaviate repository admin failed to connect at startup: %s. "
                "Chunk retrieval will retry on the next request.",
                self._error,
            )

    async def shutdown(self) -> None:
        if self._client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self._client = None


backend = WeaviateRepositoryBackend()
