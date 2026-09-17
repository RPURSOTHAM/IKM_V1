"""Lifecycle tests for the shared Weaviate repository backend client."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from src.features.repositories.infrastructure.weaviate_backend import WeaviateRepositoryBackend

_backend_mod = importlib.import_module("src.features.repositories.infrastructure.weaviate_backend")


@pytest.mark.asyncio
async def test_shared_backend_closes_on_shutdown_not_per_request() -> None:
    instance = WeaviateRepositoryBackend()
    client = MagicMock()
    client.is_ready.return_value = True

    mock_settings = MagicMock()
    mock_settings.enable_repository_admin = True
    mock_settings.weaviate_url = "http://weaviate:8080"
    mock_settings.weaviate_api_key = "key"
    mock_settings.weaviate_grpc_port = 50051

    with patch.object(_backend_mod, "settings", mock_settings), patch.object(
        _backend_mod, "connect_weaviate", return_value=client
    ):
        first = instance.require_client()
        second = instance.require_client()
        assert first is client
        assert second is client
        client.close.assert_not_called()

        await instance.shutdown()
        client.close.assert_called_once()
        assert instance._client is None

        await instance.shutdown()
        assert client.close.call_count == 1


def test_shared_backend_registers_atexit_once() -> None:
    instance = WeaviateRepositoryBackend()
    client = MagicMock()
    client.is_ready.return_value = True

    mock_settings = MagicMock()
    mock_settings.enable_repository_admin = True
    mock_settings.weaviate_url = "http://weaviate:8080"
    mock_settings.weaviate_api_key = "key"
    mock_settings.weaviate_grpc_port = 50051

    with patch.object(_backend_mod, "settings", mock_settings), patch.object(
        _backend_mod, "connect_weaviate", return_value=client
    ), patch.object(_backend_mod.atexit, "register") as register:
        instance.require_client()
        instance.require_client()
        register.assert_called_once_with(instance.close)
