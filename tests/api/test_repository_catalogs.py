"""Repository catalog endpoints (read-only)."""

from __future__ import annotations

import httpx

from api.helpers import assert_status


def test_get_embedding_models_catalog(api_client: httpx.Client) -> None:
    endpoint = "GET /api/v1/repositories/embedding-models"
    response = api_client.get("/api/v1/repositories/embedding-models")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert "providers" in body or "models" in body or "count" in body


def test_get_chunking_strategies_catalog(api_client: httpx.Client) -> None:
    endpoint = "GET /api/v1/repositories/chunking-strategies"
    response = api_client.get("/api/v1/repositories/chunking-strategies")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert "strategies" in body or "strategy_ids" in body
