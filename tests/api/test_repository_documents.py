"""Repository-linked document endpoints."""

from __future__ import annotations

import httpx

from api.helpers import assert_status


def test_list_repository_documents(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    endpoint = f"GET /api/v1/repositories/{repo_id}/documents"
    response = api_client.get(f"/api/v1/repositories/{repo_id}/documents")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert "documents" in body or "count" in body
