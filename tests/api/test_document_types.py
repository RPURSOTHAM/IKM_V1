"""Document type list and detail endpoints."""

from __future__ import annotations

import httpx
from api.helpers import assert_status


def test_list_document_types(api_client: httpx.Client) -> None:
    endpoint = "GET /api/v1/document-types"
    response = api_client.get("/api/v1/document-types")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert "document_types" in body
    assert body.get("count", 0) >= 1


def test_list_document_types_with_metadata(api_client: httpx.Client) -> None:
    endpoint = "GET /api/v1/document-types?include_metadata=true"
    response = api_client.get("/api/v1/document-types", params={"include_metadata": True})
    assert_status(response, 200, endpoint=endpoint)


def test_get_document_type(api_client: httpx.Client, base_document_type_id: str) -> None:
    endpoint = f"GET /api/v1/document-types/{base_document_type_id}"
    response = api_client.get(f"/api/v1/document-types/{base_document_type_id}")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert body.get("document_type_id") == base_document_type_id or body.get("id") == base_document_type_id


def test_patch_document_type(api_client: httpx.Client, base_document_type_id: str) -> None:
    endpoint = f"PATCH /api/v1/document-types/{base_document_type_id}"
    response = api_client.patch(
        f"/api/v1/document-types/{base_document_type_id}",
        json={"description": "updated by api test"},
    )
    assert_status(response, {200, 403}, endpoint=endpoint)
