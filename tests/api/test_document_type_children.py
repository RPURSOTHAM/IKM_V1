"""Document type child hierarchy endpoints."""

from __future__ import annotations

import httpx
from api.helpers import assert_status


def test_list_document_type_children(api_client: httpx.Client, base_document_type_id: str) -> None:
    endpoint = f"GET /api/v1/document-types/{base_document_type_id}/children"
    response = api_client.get(f"/api/v1/document-types/{base_document_type_id}/children")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert "children" in body


def test_create_document_type_child(api_client: httpx.Client, child_document_type: dict) -> None:
    assert child_document_type["id"]


def test_update_document_type_child(api_client: httpx.Client, child_document_type: dict) -> None:
    child_id = child_document_type["id"]
    endpoint = f"PATCH /api/v1/document-types/{child_id}"
    response = api_client.patch(
        f"/api/v1/document-types/{child_id}",
        json={"description": "child updated by api test"},
    )
    assert_status(response, 200, endpoint=endpoint)


def test_delete_document_type_child(api_client: httpx.Client, base_document_type_id: str, unique_name: str) -> None:
    create = api_client.post(
        f"/api/v1/document-types/{base_document_type_id}/children",
        json={"name": f"DeleteMe {unique_name}", "code": f"del_{unique_name[:6]}"},
    )
    assert_status(create, 201, endpoint=f"POST /api/v1/document-types/{base_document_type_id}/children")
    child_id = create.json().get("document_type_id") or create.json().get("id")
    endpoint = f"DELETE /api/v1/document-types/{child_id}"
    response = api_client.delete(f"/api/v1/document-types/{child_id}")
    assert_status(response, 204, endpoint=endpoint)
