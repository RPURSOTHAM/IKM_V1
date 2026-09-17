"""Document type metadata field endpoints."""

from __future__ import annotations

import httpx
import pytest

from api.helpers import assert_status


@pytest.fixture
def metadata_field(api_client: httpx.Client, child_document_type: dict, unique_name: str) -> dict:
    type_id = child_document_type["id"]
    field_name = f"testfield{unique_name.replace('-', '')[:8]}"
    endpoint = f"POST /api/v1/document-types/{type_id}/metadata-fields"
    response = api_client.post(
        f"/api/v1/document-types/{type_id}/metadata-fields",
        json={
            "field_name": field_name,
            "display_label": "Test Field",
            "data_type": "string",
            "required": False,
            "field_group": "general",
            "group_label": "General",
        },
    )
    assert_status(response, 201, endpoint=endpoint)
    body = response.json()
    field_id = body.get("metadata_field_id") or body.get("id")
    assert field_id
    yield {"id": field_id, "type_id": type_id, "field_name": field_name}
    delete = api_client.delete(f"/api/v1/document-types/{type_id}/metadata-fields/{field_id}")
    if delete.status_code not in {204, 404, 409}:
        pytest.fail(f"DELETE metadata field cleanup failed: {delete.status_code}")


def test_create_metadata_field(api_client: httpx.Client, metadata_field: dict) -> None:
    assert metadata_field["id"]


def test_patch_metadata_field(api_client: httpx.Client, metadata_field: dict) -> None:
    type_id = metadata_field["type_id"]
    field_id = metadata_field["id"]
    endpoint = f"PATCH /api/v1/document-types/{type_id}/metadata-fields/{field_id}"
    response = api_client.patch(
        f"/api/v1/document-types/{type_id}/metadata-fields/{field_id}",
        json={"display_label": "Updated Test Field", "required": True},
    )
    assert_status(response, 200, endpoint=endpoint)


def test_delete_metadata_field(
    api_client: httpx.Client,
    child_document_type: dict,
    unique_name: str,
) -> None:
    type_id = child_document_type["id"]
    field_name = f"delfield{unique_name.replace('-', '')[:8]}"
    create = api_client.post(
        f"/api/v1/document-types/{type_id}/metadata-fields",
        json={
            "field_name": field_name,
            "display_label": "Delete Me",
            "data_type": "string",
        },
    )
    assert_status(create, 201, endpoint=f"POST /api/v1/document-types/{type_id}/metadata-fields")
    field_id = create.json().get("metadata_field_id") or create.json().get("id")
    endpoint = f"DELETE /api/v1/document-types/{type_id}/metadata-fields/{field_id}"
    response = api_client.delete(f"/api/v1/document-types/{type_id}/metadata-fields/{field_id}")
    assert_status(response, 204, endpoint=endpoint)
