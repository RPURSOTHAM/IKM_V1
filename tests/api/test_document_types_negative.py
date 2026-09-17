"""Negative-path tests for document type endpoints."""

from __future__ import annotations

import uuid

import httpx

from api.helpers import assert_error, assert_status


def test_get_document_type_not_found(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"GET /api/v1/document-types/{missing_id}"
    response = api_client.get(f"/api/v1/document-types/{missing_id}")
    assert_error(response, 404, endpoint=endpoint, code="not_found")


def test_create_metadata_field_invalid_name(
    api_client: httpx.Client,
    child_document_type: dict,
) -> None:
    type_id = child_document_type["id"]
    endpoint = f"POST /api/v1/document-types/{type_id}/metadata-fields"
    response = api_client.post(
        f"/api/v1/document-types/{type_id}/metadata-fields",
        json={
            "field_name": "invalid-field-name",
            "display_label": "Bad Field",
            "data_type": "string",
        },
    )
    assert_error(
        response,
        400,
        endpoint=endpoint,
        code="invalid_request",
        message_contains="field_name",
    )


def test_create_metadata_field_on_missing_type(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"POST /api/v1/document-types/{missing_id}/metadata-fields"
    response = api_client.post(
        f"/api/v1/document-types/{missing_id}/metadata-fields",
        json={
            "field_name": "valid_field",
            "display_label": "Field",
            "data_type": "string",
        },
    )
    assert_error(response, 404, endpoint=endpoint, code="not_found")


def test_delete_system_document_type_forbidden(
    api_client: httpx.Client,
    base_document_type_id: str,
) -> None:
    endpoint = f"DELETE /api/v1/document-types/{base_document_type_id}"
    response = api_client.delete(f"/api/v1/document-types/{base_document_type_id}")
    assert_error(response, 403, endpoint=endpoint, code="system_type_immutable")


def test_create_child_on_missing_parent(api_client: httpx.Client, unique_name: str) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"POST /api/v1/document-types/{missing_id}/children"
    response = api_client.post(
        f"/api/v1/document-types/{missing_id}/children",
        json={"name": f"Child {unique_name}", "code": f"child_{unique_name[:6]}"},
    )
    assert_error(response, 404, endpoint=endpoint, code="not_found")


def test_patch_metadata_field_not_found(
    api_client: httpx.Client,
    child_document_type: dict,
) -> None:
    type_id = child_document_type["id"]
    missing_field_id = str(uuid.uuid4())
    endpoint = f"PATCH /api/v1/document-types/{type_id}/metadata-fields/{missing_field_id}"
    response = api_client.patch(
        f"/api/v1/document-types/{type_id}/metadata-fields/{missing_field_id}",
        json={"display_label": "Ghost Field"},
    )
    assert_error(response, 404, endpoint=endpoint, code="not_found")
