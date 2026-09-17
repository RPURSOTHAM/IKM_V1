"""Repository settings endpoint."""

from __future__ import annotations

import httpx

from api.helpers import assert_status


def test_patch_repository_settings(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    endpoint = f"PATCH /api/v1/repositories/{repo_id}/settings"
    response = api_client.patch(
        f"/api/v1/repositories/{repo_id}/settings",
        json={
            "chunk_size": 600,
            "chunk_overlap": 2,
            "reranking": False,
            "lexical_composition": True,
        },
    )
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    settings = body.get("settings") or body
    assert settings.get("chunk_size") == 600 or body.get("chunk_size") == 600


def test_get_put_repository_settings_with_key_field_extraction_enabled(
    api_client: httpx.Client,
    created_repository: dict,
) -> None:
    repo_id = created_repository["id"]
    put_endpoint = f"PUT /api/v1/repositories/{repo_id}/settings"
    put_response = api_client.put(
        f"/api/v1/repositories/{repo_id}/settings",
        json={
            "chunk_size": 600,
            "chunk_overlap": 2,
            "key_field_extraction_enabled": True,
            "key_fields": [{"name": "document_number", "type": "string", "required": True}],
        },
    )
    assert_status(put_response, 200, endpoint=put_endpoint)
    put_settings = (put_response.json().get("settings") or {})
    assert put_settings.get("key_field_extraction_enabled") is True
    assert put_settings.get("key_field_extraction") is True
    assert len(put_settings.get("key_fields") or []) == 1

    get_endpoint = f"GET /api/v1/repositories/{repo_id}/settings"
    get_response = api_client.get(f"/api/v1/repositories/{repo_id}/settings")
    assert_status(get_response, 200, endpoint=get_endpoint)
    get_settings = (get_response.json().get("settings") or {})
    assert get_settings.get("key_field_extraction_enabled") is True


def test_repository_key_field_crud_endpoints(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    create_endpoint = f"POST /api/v1/repositories/{repo_id}/key-fields"
    create_response = api_client.post(
        f"/api/v1/repositories/{repo_id}/key-fields",
        json={"name": "effective_date", "type": "date", "required": False},
    )
    assert_status(create_response, 200, endpoint=create_endpoint)
    fields = (create_response.json().get("settings") or {}).get("key_fields") or []
    created = next(item for item in fields if item.get("name") == "effective_date")
    field_id = created.get("field_id")
    assert field_id

    list_endpoint = f"GET /api/v1/repositories/{repo_id}/key-fields"
    list_response = api_client.get(f"/api/v1/repositories/{repo_id}/key-fields")
    assert_status(list_response, 200, endpoint=list_endpoint)
    listed = list_response.json().get("key_fields") or []
    assert any(item.get("field_id") == field_id for item in listed)

    delete_endpoint = f"DELETE /api/v1/repositories/{repo_id}/key-fields/{field_id}"
    delete_response = api_client.delete(f"/api/v1/repositories/{repo_id}/key-fields/{field_id}")
    assert_status(delete_response, 200, endpoint=delete_endpoint)
