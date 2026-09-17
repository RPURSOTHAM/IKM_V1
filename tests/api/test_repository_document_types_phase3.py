"""API tests for repository-scoped document type nested CRUD (Phase 3)."""

from __future__ import annotations

import uuid

import httpx
from api.helpers import assert_status


def _unique(prefix: str = "dt") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _create_repository(api_client: httpx.Client) -> dict:
    response = api_client.post(
        "/api/v1/repositories",
        json={"name": _unique("repo"), "owner_user_id": "admin", "status": "configuring"},
    )
    assert_status(response, {200, 201}, endpoint="POST /api/v1/repositories")
    return response.json()


def test_repository_document_types_nested_crud(api_client: httpx.Client) -> None:
    repo = _create_repository(api_client)
    repository_id = repo["repository_id"]
    basic_id = (repo.get("settings") or {}).get("document_type_id")
    assert basic_id

    list_resp = api_client.get(f"/api/v1/repositories/{repository_id}/document-types")
    assert_status(list_resp, 200, endpoint="GET repository document-types")
    body = list_resp.json()
    assert body["count"] >= 1
    assert any(item.get("name") == "Basic" for item in body.get("document_types") or [])

    create_resp = api_client.post(
        f"/api/v1/repositories/{repository_id}/document-types",
        json={"name": "SOP", "parent_document_type_id": basic_id},
    )
    assert_status(create_resp, 201, endpoint="POST repository document-types")
    sop = create_resp.json()
    sop_id = sop["document_type_id"]

    get_resp = api_client.get(f"/api/v1/repositories/{repository_id}/document-types/{sop_id}")
    assert_status(get_resp, 200, endpoint="GET repository document-type")
    assert get_resp.json()["name"] == "SOP"

    patch_resp = api_client.patch(
        f"/api/v1/repositories/{repository_id}/document-types/{sop_id}",
        json={"description": "Updated by Phase 3 API test"},
    )
    assert_status(patch_resp, 200, endpoint="PATCH repository document-type")
    assert patch_resp.json().get("description") == "Updated by Phase 3 API test"

    # Basic cannot be deleted
    delete_basic = api_client.delete(f"/api/v1/repositories/{repository_id}/document-types/{basic_id}")
    assert_status(delete_basic, {403, 409}, endpoint="DELETE Basic document-type")

    delete_sop = api_client.delete(f"/api/v1/repositories/{repository_id}/document-types/{sop_id}")
    assert_status(delete_sop, 204, endpoint="DELETE repository document-type")


def test_patch_default_document_type_on_active_repository(api_client: httpx.Client) -> None:
    repo = _create_repository(api_client)
    repository_id = repo["repository_id"]
    basic_id = (repo.get("settings") or {}).get("document_type_id")
    assert basic_id

    create_resp = api_client.post(
        f"/api/v1/repositories/{repository_id}/document-types",
        json={"name": "SOP", "parent_document_type_id": basic_id},
    )
    assert_status(create_resp, 201, endpoint="POST repository document-types")
    sop_id = create_resp.json()["document_type_id"]

    activate = api_client.post(f"/api/v1/repositories/{repository_id}/activate")
    assert_status(activate, 200, endpoint="POST /api/v1/repositories/{id}/activate")

    locked_patch = api_client.patch(
        f"/api/v1/repositories/{repository_id}/settings",
        json={"chunk_size": 700},
    )
    assert locked_patch.status_code == 409

    default_patch = api_client.patch(
        f"/api/v1/repositories/{repository_id}/settings",
        json={"document_type_id": sop_id},
    )
    assert_status(default_patch, 200, endpoint="PATCH repository default document_type_id")
    settings = (default_patch.json().get("settings") or {})
    assert settings.get("document_type_id") == sop_id
