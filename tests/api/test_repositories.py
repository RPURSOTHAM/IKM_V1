"""Repository CRUD endpoints."""

from __future__ import annotations

import httpx

from api.helpers import assert_status


def test_list_repositories(api_client: httpx.Client) -> None:
    endpoint = "GET /api/v1/repositories"
    response = api_client.get("/api/v1/repositories")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert "repositories" in body
    assert "count" in body


def test_create_repository(api_client: httpx.Client, unique_name: str, admin_credentials: tuple[str, str]) -> None:
    owner, _ = admin_credentials
    endpoint = "POST /api/v1/repositories"
    response = api_client.post(
        "/api/v1/repositories",
        json={"name": unique_name, "owner_user_id": owner},
    )
    assert_status(response, 201, endpoint=endpoint)
    repo_id = response.json().get("repository_id") or response.json().get("id")
    assert repo_id
    cleanup = api_client.delete(f"/api/v1/repositories/{repo_id}")
    assert cleanup.status_code in {200, 204, 409}


def test_get_repository(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    endpoint = f"GET /api/v1/repositories/{repo_id}"
    response = api_client.get(f"/api/v1/repositories/{repo_id}")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert body.get("repository_id") == repo_id or body.get("id") == repo_id


def test_delete_repository(api_client: httpx.Client, unique_name: str, admin_credentials: tuple[str, str]) -> None:
    owner, _ = admin_credentials
    create = api_client.post(
        "/api/v1/repositories",
        json={"name": unique_name, "owner_user_id": owner},
    )
    assert_status(create, 201, endpoint="POST /api/v1/repositories")
    repo_id = create.json().get("repository_id") or create.json().get("id")
    endpoint = f"DELETE /api/v1/repositories/{repo_id}"
    response = api_client.delete(f"/api/v1/repositories/{repo_id}")
    assert response.status_code in {200, 204}, f"{endpoint} expected 200/204 got {response.status_code}"


def test_patch_update_archive_reactivate_repository(
    api_client: httpx.Client,
    created_repository: dict,
) -> None:
    repo_id = created_repository["id"]

    patch = api_client.patch(
        f"/api/v1/repositories/{repo_id}",
        json={"owner_user_name": "API Owner"},
    )
    assert_status(patch, 200, endpoint=f"PATCH /api/v1/repositories/{repo_id}")
    assert patch.json().get("owner_user_name") == "API Owner"

    activate = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(activate, 200, endpoint=f"POST /api/v1/repositories/{repo_id}/activate")
    assert activate.json()["status"] == "active"

    archive = api_client.post(f"/api/v1/repositories/{repo_id}/archive")
    assert_status(archive, 200, endpoint=f"POST /api/v1/repositories/{repo_id}/archive")
    assert archive.json()["status"] == "archived"

    reactivate = api_client.post(f"/api/v1/repositories/{repo_id}/reactivate")
    assert_status(reactivate, 200, endpoint=f"POST /api/v1/repositories/{repo_id}/reactivate")
    assert reactivate.json()["status"] == "active"
