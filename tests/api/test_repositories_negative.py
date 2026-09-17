"""Negative-path tests for repository CRUD and settings endpoints."""

from __future__ import annotations

import uuid

import httpx
import pytest

from api.helpers import assert_error, assert_status


def test_get_repository_not_found(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"GET /api/v1/repositories/{missing_id}"
    response = api_client.get(f"/api/v1/repositories/{missing_id}")
    assert_error(response, 404, endpoint=endpoint, code="not_found")


def test_get_repository_invalid_uuid(api_client: httpx.Client) -> None:
    endpoint = "GET /api/v1/repositories/not-a-uuid"
    response = api_client.get("/api/v1/repositories/not-a-uuid")
    assert_error(response, 422, endpoint=endpoint, code="validation_error")


def test_create_repository_missing_owner(api_client: httpx.Client, unique_name: str) -> None:
    endpoint = "POST /api/v1/repositories"
    response = api_client.post("/api/v1/repositories", json={"name": unique_name})
    assert_error(response, 422, endpoint=endpoint, code="validation_error")


def test_create_repository_duplicate_name(
    api_client: httpx.Client,
    unique_name: str,
    admin_credentials: tuple[str, str],
) -> None:
    owner, _ = admin_credentials
    payload = {"name": unique_name, "owner_user_id": owner}
    first = api_client.post("/api/v1/repositories", json=payload)
    assert_status(first, 201, endpoint="POST /api/v1/repositories")
    repo_id = first.json()["repository_id"]

    endpoint = "POST /api/v1/repositories (duplicate)"
    second = api_client.post("/api/v1/repositories", json=payload)
    assert_error(second, 409, endpoint=endpoint, code="duplicate_name")

    api_client.delete(f"/api/v1/repositories/{repo_id}")


def test_delete_repository_not_found(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"DELETE /api/v1/repositories/{missing_id}"
    response = api_client.delete(f"/api/v1/repositories/{missing_id}")
    assert_error(response, 404, endpoint=endpoint, code="not_found")


def test_patch_settings_on_active_repository_locked(
    api_client: httpx.Client,
    created_repository: dict,
) -> None:
    repo_id = created_repository["id"]
    activate = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(activate, 200, endpoint=f"POST /api/v1/repositories/{repo_id}/activate")

    endpoint = f"PATCH /api/v1/repositories/{repo_id}/settings"
    response = api_client.patch(
        f"/api/v1/repositories/{repo_id}/settings",
        json={"chunk_size": 700},
    )
    assert_error(response, 409, endpoint=endpoint, code="settings_read_only")


def test_patch_settings_invalid_chunk_size(
    api_client: httpx.Client,
    created_repository: dict,
) -> None:
    repo_id = created_repository["id"]
    endpoint = f"PATCH /api/v1/repositories/{repo_id}/settings"
    response = api_client.patch(
        f"/api/v1/repositories/{repo_id}/settings",
        json={"chunk_size": 50},
    )
    assert_error(response, 422, endpoint=endpoint, code="validation_error")


def test_unauthenticated_request_rejected(base_url: str) -> None:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        response = client.get("/api/v1/repositories")
    assert_error(response, 401, endpoint="GET /api/v1/repositories", code="unauthorized")
