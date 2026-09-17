"""Repository registration and activation endpoints."""

from __future__ import annotations

import httpx
import pytest

from api.helpers import assert_status


def test_submit_repository_registration_request(api_client: httpx.Client, unique_name: str) -> None:
    endpoint = "POST /api/v1/repositories/requests"
    response = api_client.post(
        "/api/v1/repositories/requests",
        json={
            "proposed_name": f"req-{unique_name}",
            "description": "one-off registration test",
        },
    )
    assert_status(response, 201, endpoint=endpoint)


def test_list_repository_registration_requests(api_client: httpx.Client) -> None:
    endpoint = "GET /api/v1/repositories/requests"
    response = api_client.get("/api/v1/repositories/requests")
    assert_status(response, 200, endpoint=endpoint)


def test_approve_repository_registration_request(
    api_client: httpx.Client,
    registration_request: dict,
    admin_credentials: tuple[str, str],
) -> None:
    owner, _ = admin_credentials
    endpoint = f"POST /api/v1/repositories/requests/{registration_request['id']}/approve"
    response = api_client.post(
        f"/api/v1/repositories/requests/{registration_request['id']}/approve",
        json={"owner_user_id": owner, "review_notes": "approved by api test"},
    )
    assert_status(response, 200, endpoint=endpoint)


def test_activate_repository(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    assert created_repository["raw"]["status"] == "configuring"

    endpoint = f"POST /api/v1/repositories/{repo_id}/activate"
    response = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert body["status"] == "active"
    assert body.get("settings_locked") is True
    assert body.get("settings_locked_at")

    get_resp = api_client.get(f"/api/v1/repositories/{repo_id}")
    assert_status(get_resp, 200, endpoint=f"GET /api/v1/repositories/{repo_id}")
    assert get_resp.json()["status"] == "active"

    # Idempotent when already active
    again = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(again, 200, endpoint=endpoint)
    assert again.json()["status"] == "active"


def test_activate_repository_from_approved_registration(
    api_client: httpx.Client,
    registration_request: dict,
    admin_credentials: tuple[str, str],
) -> None:
    owner, _ = admin_credentials
    approve_endpoint = (
        f"POST /api/v1/repositories/requests/{registration_request['id']}/approve"
    )
    approve = api_client.post(
        f"/api/v1/repositories/requests/{registration_request['id']}/approve",
        json={"owner_user_id": owner, "review_notes": "activate path test"},
    )
    assert_status(approve, 200, endpoint=approve_endpoint)
    repo = approve.json()["repository"]
    repo_id = repo["repository_id"]
    assert repo["status"] == "configuring"

    settings_endpoint = f"PATCH /api/v1/repositories/{repo_id}/settings"
    settings = api_client.patch(
        f"/api/v1/repositories/{repo_id}/settings",
        json={
            "chunking_strategy": "section-based",
            "chunk_size": 512,
            "chunk_overlap": 2,
            "embedding_model": {
                "provider": "local",
                "model_id": "all-MiniLM-L6-v2",
                "local_model_dir": "models--sentence-transformers--all-MiniLM-L6-v2",
            },
        },
    )
    assert_status(settings, 200, endpoint=settings_endpoint)

    activate_endpoint = f"POST /api/v1/repositories/{repo_id}/activate"
    activate = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(activate, 200, endpoint=activate_endpoint)
    assert activate.json()["status"] == "active"
