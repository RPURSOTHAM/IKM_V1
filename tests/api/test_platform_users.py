"""Admin-only platform user and repository-role HTTP endpoints."""

from __future__ import annotations

import uuid

import httpx

from api.helpers import assert_status


def test_create_platform_user_and_list(api_client: httpx.Client) -> None:
    user_id = f"api-user-{uuid.uuid4().hex[:8]}"
    create = api_client.post(
        "/api/v1/platform/users",
        json={
            "user_id": user_id,
            "password": "IsolationA1!",
            "display_name": "API Test User",
        },
    )
    assert_status(create, 201, endpoint="POST /api/v1/platform/users")
    body = create.json()
    assert body.get("user_id") == user_id
    assert "password" not in body
    assert "password_hash" not in str(body).lower()

    listed = api_client.get("/api/v1/platform/users")
    assert_status(listed, 200, endpoint="GET /api/v1/platform/users")
    users = listed.json().get("users") or []
    assert any(u.get("user_id") == user_id for u in users)
    assert all("password_hash" not in u for u in users)


def test_non_admin_cannot_create_user(
    api_client: httpx.Client,
    base_url: str,
    created_repository: dict,
) -> None:
    user_id = f"api-restricted-{uuid.uuid4().hex[:8]}"
    created = api_client.post(
        "/api/v1/platform/users",
        json={"user_id": user_id, "password": "IsolationA1!", "display_name": "Restricted"},
    )
    assert_status(created, 201, endpoint="POST /api/v1/platform/users")
    repo_id = created_repository["id"]
    grant = api_client.post(
        f"/api/v1/repositories/{repo_id}/access",
        json={"user_id": user_id, "role": "contributor"},
    )
    assert_status(grant, {201, 200}, endpoint=f"POST /api/v1/repositories/{repo_id}/access")
    assert "password_hash" not in str(grant.json()).lower()

    token_resp = httpx.post(
        f"{base_url}/api/v1/auth/token",
        json={"user_id": user_id, "password": "IsolationA1!"},
        timeout=30.0,
    )
    assert_status(token_resp, 200, endpoint="POST /api/v1/auth/token")
    token = token_resp.json()["access_token"]
    with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0) as client:
        denied = client.post(
            "/api/v1/platform/users",
            json={"user_id": f"blocked-{uuid.uuid4().hex[:6]}", "password": "IsolationB1!"},
        )
    assert denied.status_code == 403


def test_revoke_repository_access(api_client: httpx.Client, created_repository: dict) -> None:
    user_id = f"api-revoke-{uuid.uuid4().hex[:8]}"
    api_client.post(
        "/api/v1/platform/users",
        json={"user_id": user_id, "password": "IsolationA1!"},
    )
    repo_id = created_repository["id"]
    api_client.post(
        f"/api/v1/repositories/{repo_id}/access",
        json={"user_id": user_id, "role": "contributor"},
    )
    revoked = api_client.delete(f"/api/v1/repositories/{repo_id}/access/{user_id}")
    assert_status(revoked, {200, 204}, endpoint=f"DELETE /api/v1/repositories/{repo_id}/access/{user_id}")
    assert revoked.json().get("revoked") is True
