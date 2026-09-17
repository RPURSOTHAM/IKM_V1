"""Negative-path tests for repository access endpoints."""

from __future__ import annotations

import uuid

import httpx

from api.helpers import assert_error, assert_status


def test_grant_invalid_role(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    endpoint = f"POST /api/v1/repositories/{repo_id}/grants"
    response = api_client.post(
        f"/api/v1/repositories/{repo_id}/grants",
        json={"user_id": "api-test-user", "role": "owner"},
    )
    assert_error(response, 422, endpoint=endpoint, code="validation_error")


def test_revoke_nonexistent_grant(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    endpoint = f"DELETE /api/v1/repositories/{repo_id}/grants/no-such-user"
    response = api_client.delete(
        f"/api/v1/repositories/{repo_id}/grants/no-such-user",
        params={"role": "consumer"},
    )
    assert_error(response, 404, endpoint=endpoint, code="not_found")


def test_list_members_repository_not_found(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"GET /api/v1/repositories/{missing_id}/members"
    response = api_client.get(f"/api/v1/repositories/{missing_id}/members")
    assert_error(response, 404, endpoint=endpoint, code="not_found")
