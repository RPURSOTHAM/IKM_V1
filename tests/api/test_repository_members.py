"""Repository access and member endpoints."""

from __future__ import annotations

import httpx

from api.helpers import assert_status


def test_list_repository_members(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    endpoint = f"GET /api/v1/repositories/{repo_id}/members"
    response = api_client.get(f"/api/v1/repositories/{repo_id}/members")
    assert_status(response, 200, endpoint=endpoint)


def test_grant_repository_consumer_role(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    endpoint = f"POST /api/v1/repositories/{repo_id}/grants"
    response = api_client.post(
        f"/api/v1/repositories/{repo_id}/grants",
        json={"user_id": "api-test-consumer", "role": "consumer"},
    )
    assert_status(response, {201, 403, 409}, endpoint=endpoint)


def test_revoke_repository_grant(api_client: httpx.Client, created_repository: dict) -> None:
    repo_id = created_repository["id"]
    api_client.post(
        f"/api/v1/repositories/{repo_id}/grants",
        json={"user_id": "api-test-consumer", "role": "consumer"},
    )
    endpoint = f"DELETE /api/v1/repositories/{repo_id}/grants/api-test-consumer"
    response = api_client.delete(
        f"/api/v1/repositories/{repo_id}/grants/api-test-consumer",
        params={"role": "consumer"},
    )
    assert_status(response, {200, 204, 404}, endpoint=endpoint)
