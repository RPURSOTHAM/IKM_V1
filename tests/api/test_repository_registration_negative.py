"""Negative-path tests for repository registration and activation."""

from __future__ import annotations

import uuid

import httpx
import pytest

from api.helpers import assert_error, assert_status


@pytest.fixture
def bare_repository(
    api_client: httpx.Client,
    unique_name: str,
    admin_credentials: tuple[str, str],
) -> dict:
    owner, _ = admin_credentials
    response = api_client.post(
        "/api/v1/repositories",
        json={"name": unique_name, "owner_user_id": owner},
    )
    assert_status(response, 201, endpoint="POST /api/v1/repositories")
    repo_id = response.json()["repository_id"]
    return {"id": repo_id, "name": unique_name}


def test_activate_repository_not_found(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"POST /api/v1/repositories/{missing_id}/activate"
    response = api_client.post(f"/api/v1/repositories/{missing_id}/activate")
    assert_error(response, 404, endpoint=endpoint, code="not_found")


def test_activate_repository_incomplete_configuration(
    api_client: httpx.Client,
    bare_repository: dict,
) -> None:
    repo_id = bare_repository["id"]
    endpoint = f"POST /api/v1/repositories/{repo_id}/activate"
    response = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    error = assert_error(
        response,
        422,
        endpoint=endpoint,
        code="validation_error",
        message_contains="incomplete",
    )
    missing = (error.get("details") or {}).get("missing")
    assert isinstance(missing, list)
    assert "embedding_model" in missing

    api_client.delete(f"/api/v1/repositories/{repo_id}")


def test_activate_repository_invalid_status_archived(
    api_client: httpx.Client,
    created_repository: dict,
) -> None:
    repo_id = created_repository["id"]
    archive = api_client.patch(
        f"/api/v1/repositories/{repo_id}/settings",
        json={"status": "archived"},
    )
    assert_status(archive, 200, endpoint=f"PATCH /api/v1/repositories/{repo_id}/settings")

    endpoint = f"POST /api/v1/repositories/{repo_id}/activate"
    response = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_error(
        response,
        422,
        endpoint=endpoint,
        code="validation_error",
        message_contains="cannot be activated",
        details_key="status",
    )


def test_approve_registration_request_twice(
    api_client: httpx.Client,
    registration_request: dict,
    admin_credentials: tuple[str, str],
) -> None:
    owner, _ = admin_credentials
    request_id = registration_request["id"]
    approve_endpoint = f"POST /api/v1/repositories/requests/{request_id}/approve"
    first = api_client.post(
        f"/api/v1/repositories/requests/{request_id}/approve",
        json={"owner_user_id": owner, "review_notes": "first approval"},
    )
    assert_status(first, 200, endpoint=approve_endpoint)

    second = api_client.post(
        f"/api/v1/repositories/requests/{request_id}/approve",
        json={"owner_user_id": owner, "review_notes": "duplicate approval"},
    )
    assert_error(
        second,
        422,
        endpoint=approve_endpoint,
        code="validation_error",
        message_contains="not pending",
    )

    repo_id = first.json()["repository"]["repository_id"]
    api_client.delete(f"/api/v1/repositories/{repo_id}")


def test_reject_registration_request_after_approval(
    api_client: httpx.Client,
    registration_request: dict,
    admin_credentials: tuple[str, str],
) -> None:
    owner, _ = admin_credentials
    request_id = registration_request["id"]
    approve = api_client.post(
        f"/api/v1/repositories/requests/{request_id}/approve",
        json={"owner_user_id": owner},
    )
    assert_status(approve, 200, endpoint=f"POST .../requests/{request_id}/approve")
    repo_id = approve.json()["repository"]["repository_id"]

    reject_endpoint = f"POST /api/v1/repositories/requests/{request_id}/reject"
    reject = api_client.post(
        f"/api/v1/repositories/requests/{request_id}/reject",
        json={"review_notes": "too late"},
    )
    assert_error(
        reject,
        422,
        endpoint=reject_endpoint,
        code="validation_error",
        message_contains="not pending",
    )

    api_client.delete(f"/api/v1/repositories/{repo_id}")


def test_approve_registration_request_not_found(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = f"POST /api/v1/repositories/requests/{missing_id}/approve"
    response = api_client.post(
        f"/api/v1/repositories/requests/{missing_id}/approve",
        json={"owner_user_id": "admin"},
    )
    assert_error(response, 404, endpoint=endpoint, code="not_found")
