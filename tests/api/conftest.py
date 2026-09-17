from __future__ import annotations

import os
import uuid
from collections.abc import Generator

import httpx
import pytest

from api.helpers import assert_status

DEFAULT_BASE_URL = "http://localhost:8088"
DEFAULT_USER = "admin"
DEFAULT_PASSWORD = os.getenv("DMS_API_PASSWORD", "changeme-local-admin")


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.getenv("DMS_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


@pytest.fixture(scope="session")
def admin_credentials() -> tuple[str, str]:
    user = os.getenv("DMS_API_USER", DEFAULT_USER)
    password = os.getenv("DMS_API_PASSWORD", DEFAULT_PASSWORD)
    return user, password


@pytest.fixture(scope="session")
def admin_token(base_url: str, admin_credentials: tuple[str, str]) -> str:
    user, password = admin_credentials
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        response = client.post(
            "/api/v1/auth/token",
            json={"user_id": user, "password": password},
        )
        if response.status_code != 200:
            pytest.fail(
                f"Authentication failed ({response.status_code}): {response.text}"
            )
        token = response.json().get("access_token")
        if not token:
            pytest.fail("Authentication response missing access_token")
        return token


@pytest.fixture(scope="session")
def api_client(base_url: str, admin_token: str) -> Generator[httpx.Client, None, None]:
    headers = {"Authorization": f"Bearer {admin_token}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        yield client


@pytest.fixture
def unique_name() -> str:
    return f"api-test-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def registration_request(api_client: httpx.Client, unique_name: str) -> dict:
    endpoint = "POST /api/v1/repositories/requests"
    response = api_client.post(
        "/api/v1/repositories/requests",
        json={
            "proposed_name": f"req-{unique_name}",
            "description": "API integration test registration",
        },
    )
    assert_status(response, 201, endpoint=endpoint)
    body = response.json()
    request_id = body.get("request_id") or body.get("id")
    assert request_id
    return {"id": request_id, "proposed_name": f"req-{unique_name}"}


@pytest.fixture
def created_repository(
    api_client: httpx.Client,
    unique_name: str,
    admin_credentials: tuple[str, str],
) -> Generator[dict, None, None]:
    owner, _ = admin_credentials
    endpoint = "POST /api/v1/repositories"
    response = api_client.post(
        "/api/v1/repositories",
        json={
            "name": unique_name,
            "owner_user_id": owner,
            "settings": {
                "chunking_strategy": "section-based",
                "chunk_size": 512,
                "chunk_overlap": 2,
                "embedding_model": {
                    "provider": "local",
                    "model_id": "all-MiniLM-L6-v2",
                    "local_model_dir": "models--sentence-transformers--all-MiniLM-L6-v2",
                },
            },
        },
    )
    assert_status(response, 201, endpoint=endpoint)
    repo = response.json()
    repo_id = repo.get("repository_id") or repo.get("id")
    assert repo_id, "create response missing repository_id"
    yield {"id": repo_id, "name": unique_name, "raw": repo}
    delete = api_client.delete(f"/api/v1/repositories/{repo_id}")
    if delete.status_code not in {200, 204, 404, 409}:
        pytest.fail(
            f"DELETE /api/v1/repositories/{{id}} cleanup failed: {delete.status_code} {delete.text[:300]}"
        )


@pytest.fixture(scope="module")
def base_document_type_id(api_client: httpx.Client) -> str:
    response = api_client.get("/api/v1/document-types")
    assert_status(response, 200, endpoint="GET /api/v1/document-types")
    types = response.json().get("document_types") or []
    assert types, "expected at least one seeded document type"
    base = next((t for t in types if t.get("is_system")), types[0])
    type_id = base.get("document_type_id") or base.get("id")
    assert type_id
    return type_id


@pytest.fixture
def child_document_type(
    api_client: httpx.Client,
    base_document_type_id: str,
    unique_name: str,
) -> Generator[dict, None, None]:
    endpoint = f"POST /api/v1/document-types/{base_document_type_id}/children"
    response = api_client.post(
        f"/api/v1/document-types/{base_document_type_id}/children",
        json={
            "name": f"Child {unique_name}",
            "code": f"child_{unique_name[:8]}",
            "description": "API test child type",
        },
    )
    assert_status(response, 201, endpoint=endpoint)
    body = response.json()
    child_id = body.get("document_type_id") or body.get("id")
    assert child_id
    yield {"id": child_id, "parent_id": base_document_type_id}
    delete = api_client.delete(f"/api/v1/document-types/{child_id}")
    if delete.status_code not in {204, 404, 409}:
        pytest.fail(f"DELETE child cleanup failed: {delete.status_code} {delete.text[:300]}")
