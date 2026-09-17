"""Integration coverage for Phase 2 runtime gaps."""

from __future__ import annotations

import uuid
from collections.abc import Generator

import httpx
import pytest

from api.helpers import assert_status


def _list_repositories(api_client: httpx.Client, *, status: str | None = None) -> list[dict]:
    response = api_client.get("/api/v1/repositories", params={"status": status} if status else None)
    assert_status(response, 200, endpoint="GET /api/v1/repositories")
    return response.json().get("repositories") or []


@pytest.fixture
def phase2_repo(
    api_client: httpx.Client,
    unique_name: str,
) -> Generator[dict, None, None]:
    repositories = _list_repositories(api_client, status="configuring")
    candidate = None
    for repo in repositories:
        if bool(repo.get("settings_locked")):
            continue
        repo_id = str(repo.get("repository_id") or "")
        if not repo_id:
            continue
        settings_response = api_client.get(f"/api/v1/repositories/{repo_id}/settings")
        if settings_response.status_code != 200:
            continue
        settings = settings_response.json().get("settings") or {}
        chunk_size = settings.get("chunk_size")
        if isinstance(chunk_size, int) and 200 <= chunk_size <= 2000:
            candidate = repo
            break
    if candidate is None:
        pytest.skip("No mutable configuring repository with valid settings available for Phase 2 gap tests.")
    yield {"repository_id": candidate["repository_id"], "payload": candidate, "unique_name": unique_name}


def test_patch_repository_key_fields_endpoint(phase2_repo: dict, api_client: httpx.Client) -> None:
    repo_id = phase2_repo["repository_id"]
    endpoint = f"PATCH /api/v1/repositories/{repo_id}/key-fields"
    response = api_client.patch(
        f"/api/v1/repositories/{repo_id}/key-fields",
        json={
            "key_fields": [
                {"name": "document_number", "type": "string", "required": True},
                {"name": "effective_date", "type": "date", "required": False},
            ]
        },
    )
    assert_status(response, 200, endpoint=endpoint)
    key_fields = response.json().get("key_fields") or []
    assert len(key_fields) == 2
    assert {item.get("name") for item in key_fields} == {"document_number", "effective_date"}


def test_inactive_repository_returns_structured_error(phase2_repo: dict, api_client: httpx.Client) -> None:
    repo_id = phase2_repo["repository_id"]
    endpoint = "POST /api/v1/documents/upload (inactive repository)"
    response = api_client.post(
        "/api/v1/documents/upload",
        data={"repository_id": repo_id, "submit_for_processing": "false"},
        files=[("files", ("inactive-upload.txt", b"inactive repository upload", "text/plain"))],
    )
    if response.status_code == 500:
        pytest.fail(f"{endpoint} -> expected non-500 response, got {response.status_code}: {response.text[:300]}")
    assert response.status_code in {400, 409}
    body = response.json()
    error = body.get("error") or {}
    assert error.get("code") == "repository_inactive"


def test_metadata_api_exposes_extracted_fields(
    phase2_repo: dict,
    api_client: httpx.Client,
) -> None:
    repo_id = phase2_repo["repository_id"]

    activate = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(activate, 200, endpoint=f"POST /api/v1/repositories/{repo_id}/activate")

    upload = api_client.post(
        "/api/v1/documents/upload",
        data={"repository_id": repo_id, "submit_for_processing": "false"},
        files=[("files", (f"phase2-meta-{uuid.uuid4().hex[:8]}.txt", b"metadata exposure test", "text/plain"))],
    )
    assert_status(upload, 200, endpoint="POST /api/v1/documents/upload")
    doc_id = (upload.json().get("documents") or [{}])[0].get("document_id")
    assert doc_id, "upload response missing document_id"

    try:
        metadata = api_client.get(f"/api/v1/documents/{doc_id}/metadata")
        assert_status(metadata, 200, endpoint=f"GET /api/v1/documents/{doc_id}/metadata")
        body = metadata.json()
        extracted = body.get("extracted_metadata")
        key_field_metadata = body.get("key_field_metadata") or {}
        assert isinstance(extracted, dict)
        assert key_field_metadata.get("processor") == "key_field_extraction"

        document = api_client.get(f"/api/v1/documents/{doc_id}")
        assert_status(document, 200, endpoint=f"GET /api/v1/documents/{doc_id}")
        doc_meta = (document.json().get("metadata") or {})
        assert isinstance(doc_meta.get("extracted_metadata"), dict)
        assert (doc_meta.get("key_field_metadata") or {}).get("processor") == "key_field_extraction"
    finally:
        try:
            api_client.delete(f"/api/v1/documents/{doc_id}")
        except Exception:
            pass
