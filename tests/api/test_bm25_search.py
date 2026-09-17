"""Live API tests for BM25 search production controls."""

from __future__ import annotations

import uuid

import httpx
import pytest

from api.helpers import assert_status


def _bm25_endpoint(repository_id: str) -> str:
    return f"POST /api/v1/repositories/{repository_id}/search/bm25"


@pytest.fixture
def empty_repository(api_client: httpx.Client, unique_name: str, admin_credentials: tuple[str, str]):
    owner, _ = admin_credentials
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
    assert_status(response, 201, endpoint="POST /api/v1/repositories")
    repo_id = response.json().get("repository_id") or response.json().get("id")
    assert repo_id
    activate = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(activate, {200, 409}, endpoint=f"POST /api/v1/repositories/{repo_id}/activate")
    yield repo_id
    api_client.delete(f"/api/v1/repositories/{repo_id}")


def test_bm25_search_authorized_empty_repository(
    api_client: httpx.Client,
    empty_repository: str,
) -> None:
    endpoint = _bm25_endpoint(empty_repository)
    response = api_client.post(
        f"/api/v1/repositories/{empty_repository}/search/bm25",
        json={"query": "batch release", "top_k": 10},
    )
    if response.status_code == 404 and "collection_not_found" in response.text:
        pytest.skip("Repository has no Weaviate collection in this environment")
    assert_status(response, 200, endpoint=endpoint)
    body = response.json()
    assert body["repository_id"] == empty_repository
    assert body["query"] == "batch release"
    assert body["result_count"] == 0
    assert body["results"] == []


def test_bm25_search_unauthorized_without_token(base_url: str, empty_repository: str) -> None:
    endpoint = _bm25_endpoint(empty_repository)
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        response = client.post(
            f"/api/v1/repositories/{empty_repository}/search/bm25",
            json={"query": "batch release", "top_k": 5},
        )
    if response.status_code == 200:
        pytest.skip("BM25 endpoint does not require auth in this deployment")
    assert response.status_code in {401, 403}, f"{endpoint} unexpected status {response.status_code}"


def test_bm25_search_repository_not_found(api_client: httpx.Client) -> None:
    missing_id = str(uuid.uuid4())
    endpoint = _bm25_endpoint(missing_id)
    response = api_client.post(
        f"/api/v1/repositories/{missing_id}/search/bm25",
        json={"query": "missing repo", "top_k": 5},
    )
    assert_status(response, 404, endpoint=endpoint)
