"""Live API smoke tests for evidence endpoints (skip if DMS unavailable)."""

from __future__ import annotations

import httpx
import pytest

from api.helpers import assert_status


@pytest.fixture
def evidence_repo(api_client: httpx.Client, created_repository: dict) -> str:
    repo_id = created_repository["id"]
    activate = api_client.post(f"/api/v1/repositories/{repo_id}/activate")
    assert_status(activate, 200, endpoint=f"POST /api/v1/repositories/{repo_id}/activate")
    assert activate.json().get("status") == "active"
    return repo_id


def test_evidence_pubmed_search_live_or_mocked(
    api_client: httpx.Client,
    evidence_repo: str,
) -> None:
    """Search must not auto-import; uses live NCBI when reachable, else soft-fail."""
    response = api_client.post(
        f"/api/v1/repositories/{evidence_repo}/evidence/pubmed/search",
        json={"query": "metformin diabetes", "max_results": 3},
        timeout=60.0,
    )
    # Endpoint must exist (not 404). External provider outages may yield 502/503/504.
    assert response.status_code != 404, response.text
    if response.status_code == 200:
        body = response.json()
        assert body["source"] == "pubmed"
        assert body["max_results"] == 3
        assert "results" in body
        for item in body["results"]:
            assert "record_id" in item
            assert "already_imported" in item
    else:
        assert response.status_code in {422, 429, 502, 503, 504}, response.text


def test_evidence_clinical_trials_search_live(
    api_client: httpx.Client,
    evidence_repo: str,
) -> None:
    response = api_client.post(
        f"/api/v1/repositories/{evidence_repo}/evidence/clinical-trials/search",
        json={"condition": "diabetes", "max_results": 3},
        timeout=60.0,
    )
    assert response.status_code != 404, response.text
    if response.status_code == 200:
        body = response.json()
        assert body["source"] == "clinicaltrials.gov"
        assert body["max_results"] == 3
    else:
        assert response.status_code in {422, 429, 502, 503, 504}, response.text


def test_evidence_imports_list_empty(
    api_client: httpx.Client,
    evidence_repo: str,
) -> None:
    response = api_client.get(f"/api/v1/repositories/{evidence_repo}/evidence/imports")
    assert_status(response, 200, endpoint="GET .../evidence/imports")
    body = response.json()
    assert body["repository_id"] == evidence_repo
    assert isinstance(body["items"], list)


def test_evidence_import_validation(
    api_client: httpx.Client,
    evidence_repo: str,
) -> None:
    response = api_client.post(
        f"/api/v1/repositories/{evidence_repo}/evidence/import",
        json={"items": []},
    )
    assert response.status_code == 422


def test_pubmed_agent_endpoint_exists(
    api_client: httpx.Client,
    evidence_repo: str,
) -> None:
    response = api_client.post(
        f"/api/v1/repositories/{evidence_repo}/pubmed-agent/chat",
        json={"query": "Summarize repository evidence about diabetes", "provider": "extractive"},
        timeout=60.0,
    )
    assert response.status_code != 404, response.text
    if response.status_code == 200:
        body = response.json()
        assert body["repository_id"] == evidence_repo
        assert "answer" in body
        assert body.get("citation_format") == "[REPO:DOCUMENT:PAGE]"
