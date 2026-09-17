"""HTTP route tests for external evidence + PubMed agent (auth/RBAC/validation)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.application.consumer_api.context import RequestActor, set_current_user_in_context
from src.application.consumer_api.errors.handlers import register_exception_handlers
from src.features.medical_literature.api.medical_literature_routes import (
    pubmed_router,
    router as evidence_router,
)
from src.features.medical_literature.schemas.literature_schemas import (
    EvidenceImportResponse,
    EvidenceImportResultItem,
    EvidencePreviewItem,
    EvidenceRegistryListResponse,
    EvidenceSearchResponse,
    PubmedAgentResponse,
)
from src.features.authentication.domain.authentication_exceptions import AuthorizationError

_REPO_ID = "11111111-1111-4111-8111-111111111111"
_SECURITY_PATCH = "src.features.users.application.user_service.get_platform_security_service"
_EVIDENCE_PATCH = "src.features.medical_literature.api.medical_literature_routes.get_evidence_service"


def _client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(evidence_router, prefix="/api/v1")
    app.include_router(pubmed_router, prefix="/api/v1")
    return TestClient(app)


def _admin() -> RequestActor:
    return RequestActor(user_id="admin", platform_role="administrator", auth_method="jwt")


def test_pubmed_search_endpoint() -> None:
    client = _client()
    security = MagicMock()
    security.check_upload_access.return_value = "owner"
    evidence = MagicMock()
    evidence.search_pubmed.return_value = EvidenceSearchResponse(
        source="pubmed",
        query="diabetes",
        max_results=25,
        count=1,
        results=[
            EvidencePreviewItem(
                source="pubmed",
                record_id="123",
                title="Title",
                already_imported=False,
                pdf_available=True,
                full_text_available=True,
                external_url="https://pubmed.ncbi.nlm.nih.gov/123/",
            )
        ],
    )
    set_current_user_in_context(_admin())
    try:
        with patch(_SECURITY_PATCH, return_value=security), patch(_EVIDENCE_PATCH, return_value=evidence):
            response = client.post(
                f"/api/v1/repositories/{_REPO_ID}/evidence/pubmed/search",
                json={"query": "diabetes", "mesh_terms": ["Diabetes Mellitus"], "max_results": 10},
            )
    finally:
        set_current_user_in_context(None)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["results"][0]["record_id"] == "123"
    security.check_upload_access.assert_called_once()
    evidence.search_pubmed.assert_called_once()


def test_clinical_trials_search_endpoint() -> None:
    client = _client()
    security = MagicMock()
    security.check_upload_access.return_value = "contributor"
    evidence = MagicMock()
    evidence.search_clinical_trials.return_value = EvidenceSearchResponse(
        source="clinicaltrials.gov",
        query="NCT01234567",
        max_results=25,
        count=1,
        results=[
            EvidencePreviewItem(
                source="clinicaltrials.gov",
                record_id="NCT01234567",
                title="Trial",
                already_imported=False,
            )
        ],
    )
    set_current_user_in_context(_admin())
    try:
        with patch(_SECURITY_PATCH, return_value=security), patch(_EVIDENCE_PATCH, return_value=evidence):
            response = client.post(
                f"/api/v1/repositories/{_REPO_ID}/evidence/clinical-trials/search",
                json={"nct_id": "NCT01234567"},
            )
    finally:
        set_current_user_in_context(None)
    assert response.status_code == 200
    assert response.json()["results"][0]["record_id"] == "NCT01234567"


def test_evidence_import_endpoint() -> None:
    client = _client()
    security = MagicMock()
    security.check_upload_access.return_value = "owner"
    evidence = MagicMock()
    evidence.import_selected = AsyncMock(
        return_value=EvidenceImportResponse(
            repository_id=_REPO_ID,
            results=[
                EvidenceImportResultItem(
                    source="pubmed",
                    record_id="123",
                    status="imported",
                    document_id="22222222-2222-4222-8222-222222222222",
                )
            ],
        )
    )
    set_current_user_in_context(_admin())
    try:
        with patch(_SECURITY_PATCH, return_value=security), patch(_EVIDENCE_PATCH, return_value=evidence):
            response = client.post(
                f"/api/v1/repositories/{_REPO_ID}/evidence/import",
                json={"items": [{"source": "pubmed", "record_id": "123"}]},
            )
    finally:
        set_current_user_in_context(None)
    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "imported"


def test_list_imports_endpoint() -> None:
    client = _client()
    security = MagicMock()
    security.check_retrieval_access.return_value = "consumer"
    evidence = MagicMock()
    evidence.list_imports.return_value = EvidenceRegistryListResponse(
        repository_id=_REPO_ID,
        count=0,
        items=[],
    )
    set_current_user_in_context(_admin())
    try:
        with patch(_SECURITY_PATCH, return_value=security), patch(_EVIDENCE_PATCH, return_value=evidence):
            response = client.get(f"/api/v1/repositories/{_REPO_ID}/evidence/imports")
    finally:
        set_current_user_in_context(None)
    assert response.status_code == 200
    assert response.json()["count"] == 0


def test_evidence_search_forbidden_cross_repo() -> None:
    client = _client()
    security = MagicMock()
    security.check_upload_access.side_effect = AuthorizationError("No repository access")
    set_current_user_in_context(RequestActor(user_id="user-b", platform_role=None, auth_method="jwt"))
    try:
        with patch(_SECURITY_PATCH, return_value=security):
            response = client.post(
                f"/api/v1/repositories/{_REPO_ID}/evidence/pubmed/search",
                json={"query": "x"},
            )
    finally:
        set_current_user_in_context(None)
    assert response.status_code == 403


def test_evidence_requires_auth() -> None:
    client = _client()
    set_current_user_in_context(None)
    response = client.post(
        f"/api/v1/repositories/{_REPO_ID}/evidence/pubmed/search",
        json={"query": "x"},
    )
    assert response.status_code in {401, 403}


def test_invalid_repository_uuid() -> None:
    client = _client()
    set_current_user_in_context(_admin())
    try:
        response = client.post(
            "/api/v1/repositories/not-a-uuid/evidence/pubmed/search",
            json={"query": "x"},
        )
    finally:
        set_current_user_in_context(None)
    assert response.status_code == 422


def test_invalid_nct_returns_422_not_500() -> None:
    client = _client()
    security = MagicMock()
    security.check_upload_access.return_value = "owner"
    set_current_user_in_context(_admin())
    try:
        with patch(_SECURITY_PATCH, return_value=security):
            response = client.post(
                f"/api/v1/repositories/{_REPO_ID}/evidence/clinical-trials/search",
                json={"nct_id": "NOTANCT", "max_results": 1},
            )
    finally:
        set_current_user_in_context(None)
    assert response.status_code == 422
    body = response.json()
    assert body.get("error", {}).get("code") == "validation_error" or "error" in body


def test_pubmed_agent_out_of_corpus() -> None:
    client = _client()
    security = MagicMock()
    security.check_retrieval_access.return_value = "consumer"
    retrieval = MagicMock()
    retrieval.retrieve = AsyncMock(return_value=MagicMock(blocked=False, results=[], block_reason=None))

    set_current_user_in_context(_admin())
    try:
        with patch(_SECURITY_PATCH, return_value=security), patch(
            "src.features.retrieval.application.retrieval_service.get_retrieval_service",
            return_value=retrieval,
        ), patch(
            "src.features.security.moderation.hybrid_prompt_guard.HybridPromptGuard.check_prompt",
            return_value={"allowed": True},
        ), patch(
            "src.features.security.dlp.policy_loader.scan_keyword_policies",
            return_value={"action": "allow"},
        ), patch(
            "src.features.repositories.infrastructure.repository_repository.get_repository_store",
            return_value=None,
        ):
            response = client.post(
                f"/api/v1/repositories/{_REPO_ID}/pubmed-agent/chat",
                json={"query": "What is the capital of Mars?", "provider": "extractive"},
            )
    finally:
        set_current_user_in_context(None)

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_sufficient"] is False
    assert "couldn't find sufficient evidence" in (body.get("answer") or "").lower()
