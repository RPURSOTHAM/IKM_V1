"""Unit tests for Azure deployment aliases and PubMed retrieval knob overrides."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.application.consumer_api.context import RequestActor, set_current_user_in_context
from src.application.consumer_api.errors.handlers import register_exception_handlers
from src.features.medical_literature.api.medical_literature_routes import (
    pubmed_router,
    router as evidence_router,
)
from src.features.medical_literature.schemas.literature_schemas import PubmedAgentResponse
from src.features.generation.providers.llm_client import OpenAICompatibleGenerator
from src.features.generation.configuration.generation_config import GenerationConfig
from src.features.generation.domain.models import GenerationRequest

_REPO_ID = "11111111-1111-4111-8111-111111111111"
_SECURITY_PATCH = "src.features.users.application.user_service.get_platform_security_service"
_RETRIEVAL_PATCH = "src.features.retrieval.application.retrieval_service.get_retrieval_service"


def test_azure_deployment_name_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "my-deploy")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    captured: dict[str, object] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(req, timeout=0):  # noqa: ARG001
        captured["url"] = req.full_url
        return _Resp()

    gen = OpenAICompatibleGenerator()
    cfg = GenerationConfig(
        provider="azure_openai",
        model_id="gpt-4o",
        base_url="https://example.openai.azure.com",
        api_key_env="AZURE_OPENAI_API_KEY",
        timeout_seconds=5,
        max_output_tokens=64,
    )
    req = GenerationRequest(question="hi", provider="azure_openai", model_id="gpt-4o")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = gen.generate(req, cfg, messages=[{"role": "user", "content": "hi"}])
    assert "ok" in (result.answer or "")
    assert "my-deploy" in str(captured.get("url"))


def _pubmed_client_harness(*, settings: dict, body: dict):
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(evidence_router, prefix="/api/v1")
    app.include_router(pubmed_router, prefix="/api/v1")
    client = TestClient(app)

    security = MagicMock()
    security.check_retrieval_access.return_value = "owner"
    retrieval = MagicMock()
    ret_res = MagicMock()
    ret_res.blocked = False
    ret_res.results = []
    retrieval.retrieve = AsyncMock(return_value=ret_res)

    settings_store = MagicMock()
    settings_store.get_settings.return_value = settings

    pg = MagicMock()
    pg.check_prompt.return_value = {"allowed": True}

    actor = RequestActor(user_id="admin", platform_role="administrator", auth_method="jwt")
    set_current_user_in_context(actor)
    try:
        with (
            patch(_SECURITY_PATCH, return_value=security),
            patch(_RETRIEVAL_PATCH, return_value=retrieval),
            patch(
                "src.features.repositories.infrastructure.repository_repository.get_repository_store",
                return_value=settings_store,
            ),
            patch(
                "src.features.security.moderation.hybrid_prompt_guard.HybridPromptGuard",
                return_value=pg,
            ),
            patch(
                "src.features.security.dlp.policy_loader.scan_keyword_policies",
                return_value={"action": "allow"},
            ),
            patch(
                "src.features.medical_literature.agents.medical_literature_agent.PubmedAgentService.answer",
                new=AsyncMock(
                    return_value=PubmedAgentResponse(
                        allowed=True,
                        answer="Insufficient evidence.",
                        action="refuse",
                        repository_id=_REPO_ID,
                        evidence_sufficient=False,
                        grounded=True,
                    )
                ),
            ),
        ):
            response = client.post(
                f"/api/v1/repositories/{_REPO_ID}/pubmed-agent/chat",
                json=body,
            )
    finally:
        set_current_user_in_context(None)
    return response, retrieval


def test_pubmed_request_overrides_beat_repository_settings() -> None:
    response, retrieval = _pubmed_client_harness(
        settings={
            "retrieval_top_k": 5,
            "reranking": False,
            "query_expansion_enabled": False,
            "parallel_retrieval_enabled": False,
        },
        body={
            "query": "What is the dose?",
            "provider": "extractive",
            "top_k": 17,
            "reranking_enabled": True,
            "query_expansion_enabled": True,
            "parallel_retrieval_enabled": True,
        },
    )
    assert response.status_code == 200
    retrieve_req = retrieval.retrieve.await_args.args[0]
    assert retrieve_req.top_k == 17
    assert retrieve_req.use_rerank is True
    assert retrieve_req.expand_query is True
    assert retrieve_req.use_production_pipeline is True


def test_pubmed_settings_fallback_when_overrides_omitted() -> None:
    response, retrieval = _pubmed_client_harness(
        settings={
            "retrieval_top_k": 11,
            "reranking": True,
            "query_expansion_enabled": True,
            "parallel_retrieval_enabled": True,
        },
        body={"query": "What is the dose?", "provider": "extractive"},
    )
    assert response.status_code == 200
    retrieve_req = retrieval.retrieve.await_args.args[0]
    assert retrieve_req.top_k == 11
    assert retrieve_req.use_rerank is True
    assert retrieve_req.expand_query is True
    assert retrieve_req.use_production_pipeline is True
