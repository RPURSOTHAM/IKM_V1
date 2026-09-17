"""Unit tests for reranker enablement precedence and execution gating."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from src.features.retrieval.reranking.config import RerankerConfig, resolve_rerank_enabled
from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest
from src.features.retrieval.application.retrieval_service import RetrievalService, _effective_use_rerank


def test_reranker_config_enabled_true_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        cfg = RerankerConfig.from_env()
    assert cfg.enabled is True


def test_reranker_config_enabled_false_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        cfg = RerankerConfig.from_env()
    assert cfg.enabled is False


def test_reranker_config_respects_deployment_gate(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=False,
    ):
        cfg = RerankerConfig.from_env({"enabled": True})
    assert cfg.enabled is False


def test_resolve_uses_env_when_request_and_repo_unset(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        assert resolve_rerank_enabled(use_rerank=None, repo_settings={"reranking": None}) is True
        assert resolve_rerank_enabled(use_rerank=None, repo_settings={}) is True


def test_resolve_env_false_disables_even_with_repo_true(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        assert (
            resolve_rerank_enabled(use_rerank=None, repo_settings={"reranking": True}) is False
        )


def test_resolve_repo_false_overrides_env_true(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        assert (
            resolve_rerank_enabled(use_rerank=None, repo_settings={"reranking": False}) is False
        )


def test_resolve_repo_true_with_env_true(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        assert resolve_rerank_enabled(use_rerank=None, repo_settings={"reranking": True}) is True


def test_resolve_request_override(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        assert resolve_rerank_enabled(use_rerank=False, repo_settings={"reranking": True}) is False
        assert resolve_rerank_enabled(use_rerank=True, repo_settings={"reranking": False}) is True


def test_effective_use_rerank_reads_request_body(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    with patch(
        "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
        return_value=True,
    ):
        body = RetrieveRequest(
            query="q",
            repository_id="12345678-1234-1234-1234-123456789012",
            use_rerank=True,
        )
        assert _effective_use_rerank(body, {"reranking": False}) is True


def test_valid_local_model_path_used(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "bge-reranker-base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("RERANKER_MODEL", str(model_dir))
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    from src.features.retrieval.reranking.reranker_model_resolver import resolve_reranker_model_path

    resolved = resolve_reranker_model_path(None, config=RerankerConfig(model=str(model_dir)))
    assert resolved == str(model_dir.resolve())


def test_missing_model_path_offline_returns_empty(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RERANKER_MODEL", str(tmp_path / "missing-reranker"))
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from src.features.retrieval.reranking.reranker_model_resolver import resolve_reranker_model_path

    assert resolve_reranker_model_path(None) == ""


class _FakeReranker:
    def predict(self, pairs):
        return [0.1 if "first" in text.lower() else 4.0 for _query, text in pairs]


class _FakeQuery:
    def __init__(self, objects):
        self.objects = objects

    def hybrid(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)

    def bm25(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)

    def near_vector(self, **_kwargs):
        return SimpleNamespace(objects=self.objects)


def _fake_obj(uuid: str, text: str, score: float):
    return SimpleNamespace(
        uuid=uuid,
        properties={
            "chunk_id": uuid,
            "doc_name": "doc.pdf",
            "section_name": "Test",
            "page": 1,
            "text": text,
        },
        metadata=SimpleNamespace(score=score, distance=None),
    )


async def _run_retrieve(
    *,
    use_rerank: bool | None,
    repo_reranking: bool | None,
    env_enabled: bool,
):
    objects = [
        _fake_obj("first", "First more relevant but lower rerank result", 0.95),
        _fake_obj("second", "Second more relevant result", 0.20),
    ]
    collection = SimpleNamespace(query=_FakeQuery(objects))
    service = RetrievalService()
    service.ready = True
    service._client = SimpleNamespace(collections=SimpleNamespace(get=lambda _n: collection))
    service._model_error = None
    service._query_vector = lambda *_a, **_k: [0.1, 0.2]
    cfg = SimpleNamespace(
        enable_retrieval=True,
        hybrid_alpha=0.75,
        default_collection_name="DefaultCollection",
        retrieval_model_dir=None,
        retrieval_model_name="test-model",
    )
    request_kwargs = {
        "query": "more relevant",
        "repository_id": "12345678-1234-1234-1234-123456789012",
    }
    if use_rerank is not None:
        request_kwargs["use_rerank"] = use_rerank
    body = RetrieveRequest(**request_kwargs)
    settings = {"lexical_composition": True, "reranker_model": "fake-reranker"}
    if repo_reranking is not None:
        settings["reranking"] = repo_reranking

    with (
        patch("src.features.retrieval.application.retrieval_service.get_settings", return_value=cfg),
        patch("src.application.consumer_api.context.get_current_user_from_context", return_value=None),
        patch("src.features.retrieval.application.retrieval_service._build_repository_citation_lookup", return_value={}),
        patch.object(service, "_collection_exists", return_value=True),
        patch.object(
            service,
            "_resolve_repository_context",
            return_value={"weaviate_collection": "TestCollection", "settings": settings},
        ),
        patch.object(service, "_get_reranker", return_value=_FakeReranker()),
        patch.dict(
            "os.environ",
            {"RERANKER_ENABLED": "true" if env_enabled else "false"},
            clear=False,
        ),
        patch(
            "src.features.document_processing.shared_processor.deployment.ProcessorConfigurationProvider.is_enabled",
            return_value=True,
        ),
    ):
        return await service.retrieve(body)


def test_retrieve_executes_rerank_when_env_enabled_and_repo_unset() -> None:
    response = asyncio.run(_run_retrieve(use_rerank=None, repo_reranking=None, env_enabled=True))
    assert "rerank" in response.pipeline_stages_executed
    assert [item.id for item in response.results] == ["second", "first"]
    assert response.results[0].retrieval_signals["rerank_score"] == 4.0


def test_retrieve_skips_rerank_when_env_disabled() -> None:
    response = asyncio.run(_run_retrieve(use_rerank=None, repo_reranking=None, env_enabled=False))
    assert "rerank" not in response.pipeline_stages_executed
    assert [item.id for item in response.results] == ["first", "second"]


def test_retrieve_repo_false_disables_despite_env() -> None:
    response = asyncio.run(_run_retrieve(use_rerank=None, repo_reranking=False, env_enabled=True))
    assert "rerank" not in response.pipeline_stages_executed


def test_retrieve_does_not_load_model_when_disabled() -> None:
    service = RetrievalService()
    service.ready = True
    loader = MagicMock(side_effect=AssertionError("model should not load"))
    service._get_reranker = loader  # type: ignore[method-assign]
    from src.features.retrieval.reranking import CrossEncoderReranker
    from src.features.retrieval.schemas.retrieval_schemas import ChunkOut

    reranker = CrossEncoderReranker(loader, config=RerankerConfig(enabled=False))
    chunk = ChunkOut(
        chunk_id="c1",
        id="c1",
        text="hello",
        doc_name="d.pdf",
        section_name="s",
        page=1,
        score=0.5,
        source="d.pdf",
        graph_context=[],
        highlight_spans=[],
    )
    result = reranker.rerank("q", [chunk])
    assert result.applied is False
    loader.assert_not_called()
