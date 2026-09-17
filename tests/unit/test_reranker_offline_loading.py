"""Unit tests for offline CrossEncoder model resolution and loading."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.features.retrieval.reranking.config import RerankerConfig
from src.features.retrieval.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.features.retrieval.reranking.reranker_model_resolver import (
    has_required_model_files,
    resolve_reranker_model_path,
    resolve_snapshot_dir,
)
from src.features.retrieval.schemas.retrieval_schemas import ChunkOut
from src.features.retrieval.application.retrieval_service import RetrievalService


def _write_fake_model(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}), encoding="utf-8")
    (path / "modules.json").write_text("[]", encoding="utf-8")
    (path / "pytorch_model.bin").write_bytes(b"fake-weights")
    return path


def test_has_required_model_files(tmp_path: Path) -> None:
    model_dir = _write_fake_model(tmp_path / "bge-reranker-base")
    assert has_required_model_files(model_dir) is True
    assert has_required_model_files(tmp_path / "missing") is False


def test_resolve_local_model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = _write_fake_model(tmp_path / "bge-reranker-base")
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("RERANKER_MODEL", str(model_dir))
    resolved = resolve_reranker_model_path(config=RerankerConfig(model="BAAI/bge-reranker-base"))
    assert Path(resolved) == model_dir


def test_resolve_models_baai_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = _write_fake_model(tmp_path / "models--BAAI--bge-reranker-base")
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    monkeypatch.delenv("RERANKER_MODEL_DIR", raising=False)
    resolved = resolve_reranker_model_path(config=RerankerConfig(model="BAAI/bge-reranker-base"))
    assert Path(resolved) == model_dir


def test_resolve_invalid_path_offline_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODELS_ROOT", str(tmp_path))
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("RERANKER_MODEL", str(tmp_path / "does-not-exist"))
    resolved = resolve_reranker_model_path(config=RerankerConfig(model="BAAI/bge-reranker-base"))
    assert resolved == ""


def test_resolve_snapshot_dir(tmp_path: Path) -> None:
    snap = _write_fake_model(tmp_path / "models--BAAI--bge-reranker-base" / "snapshots" / "abc123")
    root = tmp_path / "models--BAAI--bge-reranker-base"
    assert resolve_snapshot_dir(root) == snap


def test_service_loader_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = _write_fake_model(tmp_path / "bge-reranker-base")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    service = RetrievalService()
    fake = MagicMock(name="cross-encoder")
    with patch("sentence_transformers.CrossEncoder", return_value=fake) as ctor:
        first = service._get_reranker(str(model_dir))
        second = service._get_reranker(str(model_dir))
    assert first is fake
    assert second is fake
    assert ctor.call_count == 1


def test_service_loader_invalid_path_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    service = RetrievalService()
    assert service._get_reranker("/app/src/models/missing-reranker") is None


def test_graceful_fallback_when_model_missing() -> None:
    reranker = CrossEncoderReranker(lambda _path: None, config=RerankerConfig(enabled=True))
    chunks = [
        ChunkOut(
            chunk_id="a",
            id="a",
            text="alpha",
            doc_name="d.pdf",
            section_name="s",
            page=1,
            score=0.03,
            source="d.pdf",
            graph_context=[],
            highlight_spans=[],
            retrieval_signals={"rrf_score": 0.03},
        )
    ]
    with patch(
        "src.features.retrieval.reranking.cross_encoder_reranker.resolve_reranker_model_path",
        return_value="",
    ):
        result = reranker.rerank("q", chunks, top_k=1)
    assert result.applied is False
    assert result.chunks[0].chunk_id == "a"
    assert "rerank_score" not in (result.chunks[0].retrieval_signals or {})


def test_rerank_metrics_populated_when_model_loads() -> None:
    model = MagicMock()
    model.predict.return_value = [2.0]
    reranker = CrossEncoderReranker(lambda _path: model, config=RerankerConfig(enabled=True, batch_size=8))
    chunks = [
        ChunkOut(
            chunk_id="a",
            id="a",
            text="alpha",
            doc_name="d.pdf",
            section_name="s",
            page=1,
            score=0.03,
            source="d.pdf",
            graph_context=[],
            highlight_spans=[],
            retrieval_signals={"rrf_score": 0.03},
        )
    ]
    with patch(
        "src.features.retrieval.reranking.cross_encoder_reranker.resolve_reranker_model_path",
        return_value="/models/bge-reranker-base",
    ):
        result = reranker.rerank("q", chunks, top_k=1)
    assert result.applied is True
    signals = result.chunks[0].retrieval_signals
    assert signals["rerank_score"] == pytest.approx(2.0)
    assert "rerank_score_normalized" in signals
    assert signals["original_score"] == pytest.approx(0.03)
    assert signals["rerank_model"] == "/models/bge-reranker-base"
    assert result.metrics.batch_count >= 1
    assert result.metrics.top_score == pytest.approx(2.0)
    trace = result.metrics.as_trace()
    assert "rerank_ms" in trace
    assert "rerank_top_score" in trace
