from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.features.document_processing.core.config import Settings
from src.features.indexing.application import embedding


def _settings(models_root: Path, *, model_dir: str | None = None, model_name: str = "bge-base-en") -> Settings:
    return replace(
        Settings(),
        models_root=str(models_root),
        model_dir=model_dir,
        model_name=model_name,
    )


def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr(embedding, "get_settings", lambda: settings)


def _write_sentence_transformer_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "modules.json").write_text("[]", encoding="utf-8")
    (path / "config_sentence_transformers.json").write_text("{}", encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "sentence_bert_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("", encoding="utf-8")


def test_resolves_valid_local_embedding_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    _write_sentence_transformer_files(model_root / "bge-base-en")
    _patch_settings(monkeypatch, _settings(model_root))

    resolved = embedding._sentence_transformer_source("bge-base-en", model_dir="bge-base-en")

    assert resolved == str((model_root / "bge-base-en").resolve())


def test_resolves_short_name_to_huggingface_cache_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    snapshot = model_root / "models--BAAI--bge-base-en" / "snapshots" / "abc123"
    _write_sentence_transformer_files(snapshot)
    _patch_settings(monkeypatch, _settings(model_root))

    resolved = embedding._sentence_transformer_source("bge-base-en", model_dir="bge-base-en")

    assert resolved == str(snapshot.resolve())


def test_missing_models_root_reports_mount_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_root = tmp_path / "missing-models"
    _patch_settings(monkeypatch, _settings(missing_root))

    with pytest.raises(embedding.EmbeddingModelConfigurationError, match="not mounted or does not exist"):
        embedding._sentence_transformer_source("bge-base-en", model_dir="bge-base-en")


def test_missing_model_name_reports_available_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    _write_sentence_transformer_files(model_root / "models--BAAI--bge-base-en" / "snapshots" / "abc123")
    _patch_settings(monkeypatch, _settings(model_root))

    with pytest.raises(embedding.EmbeddingModelConfigurationError, match="Available model directories"):
        embedding._sentence_transformer_source("unknown-model", model_dir="unknown-model")


def test_incomplete_local_model_reports_configuration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "models"
    (model_root / "bge-base-en").mkdir(parents=True)
    _patch_settings(monkeypatch, _settings(model_root))

    with pytest.raises(embedding.EmbeddingModelConfigurationError, match="not a complete"):
        embedding._sentence_transformer_source("bge-base-en", model_dir="bge-base-en")


def test_embedding_diagnostics_for_valid_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    snapshot = model_root / "models--BAAI--bge-base-en" / "snapshots" / "abc123"
    _write_sentence_transformer_files(snapshot)
    _patch_settings(monkeypatch, _settings(model_root))

    diagnostics = embedding.embedding_model_diagnostics("bge-base-en", "bge-base-en")

    assert diagnostics["resolved_path"] == str(snapshot.resolve())
    assert diagnostics["exists"] is True
    assert "config.json" in diagnostics["files"]
