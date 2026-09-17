"""Integration tests: reload processor config without restarting the app."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.system.api.system_routes import router as system_router
from src.features.document_processing.shared_processor.deployment import (
    DynamicProcessingPipeline,
    ProcessorConfigurationProvider,
    bootstrap_deployment_processors,
    get_active_pipeline,
    get_processor_registry,
    reload_deployment_processors,
)
from src.features.document_processing.shared_processor.deployment.config_provider import ProcessorConfigurationProvider as PCP
from src.features.document_processing.shared_processor.deployment.pipeline import reset_active_pipeline
from src.features.document_processing.shared_processor.deployment.registry import ProcessorRegistry as PR
from src.features.document_processing.shared_processor.deployment.watcher import (
    ProcessorConfigWatcher,
    stop_processor_config_watcher,
)


def _write_processors(path: Path, **flags: bool) -> None:
    lines = ["processors:"]
    defaults = {
        "pdf": True,
        "docx": True,
        "ocr": True,
        "security": True,
        "key_fields": True,
        "reference_extraction": True,
        "image": True,
        "embedding": True,
        "reranker": True,
    }
    defaults.update(flags)
    for key, value in defaults.items():
        lines.append(f"  {key}: {'true' if value else 'false'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    config = tmp_path / "processors.yaml"
    _write_processors(config)
    monkeypatch.setenv("PROCESSORS_CONFIG_PATH", str(config))
    monkeypatch.setenv("PROCESSORS_CONFIG_WATCH", "false")
    for key in list(__import__("os").environ):
        if key.startswith("PROCESSOR_") and key.endswith("_ENABLED"):
            monkeypatch.delenv(key, raising=False)
        if key in {"IKM_ENABLED_PROCESSORS", "IKM_DISABLED_PROCESSORS"}:
            monkeypatch.delenv(key, raising=False)
    stop_processor_config_watcher()
    PCP.reset_instance()
    PR.reset_instance()
    reset_active_pipeline()
    yield
    stop_processor_config_watcher()
    PCP.reset_instance()
    PR.reset_instance()
    reset_active_pipeline()


@pytest.fixture
def config_path(tmp_path) -> Path:
    return tmp_path / "processors.yaml"


def test_reload_enables_and_disables_without_restart(config_path: Path):
    bootstrap_deployment_processors(initialize=True)
    assert ProcessorConfigurationProvider.is_enabled("ocr")
    assert get_processor_registry().get("ocr") is not None

    _write_processors(config_path, ocr=False, image=False)
    result = reload_deployment_processors(initialize=True, reason="test")

    assert result.ok
    assert "ocr" in result.newly_disabled
    assert "image" in result.newly_disabled
    assert not ProcessorConfigurationProvider.is_enabled("ocr")
    assert get_processor_registry().get("ocr") is None
    assert "ocr" not in get_active_pipeline().pipeline_ids

    _write_processors(config_path, ocr=True, image=False)
    result2 = reload_deployment_processors(initialize=True, reason="test")
    assert result2.ok
    assert "ocr" in result2.newly_enabled
    assert ProcessorConfigurationProvider.is_enabled("ocr")
    ocr = get_processor_registry().get("ocr")
    assert ocr is not None
    assert ocr.initialized is True


def test_pipeline_changes_after_reload(config_path: Path):
    bootstrap_deployment_processors(initialize=True)
    before = get_active_pipeline().pipeline_ids
    assert "reranker" in before

    _write_processors(config_path, reranker=False, ocr=False)
    reload_deployment_processors(initialize=True, reason="test")
    after = get_active_pipeline().pipeline_ids
    assert "reranker" not in after
    assert "ocr" not in after

    # Live pipeline run must not execute disabled processors.
    pipeline = get_active_pipeline()
    result = pipeline.run({"document_path": "notes.txt", "file_extension": "txt"})
    assert "reranker" not in result["pipeline_executed"]
    assert "ocr" not in result["pipeline_executed"]
    assert "security" in result["pipeline_executed"]


def test_models_initialized_only_for_enabled(config_path: Path):
    _write_processors(config_path, ocr=False, embedding=True, reranker=False)
    PCP.reset_instance()
    PR.reset_instance()
    reset_active_pipeline()

    registry = bootstrap_deployment_processors(initialize=True)
    assert registry.get("ocr") is None
    embedding = registry.get("embedding")
    assert embedding is not None and embedding.initialized
    assert registry.get("reranker") is None


def test_invalid_reload_keeps_previous_config(config_path: Path):
    bootstrap_deployment_processors(initialize=True)
    previous = ProcessorConfigurationProvider.instance().enabled_processors()

    config_path.write_text("processors:\n  pdf: true\n  not_a_real_processor: true\n", encoding="utf-8")
    result = reload_deployment_processors(initialize=True, keep_previous_on_error=True, reason="test")

    assert result.ok is False
    assert result.kept_previous_config is True
    assert "unknown processor" in (result.error or "").lower()
    assert ProcessorConfigurationProvider.instance().enabled_processors() == previous
    assert get_processor_registry().get("pdf") is not None


def test_api_reload_endpoint(config_path: Path):
    bootstrap_deployment_processors(initialize=False)
    app = FastAPI()
    app.include_router(system_router, prefix="/api/v1")
    client = TestClient(app)

    _write_processors(config_path, ocr=False, reference_extraction=False)
    response = client.post("/api/v1/system/processors/reload")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "ocr" in body["disabled"]
    assert "reference_extraction" in body["disabled"]
    assert "ocr" not in body["pipeline"]

    listed = client.get("/api/v1/system/processors")
    assert listed.status_code == 200
    assert listed.json()["disabled"] == body["disabled"]


def test_api_reload_rejects_invalid_config(config_path: Path):
    bootstrap_deployment_processors(initialize=False)
    app = FastAPI()
    app.include_router(system_router, prefix="/api/v1")
    client = TestClient(app)

    config_path.write_text("processors:\n  pdf: true\n  bogus: false\n", encoding="utf-8")
    response = client.post("/api/v1/system/processors/reload")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["kept_previous_config"] is True
    assert "bogus" in detail["message"]


def test_file_watcher_reloads_pipeline(config_path: Path):
    bootstrap_deployment_processors(initialize=True)
    assert ProcessorConfigurationProvider.is_enabled("ocr")

    reloads: list[str] = []

    def _on_change() -> None:
        result = reload_deployment_processors(initialize=True, reason="watcher_test")
        reloads.append("ok" if result.ok else "err")

    watcher = ProcessorConfigWatcher(config_path, _on_change, interval_seconds=0.25)
    watcher.start()
    try:
        time.sleep(0.35)
        _write_processors(config_path, ocr=False)
        deadline = time.time() + 5.0
        while time.time() < deadline and not reloads:
            time.sleep(0.1)
        assert reloads, "watcher did not fire"
        assert not ProcessorConfigurationProvider.is_enabled("ocr")
        assert get_processor_registry().get("ocr") is None
    finally:
        watcher.stop()


def test_disabled_processor_not_executed_after_reload(config_path: Path):
    bootstrap_deployment_processors(initialize=True)
    pipeline = DynamicProcessingPipeline.from_registry()
    first = pipeline.run({"document_path": "a.txt", "file_extension": "txt"})
    assert "ocr" in first["pipeline_executed"]

    _write_processors(config_path, ocr=False)
    reload_deployment_processors(initialize=True, reason="test")
    second = get_active_pipeline().run({"document_path": "a.txt", "file_extension": "txt"})
    assert "ocr" not in second["pipeline_executed"]
    assert "ocr" not in second.get("pipeline_skipped", [])
