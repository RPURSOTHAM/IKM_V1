"""Unit tests for deployment-level processor configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.features.document_processing.shared_processor.deployment import (
    DynamicProcessingPipeline,
    ProcessorConfigurationError,
    ProcessorConfigurationProvider,
    ProcessorRegistry,
    bootstrap_deployment_processors,
    format_processor_startup_banner,
)
from src.features.document_processing.shared_processor.deployment.config_provider import ProcessorConfigurationProvider as PCP
from src.features.document_processing.shared_processor.deployment.pipeline import reset_active_pipeline
from src.features.document_processing.shared_processor.deployment.registry import ProcessorRegistry as PR
from src.features.document_processing.shared_processor.types import enabled_processor_types


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch, tmp_path):
    """Isolate config/registry state per test."""
    # Point at a non-existent config by default so tests control enablement via env.
    missing = tmp_path / "missing-processors.yaml"
    monkeypatch.setenv("PROCESSORS_CONFIG_PATH", str(missing))
    monkeypatch.setenv("PROCESSORS_CONFIG_WATCH", "false")
    for key in list(__import__("os").environ):
        if key.startswith("PROCESSOR_") and key.endswith("_ENABLED"):
            monkeypatch.delenv(key, raising=False)
        if key in {"IKM_ENABLED_PROCESSORS", "IKM_DISABLED_PROCESSORS"}:
            monkeypatch.delenv(key, raising=False)
    PCP.reset_instance()
    PR.reset_instance()
    reset_active_pipeline()
    yield
    PCP.reset_instance()
    PR.reset_instance()
    reset_active_pipeline()


def test_absent_config_enables_all_processors():
    provider = ProcessorConfigurationProvider.instance()
    enabled = provider.enabled_processors()
    disabled = provider.disabled_processors()
    assert disabled == []
    assert set(enabled) == {
        "pdf",
        "docx",
        "ocr",
        "security",
        "key_fields",
        "reference_extraction",
        "image",
        "embedding",
        "reranker",
    }


def test_yaml_disables_selected_processors(tmp_path, monkeypatch):
    config = tmp_path / "processors.yaml"
    config.write_text(
        "processors:\n"
        "  pdf: true\n"
        "  docx: true\n"
        "  security: true\n"
        "  ocr: false\n"
        "  key_fields: true\n"
        "  reference_extraction: false\n"
        "  image: false\n"
        "  embedding: true\n"
        "  reranker: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PROCESSORS_CONFIG_PATH", str(config))
    PCP.reset_instance()

    provider = ProcessorConfigurationProvider.instance()
    assert provider.enabled_processors() == [
        "pdf",
        "docx",
        "security",
        "key_fields",
        "embedding",
        "reranker",
    ]
    assert provider.disabled_processors() == [
        "ocr",
        "reference_extraction",
        "image",
    ]


def test_unknown_processor_fails_startup(tmp_path, monkeypatch):
    config = tmp_path / "processors.yaml"
    config.write_text("processors:\n  pdf: true\n  unknown_thing: true\n", encoding="utf-8")
    monkeypatch.setenv("PROCESSORS_CONFIG_PATH", str(config))
    PCP.reset_instance()

    with pytest.raises(ProcessorConfigurationError, match="unknown processor"):
        ProcessorConfigurationProvider.instance().enabled_processors()


def test_env_override_disables_processor(monkeypatch):
    monkeypatch.setenv("PROCESSOR_OCR_ENABLED", "false")
    PCP.reset_instance()
    provider = ProcessorConfigurationProvider.instance()
    assert not provider.is_processor_enabled("ocr")
    assert "ocr" in provider.disabled_processors()


def test_registry_instantiates_only_enabled(monkeypatch):
    monkeypatch.setenv("IKM_DISABLED_PROCESSORS", "ocr,image,reranker")
    PCP.reset_instance()
    PR.reset_instance()

    registry = bootstrap_deployment_processors(initialize=True)
    ids = [p.id for p in registry.enabled_instances()]
    assert "ocr" not in ids
    assert "image" not in ids
    assert "reranker" not in ids
    assert "pdf" in ids
    assert registry.get("ocr") is None


def test_startup_banner_format(monkeypatch):
    monkeypatch.setenv("IKM_DISABLED_PROCESSORS", "ocr,image")
    PCP.reset_instance()
    banner = format_processor_startup_banner()
    assert "Processor Configuration Loaded" in banner
    assert "Enabled:" in banner
    assert "✓ PDF" in banner
    assert "Disabled:" in banner
    assert "✗ OCR" in banner
    assert "✗ Image" in banner
    assert "Pipeline:" in banner
    assert "→" in banner


def test_dynamic_pipeline_skips_unsupported(monkeypatch):
    monkeypatch.setenv(
        "IKM_ENABLED_PROCESSORS",
        "pdf,docx,ocr,security,key_fields,reference_extraction,image,embedding,reranker",
    )
    PCP.reset_instance()
    PR.reset_instance()
    registry = bootstrap_deployment_processors(initialize=True)
    pipeline = DynamicProcessingPipeline(registry.enabled_instances())
    result = pipeline.run({"document_path": "notes.txt", "file_extension": "txt"})
    # PDF/DOCX supports() is false for .txt — they should be skipped, others run.
    assert "pdf" in result["pipeline_skipped"]
    assert "docx" in result["pipeline_skipped"]
    assert "security" in result["pipeline_executed"]
    assert "embedding" in result["pipeline_executed"]


def test_enabled_processor_types_respects_deployment_gate(monkeypatch):
    monkeypatch.setenv("PROCESSOR_KEY_FIELDS_ENABLED", "false")
    monkeypatch.setenv("PROCESSOR_REFERENCE_EXTRACTION_ENABLED", "false")
    PCP.reset_instance()

    types = enabled_processor_types(
        {
            "key_field_extraction": True,
            "validation_enabled": True,
            "reference_document_extraction": True,
            "metadata_extraction": True,
        }
    )
    assert "chunking_vectorizing" in types
    assert "metadata_extraction" in types
    assert "key_field_extraction" not in types
    assert "document_validation" not in types
    assert "reference_document_extraction" not in types


def test_embedding_disabled_blocks_chunking_queue(monkeypatch):
    monkeypatch.setenv("PROCESSOR_EMBEDDING_ENABLED", "false")
    PCP.reset_instance()
    types = enabled_processor_types({})
    assert "chunking_vectorizing" not in types


def test_api_status_shape(tmp_path, monkeypatch):
    config = tmp_path / "processors.yaml"
    config.write_text(
        "processors:\n  pdf: true\n  ocr: false\n  docx: true\n  security: true\n"
        "  key_fields: true\n  reference_extraction: false\n  image: false\n"
        "  embedding: true\n  reranker: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PROCESSORS_CONFIG_PATH", str(config))
    PCP.reset_instance()
    status = ProcessorConfigurationProvider.instance().status()
    assert status["enabled"] == [
        "pdf",
        "docx",
        "security",
        "key_fields",
        "embedding",
        "reranker",
    ]
    assert status["disabled"] == ["ocr", "reference_extraction", "image"]


def test_bootstrap_logs_banner(caplog, monkeypatch):
    monkeypatch.setenv("IKM_DISABLED_PROCESSORS", "ocr")
    PCP.reset_instance()
    PR.reset_instance()
    with caplog.at_level(logging.INFO):
        bootstrap_deployment_processors(initialize=True, log=logging.getLogger("test"))
    joined = "\n".join(caplog.messages)
    assert "Processor Configuration Loaded" in joined
    assert "✗ OCR" in joined
    assert "Pipeline:" in joined


def test_reload_keeps_previous_on_unknown(tmp_path, monkeypatch):
    config = tmp_path / "processors.yaml"
    config.write_text(
        "processors:\n  pdf: true\n  docx: true\n  ocr: false\n  security: true\n"
        "  key_fields: true\n  reference_extraction: true\n  image: true\n"
        "  embedding: true\n  reranker: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PROCESSORS_CONFIG_PATH", str(config))
    PCP.reset_instance()
    PR.reset_instance()
    reset_active_pipeline()
    from src.features.document_processing.shared_processor.deployment.reload import reload_deployment_processors

    bootstrap_deployment_processors(initialize=True)
    before = ProcessorConfigurationProvider.instance().enabled_processors()

    config.write_text("processors:\n  pdf: true\n  totally_unknown: true\n", encoding="utf-8")
    result = reload_deployment_processors(keep_previous_on_error=True, reason="test")
    assert result.ok is False
    assert result.kept_previous_config is True
    assert ProcessorConfigurationProvider.instance().enabled_processors() == before
