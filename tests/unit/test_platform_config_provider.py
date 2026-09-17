"""Unit tests for platform config provider normalization."""

from src.features.configuration.application.config_provider import PlatformConfigProvider


def test_platform_config_provider_normalizes_string_values():
    provider = PlatformConfigProvider(store=None)
    normalized = provider._normalize_value("processor", "embedding.model_name", "  BAAI/bge-base-en  ")
    assert normalized == "BAAI/bge-base-en"
