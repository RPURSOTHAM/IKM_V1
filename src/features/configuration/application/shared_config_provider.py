from __future__ import annotations

"""Lightweight platform config reader for scheduler and processor tiers."""

from src.features.configuration.application.config_provider import (
    NamespaceView,
    PlatformConfigProvider,
    get_platform_config,
)

SharedPlatformConfigProvider = PlatformConfigProvider


def get_shared_platform_config() -> PlatformConfigProvider:
    return get_platform_config()


__all__ = ["NamespaceView", "PlatformConfigProvider", "get_shared_platform_config"]
