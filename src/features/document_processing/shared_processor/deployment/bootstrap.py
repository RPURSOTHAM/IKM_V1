"""Startup bootstrap + logging for deployment processors."""

from __future__ import annotations

import logging

from src.features.document_processing.shared_processor.deployment.config_provider import (
    ProcessorConfigurationError,
    ProcessorConfigurationProvider,
)
from src.features.document_processing.shared_processor.deployment.ids import display_name
from src.features.document_processing.shared_processor.deployment.pipeline import rebuild_active_pipeline
from src.features.document_processing.shared_processor.deployment.registry import ProcessorRegistry, get_processor_registry

logger = logging.getLogger(__name__)


def format_processor_startup_banner(
    *,
    enabled: list[str] | None = None,
    disabled: list[str] | None = None,
    pipeline: list[str] | None = None,
) -> str:
    provider = ProcessorConfigurationProvider.instance()
    enabled_ids = enabled if enabled is not None else provider.enabled_processors()
    disabled_ids = disabled if disabled is not None else provider.disabled_processors()
    if pipeline is None:
        pipeline_ids = list(enabled_ids)
    else:
        pipeline_ids = list(pipeline)

    lines = [
        "------------------------------------------------",
        "Processor Configuration Loaded",
        "",
        "Enabled:",
    ]
    if enabled_ids:
        for pid in enabled_ids:
            lines.append(f"  ✓ {display_name(pid)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Disabled:")
    if disabled_ids:
        for pid in disabled_ids:
            lines.append(f"  ✗ {display_name(pid)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Pipeline:")
    if pipeline_ids:
        lines.append("  " + " → ".join(display_name(pid) for pid in pipeline_ids))
    else:
        lines.append("  (none)")
    lines.append("------------------------------------------------")
    return "\n".join(lines)


def log_processor_startup_banner(log: logging.Logger | None = None) -> None:
    target = log or logger
    banner = format_processor_startup_banner()
    for line in banner.splitlines():
        target.info(line)


def bootstrap_deployment_processors(
    *,
    initialize: bool = True,
    log: logging.Logger | None = None,
    start_watcher: bool = False,
) -> ProcessorRegistry:
    """Read deployment config, instantiate enabled processors only, log status.

    Raises ``ProcessorConfigurationError`` on unknown processor IDs so startup fails loudly.
    """
    target = log or logger
    try:
        provider = ProcessorConfigurationProvider.instance()
        # Force load + validation before registry bootstrap.
        # Initial load must not keep a prior empty state — fail closed on bad config.
        provider.reload(keep_previous_on_error=False)
    except ProcessorConfigurationError:
        target.error("Failed to load deployment processor configuration.")
        raise

    registry = get_processor_registry()
    registry.bootstrap(initialize=initialize)
    rebuild_active_pipeline(registry)
    log_processor_startup_banner(target)

    if start_watcher:
        try:
            from src.features.document_processing.shared_processor.deployment.watcher import start_processor_config_watcher

            start_processor_config_watcher(initialize=initialize, log=target)
        except Exception:
            target.exception("Failed to start processor config file watcher.")

    return registry
