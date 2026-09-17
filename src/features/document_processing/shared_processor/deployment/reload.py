"""Thread-safe reload of deployment processor configuration + runtime rebuild."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from src.features.document_processing.shared_processor.deployment.bootstrap import log_processor_startup_banner
from src.features.document_processing.shared_processor.deployment.config_provider import (
    ProcessorConfigurationError,
    ProcessorConfigurationProvider,
)
from src.features.document_processing.shared_processor.deployment.ids import display_name
from src.features.document_processing.shared_processor.deployment.pipeline import (
    DynamicProcessingPipeline,
    get_active_pipeline,
    rebuild_active_pipeline,
)
from src.features.document_processing.shared_processor.deployment.registry import ProcessorRegistry, get_processor_registry

logger = logging.getLogger(__name__)

_RELOAD_LOCK = threading.RLock()


@dataclass
class ProcessorReloadResult:
    """Outcome of a configuration reload attempt."""

    ok: bool
    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    pipeline: list[str] = field(default_factory=list)
    newly_enabled: list[str] = field(default_factory=list)
    newly_disabled: list[str] = field(default_factory=list)
    previous_enabled: list[str] = field(default_factory=list)
    config_path: str = ""
    error: str | None = None
    kept_previous_config: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "enabled": list(self.enabled),
            "disabled": list(self.disabled),
            "pipeline": list(self.pipeline),
            "newly_enabled": list(self.newly_enabled),
            "newly_disabled": list(self.newly_disabled),
            "previous_enabled": list(self.previous_enabled),
            "config_path": self.config_path,
            "kept_previous_config": self.kept_previous_config,
        }
        if self.error:
            payload["error"] = self.error
        return payload


def _clear_disabled_model_caches(disabled_ids: list[str]) -> None:
    """Best-effort release of models for processors that were just disabled."""
    if "embedding" in disabled_ids:
        try:
            from src.features.embeddings.application.embedding_service import clear_embedding_model_cache

            clear_embedding_model_cache()
            logger.info("Cleared embedding model cache after disabling embedding processor.")
        except Exception:
            logger.exception("Failed to clear embedding model cache.")

    if "reranker" in disabled_ids:
        try:
            from src.features.retrieval.application.retrieval_service import clear_reranker_model_cache

            clear_reranker_model_cache()
            logger.info("Cleared reranker model cache after disabling reranker processor.")
        except Exception:
            # Retrieval service may not be imported in processor-only workers.
            logger.debug("Reranker cache clear skipped or failed.", exc_info=True)

    # Drop lazily constructed typed BaseProcessor instances so disabled jobs cannot reuse them.
    try:
        from src.features.document_processing.processors import reset_processor_cache

        reset_processor_cache()
    except Exception:
        logger.debug("Typed processor cache reset skipped.", exc_info=True)


def reload_deployment_processors(
    *,
    initialize: bool = True,
    keep_previous_on_error: bool = True,
    log: logging.Logger | None = None,
    reason: str = "manual",
) -> ProcessorReloadResult:
    """Reload YAML config, rebuild registry + pipeline under a process-wide lock.

    On invalid configuration (unknown processor IDs, bad YAML), when
    ``keep_previous_on_error`` is True and a prior valid config exists, the previous
    configuration remains active and an error result is returned / raised via
    ``ProcessorConfigurationError`` after restoring state.
    """
    target = log or logger
    with _RELOAD_LOCK:
        provider = ProcessorConfigurationProvider.instance()
        previous_enabled = list(provider.enabled_processors()) if provider.has_loaded_flags() else []
        config_path = str(provider.config_path)

        try:
            provider.reload(keep_previous_on_error=keep_previous_on_error)
        except ProcessorConfigurationError as exc:
            target.error(
                "Processor configuration reload failed (%s); keeping previous configuration: %s",
                reason,
                exc,
            )
            current = provider.status() if provider.has_loaded_flags() else {
                "enabled": previous_enabled,
                "disabled": [],
            }
            pipeline = [p.id for p in get_active_pipeline().processors] if provider.has_loaded_flags() else []
            return ProcessorReloadResult(
                ok=False,
                enabled=list(current.get("enabled") or previous_enabled),
                disabled=list(current.get("disabled") or []),
                pipeline=pipeline,
                previous_enabled=previous_enabled,
                config_path=config_path,
                error=str(exc),
                kept_previous_config=bool(previous_enabled) or provider.has_loaded_flags(),
            )

        registry = get_processor_registry()
        before_ids = {p.id for p in registry.enabled_instances()}
        registry.bootstrap(initialize=initialize)
        after_ids = {p.id for p in registry.enabled_instances()}

        newly_enabled = sorted(after_ids - before_ids)
        newly_disabled = sorted(before_ids - after_ids)

        if newly_enabled:
            target.info(
                "Newly enabled processors: %s",
                ", ".join(display_name(pid) for pid in newly_enabled),
            )
        if newly_disabled:
            target.info(
                "Newly disabled processors: %s",
                ", ".join(display_name(pid) for pid in newly_disabled),
            )
            _clear_disabled_model_caches(newly_disabled)

        pipeline = rebuild_active_pipeline(registry)
        pipeline_ids = [p.id for p in pipeline.processors]

        log_processor_startup_banner(target)
        target.info("Processor configuration reloaded (%s).", reason)

        status = provider.status()
        return ProcessorReloadResult(
            ok=True,
            enabled=list(status["enabled"]),
            disabled=list(status["disabled"]),
            pipeline=pipeline_ids,
            newly_enabled=newly_enabled,
            newly_disabled=newly_disabled,
            previous_enabled=previous_enabled,
            config_path=config_path,
            kept_previous_config=False,
        )


def reload_deployment_processors_or_raise(
    *,
    initialize: bool = True,
    keep_previous_on_error: bool = True,
    log: logging.Logger | None = None,
    reason: str = "manual",
) -> ProcessorReloadResult:
    """Same as ``reload_deployment_processors`` but raises on invalid configuration."""
    result = reload_deployment_processors(
        initialize=initialize,
        keep_previous_on_error=keep_previous_on_error,
        log=log,
        reason=reason,
    )
    if not result.ok:
        raise ProcessorConfigurationError(result.error or "Processor configuration reload failed.")
    return result
