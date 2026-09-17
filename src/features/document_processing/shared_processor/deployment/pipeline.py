"""Dynamic processing pipeline over enabled deployment processors."""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

from src.features.document_processing.shared_processor.deployment.interface import DeploymentProcessor
from src.features.document_processing.shared_processor.deployment.registry import ProcessorRegistry, get_processor_registry

logger = logging.getLogger(__name__)

_PIPELINE_LOCK = threading.RLock()
_ACTIVE_PIPELINE: "DynamicProcessingPipeline | None" = None


class DynamicProcessingPipeline:
    """Iterate the enabled processor list instead of a hardcoded stage order."""

    def __init__(self, processors: Iterable[DeploymentProcessor] | None = None) -> None:
        self._processors = list(processors) if processors is not None else None

    @classmethod
    def from_registry(cls, registry: ProcessorRegistry | None = None) -> "DynamicProcessingPipeline":
        reg = registry or get_processor_registry()
        reg.ensure_bootstrapped()
        return cls(reg.enabled_instances())

    @property
    def processors(self) -> list[DeploymentProcessor]:
        if self._processors is None:
            return list(get_active_pipeline().processors)
        return list(self._processors)

    @property
    def pipeline_ids(self) -> list[str]:
        return [p.id for p in self.processors]

    def describe(self) -> str:
        ids = self.pipeline_ids
        if not ids:
            return "(none)"
        from src.features.document_processing.shared_processor.deployment.ids import display_name

        return " → ".join(display_name(pid) for pid in ids)

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        # Snapshot processors under lock so a concurrent reload cannot mutate mid-run.
        with _PIPELINE_LOCK:
            processors = list(self.processors)

        state: dict[str, Any] = dict(context or {})
        executed: list[str] = []
        skipped: list[str] = []

        for processor in processors:
            if not processor.supports(state):
                skipped.append(processor.id)
                logger.debug("Skipping processor %s (supports=False)", processor.id)
                continue
            if not processor.initialized:
                processor.initialize()
            logger.debug("Running deployment processor: %s", processor.id)
            updates = processor.process(state) or {}
            if isinstance(updates, dict):
                state.update(updates)
            executed.append(processor.id)

        state["pipeline_executed"] = executed
        state["pipeline_skipped"] = skipped
        return state


def get_active_pipeline() -> DynamicProcessingPipeline:
    """Return the process-wide pipeline, building it from the registry if needed."""
    global _ACTIVE_PIPELINE
    with _PIPELINE_LOCK:
        if _ACTIVE_PIPELINE is None:
            _ACTIVE_PIPELINE = DynamicProcessingPipeline.from_registry()
        return _ACTIVE_PIPELINE


def rebuild_active_pipeline(registry: ProcessorRegistry | None = None) -> DynamicProcessingPipeline:
    """Replace the active pipeline after a configuration reload (thread-safe)."""
    global _ACTIVE_PIPELINE
    with _PIPELINE_LOCK:
        _ACTIVE_PIPELINE = DynamicProcessingPipeline.from_registry(registry)
        return _ACTIVE_PIPELINE


def reset_active_pipeline() -> None:
    """Test helper — drop the cached pipeline."""
    global _ACTIVE_PIPELINE
    with _PIPELINE_LOCK:
        _ACTIVE_PIPELINE = None
