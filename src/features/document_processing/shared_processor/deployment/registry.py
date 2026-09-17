"""Central registry of deployment processors.

Avoids hardcoded processor initialization throughout the codebase.
Only processors enabled by ``ProcessorConfigurationProvider`` are instantiated.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

from src.features.document_processing.shared_processor.deployment.config_provider import (
    ProcessorConfigurationError,
    ProcessorConfigurationProvider,
)
from src.features.document_processing.shared_processor.deployment.ids import (
    DEPLOYMENT_PROCESSOR_ORDER,
    DeploymentProcessorId,
)
from src.features.document_processing.shared_processor.deployment.implementations import PROCESSOR_CLASSES
from src.features.document_processing.shared_processor.deployment.interface import DeploymentProcessor

logger = logging.getLogger(__name__)


class ProcessorRegistry:
    """Catalog of available processors + runtime instances for enabled ones."""

    _lock = threading.RLock()
    _instance: "ProcessorRegistry | None" = None

    def __init__(
        self,
        *,
        config_provider: ProcessorConfigurationProvider | None = None,
    ) -> None:
        self._config = config_provider or ProcessorConfigurationProvider.instance()
        self._available: dict[str, type[DeploymentProcessor]] = {
            pid.value: cls for pid, cls in PROCESSOR_CLASSES.items()
        }
        self._instances: dict[str, DeploymentProcessor] = {}
        self._bootstrapped = False

    @classmethod
    def instance(cls) -> "ProcessorRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown_all()
            cls._instance = None

    def available_ids(self) -> list[str]:
        return [pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER if pid.value in self._available]

    def register(self, processor_id: str, cls: type[DeploymentProcessor]) -> None:
        name = str(processor_id).strip().lower()
        if name not in {p.value for p in DeploymentProcessorId}:
            raise ProcessorConfigurationError(
                f"Cannot register unknown deployment processor '{processor_id}'."
            )
        self._available[name] = cls

    def get(self, processor_id: str | DeploymentProcessorId) -> DeploymentProcessor | None:
        name = processor_id.value if isinstance(processor_id, DeploymentProcessorId) else str(processor_id)
        return self._instances.get(name)

    def enabled_instances(self) -> list[DeploymentProcessor]:
        order = [pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER]
        return [self._instances[name] for name in order if name in self._instances]

    def bootstrap(self, *, initialize: bool = True) -> list[DeploymentProcessor]:
        """Instantiate and optionally initialize only enabled processors."""
        with self._lock:
            enabled = set(self._config.enabled_processors())
            disabled = set(self._config.disabled_processors())

            # Tear down anything that is no longer enabled.
            for name in list(self._instances.keys()):
                if name not in enabled:
                    self._instances[name].shutdown()
                    del self._instances[name]

            created: list[DeploymentProcessor] = []
            for pid in DEPLOYMENT_PROCESSOR_ORDER:
                name = pid.value
                if name in disabled or name not in enabled:
                    continue
                cls = self._available.get(name)
                if cls is None:
                    raise ProcessorConfigurationError(
                        f"Enabled processor '{name}' has no registered implementation."
                    )
                if name not in self._instances:
                    instance = cls()
                    self._instances[name] = instance
                    created.append(instance)
                    if initialize:
                        instance.initialize()
                elif initialize and not self._instances[name].initialized:
                    self._instances[name].initialize()

            self._bootstrapped = True
            return list(self.enabled_instances())

    def ensure_bootstrapped(self) -> None:
        if not self._bootstrapped:
            self.bootstrap(initialize=True)

    def shutdown_all(self) -> None:
        with self._lock:
            for instance in list(self._instances.values()):
                try:
                    instance.shutdown()
                except Exception:
                    logger.exception("Failed shutting down processor %s", instance.id)
            self._instances.clear()
            self._bootstrapped = False

    def require_enabled(self, processor_id: str | DeploymentProcessorId) -> None:
        name = (
            processor_id.value
            if isinstance(processor_id, DeploymentProcessorId)
            else str(processor_id).strip().lower()
        )
        if not self._config.is_processor_enabled(name):
            raise ProcessorConfigurationError(
                f"Deployment processor '{name}' is disabled for this IKM instance."
            )


def get_processor_registry() -> ProcessorRegistry:
    return ProcessorRegistry.instance()


def iter_enabled_processors() -> Iterable[DeploymentProcessor]:
    registry = get_processor_registry()
    registry.ensure_bootstrapped()
    return registry.enabled_instances()
