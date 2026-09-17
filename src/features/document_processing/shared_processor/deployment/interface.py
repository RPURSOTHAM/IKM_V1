"""Common interface for deployment-scoped processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.features.document_processing.shared_processor.deployment.ids import DeploymentProcessorId, display_name


class DeploymentProcessor(ABC):
    """Lifecycle contract for processors gated by deployment configuration.

    Implementations must be cheap to construct; heavy work (model loads, clients)
    belongs in ``initialize()``, which is only called when the processor is enabled.
    """

    processor_id: DeploymentProcessorId

    def __init__(self) -> None:
        self._initialized = False

    @property
    def id(self) -> str:
        return self.processor_id.value

    @property
    def display_name(self) -> str:
        return display_name(self.processor_id)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Load models / clients required by this processor."""
        if self._initialized:
            return
        self._do_initialize()
        self._initialized = True

    def shutdown(self) -> None:
        """Release resources acquired during initialize()."""
        if not self._initialized:
            return
        self._do_shutdown()
        self._initialized = False

    @abstractmethod
    def _do_initialize(self) -> None:
        raise NotImplementedError

    def _do_shutdown(self) -> None:
        """Optional cleanup; default is a no-op."""

    @abstractmethod
    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute this processor against a shared pipeline context.

        Returns a dict of updates to merge into the context (or the full context).
        """
        raise NotImplementedError

    def supports(self, context: dict[str, Any]) -> bool:
        """Return False to skip this processor for the current document/context."""
        return True
