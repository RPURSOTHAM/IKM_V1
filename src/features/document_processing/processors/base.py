from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.features.document_processing.shared_processor.types import PROCESSOR_STORAGE, ProcessorType


class BaseProcessor(ABC):
    processor_type: ProcessorType

    def initialize(self) -> None:
        """Optional startup hook (models/clients). Default is a no-op."""

    def shutdown(self) -> None:
        """Optional cleanup hook. Default is a no-op."""

    @abstractmethod
    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        raise NotImplementedError

    # Alias so typed processors share the deployment-facing process() vocabulary.
    def process(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        return self.run(
            request,
            document_path,
            set_status=set_status,
            check_stop=check_stop,
        )

    @property
    def storage_backend(self) -> str:
        return PROCESSOR_STORAGE[self.processor_type]
