from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.features.document_processing.processors.base import BaseProcessor
from src.features.references.application.reference_pipeline import run_pre_chunk_reference_extraction
from src.features.document_processing.shared_processor.types import ProcessorType


class ReferenceDocumentExtractionProcessor(BaseProcessor):
    processor_type = ProcessorType.REFERENCE_DOCUMENT_EXTRACTION

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        return run_pre_chunk_reference_extraction(
            request,
            document_path,
            set_status=set_status,
            check_stop=check_stop,
        )
