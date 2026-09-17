from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.features.references.application.reference_pipeline import run_pre_chunk_reference_extraction


class ReferenceExtractionProcessor:
    processor_type = "reference_extraction"
    storage_backend = "neo4j"

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
