from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.infrastructure.document_databases.redis_store import store_render_payload
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.document_processing.processors.base import BaseProcessor
from src.features.documents.infrastructure.content.convert_for_rendering import convert_document_for_rendering
from src.features.documents.infrastructure.content.render_payload import build_render_payload
from src.features.document_processing.shared_processor.types import ProcessorType


class ConversionForRenderingProcessor(BaseProcessor):
    processor_type = ProcessorType.CONVERSION_FOR_RENDERING

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        set_status("loading_document", 10.0)
        blocks, doc_meta = load_document_blocks(document_path)
        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        set_status("converting", 45.0)
        title = str(request.original_file_name or request.document_name or document_path.name)
        documents_root = Path(os.getenv("DOCUMENT_ROOT") or document_path.parent)
        result = convert_document_for_rendering(
            document_path,
            document_id=request.document_id,
            title=title,
            blocks=blocks,
            doc_meta=doc_meta,
            documents_root=documents_root,
        )
        payload = build_render_payload(
            document_id=request.document_id,
            document_name=request.document_name,
            result=result,
        )

        set_status("storing", 85.0)
        ttl = int(os.getenv("RENDER_CACHE_TTL_SECONDS") or "86400")
        location = store_render_payload(document_id=request.document_id, payload=payload, ttl_seconds=ttl)
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata={
                **doc_meta,
                "render_format": result.format,
                "conversion_method": result.conversion_method,
            },
            artifacts={
                "format": result.format,
                "conversion_method": result.conversion_method,
                "block_count": len(blocks),
            },
        )
