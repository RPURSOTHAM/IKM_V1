"""Media transcription processor — Whisper STT for audio/video CIH uploads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.features.document_processing.media.cih_media import (
    is_media_suffix,
    load_transcript_sidecar,
    sidecar_is_reusable,
    transcribe_media_file,
    write_transcript_sidecars,
)
from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_processing.shared_processor.types import ProcessorType
from src.infrastructure.document_databases.neo4j_store import store_document_graph

logger = logging.getLogger(__name__)


class MediaTranscriptionProcessor(BaseProcessor):
    processor_type = ProcessorType.MEDIA_TRANSCRIPTION

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        suffix = document_path.suffix.lower()
        if not is_media_suffix(suffix):
            set_status("skipped_not_media", 100.0)
            return ProcessorResult(
                processor_type=self.processor_type.value,
                document_id=request.document_id,
                storage_backend=self.storage_backend,
                result_location=None,
                document_metadata={
                    "skipped": True,
                    "skip_reason": "not_audio_or_video",
                    "file_extension": suffix.lstrip("."),
                },
                artifacts={"skipped": True},
            )

        existing = load_transcript_sidecar(document_path)
        if sidecar_is_reusable(existing):
            payload = existing
            set_status("reusing_transcript_sidecar", 40.0)
        else:
            if existing:
                logger.info(
                    "Ignoring stale transcript sidecar for %s (model/task mismatch or force retranscribe)",
                    document_path.name,
                )
            set_status("transcribing", 20.0)
            if check_stop():
                raise RuntimeError("Stop requested during media transcription")
            payload = transcribe_media_file(document_path)
            write_transcript_sidecars(document_path, payload)
            set_status("transcript_written", 70.0)

        location = store_document_graph(
            document_id=request.document_id,
            repository_id=request.repository_id,
            processor_type=self.processor_type.value,
            payload=payload,
            document_type_id=request.document_type_id,
            document_type_name=request.document_type_name,
        )
        set_status("completed", 100.0)
        logger.info(
            "Media transcription complete document_id=%s segments=%s",
            request.document_id,
            payload.get("segment_count"),
        )
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata={
                "media_type": payload.get("media_type"),
                "segment_count": payload.get("segment_count"),
                "provider": payload.get("provider"),
                "model": payload.get("model"),
                "task": payload.get("task"),
                "language": payload.get("language"),
                "output_language": payload.get("output_language"),
                "character_count": len(str(payload.get("full_text") or "")),
            },
            artifacts={
                "segments": payload.get("segments") or [],
                "full_text": payload.get("full_text") or "",
            },
        )
