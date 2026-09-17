"""Unit tests for independent processor outcome aggregation on document_job."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.infrastructure.database.document_jobs import DocumentJobStore


def _store_with_meta(meta: dict) -> tuple[DocumentJobStore, MagicMock]:
    store = DocumentJobStore.__new__(DocumentJobStore)
    store._scheduling_metadata = MagicMock(return_value=dict(meta))
    store._write_scheduling_metadata = MagicMock()
    store.heartbeat = MagicMock()
    store.update_failed = MagicMock()
    store.update_completed = MagicMock()
    store._sync_processor_results_to_intake = MagicMock()
    return store, store._write_scheduling_metadata


def test_chunking_failure_does_not_fail_job_while_template_pending() -> None:
    store, write_meta = _store_with_meta(
        {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {},
        }
    )

    store.record_processor_outcome(
        "doc-1",
        processor_type="chunking_vectorizing",
        status="FAILED",
        error_details="EmbeddingModelConfigurationError: bge-base-en missing",
    )

    store.update_failed.assert_not_called()
    store.heartbeat.assert_called_once()
    written = write_meta.call_args.args[1]
    assert written["processor_results"]["chunking_vectorizing"]["status"] == "FAILED"


def test_template_success_then_chunking_failure_finalizes_failed_with_both_results() -> None:
    store, write_meta = _store_with_meta(
        {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "template_extraction": {
                    "status": "COMPLETED",
                    "result_location": "neo4j://DocumentTemplate/doc-1",
                    "document_metadata": {},
                    "error_details": None,
                    "completed_at": "2026-01-01T00:00:00",
                }
            },
        }
    )

    store.record_processor_outcome(
        "doc-1",
        processor_type="chunking_vectorizing",
        status="FAILED",
        error_details="model missing",
    )

    store.update_failed.assert_called_once()
    store.update_completed.assert_not_called()
    written = write_meta.call_args.args[1]
    results = written["processor_results"]
    assert results["template_extraction"]["status"] == "COMPLETED"
    assert results["template_extraction"]["result_location"] == "neo4j://DocumentTemplate/doc-1"
    assert results["chunking_vectorizing"]["status"] == "FAILED"


def test_all_processors_completed_marks_job_completed() -> None:
    store, _write_meta = _store_with_meta(
        {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "chunking_vectorizing": {
                    "status": "COMPLETED",
                    "result_location": "weaviate://DocumentChunk/doc-1",
                    "document_metadata": {},
                    "error_details": None,
                    "completed_at": "2026-01-01T00:00:00",
                }
            },
        }
    )

    store.record_processor_outcome(
        "doc-1",
        processor_type="template_extraction",
        status="COMPLETED",
        result_location="neo4j://DocumentTemplate/doc-1",
    )

    store.update_completed.assert_called_once()
    store.update_failed.assert_not_called()


def test_list_pending_skips_failed_processors() -> None:
    store = DocumentJobStore.__new__(DocumentJobStore)
    store._scheduling_metadata = MagicMock(
        return_value={
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "chunking_vectorizing": {"status": "FAILED"},
            },
        }
    )
    pending = store.list_pending_processor_types("doc-1")
    assert pending == ["template_extraction"]
