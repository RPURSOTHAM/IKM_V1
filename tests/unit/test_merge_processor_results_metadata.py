"""merge_job_into_record must expose processor_results on document metadata."""

from __future__ import annotations

from src.features.documents.application.document_service import DocumentReceiverService


def test_merge_job_copies_processor_results_into_metadata() -> None:
    svc = DocumentReceiverService()
    record = {
        "document_id": "doc-1",
        "status": "processing",
        "metadata": {"job_id": "job-1", "batch_id": "batch-1"},
    }
    job = {
        "job_id": "job-1",
        "status": "FAILED",
        "error_details": "embedding missing",
        "scheduling_metadata": {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "chunking_vectorizing": {
                    "status": "FAILED",
                    "error_details": "embedding missing",
                    "result_location": None,
                },
                "template_extraction": {
                    "status": "COMPLETED",
                    "result_location": "neo4j://DocumentTemplate/doc-1",
                    "completed_at": "2026-07-30T00:00:00",
                },
            },
        },
    }

    merged = svc.merge_job_into_record(record, job)
    assert merged["status"] == "failed"
    assert merged["error_details"] == "embedding missing"
    results = merged["metadata"]["processor_results"]
    assert results["chunking_vectorizing"]["status"] == "FAILED"
    assert results["template_extraction"]["status"] == "COMPLETED"
    assert merged["metadata"]["template_extraction"]["result_location"] == "neo4j://DocumentTemplate/doc-1"
    assert merged["metadata"]["enabled_processor_types"] == [
        "chunking_vectorizing",
        "template_extraction",
    ]
