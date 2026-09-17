"""document_job effective status when parallel processors share one row."""

from __future__ import annotations

from src.infrastructure.database.document_jobs import (
    aggregate_processor_plan_status,
    effective_job_status,
)


def test_effective_job_status_keeps_in_progress_while_sibling_pending() -> None:
    job = {
        "status": "IN_PROGRESS",
        "scheduling_metadata": {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "chunking_vectorizing": {"status": "FAILED", "error_details": "model missing"},
            },
        },
    }
    assert effective_job_status(job) == "IN_PROGRESS"


def test_effective_job_status_failed_only_when_all_processors_terminal() -> None:
    job = {
        "status": "IN_PROGRESS",
        "scheduling_metadata": {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "chunking_vectorizing": {"status": "FAILED", "error_details": "model missing"},
                "template_extraction": {
                    "status": "COMPLETED",
                    "result_location": "neo4j://DocumentTemplate/doc-1",
                },
            },
        },
    }
    assert effective_job_status(job) == "FAILED"
    assert aggregate_processor_plan_status(job["scheduling_metadata"]) == "FAILED"


def test_effective_job_status_completed_when_all_succeed() -> None:
    job = {
        "status": "IN_PROGRESS",
        "scheduling_metadata": {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "chunking_vectorizing": {"status": "COMPLETED"},
                "template_extraction": {"status": "COMPLETED"},
            },
        },
    }
    assert effective_job_status(job) == "COMPLETED"


def test_effective_job_status_recovers_premature_failed_row() -> None:
    job = {
        "status": "FAILED",
        "scheduling_metadata": {
            "enabled_processor_types": ["chunking_vectorizing", "template_extraction"],
            "processor_results": {
                "chunking_vectorizing": {"status": "FAILED"},
            },
        },
    }
    assert effective_job_status(job) == "IN_PROGRESS"


def test_effective_job_status_maps_assigned_to_in_progress_while_plan_pending() -> None:
    job = {
        "status": "ASSIGNED",
        "scheduling_metadata": {
            "enabled_processor_types": ["chunking_vectorizing", "metadata_extraction"],
            "processor_results": {},
        },
    }
    assert effective_job_status(job) == "IN_PROGRESS"


def test_effective_job_status_keeps_completed_terminal_when_plan_complete() -> None:
    job = {
        "status": "COMPLETED",
        "scheduling_metadata": {
            "enabled_processor_types": ["chunking_vectorizing"],
            "processor_results": {
                "chunking_vectorizing": {"status": "COMPLETED"},
            },
        },
    }
    assert effective_job_status(job) == "COMPLETED"
