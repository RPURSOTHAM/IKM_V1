from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features.documents.application.document_service import DocumentReceiverService
from src.features.security.review.human_review_queue import (
    HumanReviewQueue,
    ensure_human_review_queue_entry,
)
from src.features.security.review.review_api import _status_filter_matches


def test_status_filter_matches_human_review_alias() -> None:
    item = {"status": "PENDING", "review_status": "PENDING"}
    assert _status_filter_matches(item, "human_review") is True
    assert _status_filter_matches(item, "APPROVED") is False


def test_ensure_human_review_queue_entry_creates_pending_review(tmp_path: Path) -> None:
    queue_file = tmp_path / "human_review_queue.json"
    scan = {
        "status": "human_review",
        "severity": "medium",
        "risk_level": "MEDIUM",
        "dlp_decision": "HUMAN_REVIEW",
        "requires_human_review": True,
        "available_reviewer_actions": ["ALLOW", "MASK_AND_ALLOW", "BLOCK"],
        "reason": "Needs review",
        "detections": [],
    }
    review_id = ensure_human_review_queue_entry(
        document_id="doc-123",
        document_name="doc-123.pdf",
        security_scan=scan,
        queue_file=str(queue_file),
    )
    assert review_id == "rev_doc-123"
    rows = json.loads(queue_file.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["status"] == "PENDING"
    assert rows[0]["document_id"] == "doc-123"


def test_merge_job_preserves_human_review_over_received() -> None:
    svc = DocumentReceiverService()
    record = {
        "document_id": "doc-456",
        "status": "human_review",
        "metadata": {
            "requires_human_review": True,
            "security_scan": {
                "status": "human_review",
                "dlp_decision": "HUMAN_REVIEW",
                "requires_human_review": True,
            },
        },
    }
    job = {"job_id": "job-1", "status": "RECEIVED"}
    merged = svc.merge_job_into_record(record, job)
    assert merged["status"] == "human_review"
    assert merged["job_id"] == "job-1"


def test_merge_job_preserves_human_review_over_queued_and_completed() -> None:
    """Job progress must not hide a pending Security Human Review hold."""
    svc = DocumentReceiverService()
    record = {
        "document_id": "doc-hold-1",
        "status": "human_review",
        "metadata": {
            "requires_human_review": True,
            "security_scan": {
                "status": "human_review",
                "dlp_decision": "HUMAN_REVIEW",
                "requires_human_review": True,
            },
        },
    }
    for job_status in ("QUEUED", "DISPATCHING", "ASSIGNED", "IN_PROGRESS", "COMPLETED"):
        merged = svc.merge_job_into_record(record, {"job_id": "job-x", "status": job_status})
        assert merged["status"] == "human_review", job_status


def test_register_human_review_queue_entry_writes_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue_file = tmp_path / "human_review_queue.json"
    monkeypatch.setenv("HOST_DOCUMENTS_DIR", str(tmp_path))
    record = {
        "document_id": "doc-789",
        "document_name": "doc-789.pdf",
        "metadata": {
            "security_scan": {
                "status": "human_review",
                "severity": "medium",
                "risk_level": "MEDIUM",
                "dlp_decision": "HUMAN_REVIEW",
                "requires_human_review": True,
                "available_reviewer_actions": ["ALLOW", "MASK_AND_ALLOW", "BLOCK"],
                "reason": "Review me",
                "detections": [],
            }
        },
    }
    review_id = DocumentReceiverService._register_human_review_queue_entry(record, record["metadata"]["security_scan"])
    assert review_id == "rev_doc-789"
    assert record["metadata"]["review_id"] == "rev_doc-789"
    queue = HumanReviewQueue(queue_file=str(queue_file))
    assert queue.get_pending_reviews()[0]["review_id"] == "rev_doc-789"
