from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.features.security.review.human_review_queue import HumanReviewQueue
from src.features.security.application.upload_security_pipeline import reset_security_pipeline, run_security_pipeline


@pytest.fixture
def temp_txt_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CONFIDENTIAL_CORPUS_DIR", str(tmp_path / "isolated_corpus"))
    created: list[Path] = []

    def _make(content: str, *, suffix: str = ".txt") -> Path:
        fd, name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        path = Path(name)
        path.write_text(content, encoding="utf-8")
        created.append(path)
        return path

    yield _make

    for path in created:
        path.unlink(missing_ok=True)


def _scan(content: str, maker, document_id: str) -> dict:
    reset_security_pipeline()
    return run_security_pipeline(maker(content), document_id=document_id)


def _assert_clean_allow(result: dict) -> None:
    assert result["status"] == "allow"
    assert result["severity"] == "low"
    assert result.get("risk_level") == "LOW"
    assert result.get("dlp_decision") == "ALLOW"
    assert result.get("requires_human_review") is False
    assert "route_to_human_review" not in (result.get("actions") or [])


def test_clean_pdf_text_allowed(temp_txt_file) -> None:
    text = (
        "Standard Operating Procedure for equipment cleaning.\n"
        "Purpose: maintain validated manufacturing areas.\n"
        "Scope: applies to all production staff.\n"
        "This document contains no confidential or sensitive data.\n"
    )
    result = _scan(text, temp_txt_file, "clean-pdf-text")
    _assert_clean_allow(result)
    assert result.get("pipeline_debug")[-1]["stage"] == "final_aggregation"
    assert result["pipeline_debug"][-1]["severity"] == "low"


def test_business_report_allowed(temp_txt_file) -> None:
    text = (
        "Business Report Q4 2025\n"
        "Executive Summary\n"
        "Revenue increased by 12 percent across business units.\n"
        "Marketing and sales teams delivered strong operational results.\n"
    )
    result = _scan(text, temp_txt_file, "clean-business-report")
    _assert_clean_allow(result)


def test_empty_document_allowed(temp_txt_file) -> None:
    result = _scan("", temp_txt_file, "clean-empty-doc")
    _assert_clean_allow(result)
    assert "Unable to verify" not in str(result.get("reason") or "")


def test_person_names_only_allowed(temp_txt_file) -> None:
    text = (
        "Prepared by Dr. Jane Smith for internal distribution.\n"
        "Contact John Doe for follow-up questions about the process.\n"
    )
    result = _scan(text, temp_txt_file, "clean-person-names")
    _assert_clean_allow(result)


def test_organization_names_only_allowed(temp_txt_file) -> None:
    text = (
        "Manufacturing guidelines prepared for PharmaCorp Inc.\n"
        "Distribution handled by Global Health Systems LLC.\n"
    )
    result = _scan(text, temp_txt_file, "clean-org-names")
    _assert_clean_allow(result)


def test_clean_document_does_not_create_review_queue(temp_txt_file, tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "security_docs"
    docs_dir.mkdir()
    monkeypatch.setenv("HOST_DOCUMENTS_DIR", str(docs_dir))
    reset_security_pipeline()
    text = "Routine manufacturing checklist for validated equipment cleaning."
    result = run_security_pipeline(temp_txt_file(text), document_id="clean-no-queue")
    _assert_clean_allow(result)

    queue = HumanReviewQueue(queue_file=str(docs_dir / "human_review_queue.json"))
    pending = queue.get_pending_reviews()
    assert all(item.get("document_id") != "clean-no-queue" for item in pending)
