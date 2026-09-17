from __future__ import annotations

from pathlib import Path
import pytest

from src.shared.errors import DmsServiceError
from src.features.security.compliance.compliance_validator import (
    ComplianceScanner,
    check_document_compliance,
    security_scan_requires_review,
)


def test_luhn_algorithm() -> None:
    scanner = ComplianceScanner()
    # Valid credit cards
    assert scanner.check_luhn("49927398716")  # Standard Luhn test case
    assert scanner.check_luhn("4111111111111111")  # Typical Visa
    # Invalid credit card
    assert not scanner.check_luhn("4111111111111112")


def test_masking_logic() -> None:
    scanner = ComplianceScanner()
    assert scanner.mask_value("4111-1111-1111-1234", "Credit/Debit Card Number") == "XXXX-XXXX-XXXX-1234"
    assert scanner.mask_value("test.user@domain.com", "Email Address") == "t***@domain.com"
    assert scanner.mask_value("123-45-6789", "Social Security Number (SSN)") == "XXX-XX-6789"
    assert scanner.mask_value("1234 5678 9012", "Aadhaar Number") == "XXXX-XXXX-9012"
    assert scanner.mask_value("ABCDE1234F", "PAN Number") == "ABXXXXXX4F"
    assert scanner.mask_value("AKIAIOSFODNN7EXAMPLE", "Authentication Credentials/API Keys") == "AKIA************MPLE"


def test_compliance_scanner_pii_detection() -> None:
    scanner = ComplianceScanner()
    
    # SSN
    res = scanner.scan_text("Contact me at SSN 000-12-3456.")
    assert len(res) == 1
    assert res[0]["category"] == "Social Security Number (SSN)"
    assert res[0]["severity"] == "High"

    # Aadhaar
    res = scanner.scan_text("Aadhaar Number is 1234-5678-9012.")
    assert len(res) == 1
    assert res[0]["category"] == "Aadhaar Number"

    # PAN
    res = scanner.scan_text("PAN Number: ABCDE1234F")
    assert len(res) == 1
    assert res[0]["category"] == "PAN Number"


def test_compliance_scanner_credentials_detection() -> None:
    scanner = ComplianceScanner()
    
    # AWS key
    res = scanner.scan_text("My key is AKIA1234567890ABCDEF")
    assert len(res) == 1
    assert res[0]["category"] == "Authentication Credentials/API Keys"
    assert res[0]["severity"] == "Critical"

    # Private key block
    res = scanner.scan_text("-----BEGIN RSA PRIVATE KEY-----\nSomeData\n-----END RSA PRIVATE KEY-----")
    assert len(res) == 1
    assert res[0]["category"] == "Private Key"


def test_compliance_scanner_confidentiality_detection() -> None:
    scanner = ComplianceScanner()
    
    # Strictly confidential keyword
    res = scanner.scan_text("This document is strictly confidential and proprietary.")
    categories = [d["category"] for d in res]
    assert "Confidential Business Information" in categories
    assert "Intellectual Property / Trade Secrets" in categories


def test_check_document_compliance_violating(tmp_path: Path) -> None:
    # Write a violating file — with auto_block disabled, DLP routes to human_review (not hard block).
    violating_file = tmp_path / "sensitive.txt"
    violating_file.write_text("Customer SSN: 999-99-9999. Do not share.", encoding="utf-8")

    result = check_document_compliance(violating_file)
    assert result is not None
    assert result["status"] == "human_review"
    assert security_scan_requires_review(result)


def test_check_document_compliance_clean(tmp_path: Path) -> None:
    # Write a clean file
    clean_file = tmp_path / "clean.txt"
    clean_file.write_text("This is standard documentation describing the architecture of the platform.", encoding="utf-8")

    # Should run without raising any exceptions
    check_document_compliance(clean_file)


def test_check_document_compliance_human_review_does_not_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DLP human_review must not raise — intake saves the file but defers processor queue."""
    flagged = tmp_path / "flagged.txt"
    flagged.write_text("Clinical notes for patient follow-up.", encoding="utf-8")

    monkeypatch.setattr(
        "src.features.security.application.upload_security_pipeline.run_security_pipeline",
        lambda *args, **kwargs: {
            "status": "human_review",
            "severity": "medium",
            "reason": "topic_policy_human_review",
            "detected_categories": ["Medical / Clinical"],
            "document_id": "flagged",
        },
    )
    result = check_document_compliance(flagged)
    assert result is not None
    assert result["status"] == "human_review"


def test_security_scan_requires_review_medium_dlp() -> None:
    assert security_scan_requires_review(
        {
            "status": "human_review",
            "severity": "medium",
            "risk_level": "MEDIUM",
            "dlp_decision": "HUMAN_REVIEW",
            "requires_human_review": True,
        }
    )


def test_security_scan_requires_review_clean_allow() -> None:
    assert not security_scan_requires_review(
        {
            "status": "allow",
            "severity": "low",
            "risk_level": "LOW",
            "dlp_decision": "ALLOW",
        }
    )


def test_processing_allowed_blocked_until_reviewer_approves() -> None:
    from src.features.documents.application.document_service import DocumentReceiverService
    from src.features.security.compliance.compliance_validator import (
        processing_allowed_after_security_review,
    )

    record = {
        "document_id": "doc-review-1",
        "status": "human_review",
        "metadata": {
            "requires_human_review": True,
            "security_scan": {
                "status": "human_review",
                "severity": "medium",
                "risk_level": "MEDIUM",
                "dlp_decision": "HUMAN_REVIEW",
                "requires_human_review": True,
            }
        },
    }
    assert DocumentReceiverService._processing_allowed_after_security_review(record) is False
    assert processing_allowed_after_security_review(record) is False


def test_submit_record_to_queue_skips_pending_human_review(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.features.documents.application import job_republish

    published: list[dict] = []

    class _FakePublisher:
        def publish_document_job(self, payload: dict) -> None:
            published.append(payload)

    monkeypatch.setattr(job_republish, "_publisher_from_settings", lambda: _FakePublisher())
    monkeypatch.setattr(job_republish, "get_document_job_store", lambda: None)

    record = {
        "document_id": "doc-orphan-1",
        "document_name": "sensitive.pdf",
        "collection_name": "demo",
        "status": "human_review",
        "metadata": {
            "requires_human_review": True,
            "security_scan": {
                "status": "human_review",
                "dlp_decision": "HUMAN_REVIEW",
                "requires_human_review": True,
            },
        },
        "processing": {"enabled_processor_types": ["chunking_vectorizing"]},
    }
    # Avoid MetadataStore writes when skipping.
    class _FakeStore:
        def update_document_status(self, *_a, **_k):
            return None

    monkeypatch.setattr(job_republish, "MetadataStore", lambda *_a, **_k: _FakeStore())
    monkeypatch.setattr(job_republish, "get_settings", lambda: type("S", (), {"db_path": ":memory:"})())

    job_republish.submit_record_to_queue(record)
    assert published == []
