from __future__ import annotations

from pathlib import Path
import pytest

from src.features.security.dlp.sensitive_data_detector import (
    ComplianceScanner,
    DocumentComplianceError,
    validate_document_content,
    scan_document_for_api,
    mask_text_content,
)


def test_low_risk_policy(tmp_path: Path) -> None:
    # Low-risk elements (SOP, Batch, Product, Equipment, Company names) are allowed normally
    low_risk_file = tmp_path / "low_risk.txt"
    low_risk_file.write_text(
        "Standard Operating Procedure SOP-101 for Batch #4002. "
        "Product Model-9 and Facility Equipment EQ-501 are owned by Company TechCorp.",
        encoding="utf-8"
    )

    res = scan_document_for_api(low_risk_file)
    assert res["status"] == "accepted"
    # Should not raise exception
    validate_document_content(low_risk_file)


def test_medium_risk_policy(tmp_path: Path) -> None:
    # Medium-risk elements (email, phone, employee ID, project name, intranet URLs)
    # are allowed with warnings and masked downstream
    medium_risk_file = tmp_path / "medium_risk.txt"
    medium_risk_file.write_text(
        "Contact project coordinator at user@intranet.company.com or 555-555-0199. "
        "Employee ID: EMP-12345. Project Name: Project Alpha. URL: intranet.domain.local",
        encoding="utf-8"
    )

    res = scan_document_for_api(medium_risk_file)
    assert res["status"] == "allowed_with_warning"
    assert res["severity"] == "medium"
    assert "Email address" in res["detected_categories"]
    assert "Phone number" in res["detected_categories"]
    assert "Employee ID" in res["detected_categories"]

    # Should not raise exception during validation (allowed to upload)
    validate_document_content(medium_risk_file)

    # Verify masking of all medium risk data in text content
    raw_text = medium_risk_file.read_text(encoding="utf-8")
    masked = mask_text_content(raw_text)

    assert "user@intranet.company.com" not in masked
    assert "555-555-0199" not in masked
    assert "EMP-12345" not in masked
    assert "intranet.domain.local" not in masked
    assert "Project Alpha" not in masked


def test_generic_project_update_phrase_not_medium_risk(tmp_path: Path) -> None:
    generic_file = tmp_path / "generic_project_update.txt"
    generic_file.write_text(
        "This is a general project update with no secrets or instructions.",
        encoding="utf-8",
    )
    res = scan_document_for_api(generic_file)
    assert res["status"] == "accepted"
    assert not (res.get("detected_categories") or [])


def test_high_risk_policy(tmp_path: Path) -> None:
    # High-risk elements (Aadhaar, PAN, Passport, Bank, API Key, CVV, patient info, NDAs)
    # are blocked immediately
    high_risk_file = tmp_path / "high_risk.txt"
    high_risk_file.write_text(
        "The client Aadhaar is 1234-5678-9012. PAN: ABCDE1234F. CVV: 123.",
        encoding="utf-8"
    )

    res = scan_document_for_api(high_risk_file)
    assert res["status"] == "upload_blocked"
    assert res["severity"] == "high"
    assert "Aadhaar" in res["detected_categories"]
    assert "PAN" in res["detected_categories"]
    assert "CVV" in res["detected_categories"]

    # Should raise compliance error exception
    with pytest.raises(DocumentComplianceError) as exc_info:
        validate_document_content(high_risk_file)
    
    assert exc_info.value.response_payload["status"] == "upload_blocked"


def test_masking_high_risk_safeguard() -> None:
    # Even if high risk content is allowed by config/bypassed, mask it as a safeguard
    raw_text = "Aadhaar is 9876 5432 1098. PAN: FGHIJ9876K. Visa card: 4111 1111 1111 1111. Private Key: -----BEGIN PRIVATE KEY-----"
    masked = mask_text_content(raw_text)

    assert "9876 5432 1098" not in masked
    assert "FGHIJ9876K" not in masked
    assert "4111 1111 1111 1111" not in masked
    assert "-----BEGIN PRIVATE KEY-----" not in masked


def test_confidentiality_labels_false_positives(tmp_path: Path) -> None:
    # General confidentiality labels or watermarks should not block upload
    conf_file = tmp_path / "confidentiality_labels.txt"
    conf_file.write_text(
        "Header: Confidential Work Product\n"
        "Footer: Internal Use Only - Reference Copy\n"
        "Draft version 1.0 (Confidential Draft)",
        encoding="utf-8"
    )

    res = scan_document_for_api(conf_file)
    assert res["status"] == "allowed_with_warning"
    assert res["severity"] == "medium"
    
    # Should not raise exception (i.e. not blocked)
    validate_document_content(conf_file)


def test_confidentiality_labels_combined_with_high_risk(tmp_path: Path) -> None:
    # "Confidential – Attorney-Client Privileged" -> High (Block)
    privileged_file = tmp_path / "privileged.txt"
    privileged_file.write_text(
        "Confidential – Attorney-Client Privileged Work Product",
        encoding="utf-8"
    )
    res = scan_document_for_api(privileged_file)
    assert res["status"] == "upload_blocked"
    assert res["severity"] == "high"

    # "Confidential Customer Database containing Aadhaar numbers" -> High (Block)
    aadhaar_file = tmp_path / "conf_aadhaar.txt"
    aadhaar_file.write_text(
        "Confidential Customer Database containing Aadhaar numbers: 1234 5678 9012",
        encoding="utf-8"
    )
    res2 = scan_document_for_api(aadhaar_file)
    assert res2["status"] == "upload_blocked"
    assert res2["severity"] == "high"


def test_short_and_long_credentials_detected(tmp_path: Path) -> None:
    cred_file = tmp_path / "credentials.txt"
    cred_file.write_text(
        "password=Admin123\n"
        "api_key=test-key\n"
        "access_token=token123\n"
        "authentication_token=authToken123\n"
        "secret=UltraSecret987\n"
        "private_key=abcDEF123456\n",
        encoding="utf-8",
    )
    res = scan_document_for_api(cred_file)
    assert res["status"] == "upload_blocked"
    categories = set(res["detected_categories"])
    assert "Token" in categories


def test_violence_operational_phrases_detected(tmp_path: Path) -> None:
    vio_file = tmp_path / "violence_terms.txt"
    vio_file.write_text(
        "Incident report: physical assault, violent attack, serious injury, "
        "shooting incident, stabbing event, and threats of murder.",
        encoding="utf-8",
    )
    from src.features.security.dlp.policy_loader import scan_keyword_policies

    kw = scan_keyword_policies(vio_file.read_text(encoding="utf-8"))
    assert kw["action"] == "human_review"
    assert "violence" in set(kw.get("categories") or [])


def test_passport_formats_detected(tmp_path: Path) -> None:
    passport_file = tmp_path / "passport_formats.txt"
    passport_file.write_text(
        "Passport Number: X1234567\n"
        "Alternate Passport: AB1234567\n"
        "Legacy passport A1234567\n",
        encoding="utf-8",
    )
    res = scan_document_for_api(passport_file)
    assert res["status"] == "upload_blocked"
    assert "Passport" in set(res["detected_categories"])

