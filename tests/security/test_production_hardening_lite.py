"""Final production hardening — LITE must not skip NER/classification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("LLM_MODERATION_PROVIDER", "local")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def test_lite_still_runs_ner_and_classification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECURITY_PIPELINE_LITE", "true")
    from src.features.security.application.upload_security_pipeline import (
        reset_security_pipeline,
        run_security_pipeline,
    )

    reset_security_pipeline()
    path = tmp_path / "sop.txt"
    path.write_text(
        "Standard Operating Procedure for Cleanroom Operations.\n"
        "Purpose: Define entry guidelines.\n"
        "Scope: All operators.\n"
        "Patient ID ABC-123 is not present in this SOP training example.\n"
        "Authored by Dr. Jane Smith for PharmaCorp Inc.\n",
        encoding="utf-8",
    )
    result = run_security_pipeline(path, document_id="lite-1")
    assert "ner" in result["pipeline_stages"] or "document_classification" in str(result["pipeline_stages"])
    assert result["document_type"] != "Unknown" or result.get("document_type") in {
        "SOP",
        "Other",
        "Manual",
        "Report",
        "Policy",
    }
    # NER must have executed (person/org/patient context from rules strategy)
    ner_types = {d.get("type") for d in result.get("detections") or [] if d.get("category") == "NER"}
    assert ner_types  # at least one NER detection


def test_classify_document_lite_nda() -> None:
    from src.features.security.classification.document_classifier import classify_document_lite

    result = classify_document_lite(
        "This Non-Disclosure Agreement (NDA) binds the parties to confidentiality."
    )
    assert result["document_type"] == "NDA"
    assert result["strategy"] == "lite_heuristic"
    assert result["confidence"] >= 0.7


def test_ner_rules_strategy_always_runs() -> None:
    from src.features.security.dlp.ner_detector import NERDetector

    dets = NERDetector().detect_entities(
        "Dr. Alice Wonder works at TechCorp Inc. Patient ID P-99.",
        strategy="rules",
    )
    types = {d["type"] for d in dets}
    assert "PERSON" in types or "ORG" in types
    assert "PATIENT_CONTEXT" in types


def test_output_moderation_from_env_cannot_disable_regex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODERATION_ENABLE_REGEX", "false")
    monkeypatch.setenv("MODERATION_ENABLE_POLICY_ENGINE", "false")
    from src.features.security.moderation.output_moderator import OutputModerationConfig

    cfg = OutputModerationConfig.from_env()
    assert cfg.enable_regex is True
    assert cfg.enable_policy_engine is True


def test_lite_medical_record_escalates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECURITY_PIPELINE_LITE", "true")
    policy = tmp_path / "sec"
    policy.mkdir()
    (policy / "blacklist.yaml").write_text("{}\n", encoding="utf-8")
    (policy / "allowlist.yaml").write_text("{}\n", encoding="utf-8")
    (policy / "dlp_policies.yaml").write_text(
        "similarity:\n  block_above: 0.95\n  human_review_min: 0.85\n  warning_min: 0.75\n"
        "document_types:\n  Medical Record:\n    escalate: high\n"
        "blacklist_categories: {}\n"
        "topics: {}\n",
        encoding="utf-8",
    )
    (policy / "security_policy.yaml").write_text(
        "prompt_guard:\n  fail_open: false\n  unavailable_action: human_review\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SECURITY_POLICY_DIR", str(policy))
    from src.features.security import policy_loader
    from src.features.security.application.upload_security_pipeline import (
        reset_security_pipeline,
        run_security_pipeline,
    )

    policy_loader._engine = None
    reset_security_pipeline()
    path = tmp_path / "mr.txt"
    path.write_text(
        "Medical Record for outpatient visit.\n"
        "Patient diagnosis and clinical notes for PHI review.\n"
        "No bank account listed.\n",
        encoding="utf-8",
    )
    result = run_security_pipeline(path, document_id="mr-1")
    assert result["document_type"] == "Medical Record"
    # Escalation should elevate beyond a clean allow
    assert result["status"] in {"block", "human_review", "mask_and_allow"}
