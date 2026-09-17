"""Tests for decision-based human review workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.features.security.review.review_decisions import (
    REVIEWER_ALLOW,
    REVIEWER_BLOCK,
    REVIEWER_MASK_AND_ALLOW,
    STATE_APPROVED,
    STATE_APPROVED_MASKED,
    STATE_PENDING,
    STATE_REJECTED,
    available_actions_for_risk,
    enrich_policy_result,
    reviewer_decision_to_status,
    reviewer_override_applies_on_scan,
    should_queue_for_review,
    validate_reviewer_decision,
)


def test_human_review_not_reviewer_action() -> None:
    assert "HUMAN_REVIEW" not in available_actions_for_risk("MEDIUM")
    assert "HUMAN_REVIEW" not in available_actions_for_risk("HIGH")
    assert not validate_reviewer_decision("MEDIUM", "HUMAN_REVIEW")


def test_should_queue_only_human_review_dlp() -> None:
    assert should_queue_for_review("human_review") is True
    assert should_queue_for_review("mask_and_allow") is False
    assert should_queue_for_review("block") is False
    assert should_queue_for_review("allow") is False


def test_reviewer_override_skipped_for_high_risk_human_review() -> None:
    """Stale APPROVED rows must not replay over a fresh HIGH/MEDIUM human_review scan."""
    assert reviewer_override_applies_on_scan(
        live_status="human_review",
        live_dlp_decision="HUMAN_REVIEW",
        live_risk_level="HIGH",
        requires_human_review=True,
    ) is False
    assert reviewer_override_applies_on_scan(
        live_status="allow",
        live_dlp_decision="ALLOW",
        live_risk_level="HIGH",
        requires_human_review=False,
    ) is False


def test_reviewer_override_applies_only_for_low_risk_allow() -> None:
    assert reviewer_override_applies_on_scan(
        live_status="allow",
        live_dlp_decision="ALLOW",
        live_risk_level="LOW",
        requires_human_review=False,
    ) is True
    assert reviewer_override_applies_on_scan(
        live_status="allow",
        live_dlp_decision="ALLOW",
        live_risk_level="MEDIUM",
        requires_human_review=False,
    ) is False


def test_reviewer_decision_status_mapping() -> None:
    assert reviewer_decision_to_status("ALLOW") == STATE_APPROVED
    assert reviewer_decision_to_status("MASK_AND_ALLOW") == STATE_APPROVED_MASKED
    assert reviewer_decision_to_status("BLOCK") == STATE_REJECTED


def test_available_actions_by_risk() -> None:
    assert available_actions_for_risk("LOW") == [REVIEWER_ALLOW]
    assert set(available_actions_for_risk("MEDIUM")) == {
        REVIEWER_ALLOW,
        REVIEWER_MASK_AND_ALLOW,
        REVIEWER_BLOCK,
    }
    assert set(available_actions_for_risk("HIGH")) == {
        REVIEWER_ALLOW,
        REVIEWER_MASK_AND_ALLOW,
        REVIEWER_BLOCK,
    }
    assert set(available_actions_for_risk("CRITICAL")) == {
        REVIEWER_ALLOW,
        REVIEWER_MASK_AND_ALLOW,
        REVIEWER_BLOCK,
    }

def test_enrich_policy_result_human_review_queue() -> None:
    out = enrich_policy_result({"status": "human_review", "severity": "medium", "reason": "x"})
    assert out["risk_level"] == "MEDIUM"
    assert out["dlp_decision"] == "HUMAN_REVIEW"
    assert "HUMAN_REVIEW" not in out["available_reviewer_actions"]
    assert REVIEWER_BLOCK in out["available_reviewer_actions"]


def test_enrich_human_review_floors_low_severity_to_medium() -> None:
    out = enrich_policy_result({"status": "human_review", "severity": "low", "reason": "labels"})
    assert out["severity"] == "medium"
    assert out["risk_level"] == "MEDIUM"
    assert set(out["available_reviewer_actions"]) == {
        REVIEWER_ALLOW,
        REVIEWER_MASK_AND_ALLOW,
        REVIEWER_BLOCK,
    }


def test_enrich_hard_block_has_no_reviewer_actions() -> None:
    out = enrich_policy_result({"status": "block", "severity": "high", "reason": "secrets"})
    assert out["dlp_decision"] == "BLOCK"
    assert out["available_reviewer_actions"] == []


def test_effective_actions_expand_stale_allow_only_list() -> None:
    from src.features.security.review.review_decisions import effective_available_actions

    actions = effective_available_actions(
        {
            "severity": "low",
            "risk_level": "LOW",
            "dlp_decision": "HUMAN_REVIEW",
            "available_actions": ["ALLOW"],
        }
    )
    assert REVIEWER_ALLOW in actions
    assert REVIEWER_MASK_AND_ALLOW in actions
    assert REVIEWER_BLOCK in actions


def test_dlp_human_review_creates_pending_record(tmp_path: Path) -> None:
    from src.features.security.review.human_review_queue import HumanReviewQueue

    queue = HumanReviewQueue(queue_file=str(tmp_path / "q.json"))
    rid = queue.add_to_queue(
        "doc-hr",
        "doc.txt",
        "medium",
        "ambiguous labels",
        [],
        risk_level="MEDIUM",
        dlp_decision="human_review",
    )
    assert rid == "rev_doc-hr"
    item = queue.queue[0]
    assert item["status"] == STATE_PENDING
    assert item["severity"] == "medium"
    assert item["dlp_decision"] == "HUMAN_REVIEW"


def test_review_queue_persists_document_type_context(tmp_path: Path) -> None:
    from src.features.security.review.human_review_queue import HumanReviewQueue

    queue = HumanReviewQueue(queue_file=str(tmp_path / "q-with-type.json"))
    queue.add_to_queue(
        "doc-typed",
        "typed.txt",
        "medium",
        "needs review",
        [],
        document_type_id="type-123",
        document_type_name="SOP",
        dlp_decision="human_review",
    )
    item = queue.queue[0]
    assert item["document_type_id"] == "type-123"
    assert item["document_type_name"] == "SOP"


def test_review_service_backfills_document_type_context_for_legacy_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.features.security.review.human_review_queue import HumanReviewQueue
    from src.features.security.review.human_review_service import HumanReviewService

    queue = HumanReviewQueue(queue_file=str(tmp_path / "legacy-q.json"))
    queue.queue = [
        {
            "review_id": "rev_legacy-1",
            "document_id": "legacy-1",
            "document_name": "legacy.txt",
            "severity": "medium",
            "reason": "legacy row",
            "dlp_decision": "HUMAN_REVIEW",
            "status": "PENDING",
            "detections": [],
        }
    ]
    queue._save_queue()

    fake_store = SimpleNamespace(
        get_document_instance=lambda document_id: {"document_type_id": "type-legacy"} if document_id == "legacy-1" else None,
        get_type_by_id=lambda type_id: SimpleNamespace(name="Legacy SOP") if type_id == "type-legacy" else None,
    )
    monkeypatch.setattr(
        "src.features.document_types.infrastructure.document_type_repository.get_document_type_store",
        lambda: fake_store,
    )

    svc = HumanReviewService(queue=queue)
    rows = svc.list_all()
    assert len(rows) == 1
    assert rows[0]["document_type_id"] == "type-legacy"
    assert rows[0]["document_type_name"] == "Legacy SOP"


def test_apply_decision_transitions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.features.security.review.human_review_service.HumanReviewService._trigger_processing",
        lambda self, item, reason, apply_masking=False: {
            "triggered": True,
            "continues": True,
            "masking_applied": apply_masking,
        },
    )
    from src.features.security.review.human_review_queue import HumanReviewQueue
    from src.features.security.review.human_review_service import HumanReviewService

    queue = HumanReviewQueue(queue_file=str(tmp_path / "q.json"))
    rid = queue.add_to_queue(
        "d99",
        "doc.txt",
        "medium",
        "test",
        [],
        risk_level="MEDIUM",
        dlp_decision="human_review",
    )
    svc = HumanReviewService(queue=queue)
    item = svc.get(rid)
    assert item is not None
    assert item["status"] == STATE_PENDING

    allow = svc.apply_decision(rid, decision=REVIEWER_ALLOW, reviewer="alice", comments="ok")
    assert allow["status"] == STATE_APPROVED
    assert allow["reviewer_decision"] == REVIEWER_ALLOW
    assert allow["processing"]["continues"] is True
    assert allow["processing"]["masking_applied"] is False

    rid2 = queue.add_to_queue("d100", "d2.txt", "medium", "t", [], risk_level="MEDIUM", dlp_decision="human_review")
    masked = svc.apply_decision(rid2, decision=REVIEWER_MASK_AND_ALLOW, reviewer="bob", comments="mask")
    assert masked["status"] == STATE_APPROVED_MASKED
    assert masked["processing"]["masking_applied"] is True

    rid3 = queue.add_to_queue("d101", "d3.txt", "medium", "t", [], risk_level="MEDIUM", dlp_decision="human_review")
    blocked = svc.apply_decision(rid3, decision=REVIEWER_BLOCK, reviewer="carol", comments="no")
    assert blocked["status"] == STATE_REJECTED
    assert blocked["processing"]["blocked"] is True
    assert blocked["processing"]["continues"] is False


def test_audit_log_on_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SECURITY_AUDIT_LOG_PATH", str(audit_path))
    monkeypatch.setattr(
        "src.features.security.review.human_review_service.HumanReviewService._trigger_processing",
        lambda self, item, reason, apply_masking=False: {"triggered": False},
    )
    from src.features.security.review.human_review_queue import HumanReviewQueue
    from src.features.security.review.human_review_service import HumanReviewService

    queue = HumanReviewQueue(queue_file=str(tmp_path / "q.json"))
    rid = queue.add_to_queue("aud1", "a.txt", "medium", "r", [], dlp_decision="human_review")
    svc = HumanReviewService(queue=queue)
    svc.apply_decision(rid, decision=REVIEWER_BLOCK, reviewer="auditor", comments="deny")

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    row = json.loads(lines[-1])
    assert row["event_type"] == "HUMAN_REVIEW"
    assert row["decision"] == REVIEWER_BLOCK
    assert row["metadata"]["queue_status"] == STATE_REJECTED
    assert row["metadata"]["ingestion_blocked"] is True


def test_reviewer_override_changes_pipeline_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.features.security.review.human_review_queue import HumanReviewQueue, get_reviewer_override_for_document
    from src.features.security.review.human_review_service import HumanReviewService

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "src.shared.networking.document_paths.resolve_security_documents_dir",
        lambda: tmp_path,
    )
    queue_file = str(tmp_path / "human_review_queue.json")
    queue = HumanReviewQueue(queue_file=queue_file)
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        "Employee PAN ABCDE1234F is recorded for payroll compliance.",
        encoding="utf-8",
    )
    rid = queue.add_to_queue(
        "override-doc",
        "sample.txt",
        "medium",
        "human review",
        [],
        dlp_decision="human_review",
    )
    svc = HumanReviewService(queue=queue)
    monkeypatch.setattr(
        HumanReviewService,
        "_trigger_processing",
        lambda self, item, reason, apply_masking=False: {
            "triggered": True,
            "continues": True,
            "masking_applied": apply_masking,
        },
    )
    svc.apply_decision(rid, decision=REVIEWER_MASK_AND_ALLOW, reviewer="tester", comments="mask it")

    override = get_reviewer_override_for_document("override-doc", queue_file=queue_file)
    assert override is not None
    assert override["pipeline_status"] == "mask_and_allow"

    from src.features.security.application.upload_security_pipeline import reset_security_pipeline, run_security_pipeline

    reset_security_pipeline()
    result = run_security_pipeline(doc_path, document_id="override-doc")
    # Live medium-risk DLP human_review wins over a prior MASK_AND_ALLOW so the
    # reviewer dropdown is shown again on re-upload.
    assert result["status"] == "human_review"
    override_meta = (result.get("stage_results") or {}).get("reviewer_override")
    if override_meta:
        assert override_meta.get("skipped") is True
    else:
        # Queue reopened to PENDING before override read — stale decision not replayed.
        assert get_reviewer_override_for_document("override-doc", queue_file=queue_file) is None
    reopened = queue.get_pending_reviews()
    assert any(r.get("review_id") == rid for r in reopened)


def test_reviewer_block_override(tmp_path: Path) -> None:
    from src.features.security.review.human_review_queue import HumanReviewQueue, get_reviewer_override_for_document
    from src.features.security.review.human_review_service import HumanReviewService

    queue_file = str(tmp_path / "q.json")
    queue = HumanReviewQueue(queue_file=queue_file)
    rid = queue.add_to_queue("blocked-doc", "blocked.txt", "medium", "review", [], dlp_decision="human_review")
    svc = HumanReviewService(queue=queue)
    svc.apply_decision(rid, decision=REVIEWER_BLOCK, reviewer="tester", comments="stop")

    override = get_reviewer_override_for_document("blocked-doc", queue_file=queue_file)
    assert override is not None
    assert override["pipeline_status"] == "block"


def test_block_allowed_when_queue_severity_low_but_human_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.features.security.review.human_review_queue import HumanReviewQueue
    from src.features.security.review.human_review_service import HumanReviewService
    from src.features.security.review.review_decisions import effective_risk_level

    monkeypatch.setattr(
        "src.features.security.review.human_review_service.HumanReviewService._trigger_processing",
        lambda self, item, reason, apply_masking=False: {"triggered": False, "blocked": True},
    )
    queue = HumanReviewQueue(queue_file=str(tmp_path / "q-stale.json"))
    queue.queue = [
        {
            "review_id": "rev_stale-medium",
            "document_id": "stale-medium",
            "document_name": "sample.txt",
            "severity": "low",
            "reason": "ambiguous labels",
            "dlp_decision": "HUMAN_REVIEW",
            "available_actions": ["ALLOW", "MASK_AND_ALLOW", "BLOCK"],
            "status": "PENDING",
            "detections": [],
        }
    ]
    queue._save_queue()
    assert effective_risk_level(queue.queue[0]) == "MEDIUM"

    svc = HumanReviewService(queue=queue)
    result = svc.apply_decision("rev_stale-medium", decision=REVIEWER_BLOCK, reviewer="tester", comments="stop")
    assert result["status"] == STATE_REJECTED
    assert result["reviewer_decision"] == REVIEWER_BLOCK


def test_critical_allows_full_reviewer_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.features.security.review.human_review_service.HumanReviewService._trigger_processing",
        lambda self, item, reason, apply_masking=False: {"triggered": False},
    )
    from src.features.security.review.human_review_queue import HumanReviewQueue
    from src.features.security.review.human_review_service import HumanReviewService

    queue = HumanReviewQueue(queue_file=str(tmp_path / "q2.json"))
    rid = queue.add_to_queue(
        "crit1",
        "blocked.txt",
        "critical",
        "blocked",
        [],
        risk_level="CRITICAL",
        dlp_decision="human_review",
    )
    svc = HumanReviewService(queue=queue)
    result = svc.apply_decision(rid, decision=REVIEWER_ALLOW, reviewer="r1", comments="inspected")
    assert result["status"] == STATE_APPROVED
    assert result["reviewer_decision"] == REVIEWER_ALLOW
