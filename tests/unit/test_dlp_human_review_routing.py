"""DLP routing: low auto-allows; everything else → human review (unless auto_block enabled)."""

from __future__ import annotations

from src.features.security.dlp.dlp_engine import DLPPolicyEngine
from src.features.security.dlp.policy_loader import get_keyword_policy_engine


def _eval(*, detections=None, similarity=0.0, topics=None, moderation=None, doc_type="SOP"):
    get_keyword_policy_engine().reload(force=True)
    return DLPPolicyEngine().evaluate_policy(
        detections=detections or [],
        doc_classification={"document_type": doc_type},
        topic_classification={"topics": topics or []},
        similarity_results={"max_similarity": similarity, "matches": []},
        moderation_results=moderation or {"risk": "low", "action": "allow", "confidence": 0.5},
    )


def test_medium_email_phone_routes_to_human_review() -> None:
    decision = _eval(
        detections=[
            {"category": "PII", "type": "Email address", "severity": "medium"},
            {"category": "PII", "type": "Phone number", "severity": "medium"},
        ]
    )
    assert decision["status"] == "human_review"
    assert decision["severity"] == "medium"
    assert set(decision["available_reviewer_actions"]) >= {"ALLOW", "MASK_AND_ALLOW", "BLOCK"}


def test_high_pan_routes_to_human_review_not_hard_block() -> None:
    decision = _eval(
        detections=[{"category": "PII", "type": "PAN", "severity": "high", "matched_value": "ABCDE1234F"}]
    )
    assert decision["status"] == "human_review"
    assert decision["severity"] in {"medium", "high"}
    assert "BLOCK" in decision["available_reviewer_actions"]


def test_critical_private_key_routes_to_human_review_when_auto_block_off(
    monkeypatch,
) -> None:
    monkeypatch.delenv("DLP_AUTO_BLOCK_ENABLED", raising=False)
    decision = _eval(
        detections=[
            {
                "category": "Credentials",
                "type": "Private Key",
                "severity": "critical",
                "matched_value": "BEGIN PRIVATE KEY",
            }
        ]
    )
    assert decision["status"] == "human_review"
    assert decision["severity"] == "critical"
    assert set(decision["available_reviewer_actions"]) >= {"ALLOW", "MASK_AND_ALLOW", "BLOCK"}


def test_critical_private_key_hard_blocks_when_auto_block_on(monkeypatch) -> None:
    monkeypatch.setenv("DLP_AUTO_BLOCK_ENABLED", "true")
    decision = _eval(
        detections=[
            {
                "category": "Credentials",
                "type": "Private Key",
                "severity": "critical",
                "matched_value": "BEGIN PRIVATE KEY",
            }
        ]
    )
    assert decision["status"] == "block"
    assert decision["severity"] == "critical"
    assert decision["available_reviewer_actions"] == []


def test_clear_non_sensitive_document_allows() -> None:
    decision = _eval()
    assert decision["status"] == "allow"
    assert decision["requires_human_review"] is False


def test_medical_topic_alone_does_not_force_human_review() -> None:
    decision = _eval(topics=[{"topic": "Medical / Clinical", "confidence": 0.95}])
    assert decision["status"] == "allow"
    assert decision["requires_human_review"] is False


def test_near_duplicate_similarity_routes_to_human_review() -> None:
    decision = _eval(similarity=0.97)
    assert decision["status"] == "human_review"
    assert decision["available_reviewer_actions"]



def test_multiple_gov_ids_route_to_human_review_not_hard_block() -> None:
    decision = _eval(
        detections=[
            {"category": "PII", "type": "Aadhaar", "severity": "high"},
            {"category": "PII", "type": "PAN", "severity": "high"},
        ]
    )
    assert decision["status"] == "human_review"
    assert decision["severity"] in {"medium", "high"}
    assert "ALLOW" in decision["available_reviewer_actions"]


def test_self_harm_ideation_blacklist_routes_to_human_review() -> None:
    """Natural-language ideation (not instructional) must not silently ALLOW."""
    get_keyword_policy_engine().reload(force=True)
    text = (
        "Alex writes in a private journal entry: Some days it feels like the pressure "
        "never lets up, and I've started thinking about hurting myself just to feel "
        "some kind of relief. This file exists to test self-harm risk moderation."
    )
    scan = get_keyword_policy_engine().scan_text(text)
    assert "self_harm" in (scan.get("categories") or [])
    assert scan.get("action") in {"human_review", "block"}
    decision = _eval(detections=scan.get("matches") or [])
    assert decision["status"] == "human_review"
    assert decision["requires_human_review"] is True
    assert set(decision["available_reviewer_actions"]) >= {"ALLOW", "MASK_AND_ALLOW", "BLOCK"}


def test_self_harm_local_moderation_routes_to_human_review() -> None:
    from src.features.security.moderation.llm_moderator import _local_moderate

    text = (
        "TEST FILE: self_harm_test_01.txt PRIMARY CATEGORY: Self-Harm Risk "
        "I've started thinking about hurting myself just to feel some kind of relief."
    )
    mod = _local_moderate(text)
    assert mod["risk"] in {"medium", "high"}
    assert mod["action"] in {"warn", "block"}
    decision = _eval(moderation=mod)
    assert decision["status"] == "human_review"
    assert decision["requires_human_review"] is True
