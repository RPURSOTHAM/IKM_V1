"""Decision-based human review model — DLP outcomes vs reviewer actions."""

from __future__ import annotations

from typing import Any

# Risk levels (ordered)
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"

RISK_LEVELS = (RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL)

# DLP pipeline outcomes (not reviewer-selectable)
DLP_ALLOW = "ALLOW"
DLP_MASK_AND_ALLOW = "MASK_AND_ALLOW"
DLP_HUMAN_REVIEW = "HUMAN_REVIEW"
DLP_BLOCK = "BLOCK"

# Reviewer-selectable actions only
REVIEWER_ALLOW = "ALLOW"
REVIEWER_MASK_AND_ALLOW = "MASK_AND_ALLOW"
REVIEWER_BLOCK = "BLOCK"

REVIEWER_ACTIONS = (REVIEWER_ALLOW, REVIEWER_MASK_AND_ALLOW, REVIEWER_BLOCK)

# Queue lifecycle statuses
STATE_PENDING = "PENDING"
STATE_APPROVED = "APPROVED"
STATE_APPROVED_MASKED = "APPROVED_MASKED"
STATE_REJECTED = "REJECTED"

QUEUE_STATUSES = (STATE_PENDING, STATE_APPROVED, STATE_APPROVED_MASKED, STATE_REJECTED)

# Legacy aliases (backward compatible imports)
DECISION_ALLOW = REVIEWER_ALLOW
DECISION_MASK_AND_ALLOW = REVIEWER_MASK_AND_ALLOW
DECISION_BLOCK = REVIEWER_BLOCK
DECISION_HUMAN_REVIEW = DLP_HUMAN_REVIEW  # DLP-only; not a reviewer action
REVIEW_STATUS_PENDING = STATE_PENDING
REVIEW_STATUS_ACTIONED = "ACTIONED"

# Legacy workflow statuses
STATE_REPROCESSED = "REPROCESSED"
STATE_COMPLETED = "COMPLETED"

AVAILABLE_ACTIONS_BY_RISK: dict[str, list[str]] = {
    RISK_LOW: [REVIEWER_ALLOW],
    RISK_MEDIUM: [REVIEWER_ALLOW, REVIEWER_MASK_AND_ALLOW, REVIEWER_BLOCK],
    # High/critical enter human review — reviewer chooses Allow / Mask / Block.
    RISK_HIGH: [REVIEWER_ALLOW, REVIEWER_MASK_AND_ALLOW, REVIEWER_BLOCK],
    RISK_CRITICAL: [REVIEWER_ALLOW, REVIEWER_MASK_AND_ALLOW, REVIEWER_BLOCK],
}

_REVIEWER_DECISION_TO_STATUS = {
    REVIEWER_ALLOW: STATE_APPROVED,
    REVIEWER_MASK_AND_ALLOW: STATE_APPROVED_MASKED,
    REVIEWER_BLOCK: STATE_REJECTED,
}

_SEVERITY_TO_RISK = {
    "low": RISK_LOW,
    "medium": RISK_MEDIUM,
    "high": RISK_HIGH,
    "critical": RISK_CRITICAL,
    "med": RISK_MEDIUM,
    "mid": RISK_MEDIUM,
}

_RISK_RANK = {
    RISK_LOW: 1,
    RISK_MEDIUM: 2,
    RISK_HIGH: 3,
    RISK_CRITICAL: 4,
}

_DLP_STATUS_TO_DECISION = {
    "allow": DLP_ALLOW,
    "mask_and_allow": DLP_MASK_AND_ALLOW,
    "human_review": DLP_HUMAN_REVIEW,
    "block": DLP_BLOCK,
}


def severity_to_risk_level(severity: str) -> str:
    return _SEVERITY_TO_RISK.get(str(severity or "medium").lower().strip(), RISK_MEDIUM)


def canonical_severity_for_dlp(severity: str, dlp_decision: str) -> str:
    """Normalize severity labels so human-review outcomes are never surfaced as low."""
    normalized = str(severity or "medium").strip().lower()
    if normalized not in {"low", "medium", "high", "critical"}:
        normalized = "medium"
    if str(dlp_decision or "").upper() == DLP_HUMAN_REVIEW and normalized == "low":
        return "medium"
    return normalized


def _risk_rank(level: str) -> int:
    return _RISK_RANK.get(str(level or RISK_MEDIUM).upper(), 2)


def effective_risk_level(item: dict[str, Any]) -> str:
    """Resolve risk for reviewer validation — handles stale queue rows."""
    severity_risk = severity_to_risk_level(str(item.get("severity") or "medium"))
    stored = str(item.get("risk_level") or "").upper()
    candidates = [severity_risk]
    if stored in _RISK_RANK:
        candidates.append(stored)

    dlp = str(item.get("dlp_decision") or "").upper()
    if dlp == DLP_HUMAN_REVIEW:
        # Human-review queue entries are never LOW (DLP only routes medium+ ambiguity here).
        candidates.append(RISK_MEDIUM)

    stored_actions = [str(a).upper() for a in (item.get("available_actions") or [])]
    if REVIEWER_BLOCK in stored_actions or REVIEWER_MASK_AND_ALLOW in stored_actions:
        candidates.append(RISK_MEDIUM)

    return max(candidates, key=_risk_rank)


def effective_available_actions(item: dict[str, Any]) -> list[str]:
    """Reviewer actions for effective risk — always the canonical set for that risk.

    Stale queue rows often store a truncated list (e.g. LOW→[ALLOW] while DLP is
    HUMAN_REVIEW). Preferring that subset caused the UI to show only Allow even
    when Block/Mask were required.
    """
    risk = effective_risk_level(item)
    return available_actions_for_risk(risk)


def normalize_dlp_decision(status: str) -> str:
    raw = str(status or "allow").lower()
    return _DLP_STATUS_TO_DECISION.get(raw, DLP_HUMAN_REVIEW)


def reviewer_decision_to_status(decision: str) -> str:
    """Map reviewer action to queue status."""
    key = str(decision or "").upper()
    if key not in _REVIEWER_DECISION_TO_STATUS:
        raise ValueError(f"Not a reviewer action: {decision}")
    return _REVIEWER_DECISION_TO_STATUS[key]


def available_actions_for_risk(risk_level: str) -> list[str]:
    key = str(risk_level or RISK_MEDIUM).upper()
    return list(AVAILABLE_ACTIONS_BY_RISK.get(key, AVAILABLE_ACTIONS_BY_RISK[RISK_MEDIUM]))


def validate_reviewer_decision(risk_level: str, decision: str) -> bool:
    normalized = str(decision or "").upper()
    if normalized == DLP_HUMAN_REVIEW:
        return False
    return normalized in available_actions_for_risk(str(risk_level or RISK_MEDIUM).upper())


def validate_reviewer_decision_for_item(item: dict[str, Any], decision: str) -> bool:
    return validate_reviewer_decision(effective_risk_level(item), decision)


def should_queue_for_review(dlp_status: str) -> bool:
    """Only DLP human_review outcomes enter the human review queue."""
    return str(dlp_status or "allow").lower() == "human_review"


def reviewer_override_applies_on_scan(
    *,
    live_status: str,
    live_dlp_decision: str,
    live_risk_level: str,
    requires_human_review: bool,
) -> bool:
    """Replay a prior reviewer Allow/Mask/Block only when live DLP is low-risk ALLOW.

    Prevents stale queue APPROVED rows from auto-allowing HIGH/MEDIUM re-uploads.
    """
    status_l = str(live_status or "").lower()
    dlp = str(live_dlp_decision or "").upper()
    risk = str(live_risk_level or "").upper()
    if status_l == "human_review" or requires_human_review or dlp == "HUMAN_REVIEW":
        return False
    try:
        from src.features.security.dlp.policy_loader import is_auto_allow_only_low

        if is_auto_allow_only_low():
            return status_l in {"allow", "allowed", "pass"} and dlp in {"ALLOW", ""} and risk == "LOW"
    except Exception:
        if risk != "LOW":
            return False
    return status_l in {"allow", "allowed", "pass"} and risk == "LOW"


def processing_continues(decision: str) -> bool:
    return str(decision or "").upper() in {REVIEWER_ALLOW, REVIEWER_MASK_AND_ALLOW}


def masking_applied(decision: str) -> bool:
    return str(decision or "").upper() == REVIEWER_MASK_AND_ALLOW


def ingestion_blocked(decision: str) -> bool:
    return str(decision or "").upper() == REVIEWER_BLOCK


def enrich_policy_result(policy: dict[str, Any]) -> dict[str, Any]:
    """Attach risk_level, dlp_decision, and available_reviewer_actions to a DLP result."""
    raw_severity = str(policy.get("severity") or "medium")
    status = str(policy.get("status") or "allow")
    dlp_decision = normalize_dlp_decision(status)
    severity = canonical_severity_for_dlp(raw_severity, dlp_decision)
    risk_level = severity_to_risk_level(severity)
    # Anything routed to human review is at least MEDIUM (reviewer must see Mask/Block).
    if dlp_decision == DLP_HUMAN_REVIEW and _risk_rank(risk_level) < _risk_rank(RISK_MEDIUM):
        risk_level = RISK_MEDIUM
    enriched = dict(policy)
    enriched["severity"] = severity
    enriched["risk_level"] = risk_level
    enriched["dlp_decision"] = dlp_decision
    # Reviewer actions only apply when DLP asks for human review — not on hard allow/block.
    if dlp_decision == DLP_HUMAN_REVIEW:
        enriched["available_reviewer_actions"] = available_actions_for_risk(risk_level)
    else:
        enriched["available_reviewer_actions"] = []
    return enriched


def decision_model_summary() -> dict[str, Any]:
    return {
        "risk_levels": list(RISK_LEVELS),
        "dlp_decisions": [DLP_ALLOW, DLP_MASK_AND_ALLOW, DLP_HUMAN_REVIEW, DLP_BLOCK],
        "reviewer_actions": list(REVIEWER_ACTIONS),
        "queue_statuses": list(QUEUE_STATUSES),
        "reviewer_action_to_status": dict(_REVIEWER_DECISION_TO_STATUS),
        "available_actions_by_risk": AVAILABLE_ACTIONS_BY_RISK,
    }
