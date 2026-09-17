"""Ingress vs egress security score separation for upload and output paths."""

from __future__ import annotations

from typing import Any

from src.features.security.dlp.policy_loader import load_security_score_weights


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _severity_to_score(severity: str) -> float:
    return {
        "low": 15.0,
        "medium": 45.0,
        "high": 75.0,
        "critical": 95.0,
    }.get(str(severity or "").lower(), 20.0)


def _action_to_score(action: str) -> float:
    normalized = str(action or "allow").lower()
    if normalized == "block":
        return 95.0
    if normalized in {"human_review", "quarantine", "flag"}:
        return 70.0
    if normalized in {"warning", "warn", "mask"}:
        return 40.0
    return 10.0


def build_ingress_security(
    *,
    policy_decision: dict[str, Any],
    keyword_res: dict[str, Any],
    similarity_res: dict[str, Any],
    prompt_injection_res: dict[str, Any],
    moderation_res: dict[str, Any],
    detection_count: int = 0,
) -> dict[str, Any]:
    """Upload-side scores only — never includes output/egress guard results."""
    weights = load_security_score_weights()

    dlp_score = _clamp(
        _severity_to_score(str(policy_decision.get("severity") or "low"))
        + min(20.0, detection_count * 0.5)
    )
    max_sim = float(similarity_res.get("max_similarity") or 0.0)
    similarity_score = _clamp(max_sim * 100.0)

    prompt_action = str(prompt_injection_res.get("action") or "allow").lower()
    if prompt_injection_res.get("degraded"):
        prompt_guard_score = _clamp(_action_to_score("human_review"))
    elif prompt_injection_res.get("hit_count"):
        prompt_guard_score = _clamp(_action_to_score(prompt_action))
    else:
        prompt_guard_score = _clamp(_action_to_score("allow"))

    moderation_score = _clamp(
        _action_to_score(str(moderation_res.get("action") or "allow"))
        + float(moderation_res.get("confidence") or 0.0) * 20.0
    )
    keyword_score = _clamp(_action_to_score(str(keyword_res.get("action") or "allow")))

    w = weights
    upload_security_score = _clamp(
        dlp_score * float(w.get("dlp", 0.30))
        + similarity_score * float(w.get("similarity", 0.25))
        + prompt_guard_score * float(w.get("prompt_guard", 0.20))
        + moderation_score * float(w.get("moderation", 0.15))
        + keyword_score * float(w.get("keyword", 0.10))
    )

    return {
        "upload_security_score": upload_security_score,
        "dlp_score": dlp_score,
        "similarity_score": similarity_score,
        "prompt_guard_score": prompt_guard_score,
        "moderation_score": moderation_score,
        "keyword_score": keyword_score,
    }


def build_egress_security(
    *,
    output_text: str | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Egress scores are computed separately from upload decisions.

    When no output text is available (upload scan), returns a neutral placeholder
    so upload decisions never inherit simulated output-guard results.
    """
    if not output_text or not str(output_text).strip():
        return {
            "output_guard_score": 0.0,
            "masking_required": False,
            "leakage_risk": "not_evaluated",
            "evaluated": False,
        }

    try:
        from src.features.security.moderation.output_moderator import moderate_output

        report = moderate_output(str(output_text), document_id=document_id)
        action = str(getattr(report, "final_action", None) or "allow").lower()
        output_guard_score = _clamp(_action_to_score(action))
        masking_required = action in {"mask", "block", "human_review"}
        leakage_risk = "high" if action in {"block", "human_review"} else (
            "medium" if masking_required else "low"
        )
        return {
            "output_guard_score": output_guard_score,
            "masking_required": masking_required,
            "leakage_risk": leakage_risk,
            "evaluated": True,
            "action": action,
        }
    except Exception:
        return {
            "output_guard_score": 0.0,
            "masking_required": False,
            "leakage_risk": "unknown",
            "evaluated": False,
        }
