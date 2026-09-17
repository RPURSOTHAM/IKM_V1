"""Hybrid Prompt Guard: rule-based first, then model-based (fail-closed when model unavailable)."""

from __future__ import annotations

import logging
import os
from typing import Any

from src.features.security.moderation.ai_safety_guard import (
    PROMPT_GUARD_VERSION,
    PromptGuard,
)
from src.features.security.audit.security_event_logger import (
    EVENT_MODEL_GUARD_UNAVAILABLE,
    log_security_event,
)
from src.features.security.moderation.model_prompt_guard import (
    get_model_prompt_guard,
    model_guard_metrics_snapshot,
)
from src.features.security.dlp.policy_loader import (
    prompt_guard_fail_open,
    prompt_guard_unavailable_action,
)

_logger = logging.getLogger(__name__)

HYBRID_PROMPT_GUARD_VERSION = "2026.07.21.2"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _log_model_guard_unavailable(*, pipeline: str, reason: str, metadata: dict[str, Any] | None = None) -> None:
    unavailable_action = prompt_guard_unavailable_action()
    log_security_event(
        EVENT_MODEL_GUARD_UNAVAILABLE,
        decision=unavailable_action.upper(),
        reason=reason or "LLM classifier unavailable",
        severity="high",
        policy="prompt_guard.fail_closed",
        metadata={
            "pipeline": pipeline,
            "degraded": True,
            "action": unavailable_action,
            **(metadata or {}),
        },
    )


def _apply_unavailable_policy(
    rule: dict[str, Any],
    rule_meta: dict[str, Any],
    *,
    pipeline: str,
    skip_reason: str,
    model_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve rule-guard outcome; escalate when fail_open is disabled."""
    if prompt_guard_fail_open():
        rule["model_guard_skipped"] = True
        rule["model_guard_skip_reason"] = f"{skip_reason}_fail_open"
        rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
        rule["rule_guard"] = rule_meta
        rule["degraded"] = True
        if model_result is not None:
            rule["model_guard"] = model_result
        return rule

    unavailable_action = prompt_guard_unavailable_action()
    if unavailable_action == "quarantine":
        unavailable_action = "human_review"

    # No injection detected — preserve rule allow state but escalate operationally.
    rule["model_guard_skipped"] = True
    rule["model_guard_skip_reason"] = skip_reason
    rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
    rule["rule_guard"] = rule_meta
    rule["degraded"] = True
    rule["action"] = unavailable_action
    rule["severity"] = rule.get("severity") or "medium"
    rule["reason"] = rule.get("reason") or "LLM prompt-injection classifier unavailable; routing to human review."
    if model_result is not None:
        rule["model_guard"] = model_result

    _log_model_guard_unavailable(
        pipeline=pipeline,
        reason="LLM classifier unavailable",
        metadata={"skip_reason": skip_reason},
    )
    _logger.warning(
        "MODEL_GUARD_UNAVAILABLE pipeline=%s action=%s degraded=true",
        pipeline,
        unavailable_action,
    )
    return rule


class HybridPromptGuard:
    """Rule PromptGuard → ModelPromptGuard. Model runs only if rules allow."""

    def __init__(self, rule_guard: PromptGuard | None = None) -> None:
        self.rule_guard = rule_guard or PromptGuard()
        self.version = HYBRID_PROMPT_GUARD_VERSION

    def check_prompt(self, prompt: str, *, pipeline: str = "chat") -> dict[str, Any]:
        rule = self.rule_guard.check_prompt(prompt, pipeline=pipeline)
        rule_meta = {
            "allowed": rule.get("allowed"),
            "risk_type": rule.get("risk_type"),
            "action": rule.get("action"),
            "provider": "rule_based_prompt_guard",
            "version": PROMPT_GUARD_VERSION,
        }
        if not rule.get("allowed", True):
            rule["model_guard_skipped"] = True
            rule["model_guard_skip_reason"] = "rule_blocked"
            rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
            rule["rule_guard"] = rule_meta
            return rule

        if not _env_bool("MODEL_GUARD_ENABLED", True):
            rule["model_guard_skipped"] = True
            rule["model_guard_skip_reason"] = "disabled"
            rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
            rule["rule_guard"] = rule_meta
            return rule

        model = get_model_prompt_guard()
        try:
            model_result = model.validate_prompt_sync(prompt)
        except Exception as exc:
            _logger.error("Hybrid model guard error: %s", exc)
            return _apply_unavailable_policy(
                rule,
                rule_meta,
                pipeline=pipeline,
                skip_reason=f"error:{exc}",
            )

        if model_result.degraded or model_result.metadata.get("model_unavailable"):
            return _apply_unavailable_policy(
                rule,
                rule_meta,
                pipeline=pipeline,
                skip_reason="unavailable_fail_closed",
                model_result=model_result.to_dict(),
            )

        shadow = _env_bool("MODEL_GUARD_SHADOW_MODE", False)
        if not model_result.allowed:
            if shadow:
                _logger.info(
                    "MODEL_GUARD_SHADOW_BLOCK pipeline=%s risk_type=%s confidence=%.3f "
                    "MODEL_GUARD_ACTION=shadow_allow",
                    pipeline,
                    model_result.risk_type,
                    model_result.confidence,
                )
                rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
                rule["rule_guard"] = rule_meta
                rule["model_guard"] = model_result.to_dict()
                rule["model_guard_skipped"] = False
                rule["model_guard_shadow"] = True
                rule["model_guard_would_block"] = True
                return rule

            out = model_result.to_legacy_dict()
            out["prompt_guard_version"] = PROMPT_GUARD_VERSION
            out["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
            out["rule_guard"] = rule_meta
            out["model_guard"] = model_result.to_dict()
            out["allowed"] = False
            out["action"] = "block"
            _logger.info(
                "HYBRID_PROMPT_GUARD_BLOCKED pipeline=%s by=model risk_type=%s "
                "MODEL_GUARD_CATEGORY=%s MODEL_GUARD_CONFIDENCE=%.3f MODEL_GUARD_ACTION=block latency_ms=%.1f",
                pipeline,
                model_result.risk_type,
                (model_result.metadata or {}).get("classifier_category") or model_result.risk_type,
                model_result.confidence,
                model_result.latency_ms,
            )
            return out

        rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
        rule["rule_guard"] = rule_meta
        rule["model_guard"] = model_result.to_dict()
        rule["model_guard_skipped"] = False
        return rule

    def check_document_text(self, text: str, *, pipeline: str = "upload") -> dict[str, Any]:
        rule = self.rule_guard.check_document_text(text, pipeline=pipeline)
        rule_meta = {
            "allowed": rule.get("allowed"),
            "action": rule.get("action"),
            "risk_type": rule.get("risk_type"),
            "hit_count": rule.get("hit_count"),
            "provider": "rule_based_prompt_guard",
        }

        rule_blocked = str(rule.get("action") or "").lower() == "block" and not rule.get("allowed", True)

        if rule_blocked:
            rule["model_guard_skipped"] = True
            rule["rule_guard"] = rule_meta
            rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
            return rule

        if not _env_bool("MODEL_GUARD_ENABLED", True):
            rule["model_guard_skipped"] = True
            rule["rule_guard"] = rule_meta
            rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
            return rule

        model = get_model_prompt_guard()
        try:
            model_result = model.validate_document_text_sync(text)
        except Exception as exc:
            _logger.error("Hybrid document model guard error: %s", exc)
            return _apply_unavailable_policy(
                rule,
                rule_meta,
                pipeline=pipeline,
                skip_reason=f"error:{exc}",
            )

        if model_result.degraded or model_result.metadata.get("model_unavailable"):
            return _apply_unavailable_policy(
                rule,
                rule_meta,
                pipeline=pipeline,
                skip_reason="unavailable_fail_closed",
                model_result=model_result.to_dict(),
            )

        doc_action = (
            os.getenv("MODEL_GUARD_DOCUMENT_ACTION")
            or os.getenv("DOCUMENT_PROMPT_INJECTION_ACTION")
            or "human_review"
        ).strip().lower() or "human_review"

        shadow = _env_bool("MODEL_GUARD_SHADOW_MODE", False)
        if model_result.allowed:
            rule["model_guard"] = model_result.to_dict()
            rule["rule_guard"] = rule_meta
            rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
            return rule

        if shadow:
            rule["model_guard"] = model_result.to_dict()
            rule["rule_guard"] = rule_meta
            rule["hybrid_version"] = HYBRID_PROMPT_GUARD_VERSION
            rule["model_guard_shadow"] = True
            rule["model_guard_would_block"] = True
            return rule

        action = doc_action if doc_action in {"block", "human_review", "flag", "quarantine"} else "human_review"
        if action == "quarantine":
            action = "human_review"
        allowed = action in {"allow", "flag"}
        out = {
            "allowed": allowed,
            "risk_type": model_result.risk_type or "document_prompt_injection",
            "severity": "high" if action == "block" else "medium",
            "reason": model_result.reason
            or "Model guard detected embedded prompt-injection in document text.",
            "action": action,
            "hits": [{"pattern": "model_guard", "type": "document_prompt_injection"}],
            "hit_count": max(1, int(rule.get("hit_count") or 0)),
            "prompt_guard_version": PROMPT_GUARD_VERSION,
            "hybrid_version": HYBRID_PROMPT_GUARD_VERSION,
            "safe_message": model_result.safe_message,
            "categories": model_result.categories,
            "rule_guard": rule_meta,
            "model_guard": model_result.to_dict(),
        }
        if rule.get("hits"):
            out["hits"] = list(rule.get("hits") or []) + out["hits"]
            out["hit_count"] = len(out["hits"])
        _logger.info(
            "MODEL_GUARD_EXECUTED pipeline=%s MODEL_GUARD_CATEGORY=%s MODEL_GUARD_CONFIDENCE=%.3f "
            "MODEL_GUARD_ACTION=%s",
            pipeline,
            (model_result.metadata or {}).get("classifier_category") or model_result.risk_type,
            model_result.confidence,
            action,
        )
        return out


def hybrid_prompt_guard_fingerprint() -> dict[str, Any]:
    from src.features.security.moderation.ai_safety_guard import prompt_guard_fingerprint
    from src.features.security.moderation.model_prompt_guard import get_model_prompt_guard

    return {
        "hybrid_version": HYBRID_PROMPT_GUARD_VERSION,
        "rule_guard": prompt_guard_fingerprint(),
        "model_guard": get_model_prompt_guard().health(),
        "model_guard_metrics": model_guard_metrics_snapshot(),
        "architecture": "rule_then_model",
        "shadow_mode": _env_bool("MODEL_GUARD_SHADOW_MODE", False),
        "fail_open": prompt_guard_fail_open(),
        "unavailable_action": prompt_guard_unavailable_action(),
    }
