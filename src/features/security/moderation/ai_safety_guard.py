"""Prompt / output safety guards with version fingerprinting and metrics."""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict

_logger = logging.getLogger(__name__)

# Bump when detection rules change — used for container/source parity checks.
PROMPT_GUARD_VERSION = "2026.07.20.2"
PROMPT_GUARD_IMPLEMENTATION = "rule_based_prompt_guard"
PROMPT_GUARD_MODULE = "src.features.security.moderation.ai_safety_guard"

_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, int] = {
    "blocked_requests_total": 0,
    "allowed_requests_total": 0,
    "jailbreak_attempts_total": 0,
    "prompt_extraction_attempts_total": 0,
    "document_injection_hits_total": 0,
}


def prompt_guard_fingerprint() -> dict[str, Any]:
    return {
        "version": PROMPT_GUARD_VERSION,
        "implementation": PROMPT_GUARD_IMPLEMENTATION,
        "module": PROMPT_GUARD_MODULE,
        "source_file": __file__,
    }


def prompt_guard_metrics_snapshot() -> dict[str, int]:
    with _METRICS_LOCK:
        return dict(_METRICS)


def _inc_metric(name: str, amount: int = 1) -> None:
    with _METRICS_LOCK:
        _METRICS[name] = int(_METRICS.get(name, 0)) + amount
    try:
        from src.infrastructure.application_support.metrics import get_metrics

        metrics = get_metrics()
        counter = getattr(metrics, name, None)
        if counter is not None and hasattr(counter, "inc"):
            counter.inc(amount)
    except Exception:
        pass


def _record_outcome(result: dict[str, Any], *, pipeline: str) -> None:
    allowed = bool(result.get("allowed", True))
    risk_type = str(result.get("risk_type") or "")
    severity = str(result.get("severity") or "low")
    action = "allow" if allowed else str(result.get("action") or "block")
    _logger.info(
        "PROMPT_GUARD_EXECUTED pipeline=%s allowed=%s risk_type=%s severity=%s action=%s "
        "version=%s pipeline_terminated=%s",
        pipeline,
        allowed,
        risk_type or "none",
        severity,
        action,
        PROMPT_GUARD_VERSION,
        (not allowed),
    )
    if allowed:
        _inc_metric("allowed_requests_total")
    else:
        _inc_metric("blocked_requests_total")
        if risk_type == "jailbreak":
            _inc_metric("jailbreak_attempts_total")
        if risk_type in {"prompt_extraction", "sensitive_extraction", "context_extraction", "config_disclosure"}:
            _inc_metric("prompt_extraction_attempts_total")
        if risk_type == "document_prompt_injection":
            _inc_metric("document_injection_hits_total")


class PromptGuard:
    """Canonical rule-based prompt injection / jailbreak detector.

    Single implementation — imported by chat, retrieval, model safety, and upload.
    """

    VERSION = PROMPT_GUARD_VERSION

    # Ordered: first match wins.
    _RULES: list[tuple[str, str, str, list[str]]] = [
        (
            "jailbreak",
            "high",
            "Jailbreak attempt detected.",
            [
                r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+instructions?",
                r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?|policies?)",
                r"forget\s+(?:all\s+|your\s+)?(?:previous|prior|system)\s+instructions?",
                r"override\s+(?:all\s+)?(?:safety|system|security)\s+(?:rules?|guards?|policies?)",
                r"bypass\s+(?:all\s+)?(?:rules?|safety|guardrails?|filters?|restrictions?)",
                r"pretend\s+you\s+are\s+(?:unrestricted|uncensored|jailbroken|dan)",
                r"\bdan\s+mode\b",
                r"\bjailbreak\b",
                r"developer\s+mode\s*(?:enabled|on|activated)?",
                r"you\s+are\s+now\s+(?:in\s+)?(?:developer|god|unrestricted)\s+mode",
                r"act\s+as\s+(?:if\s+)?(?:you\s+have\s+)?no\s+(?:restrictions?|limits?|rules?)",
                r"do\s+anything\s+now",
                r"role[\s-]?play\s+as\s+(?:an?\s+)?(?:unrestricted|uncensored)",
            ],
        ),
        (
            "prompt_extraction",
            "high",
            "Prompt / system-instruction extraction attempt detected.",
            [
                r"reveal\s+(?:the\s+)?(?:complete\s+|full\s+|entire\s+)?system\s+prompt",
                r"show\s+(?:me\s+)?(?:the\s+)?(?:complete\s+|full\s+|entire\s+)?system\s+prompt",
                r"print\s+(?:the\s+)?(?:complete\s+|full\s+|entire\s+)?system\s+prompt",
                r"(?:what|show|print|reveal|dump)\s+(?:is\s+)?(?:your\s+)?(?:hidden\s+)?(?:system\s+)?(?:prompt|instructions?)",
                r"reveal\s+hidden\s+(?:policies|instructions?|prompts?|memory)",
                r"show\s+(?:all\s+)?(?:hidden\s+)?(?:developer|internal)\s+instructions?",
                r"output\s+(?:any\s+)?hidden\s+(?:memory|developer\s+instructions?|policies)",
                r"print\s+(?:your\s+)?(?:initial|system|developer)\s+(?:prompt|instructions?)",
                r"repeat\s+(?:your\s+)?(?:system|initial)\s+(?:prompt|instructions?)",
                r"dump\s+(?:the\s+)?(?:system|hidden)\s+(?:prompt|context)",
            ],
        ),
        (
            "config_disclosure",
            "high",
            "Configuration / secrets disclosure attempt detected.",
            [
                r"print\s+(?:all\s+)?environment\s+variables?",
                r"show\s+(?:all\s+)?environment\s+variables?",
                r"dump\s+(?:all\s+)?(?:env|environment)\s+(?:vars?|variables?)",
                r"(?:show|print|reveal|list|dump)\s+(?:all\s+)?(?:api\s+)?keys?(?:\s+and\s+internal\s+configuration)?",
                r"show\s+(?:all\s+)?api\s+keys",
                r"reveal\s+(?:all\s+)?api\s+keys",
                r"print\s+(?:all\s+)?api\s+keys",
                r"show\s+internal\s+configuration",
                r"reveal\s+(?:hidden\s+)?policies",
                r"(?:show|print|dump)\s+(?:secrets?|credentials?|tokens?)",
                r"expose\s+secrets?",
                r"database\s+password",
            ],
        ),
        (
            "sensitive_extraction",
            "high",
            "Sensitive data extraction attempt detected.",
            [
                r"show\s+all\s+aadhaar",
                r"list\s+(?:all\s+)?(?:aadhaar|pan)\s+numbers?",
                r"list\s+pan\s+numbers?",
                r"reveal\s+passwords?",
                r"export\s+employee\s+personal\s+data",
                r"show\s+bank\s+account\s+numbers?",
                r"show\s+all\s+passwords?",
                r"list\s+all\s+aadhaar",
            ],
        ),
        (
            "context_extraction",
            "high",
            "Context / memory extraction attempt detected.",
            [
                r"(?:show|print|reveal|dump)\s+(?:all\s+)?(?:conversation|chat)\s+(?:history|memory)",
                r"(?:show|print|reveal)\s+(?:the\s+)?(?:full\s+)?(?:rag|retrieval)\s+context",
                r"what\s+(?:documents?|chunks?)\s+(?:were|are)\s+(?:retrieved|in\s+context)",
                r"output\s+(?:the\s+)?(?:raw\s+)?(?:retrieved\s+)?context\s+(?:window|payload)",
            ],
        ),
        (
            "role_override",
            "high",
            "Role / identity override attack detected.",
            [
                r"you\s+are\s+(?:now\s+)?(?:a\s+)?(?:different|new)\s+(?:ai|assistant|model)",
                r"from\s+now\s+on\s+you\s+(?:will|must|shall)\s+(?:ignore|disregard)",
                r"your\s+new\s+(?:persona|identity|role)\s+is",
                r"system\s*:\s*you\s+are",
                r"\[system\]\s*(?:ignore|override|new\s+instructions)",
                r"<\|?(?:system|im_start)\|?>",
            ],
        ),
        (
            "unsafe_request",
            "high",
            "Unsafe request detected.",
            [
                r"bypass\s+access\s+control",
                r"dump\s+confidential\s+document",
                r"bypass\s+auth",
            ],
        ),
        (
            "toxicity",
            "high",
            "Toxic or abusive content detected.",
            [
                r"\b(?:abuse|harass|exploit|insult|offensive)\b",
            ],
        ),
    ]

    # Patterns that look like embedded jailbreak instructions inside documents.
    _DOCUMENT_INJECTION_PATTERNS: list[str] = [
        r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above)\s+instructions?",
        r"system\s+prompt\s*:",
        r"\[system\]",
        r"<\|?(?:system|im_start)\|?>",
        r"you\s+are\s+now\s+(?:in\s+)?(?:developer|god|unrestricted)\s+mode",
        r"\bjailbreak\b",
        r"reveal\s+(?:the\s+)?(?:complete\s+|full\s+)?system\s+prompt",
        r"do\s+anything\s+now",
        r"bypass\s+(?:all\s+)?(?:safety|guardrails?|filters?)",
        r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|rules?)",
    ]

    def __init__(self) -> None:
        self.version = PROMPT_GUARD_VERSION

    def fingerprint(self) -> dict[str, Any]:
        return prompt_guard_fingerprint()

    def check_prompt(self, prompt: str, *, pipeline: str = "chat") -> Dict[str, Any]:
        prompt_lower = (prompt or "").lower()
        for risk_type, severity, reason, patterns in self._RULES:
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    result = {
                        "allowed": False,
                        "risk_type": risk_type,
                        "severity": severity,
                        "reason": reason,
                        "action": "block",
                        "matched_pattern": pattern,
                        "prompt_guard_version": PROMPT_GUARD_VERSION,
                        "safe_message": self._safe_message(risk_type),
                    }
                    _record_outcome(result, pipeline=pipeline)
                    return result

        result = {
            "allowed": True,
            "risk_type": None,
            "severity": "low",
            "reason": None,
            "action": "allow",
            "prompt_guard_version": PROMPT_GUARD_VERSION,
        }
        _record_outcome(result, pipeline=pipeline)
        return result

    def check_document_text(self, text: str, *, pipeline: str = "upload") -> Dict[str, Any]:
        """Scan extracted document text for embedded prompt-injection payloads."""
        sample = (text or "")[:50000]
        sample_lower = sample.lower()
        hits: list[dict[str, str]] = []
        for pattern in self._DOCUMENT_INJECTION_PATTERNS:
            if re.search(pattern, sample_lower):
                hits.append({"pattern": pattern, "type": "document_prompt_injection"})

        action = (
            os.getenv("DOCUMENT_PROMPT_INJECTION_ACTION", "human_review").strip().lower()
            or "human_review"
        )
        if action not in {"block", "human_review", "quarantine", "allow", "flag"}:
            action = "human_review"

        if not hits:
            result = {
                "allowed": True,
                "risk_type": None,
                "severity": "low",
                "reason": None,
                "action": "allow",
                "hits": [],
                "prompt_guard_version": PROMPT_GUARD_VERSION,
            }
            _record_outcome(result, pipeline=pipeline)
            return result

        severity = "high" if action == "block" else "medium"
        allowed = action in {"allow", "flag"}
        result = {
            "allowed": allowed,
            "risk_type": "document_prompt_injection",
            "severity": severity,
            "reason": (
                "Document contains embedded prompt-injection / jailbreak instructions "
                "that could influence downstream RAG prompts."
            ),
            "action": action if action != "quarantine" else "human_review",
            "hits": hits,
            "hit_count": len(hits),
            "prompt_guard_version": PROMPT_GUARD_VERSION,
            "safe_message": (
                "Upload blocked: document contains prompt-injection instructions."
                if action == "block"
                else "Document flagged for review due to embedded prompt-injection content."
            ),
        }
        _record_outcome(result, pipeline=pipeline)
        return result

    @staticmethod
    def _safe_message(risk_type: str) -> str:
        messages = {
            "jailbreak": (
                "I cannot execute instructions that bypass my safety guardrails. "
                "Please ask a direct question about the documents."
            ),
            "prompt_extraction": (
                "I cannot reveal system prompts, hidden policies, or developer instructions."
            ),
            "config_disclosure": (
                "I cannot disclose environment variables, API keys, or internal configuration."
            ),
            "sensitive_extraction": (
                "I can't help reveal sensitive information. "
                "Please ask for a summary or non-sensitive analysis."
            ),
            "context_extraction": (
                "I cannot dump raw retrieval context or hidden conversation memory."
            ),
            "role_override": (
                "I cannot change my safety role or ignore security policies."
            ),
            "unsafe_request": (
                "I cannot assist with requests to bypass security or expose secrets."
            ),
            "toxicity": (
                "I must decline answering prompts containing abusive or offensive language."
            ),
            "document_prompt_injection": (
                "Document blocked or quarantined due to embedded prompt-injection content."
            ),
        }
        return messages.get(
            risk_type,
            "Request blocked by security policy.",
        )


class OutputGuard:
    def check_output(self, response_text: str) -> Dict[str, Any]:
        from src.features.security.dlp.sensitive_data_detector import ComplianceScanner, mask_text_content

        scanner = ComplianceScanner()
        detections = scanner.scan_text(response_text)

        high_risk_detected = any(det.get("severity") == "high" for det in detections)
        if high_risk_detected:
            masked_text = mask_text_content(response_text)
            return {
                "safe": False,
                "action": "masked",
                "reason": "Generated response contained sensitive information.",
                "sanitized_text": masked_text,
            }

        return {
            "safe": True,
            "action": "allow",
            "sanitized_text": response_text,
        }
