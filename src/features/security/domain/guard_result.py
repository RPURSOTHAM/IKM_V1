"""Shared guard result contracts for rule and model Prompt Guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Canonical risk / category vocabulary for hybrid Prompt Guard.
GUARD_CATEGORIES = (
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
    "developer_prompt_extraction",
    "configuration_disclosure",
    "secrets_extraction",
    "api_key_requests",
    "environment_variable_requests",
    "data_exfiltration",
    "tool_misuse",
    "role_override",
    "indirect_prompt_injection",
    "context_extraction",
    "document_prompt_injection",
)


@dataclass
class GuardResult:
    allowed: bool
    action: str = "allow"  # allow | block | flag | human_review
    risk_type: str | None = None
    severity: str = "low"
    reason: str | None = None
    categories: list[str] = field(default_factory=list)
    confidence: float = 0.0
    provider: str = "unknown"
    latency_ms: float = 0.0
    safe_message: str | None = None
    degraded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def to_legacy_dict(self) -> dict[str, Any]:
        """Shape expected by existing chat/retrieval/upload callers."""
        return {
            "allowed": self.allowed,
            "action": self.action,
            "risk_type": self.risk_type,
            "severity": self.severity,
            "reason": self.reason,
            "safe_message": self.safe_message
            or (
                None
                if self.allowed
                else "Request blocked by security policy."
            ),
            "categories": list(self.categories),
            "confidence": self.confidence,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
            "prompt_guard_version": self.metadata.get("prompt_guard_version"),
            "model_guard": self.metadata.get("model_guard"),
            "rule_guard": self.metadata.get("rule_guard"),
            **{k: v for k, v in self.metadata.items() if k not in {"model_guard", "rule_guard", "prompt_guard_version"}},
        }
