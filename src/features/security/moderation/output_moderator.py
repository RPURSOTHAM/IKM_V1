"""Layered output moderation: Regex -> Presidio -> Llama Guard 3 -> Policy Engine.

Reuses existing OutputGuard/ComplianceScanner, ModelSafetyGuard, and DLPPolicyEngine.
Every layer returns a structured result; optional layers never block the pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

_logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class OutputModerationConfig:
    """All thresholds and layer toggles are configuration-driven."""

    enable_regex: bool = True
    enable_presidio: bool = True
    enable_llama_guard: bool = True
    enable_policy_engine: bool = True
    preferred_guard_model: str = "meta-llama/Llama-Guard-3-8B"
    block_on_prompt_leakage: bool = True
    block_on_jailbreak_response: bool = True
    presidio_score_threshold: float = 0.5
    presidio_spacy_model: str = ""
    # Only mask genuine PII/secrets; generic NER entities (PERSON/ORG/LOC)
    # would corrupt normal grounded answers.
    presidio_entities: tuple[str, ...] = (
        "CREDIT_CARD",
        "CRYPTO",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "PHONE_NUMBER",
        "MEDICAL_LICENSE",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "US_ITIN",
        "US_PASSPORT",
        "US_SSN",
        "IN_AADHAAR",
        "IN_PAN",
    )
    blocked_message: str = (
        "The response was blocked by output moderation policy."
    )
    system_prompt_markers: tuple[str, ...] = (
        "you are an enterprise document assistant",
        "answer only using the retrieved document context",
        "cite sources using bracket numbers",
    )

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> "OutputModerationConfig":
        # Regex + policy engine are mandatory security coverage; env cannot disable them.
        # Presidio / Llama Guard may be toggled as implementation strategy (fallbacks remain).
        cfg = cls(
            enable_regex=True,
            enable_presidio=_env_bool("MODERATION_ENABLE_PRESIDIO", True),
            enable_llama_guard=_env_bool("MODERATION_ENABLE_LLAMA_GUARD", True),
            enable_policy_engine=True,
            preferred_guard_model=os.getenv(
                "MODERATION_GUARD_MODEL", "meta-llama/Llama-Guard-3-8B"
            ),
            block_on_prompt_leakage=_env_bool("MODERATION_BLOCK_PROMPT_LEAKAGE", True),
            block_on_jailbreak_response=_env_bool("MODERATION_BLOCK_JAILBREAK", True),
            presidio_score_threshold=float(
                os.getenv("MODERATION_PRESIDIO_SCORE_THRESHOLD", "0.5")
            ),
            presidio_spacy_model=os.getenv("MODERATION_PRESIDIO_SPACY_MODEL", ""),
        )
        if not _env_bool("MODERATION_ENABLE_REGEX", True) or not _env_bool(
            "MODERATION_ENABLE_POLICY_ENGINE", True
        ):
            _logger.warning(
                "Ignoring attempt to disable mandatory output moderation layers "
                "(regex/policy_engine); coverage remains enforced."
            )
        entities_env = os.getenv("MODERATION_PRESIDIO_ENTITIES")
        if entities_env:
            cfg.presidio_entities = tuple(
                entity.strip().upper() for entity in entities_env.split(",") if entity.strip()
            )
        if overrides:
            for key, value in overrides.items():
                if hasattr(cfg, key) and value is not None:
                    # Unit tests may isolate layers via explicit overrides.
                    setattr(cfg, key, value)
        return cfg


@dataclass
class ModerationLayerResult:
    layer: str
    status: str  # ok | skipped | degraded | error
    action: str  # allow | mask | block
    confidence: float = 1.0
    execution_time_ms: float = 0.0
    detections: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ModerationReport:
    final_action: str  # allow | mask | block
    safe: bool
    sanitized_text: str
    severity: str = "low"
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    layers: list[ModerationLayerResult] = field(default_factory=list)
    total_execution_time_ms: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "final_action": self.final_action,
            "safe": self.safe,
            "severity": self.severity,
            "flags": list(self.flags),
            "reasons": list(self.reasons),
            "total_execution_time_ms": round(self.total_execution_time_ms, 2),
            "layers": [
                {
                    "layer": layer.layer,
                    "status": layer.status,
                    "action": layer.action,
                    "confidence": round(layer.confidence, 3),
                    "execution_time_ms": round(layer.execution_time_ms, 2),
                    "detection_count": len(layer.detections),
                    "detection_types": sorted(
                        {str(d.get("type") or d.get("entity_type") or "unknown") for d in layer.detections}
                    ),
                    "error": layer.error,
                }
                for layer in self.layers
            ],
        }


_JAILBREAK_RESPONSE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bdeveloper\s+mode\s+(?:enabled|activated)\b",
        r"\bi\s+am\s+now\s+(?:unrestricted|jailbroken|free\s+of\s+restrictions)\b",
        r"\bignoring\s+(?:my|all)\s+(?:previous\s+)?(?:instructions|guidelines|safety)\b",
        r"\bas\s+dan\b",
        r"\bmy\s+safety\s+guardrails\s+are\s+(?:disabled|off|bypassed)\b",
        r"\bbypass(?:ed|ing)?\s+(?:the\s+)?safety\b",
    )
]

_PROMPT_LEAKAGE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmy\s+system\s+prompt\s+(?:is|says|reads)\b",
        r"\bhere\s+is\s+(?:my|the)\s+system\s+prompt\b",
        r"\bthe\s+instructions\s+i\s+was\s+given\s+(?:are|say)\b",
        r"\bmy\s+(?:hidden|internal)\s+instructions\b",
        r"^\s*system\s+prompt\s*:",
    )
]


class RegexModerationLayer:
    """Layer 1: reuse existing rule-based OutputGuard and masking."""

    name = "regex"

    def run(self, text: str, config: OutputModerationConfig) -> tuple[str, ModerationLayerResult]:
        if not config.enable_regex:
            return text, ModerationLayerResult(layer=self.name, status="skipped", action="allow")
        started = time.perf_counter()
        try:
            from src.features.security.dlp.sensitive_data_detector import (
                ComplianceScanner,
                mask_text_content,
            )

            detections = ComplianceScanner().scan_text(text)
            high_risk = [d for d in detections if str(d.get("severity", "")).lower() in {"high", "critical"}]
            sanitized = mask_text_content(text) if high_risk else text
            action = "mask" if high_risk else "allow"
            return sanitized, ModerationLayerResult(
                layer=self.name,
                status="ok",
                action=action,
                confidence=1.0,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                detections=detections,
                metadata={"implementation": "compliance_scanner_regex", "masked": action == "mask"},
            )
        except Exception as exc:
            return text, ModerationLayerResult(
                layer=self.name,
                status="degraded",
                action="allow",
                confidence=0.0,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )


@lru_cache(maxsize=1)
def _cached_presidio_engines(spacy_model: str) -> tuple[Any, Any] | None:
    """Build Presidio engines against an already-installed spaCy model.

    Returns None when Presidio or a local spaCy model is unavailable; the
    default AnalyzerEngine() would otherwise try to download en_core_web_lg
    and block the pipeline for minutes.
    """
    try:
        import spacy.util  # type: ignore
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore
        from presidio_anonymizer import AnonymizerEngine  # type: ignore
    except ImportError:
        return None

    candidates = [spacy_model] if spacy_model else []
    candidates += ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
    model_name = next((m for m in candidates if m and spacy.util.is_package(m)), None)
    if model_name is None:
        return None

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    return analyzer, AnonymizerEngine()


class PresidioModerationLayer:
    """Layer 2: Microsoft Presidio when installed; graceful skip otherwise."""

    name = "presidio"

    def run(self, text: str, config: OutputModerationConfig) -> tuple[str, ModerationLayerResult]:
        if not config.enable_presidio:
            return text, ModerationLayerResult(layer=self.name, status="skipped", action="allow")
        started = time.perf_counter()
        try:
            engines = _cached_presidio_engines(config.presidio_spacy_model)
            if engines is None:
                return text, ModerationLayerResult(
                    layer=self.name,
                    status="skipped",
                    action="allow",
                    execution_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={"reason": "presidio or local spaCy model not installed"},
                )
            analyzer, anonymizer = engines
            results = analyzer.analyze(
                text=text,
                language="en",
                entities=list(config.presidio_entities) or None,
            )
            results = [r for r in results if float(r.score) >= config.presidio_score_threshold]
            if not results:
                return text, ModerationLayerResult(
                    layer=self.name,
                    status="ok",
                    action="allow",
                    execution_time_ms=(time.perf_counter() - started) * 1000,
                    metadata={"implementation": "presidio"},
                )
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            detections = [
                {
                    "entity_type": r.entity_type,
                    "score": float(r.score),
                    "start": r.start,
                    "end": r.end,
                    "severity": "high",
                }
                for r in results
            ]
            return anonymized.text, ModerationLayerResult(
                layer=self.name,
                status="ok",
                action="mask",
                confidence=max(float(r.score) for r in results),
                execution_time_ms=(time.perf_counter() - started) * 1000,
                detections=detections,
                metadata={"implementation": "presidio", "masked": True},
            )
        except ImportError:
            return text, ModerationLayerResult(
                layer=self.name,
                status="skipped",
                action="allow",
                execution_time_ms=(time.perf_counter() - started) * 1000,
                metadata={"reason": "presidio not installed"},
            )
        except Exception as exc:
            return text, ModerationLayerResult(
                layer=self.name,
                status="degraded",
                action="allow",
                confidence=0.0,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )


class LlamaGuardModerationLayer:
    """Layer 3: reuse ModelSafetyGuard; Llama Guard 3 preferred when configured."""

    name = "llama_guard"

    def run(self, text: str, config: OutputModerationConfig) -> tuple[str, ModerationLayerResult]:
        if not config.enable_llama_guard:
            return text, ModerationLayerResult(layer=self.name, status="skipped", action="allow")
        started = time.perf_counter()
        try:
            from src.features.security.moderation.llm_moderator import ModelSafetyGuard

            result = ModelSafetyGuard().validate_output(text)
            safe = bool(result.get("safe", True))
            raw_action = str(result.get("action") or ("allow" if safe else "mask")).lower()
            action = raw_action if raw_action in {"allow", "mask", "block"} else ("allow" if safe else "mask")
            sanitized = str(result.get("sanitized_text") or text)
            return sanitized, ModerationLayerResult(
                layer=self.name,
                status="ok",
                action=action,
                confidence=1.0 if safe else 0.9,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                detections=[] if safe else [{"type": "model_guard", "severity": "high", "reason": result.get("reason")}],
                metadata={
                    "preferred_model": config.preferred_guard_model,
                    "implementation": "model_safety_guard",
                    "reason": result.get("reason"),
                },
            )
        except Exception as exc:
            return text, ModerationLayerResult(
                layer=self.name,
                status="degraded",
                action="allow",
                confidence=0.0,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )


class PolicyEngineModerationLayer:
    """Layer 4: reuse DLPPolicyEngine plus response-specific leakage/jailbreak rules."""

    name = "policy_engine"

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = (system_prompt or "").strip().lower()

    def _prompt_leakage_detections(self, text: str, config: OutputModerationConfig) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        lowered = text.lower()
        for pattern in _PROMPT_LEAKAGE_PATTERNS:
            match = pattern.search(text)
            if match:
                detections.append(
                    {
                        "type": "prompt_leakage",
                        "severity": "high",
                        "matched_value": match.group(0),
                        "reason": "Response attempts to reveal system prompt or hidden instructions.",
                    }
                )
        markers = [m for m in config.system_prompt_markers if m and m in lowered]
        if self._system_prompt and len(self._system_prompt) > 40 and self._system_prompt[:60] in lowered:
            markers.append("system_prompt_verbatim")
        if markers:
            detections.append(
                {
                    "type": "prompt_leakage",
                    "severity": "high",
                    "matched_value": ", ".join(markers[:3]),
                    "reason": "Response echoes system prompt content.",
                }
            )
        return detections

    @staticmethod
    def _jailbreak_detections(text: str) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for pattern in _JAILBREAK_RESPONSE_PATTERNS:
            match = pattern.search(text)
            if match:
                detections.append(
                    {
                        "type": "jailbreak_response",
                        "severity": "critical",
                        "matched_value": match.group(0),
                        "reason": "Response indicates jailbreak/safety-bypass compliance.",
                    }
                )
        return detections

    def run(
        self,
        text: str,
        config: OutputModerationConfig,
        *,
        accumulated_detections: list[dict[str, Any]],
    ) -> tuple[str, ModerationLayerResult]:
        if not config.enable_policy_engine:
            return text, ModerationLayerResult(layer=self.name, status="skipped", action="allow")
        started = time.perf_counter()
        try:
            leakage = self._prompt_leakage_detections(text, config) if config.block_on_prompt_leakage else []
            jailbreak = self._jailbreak_detections(text) if config.block_on_jailbreak_response else []
            response_detections = leakage + jailbreak

            from src.features.security.dlp.dlp_engine import DLPPolicyEngine

            policy = DLPPolicyEngine().evaluate_policy(
                detections=accumulated_detections,
                doc_classification={},
                topic_classification={},
                similarity_results={},
                moderation_results={"risk": "low", "confidence": 0.0},
            )
            action = "allow"
            reasons: list[str] = []
            if response_detections:
                action = "block"
                reasons.extend(sorted({str(d.get("reason")) for d in response_detections}))
            elif policy.get("status") == "block":
                # High-risk data already masked upstream stays masked rather than blocked.
                action = "mask"
                reasons.append(str(policy.get("reason")))
            elif policy.get("status") in {"mask_and_allow", "human_review"}:
                action = "mask"
                reasons.append(str(policy.get("reason")))

            sanitized = config.blocked_message if action == "block" else text
            return sanitized, ModerationLayerResult(
                layer=self.name,
                status="ok",
                action=action,
                confidence=1.0,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                detections=response_detections,
                metadata={
                    "implementation": "dlp_policy_engine",
                    "policy_status": policy.get("status"),
                    "policy_severity": policy.get("severity"),
                    "reasons": reasons,
                },
            )
        except Exception as exc:
            return text, ModerationLayerResult(
                layer=self.name,
                status="degraded",
                action="allow",
                confidence=0.0,
                execution_time_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )


_ACTION_RANK = {"allow": 0, "mask": 1, "block": 2}
_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class OutputModerationPipeline:
    """Run Regex -> Presidio -> Llama Guard 3 -> Policy Engine over a response."""

    def __init__(
        self,
        config: OutputModerationConfig | None = None,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.config = config or OutputModerationConfig.from_env()
        self.regex_layer = RegexModerationLayer()
        self.presidio_layer = PresidioModerationLayer()
        self.llama_guard_layer = LlamaGuardModerationLayer()
        self.policy_layer = PolicyEngineModerationLayer(system_prompt=system_prompt)

    def moderate(self, text: str) -> ModerationReport:
        started = time.perf_counter()
        current = text or ""
        layers: list[ModerationLayerResult] = []
        accumulated: list[dict[str, Any]] = []

        for layer_fn in (
            lambda t: self.regex_layer.run(t, self.config),
            lambda t: self.presidio_layer.run(t, self.config),
            lambda t: self.llama_guard_layer.run(t, self.config),
        ):
            current, result = layer_fn(current)
            layers.append(result)
            accumulated.extend(result.detections)

        current, policy_result = self.policy_layer.run(
            current, self.config, accumulated_detections=accumulated
        )
        layers.append(policy_result)
        accumulated.extend(policy_result.detections)

        final_action = "allow"
        severity_rank = 1
        flags: list[str] = []
        reasons: list[str] = []
        for layer in layers:
            if _ACTION_RANK.get(layer.action, 0) > _ACTION_RANK[final_action]:
                final_action = layer.action
            for detection in layer.detections:
                det_type = str(detection.get("type") or detection.get("entity_type") or "unknown")
                if det_type not in flags:
                    flags.append(det_type)
                reason = detection.get("reason")
                if reason and str(reason) not in reasons:
                    reasons.append(str(reason))
                severity_rank = max(
                    severity_rank,
                    _SEVERITY_RANK.get(str(detection.get("severity", "low")).lower(), 1),
                )
        for extra_reason in policy_result.metadata.get("reasons") or []:
            if extra_reason and extra_reason not in reasons:
                reasons.append(str(extra_reason))

        if final_action == "block":
            current = self.config.blocked_message
            severity_rank = max(severity_rank, 3)

        severity = {1: "low", 2: "medium", 3: "high", 4: "critical"}[severity_rank]
        return ModerationReport(
            final_action=final_action,
            safe=final_action == "allow",
            sanitized_text=current,
            severity=severity,
            flags=flags,
            reasons=reasons,
            layers=layers,
            total_execution_time_ms=(time.perf_counter() - started) * 1000,
        )


def moderate_output(
    text: str,
    *,
    system_prompt: str | None = None,
    config: OutputModerationConfig | None = None,
) -> ModerationReport:
    # Keyword blacklist / allowlist before layered moderation.
    working = text or ""
    cfg = config or OutputModerationConfig.from_env()
    try:
        from src.features.security.dlp.policy_loader import scan_keyword_policies
        from src.features.security.dlp.sensitive_data_detector import mask_text_content
        from src.features.security.moderation.llm_moderator import ModelSafetyGuard

        kw = scan_keyword_policies(working)
        if kw.get("action") == "block":
            report = ModerationReport(
                final_action="block",
                safe=False,
                sanitized_text=cfg.blocked_message,
                severity="high",
                flags=["keyword_blacklist"],
                reasons=["Output blocked by keyword policy."],
                layers=[],
                total_execution_time_ms=0.0,
            )
        else:
            if kw.get("matches"):
                working = mask_text_content(working)
            guard = ModelSafetyGuard().validate_output(working)
            guard_action = str(guard.get("action") or "allow").lower()
            if guard_action == "block":
                report = ModerationReport(
                    final_action="block",
                    safe=False,
                    sanitized_text=cfg.blocked_message,
                    severity=str((guard.get("moderation") or {}).get("risk") or "high"),
                    flags=["model_safety_guard"],
                    reasons=[str((guard.get("moderation") or {}).get("reason") or "Model safety blocked output.")],
                    layers=[],
                    total_execution_time_ms=0.0,
                )
            else:
                if guard_action == "mask" and guard.get("sanitized_text"):
                    working = str(guard.get("sanitized_text"))
                report = OutputModerationPipeline(cfg, system_prompt=system_prompt).moderate(working)
                if kw.get("action") in {"warning", "human_review", "warn"} and report.final_action == "allow":
                    report.final_action = "mask"
                    report.safe = False
                    report.reasons = list(report.reasons) + ["Keyword policy required masking."]
                    report.flags = list(report.flags) + ["keyword_policy"]
    except Exception:
        report = OutputModerationPipeline(cfg, system_prompt=system_prompt).moderate(text)

    try:
        from src.features.security.audit.security_event_logger import EVENT_OUTPUT_SCAN, log_security_event

        log_security_event(
            EVENT_OUTPUT_SCAN,
            decision=report.final_action,
            reason="; ".join(report.reasons) if report.reasons else report.final_action,
            severity=report.severity,
            policy="output_moderation",
        )
    except Exception:
        pass
    return report
