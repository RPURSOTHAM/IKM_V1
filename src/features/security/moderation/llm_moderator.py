"""Configuration-driven LLM moderation (OpenAI Moderation or Llama Guard HTTP).

Always returns a structured decision. When remote providers are unavailable,
uses a deterministic local safety classifier — never a dead `is_configured=False`
placeholder that silently skips moderation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Local high-risk patterns used when remote APIs are unavailable.
_LOCAL_BLOCK_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bignore (all |any )?(previous|prior) (instructions|rules)\b",
        r"\bjailbreak\b",
        r"\bdan mode\b",
        r"\bhow to (make|build|create) (a )?(bomb|explosive|bioweapon)\b",
        r"\bchild (sexual|porn)",
        r"\bexfiltrate (secrets|credentials|api keys)\b",
        r"\b(suicide methods|how to kill myself|overdose guidance)\b",
    )
]
# Medium-risk: ideation / sensitive wording → warn so DLP routes to human review.
_LOCAL_WARN_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bprivileged and confidential\b",
        r"\battorney[- ]client\b",
        r"\btrade secret\b",
        r"\bunreleased clinical (trial )?data\b",
        r"\bself[-\s]?harm\b",
        r"\b(hurting|harming|hurt|harm)\s+myself\b",
        r"\bthinking about hurting\b",
        r"\b(suicidal|suicide)\b",
        r"\b(want to die|end my life|take my own life|kill myself)\b",
        r"\b(social security number|aadhaar number|passport number)\b",
        r"\b(bomb making|weapon construction|murder instructions)\b",
        r"\b(graphic violence|violent assault)\b",
    )
]


def _provider() -> str:
    return (os.getenv("LLM_MODERATION_PROVIDER") or os.getenv("MODERATION_PROVIDER") or "auto").strip().lower()


def _openai_key() -> str:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_MODERATION_API_KEY")
        or os.getenv("MODERATION_API_KEY")
        or ""
    ).strip()


def _llama_guard_url() -> str:
    return (
        os.getenv("LLAMA_GUARD_URL")
        or os.getenv("MODERATION_LLAMA_GUARD_URL")
        or ""
    ).strip()


def _local_moderate(text: str) -> dict[str, Any]:
    for pattern in _LOCAL_BLOCK_PATTERNS:
        if pattern.search(text or ""):
            return {
                "provider": "local_rules",
                "risk": "high",
                "confidence": 0.92,
                "action": "block",
                "flags": ["local_block_pattern"],
                "reason": f"Local safety rule matched: {pattern.pattern}",
                "configured": True,
            }
    for pattern in _LOCAL_WARN_PATTERNS:
        if pattern.search(text or ""):
            return {
                "provider": "local_rules",
                "risk": "medium",
                "confidence": 0.75,
                "action": "warn",
                "flags": ["local_warn_pattern"],
                "reason": f"Local safety rule matched: {pattern.pattern}",
                "configured": True,
            }
    return {
        "provider": "local_rules",
        "risk": "low",
        "confidence": 0.55,
        "action": "allow",
        "flags": [],
        "reason": "No local safety violations detected.",
        "configured": True,
    }


def _openai_moderate(text: str) -> dict[str, Any] | None:
    api_key = _openai_key()
    if not api_key:
        return None
    base = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODERATION_MODEL") or "omni-moderation-latest"
    payload = json.dumps({"model": model, "input": text[:12000]}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/moderations",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(os.getenv("MODERATION_TIMEOUT_SEC", "20"))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning("OpenAI moderation failed: %s", exc)
        return None

    results = body.get("results") or []
    if not results:
        return {
            "provider": "openai",
            "risk": "low",
            "confidence": 0.5,
            "action": "allow",
            "flags": [],
            "reason": "Empty moderation response.",
            "configured": True,
        }
    first = results[0]
    flagged = bool(first.get("flagged"))
    categories = first.get("categories") or {}
    scores = first.get("category_scores") or {}
    flags = [name for name, on in categories.items() if on]
    max_score = max((float(v) for v in scores.values()), default=0.0) if scores else (0.9 if flagged else 0.2)
    if flagged or max_score >= 0.8:
        action, risk = "block", "high"
    elif max_score >= 0.5:
        action, risk = "warn", "medium"
    else:
        action, risk = "allow", "low"
    return {
        "provider": "openai",
        "risk": risk,
        "confidence": max_score,
        "action": action,
        "flags": flags,
        "reason": "OpenAI moderation flagged content." if flagged else "OpenAI moderation clear.",
        "configured": True,
        "raw_categories": categories,
    }


def _llama_guard_moderate(text: str, *, role: str = "user") -> dict[str, Any] | None:
    url = _llama_guard_url()
    if not url:
        return None
    payload = json.dumps(
        {
            "model": os.getenv("LLAMA_GUARD_MODEL") or "meta-llama/Llama-Guard-3-8B",
            "messages": [{"role": role, "content": text[:12000]}],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = _openai_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(os.getenv("MODERATION_TIMEOUT_SEC", "30"))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Llama Guard moderation failed: %s", exc)
        return None

    # OpenAI-compatible chat completion response or plain {safe: bool}
    content = ""
    if isinstance(body.get("choices"), list) and body["choices"]:
        content = str(((body["choices"][0] or {}).get("message") or {}).get("content") or "")
    content = content or str(body.get("output") or body.get("result") or "")
    lowered = content.lower()
    unsafe = "unsafe" in lowered or body.get("safe") is False
    if unsafe:
        return {
            "provider": "llama_guard",
            "risk": "high",
            "confidence": 0.9,
            "action": "block",
            "flags": ["llama_guard_unsafe"],
            "reason": content[:500] or "Llama Guard marked content unsafe.",
            "configured": True,
        }
    return {
        "provider": "llama_guard",
        "risk": "low",
        "confidence": 0.8,
        "action": "allow",
        "flags": [],
        "reason": content[:500] or "Llama Guard marked content safe.",
        "configured": True,
    }


class LLMModerator:
    """Unified moderator for upload text, prompts, and model outputs."""

    def __init__(self) -> None:
        provider = _provider()
        self.provider = provider
        # Always "configured" — local rules are a real production fallback.
        self.is_configured = True

    def moderate_content(self, text: str, *, role: str = "user") -> dict[str, Any]:
        provider = _provider()
        result: dict[str, Any] | None = None
        if provider in {"openai", "auto"}:
            result = _openai_moderate(text)
        if result is None and provider in {"llama_guard", "llamaguard", "auto"}:
            result = _llama_guard_moderate(text, role=role)
        if result is None:
            result = _local_moderate(text)
        # Normalize action vocabulary: ALLOW / WARN / MASK / BLOCK
        action = str(result.get("action") or "allow").lower()
        if action not in {"allow", "warn", "mask", "block"}:
            risk = str(result.get("risk") or "low").lower()
            action = "block" if risk in {"high", "critical"} else ("warn" if risk == "medium" else "allow")
            result["action"] = action
        result["configured"] = True
        return result

    def moderate_prompt(self, prompt: str) -> dict[str, Any]:
        return self.moderate_content(prompt, role="user")

    def moderate_output(self, output: str) -> dict[str, Any]:
        return self.moderate_content(output, role="assistant")


# Backward-compatible aliases expected by older imports.
class LLMContentModerator(LLMModerator):
    """Alias retained for existing UploadSecurityPipeline imports."""


class ModelSafetyGuard:
    """Model-backed prompt/output guard replacing the previous placeholder."""

    def __init__(self) -> None:
        self._moderator = LLMModerator()
        self.is_configured = True

    def validate_prompt(self, prompt: str) -> dict[str, Any]:
        result = self._moderator.moderate_prompt(prompt)
        allowed = result.get("action") != "block"
        # Combine with rule PromptGuard for defense in depth.
        try:
            from src.features.security.moderation.ai_safety_guard import PromptGuard

            rule = PromptGuard().check_prompt(prompt)
            if not rule.get("allowed", True):
                return rule
        except Exception:
            pass
        return {
            "allowed": allowed,
            "risk_type": (result.get("flags") or [None])[0],
            "severity": result.get("risk", "low"),
            "reason": result.get("reason"),
            "safe_message": "Request blocked by content moderation policy." if not allowed else None,
            "moderation": result,
        }

    def validate_output(self, response_text: str) -> dict[str, Any]:
        result = self._moderator.moderate_output(response_text)
        action = result.get("action", "allow")
        text = response_text
        if action in {"mask", "warn"}:
            try:
                from src.features.security.dlp.sensitive_data_detector import mask_text_content

                text = mask_text_content(response_text)
                action = "mask"
            except Exception:
                pass
        if action == "block":
            text = "Response blocked by content moderation policy."
        return {
            "safe": action != "block",
            "action": action,
            "sanitized_text": text,
            "moderation": result,
        }
