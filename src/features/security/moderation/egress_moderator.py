"""Egress DLP helpers for preview, render, and chunk listing."""

from __future__ import annotations

from typing import Any

from src.features.security.audit.security_event_logger import EVENT_EXPORT_SCAN, log_security_event
from src.features.security.dlp.policy_loader import scan_keyword_policies
from src.features.security.dlp.sensitive_data_detector import mask_text_content


def redact_text_for_egress(text: str, *, document_id: str | None = None, channel: str = "egress") -> tuple[str, str]:
    """Return (possibly masked text, decision). Blocks high-risk keyword hits."""
    raw = text or ""
    kw = scan_keyword_policies(raw)
    decision = "allow"
    if kw.get("action") == "block":
        decision = "block"
        out = "[Content withheld by export DLP policy.]"
    elif kw.get("matches") or kw.get("action") in {"warning", "human_review", "warn"}:
        decision = "mask"
        out = mask_text_content(raw)
    else:
        out = mask_text_content(raw) if _looks_sensitive(raw) else raw
        if out != raw:
            decision = "mask"

    log_security_event(
        EVENT_EXPORT_SCAN,
        decision=decision,
        reason=f"{channel}:{kw.get('action') or decision}",
        severity="high" if decision == "block" else ("medium" if decision == "mask" else "low"),
        policy="egress_dlp",
        document_id=document_id,
        metadata={"channel": channel},
    )
    return out, decision


def _looks_sensitive(text: str) -> bool:
    lowered = (text or "").lower()
    needles = ("ssn", "aadhaar", "passport", "api key", "password", "bank account", "pan ")
    return any(n in lowered for n in needles)


def redact_preview_payload(payload: dict[str, Any], *, document_id: str | None = None) -> dict[str, Any]:
    out = dict(payload)
    body = out.get("content") or out.get("preview") or out.get("html") or ""
    if isinstance(body, str) and body:
        redacted, decision = redact_text_for_egress(body, document_id=document_id, channel="preview")
        if decision == "block":
            out["content"] = redacted
            out["preview"] = redacted
            out["html"] = f"<p>{redacted}</p>"
            out["export_dlp"] = decision
        elif decision == "mask":
            if "content" in out:
                out["content"] = redacted
            if "preview" in out:
                out["preview"] = redacted
            if "html" in out and isinstance(out["html"], str):
                out["html"] = redacted if "<" not in redacted else out["html"]  # keep structure if HTML-heavy
            out["export_dlp"] = decision
    return out


def redact_chunks_payload(payload: dict[str, Any], *, document_id: str | None = None) -> dict[str, Any]:
    out = dict(payload)
    chunks = list(out.get("chunks") or [])
    redacted_chunks = []
    worst = "allow"
    for chunk in chunks:
        item = dict(chunk) if isinstance(chunk, dict) else {"text": str(chunk)}
        text = str(item.get("text") or "")
        if text:
            new_text, decision = redact_text_for_egress(text, document_id=document_id, channel="chunks")
            item["text"] = new_text
            if decision == "block":
                worst = "block"
            elif decision == "mask" and worst != "block":
                worst = "mask"
        redacted_chunks.append(item)
    out["chunks"] = redacted_chunks
    out["export_dlp"] = worst
    return out
