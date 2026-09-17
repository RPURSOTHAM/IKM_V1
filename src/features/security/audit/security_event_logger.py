"""Unified security audit logger for upload/prompt/output/export/review decisions."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()

EVENT_UPLOAD_SCAN = "UPLOAD_SCAN"
EVENT_PROMPT_SCAN = "PROMPT_SCAN"
EVENT_OUTPUT_SCAN = "OUTPUT_SCAN"
EVENT_EXPORT_SCAN = "EXPORT_SCAN"
EVENT_HUMAN_REVIEW = "HUMAN_REVIEW"
EVENT_SIMILARITY_SCAN = "SIMILARITY_SCAN"
EVENT_BLACKLIST_SCAN = "BLACKLIST_SCAN"
EVENT_MODEL_GUARD_UNAVAILABLE = "MODEL_GUARD_UNAVAILABLE"


def _audit_path() -> Path:
    configured = (os.getenv("SECURITY_AUDIT_LOG_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    from src.shared.networking.document_paths import resolve_security_documents_dir

    return resolve_security_documents_dir() / "audit_logs.jsonl"


def log_security_event(
    event_type: str,
    *,
    decision: str,
    reason: str = "",
    severity: str = "low",
    policy: str = "",
    user: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one structured audit record. Never raises to callers."""
    record = {
        "timestamp": time.time(),
        "event_type": event_type,
        "user": user or os.getenv("SECURITY_AUDIT_USER") or "system",
        "document_id": document_id,
        "document": document_name,
        "decision": decision,
        "reason": reason,
        "policy": policy,
        "severity": severity,
        "metadata": metadata or {},
    }
    try:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=True, default=str)
        with _lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass
    return record


class SecurityAuditLogger:
    """OO wrapper used by services that prefer an injectable logger."""

    def log(self, event_type: str, **kwargs: Any) -> dict[str, Any]:
        return log_security_event(event_type, **kwargs)
