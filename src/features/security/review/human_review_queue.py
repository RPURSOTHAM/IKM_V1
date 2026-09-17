"""Persistent human review queue for security validation."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from src.features.security.review.review_decisions import (
    STATE_APPROVED,
    STATE_APPROVED_MASKED,
    STATE_PENDING,
    STATE_REJECTED,
    available_actions_for_risk,
    canonical_severity_for_dlp,
    effective_available_actions,
    effective_risk_level,
    normalize_dlp_decision,
    reviewer_decision_to_status,
    severity_to_risk_level,
)

STATE_REPROCESSED = "REPROCESSED"
STATE_COMPLETED = "COMPLETED"

_VALID_STATES = {
    STATE_PENDING,
    STATE_APPROVED,
    STATE_APPROVED_MASKED,
    STATE_REJECTED,
    STATE_REPROCESSED,
    STATE_COMPLETED,
}
_LEGACY_PENDING = {"pending_review", "pending", "PENDING", "PENDING_REVIEW"}
_TERMINAL_STATES = {STATE_APPROVED, STATE_APPROVED_MASKED, STATE_REJECTED, STATE_REPROCESSED, STATE_COMPLETED}


class HumanReviewQueue:
    """Manages and persists the human review queue for security validation."""

    def __init__(self, queue_file: str = "_documents/human_review_queue.json") -> None:
        self.queue_file = queue_file
        os.makedirs(os.path.dirname(self.queue_file) or ".", exist_ok=True)
        self._load_queue()

    def _load_queue(self) -> None:
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, "r", encoding="utf-8") as f:
                    self.queue = json.load(f)
            except Exception:
                self.queue = []
        else:
            self.queue = []

    def _save_queue(self) -> None:
        try:
            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump(self.queue, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _normalize_status(status: str) -> str:
        raw = str(status or STATE_PENDING).strip()
        upper = raw.upper()
        if raw in _LEGACY_PENDING or upper in _LEGACY_PENDING:
            return STATE_PENDING
        mapping = {
            "APPROVED": STATE_APPROVED,
            "APPROVED_MASKED": STATE_APPROVED_MASKED,
            "REJECTED": STATE_REJECTED,
            "REPROCESSED": STATE_REPROCESSED,
            "COMPLETED": STATE_COMPLETED,
            "PENDING": STATE_PENDING,
        }
        return mapping.get(upper, upper if upper in _VALID_STATES else STATE_PENDING)

    def add_to_queue(
        self,
        document_id: str,
        document_name: str,
        severity: str,
        reason: str,
        detections: List[Dict[str, Any]],
        *,
        document_type_id: str | None = None,
        document_type_name: str | None = None,
        risk_level: str | None = None,
        dlp_decision: str | None = None,
        available_actions: List[str] | None = None,
        dlp_reason: str | None = None,
    ) -> str:
        masked_detections = []
        for det in detections:
            masked_detections.append(
                {
                    "category": det.get("category"),
                    "type": det.get("type"),
                    "severity": det.get("severity"),
                    "masked_value": det.get("masked_value"),
                    "location": det.get("location"),
                }
            )

        review_id = f"rev_{document_id}"
        decision = normalize_dlp_decision(dlp_decision or "human_review")
        canonical_severity = canonical_severity_for_dlp(severity, decision)
        draft = {
            "severity": canonical_severity,
            "risk_level": risk_level,
            "dlp_decision": decision,
            "available_actions": available_actions,
        }
        risk = effective_risk_level(draft)
        actions = effective_available_actions({**draft, "risk_level": risk})

        for item in self.queue:
            if item["review_id"] == review_id:
                prior_status = self._normalize_status(str(item.get("status") or ""))
                item.update(
                    {
                        "document_type_id": document_type_id or item.get("document_type_id"),
                        "document_type_name": document_type_name or item.get("document_type_name"),
                        "severity": canonical_severity,
                        "reason": reason,
                        "detections": masked_detections,
                        "risk_level": risk,
                        "dlp_decision": decision,
                        "available_actions": actions,
                        "dlp_reason": dlp_reason or reason,
                        # Fresh DLP human-review always re-opens the queue so the
                        # Allow / Mask / Block controls are shown again.
                        "status": STATE_PENDING,
                        "review_status": STATE_PENDING,
                        "reviewer_decision": None,
                        "reviewer": None,
                        "reviewer_notes": "",
                        "comments": "",
                        "reviewed_at": None,
                    }
                )
                if prior_status in _TERMINAL_STATES:
                    trail = list(item.get("audit_trail") or [])
                    trail.append(
                        {
                            "event": "reopened_for_review",
                            "prior_status": prior_status,
                            "timestamp": time.time(),
                            "reason": "New upload security scan requires a fresh reviewer decision.",
                        }
                    )
                    item["audit_trail"] = trail
                self._save_queue()
                return review_id

        item = {
            "review_id": review_id,
            "document_id": document_id,
            "document_name": document_name,
            "document_type_id": document_type_id,
            "document_type_name": document_type_name,
            "severity": canonical_severity,
            "reason": reason,
            "dlp_reason": dlp_reason or reason,
            "risk_level": risk,
            "dlp_decision": decision,
            "available_actions": actions,
            "status": STATE_PENDING,
            "review_status": STATE_PENDING,
            "reviewer_decision": None,
            "detections": masked_detections,
            "created_at": time.time(),
            "reviewer": None,
            "reviewer_notes": "",
            "comments": "",
            "reviewed_at": None,
            "audit_trail": [],
            "decision_history": [],
        }
        self.queue.append(item)
        self._save_queue()
        return review_id

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        self._load_queue()
        return [
            item
            for item in self.queue
            if self._normalize_status(str(item.get("status") or "")) == STATE_PENDING
        ]

    def reopen_for_review(
        self,
        review_id: str,
        *,
        reason: str = "Reopened for a fresh reviewer decision.",
    ) -> Dict[str, Any] | None:
        """Force a terminal review back to PENDING so Allow/Mask/Block can be chosen again."""
        self._load_queue()
        for item in self.queue:
            if item.get("review_id") != review_id:
                continue
            prior = self._normalize_status(str(item.get("status") or ""))
            item["status"] = STATE_PENDING
            item["review_status"] = STATE_PENDING
            item["reviewer_decision"] = None
            item["reviewer"] = None
            item["reviewer_notes"] = ""
            item["comments"] = ""
            item["reviewed_at"] = None
            item["processing"] = None
            trail = list(item.get("audit_trail") or [])
            trail.append(
                {
                    "event": "reopened_for_review",
                    "prior_status": prior,
                    "timestamp": time.time(),
                    "reason": reason,
                }
            )
            item["audit_trail"] = trail
            self._save_queue()
            return item
        return None

    def apply_reviewer_decision(
        self,
        review_id: str,
        decision: str,
        *,
        reviewer: str,
        comments: str = "",
    ) -> Dict[str, Any] | None:
        """Persist reviewer decision: PENDING → APPROVED | APPROVED_MASKED | REJECTED."""
        self._load_queue()
        normalized_decision = str(decision or "").upper()
        queue_status = reviewer_decision_to_status(normalized_decision)
        for item in self.queue:
            if item["review_id"] != review_id:
                continue
            if self._normalize_status(str(item.get("status") or "")) != STATE_PENDING:
                raise ValueError(f"Review not pending: {review_id}")
            now = time.time()
            item["reviewer_decision"] = normalized_decision
            item["status"] = queue_status
            item["review_status"] = queue_status
            item["reviewer"] = reviewer
            item["reviewer_notes"] = comments
            item["comments"] = comments
            item["reviewed_at"] = now
            entry = {
                "decision": normalized_decision,
                "status": queue_status,
                "reviewer": reviewer,
                "comments": comments,
                "timestamp": now,
                "risk_level": item.get("risk_level"),
                "dlp_decision": item.get("dlp_decision"),
            }
            trail = list(item.get("audit_trail") or [])
            trail.append(entry)
            item["audit_trail"] = trail
            history = list(item.get("decision_history") or [])
            history.append(entry)
            item["decision_history"] = history
            self._save_queue()
            return item
        return None

    def resolve_review(
        self,
        review_id: str,
        status: str,
        notes: str = "",
        reviewer: str = "system",
    ) -> bool:
        """Legacy resolve — maps APPROVED / REJECTED / REPROCESSED."""
        self._load_queue()
        canonical = self._normalize_status(status)
        legacy_to_decision = {
            STATE_APPROVED: "ALLOW",
            STATE_APPROVED_MASKED: "MASK_AND_ALLOW",
            STATE_REJECTED: "BLOCK",
            STATE_REPROCESSED: "MASK_AND_ALLOW",
        }
        decision = legacy_to_decision.get(canonical)
        if decision:
            try:
                self.apply_reviewer_decision(review_id, decision, reviewer=reviewer, comments=notes)
                return True
            except ValueError:
                return False
        return False


def _default_queue_file() -> str:
    from src.shared.networking.document_paths import resolve_security_documents_dir

    return str(resolve_security_documents_dir() / "human_review_queue.json")


def ensure_human_review_queue_entry(
    *,
    document_id: str,
    document_name: str,
    security_scan: dict[str, Any] | None,
    document_type_id: str | None = None,
    document_type_name: str | None = None,
    queue_file: str | None = None,
) -> str | None:
    """Register or refresh a pending review row for a DLP human-review upload."""
    doc_id = str(document_id or "").strip()
    if not doc_id or not isinstance(security_scan, dict):
        return None
    status = str(security_scan.get("status") or "").lower()
    dlp = str(security_scan.get("dlp_decision") or "").upper()
    requires = bool(security_scan.get("requires_human_review"))
    if status != "human_review" and dlp != "HUMAN_REVIEW" and not requires:
        return None

    detections = list(security_scan.get("detections") or [])
    queue = HumanReviewQueue(queue_file=queue_file or _default_queue_file())
    return queue.add_to_queue(
        doc_id,
        str(document_name or doc_id),
        str(security_scan.get("severity") or "medium"),
        str(security_scan.get("reason") or "Pending human review"),
        detections,
        document_type_id=document_type_id,
        document_type_name=document_type_name,
        risk_level=str(security_scan.get("risk_level") or "MEDIUM"),
        dlp_decision=str(security_scan.get("dlp_decision") or "HUMAN_REVIEW"),
        available_actions=list(security_scan.get("available_reviewer_actions") or []),
        dlp_reason=str(security_scan.get("reason") or "Pending human review"),
    )


def get_reviewer_override_for_document(
    document_id: str,
    *,
    queue_file: str | None = None,
) -> dict[str, Any] | None:
    """Return pipeline override when a reviewer has actioned this document."""
    if not str(document_id or "").strip():
        return None
    queue = HumanReviewQueue(queue_file=queue_file or _default_queue_file())
    queue._load_queue()
    review_id = f"rev_{document_id}"
    for item in queue.queue:
        if item.get("review_id") != review_id and item.get("document_id") != document_id:
            continue
        queue_status = HumanReviewQueue._normalize_status(str(item.get("status") or ""))
        decision = str(item.get("reviewer_decision") or "").upper()
        reviewer = str(item.get("reviewer") or "reviewer")
        comments = str(item.get("comments") or item.get("reviewer_notes") or "")
        if queue_status == STATE_REJECTED or decision == "BLOCK":
            return {
                "pipeline_status": "block",
                "queue_status": queue_status,
                "reviewer_decision": "BLOCK",
                "reviewer": reviewer,
                "reason": comments or f"Blocked by reviewer ({reviewer})",
            }
        if queue_status == STATE_APPROVED or decision == "ALLOW":
            return {
                "pipeline_status": "allow",
                "queue_status": queue_status,
                "reviewer_decision": "ALLOW",
                "reviewer": reviewer,
                "reason": comments or f"Approved by reviewer ({reviewer})",
            }
        if queue_status == STATE_APPROVED_MASKED or decision == "MASK_AND_ALLOW":
            return {
                "pipeline_status": "mask_and_allow",
                "queue_status": queue_status,
                "reviewer_decision": "MASK_AND_ALLOW",
                "reviewer": reviewer,
                "reason": comments or f"Approved with masking by reviewer ({reviewer})",
            }
    return None
