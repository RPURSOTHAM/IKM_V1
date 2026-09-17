"""Complete human review workflow service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.security.audit.security_event_logger import EVENT_HUMAN_REVIEW, log_security_event
from src.features.security.review.human_review_queue import HumanReviewQueue
from src.features.security.review.review_decisions import (
    REVIEWER_ALLOW,
    REVIEWER_BLOCK,
    REVIEWER_MASK_AND_ALLOW,
    STATE_APPROVED,
    STATE_APPROVED_MASKED,
    STATE_PENDING,
    STATE_REJECTED,
    available_actions_for_risk,
    canonical_severity_for_dlp,
    decision_model_summary,
    effective_available_actions,
    effective_risk_level,
    ingestion_blocked,
    masking_applied,
    processing_continues,
    validate_reviewer_decision,
)

_LEGACY_PENDING = {"pending_review", "PENDING", "pending", "PENDING_REVIEW"}


def _normalize_queue_status(raw_status: str) -> str:
    return HumanReviewQueue._normalize_status(str(raw_status or STATE_PENDING))


def _docs_dir() -> Path:
    from src.shared.networking.document_paths import resolve_security_documents_dir

    return resolve_security_documents_dir()


class HumanReviewService:
    """Decision-based review workflow with audit trail and processing hooks."""

    def __init__(self, queue: HumanReviewQueue | None = None) -> None:
        docs = _docs_dir()
        docs.mkdir(parents=True, exist_ok=True)
        self.queue = queue or HumanReviewQueue(queue_file=str(docs / "human_review_queue.json"))
        self._document_type_context_cache: dict[str, tuple[str | None, str | None]] = {}

    def decision_model(self) -> dict[str, Any]:
        return decision_model_summary()

    def list_pending(self) -> list[dict[str, Any]]:
        self.queue._load_queue()
        return [
            self._normalize(item)
            for item in self.queue.queue
            if _normalize_queue_status(str(item.get("status") or "")) == STATE_PENDING
        ]

    def list_all(self) -> list[dict[str, Any]]:
        self.queue._load_queue()
        return [self._normalize(item) for item in self.queue.queue]

    def get(self, review_id: str) -> dict[str, Any] | None:
        self.queue._load_queue()
        for item in self.queue.queue:
            if item.get("review_id") == review_id:
                return self._normalize(item)
        return None

    def available_actions(self, review_id: str) -> list[str]:
        item = self.get(review_id)
        if not item:
            raise KeyError(f"Review not found: {review_id}")
        return effective_available_actions(item)

    def reopen(self, review_id: str, *, reason: str = "Reopened for a fresh reviewer decision.") -> dict[str, Any]:
        updated = self.queue.reopen_for_review(review_id, reason=reason)
        if not updated:
            raise KeyError(f"Review not found: {review_id}")
        return self._normalize(updated)

    def apply_decision(
        self,
        review_id: str,
        *,
        decision: str,
        reviewer: str,
        comments: str = "",
    ) -> dict[str, Any]:
        self.queue._load_queue()
        raw_item = next((item for item in self.queue.queue if item.get("review_id") == review_id), None)
        if not raw_item:
            raise KeyError(f"Review not found: {review_id}")

        queue_status = _normalize_queue_status(str(raw_item.get("status") or ""))
        if queue_status != STATE_PENDING:
            try:
                from src.features.security.dlp.policy_loader import is_reopen_on_reviewer_decide_enabled

                may_reopen = is_reopen_on_reviewer_decide_enabled()
            except Exception:
                may_reopen = True
            if may_reopen:
                reopened = self.queue.reopen_for_review(
                    review_id,
                    reason="Reopened to apply a new reviewer decision.",
                )
                if reopened:
                    self.queue._load_queue()
                    raw_item = next(
                        (item for item in self.queue.queue if item.get("review_id") == review_id),
                        raw_item,
                    )
                    queue_status = _normalize_queue_status(str(raw_item.get("status") or ""))
            if queue_status != STATE_PENDING:
                raise ValueError(
                    f"Review already actioned: {review_id} (status={raw_item.get('status')})"
                )

        item = self._normalize(raw_item)
        risk_level = effective_risk_level(item)
        normalized = str(decision or "").upper()
        if not validate_reviewer_decision(risk_level, normalized):
            allowed = available_actions_for_risk(risk_level)
            raise ValueError(
                f"Decision '{normalized}' is not allowed for risk level {risk_level}. "
                f"Allowed: {', '.join(allowed)}"
            )
        updated = self.queue.apply_reviewer_decision(
            review_id,
            normalized,
            reviewer=reviewer,
            comments=comments,
        )
        if not updated:
            raise KeyError(f"Review not found: {review_id}")

        document_path = self._resolve_document_path(
            str(updated.get("document_id") or ""),
            str(updated.get("document_name") or ""),
        )
        if masking_applied(normalized) and document_path:
            self._mask_document_file(document_path)

        processing: dict[str, Any] = {
            "triggered": False,
            "continues": processing_continues(normalized),
            "masking_applied": masking_applied(normalized),
            "blocked": ingestion_blocked(normalized),
        }
        if processing_continues(normalized):
            try:
                processing.update(
                    self._trigger_processing(
                        updated,
                        reason=f"reviewer_{normalized.lower()}",
                        apply_masking=masking_applied(normalized),
                    )
                )
            except Exception as exc:
                processing.update(
                    {
                        "triggered": False,
                        "continues": True,
                        "reason": f"processing_trigger_failed: {exc}",
                    }
                )
        elif ingestion_blocked(normalized):
            processing["reason"] = "reviewer_blocked"
            if document_path and document_path.exists():
                processing["document_path"] = str(document_path)
                processing["ingestion_blocked"] = True

        updated["processing"] = processing
        self.queue._save_queue()

        queue_status = str(updated.get("status") or "")
        log_security_event(
            EVENT_HUMAN_REVIEW,
            decision=normalized,
            reason=comments or normalized,
            severity=str(updated.get("severity") or "medium"),
            policy="human_review_decision",
            user=reviewer,
            document_id=str(updated.get("document_id") or ""),
            document_name=str(updated.get("document_name") or ""),
            metadata={
                "review_id": review_id,
                "queue_status": queue_status,
                "risk_level": updated.get("risk_level"),
                "dlp_decision": updated.get("dlp_decision"),
                "processing_continues": processing.get("continues", False),
                "masking_applied": processing.get("masking_applied", False),
                "ingestion_blocked": processing.get("blocked", False),
            },
        )
        return self._normalize(updated)

    def approve(
        self,
        review_id: str,
        *,
        reviewer: str,
        comments: str = "",
    ) -> dict[str, Any]:
        return self.apply_decision(review_id, decision=REVIEWER_ALLOW, reviewer=reviewer, comments=comments)

    def reject(
        self,
        review_id: str,
        *,
        reviewer: str,
        comments: str = "",
    ) -> dict[str, Any]:
        return self.apply_decision(review_id, decision=REVIEWER_BLOCK, reviewer=reviewer, comments=comments)

    def reprocess(
        self,
        review_id: str,
        *,
        reviewer: str,
        comments: str = "",
    ) -> dict[str, Any]:
        item = self.apply_decision(
            review_id,
            decision=REVIEWER_MASK_AND_ALLOW,
            reviewer=reviewer,
            comments=comments,
        )
        item["reprocess_requested"] = True
        return item

    def _processor_api_document_path(self, document_path: Path) -> str:
        """Map a host documents path to the path the chunking processor container can read."""
        normalized = str(document_path).replace("\\", "/")
        container_root = (
            os.getenv("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER")
            or os.getenv("SCHEDULER_DOCUMENT_CONTAINER_PATH")
            or "/app/documents"
        ).strip().rstrip("/")
        if normalized.startswith(f"{container_root}/") or normalized == container_root:
            return normalized
        if normalized.startswith("/app/documents/"):
            return normalized
        try:
            from src.shared.networking.document_paths import host_path_to_processor_container_path

            return host_path_to_processor_container_path(
                document_path,
                host_root=_docs_dir(),
                container_root=container_root,
            )
        except Exception:
            return str(document_path)

    def _resolve_document_path(self, document_id: str, document_name: str = "") -> Path | None:
        docs = _docs_dir()
        candidates: list[Path] = []
        if document_name:
            candidates.append(docs / document_name)
        candidates.append(docs / f"{document_id}.txt")
        candidates.append(docs / document_id)
        try:
            for path in docs.iterdir():
                if path.is_file() and document_id and document_id in path.name:
                    candidates.append(path)
        except OSError:
            pass
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _mask_document_file(document_path: Path) -> str:
        from src.features.security.dlp.sensitive_data_detector import mask_text_content

        suffix = document_path.suffix.lower()
        if suffix in {".txt", ".text"}:
            original = document_path.read_text(encoding="utf-8", errors="replace")
            masked = mask_text_content(original)
            document_path.write_text(masked, encoding="utf-8")
            return masked
        blocks, _ = load_document_blocks(document_path, mask_sensitive=False)
        original = "\n".join(block.text for block in blocks if block.text)
        masked = mask_text_content(original)
        # Persist redacted text beside binary sources so processors can fall back to it.
        sidecar = document_path.with_suffix(document_path.suffix + ".masked.txt")
        try:
            sidecar.write_text(masked, encoding="utf-8")
        except Exception:
            pass
        return masked

    def _dms_api_bases(self) -> list[str]:
        """Candidate DMS/Consumer API bases (host Streamlit + in-container)."""
        candidates = [
            os.getenv("RAG_API_BASE_URL"),
            os.getenv("CONSUMER_API_BASE_URL"),
            os.getenv("DMS_SERVICE_URL"),
            "http://localhost:8088",
            "http://rag-dms-service:8088",
            "http://localhost:8888",
        ]
        seen: set[str] = set()
        bases: list[str] = []
        for raw in candidates:
            base = str(raw or "").strip().rstrip("/")
            if not base or base in seen:
                continue
            seen.add(base)
            bases.append(base)
        return bases

    def _trigger_processing_via_dms_http(
        self,
        document_id: str,
        *,
        apply_masking: bool,
        reason: str,
    ) -> dict[str, Any] | None:
        """Queue via DMS HTTP so host-side review does not depend on in-process MySQL/RabbitMQ."""
        import json
        import urllib.error
        import urllib.request

        api_prefix = (os.getenv("RAG_API_PREFIX") or os.getenv("API_PREFIX") or "/api/v1").rstrip("/")
        api_key = (os.getenv("CONSUMER_API_KEY") or os.getenv("RAG_API_KEY") or "").strip()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        last_error = ""
        for base in self._dms_api_bases():
            for action in ("submit", "reprocess"):
                url = f"{base}{api_prefix}/documents/{document_id}/{action}"
                try:
                    request = urllib.request.Request(
                        url,
                        data=b"{}",
                        headers=headers,
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=60) as response:
                        status_code = response.getcode()
                        raw = response.read().decode("utf-8", errors="replace")
                    body: Any = {}
                    try:
                        body = json.loads(raw) if raw.strip() else {}
                    except Exception:
                        body = {"raw": raw[:500]}
                    if status_code in {200, 202}:
                        return {
                            "triggered": True,
                            "mode": f"dms_http_{action}",
                            "status": (body or {}).get("status") if isinstance(body, dict) else None,
                            "status_code": status_code,
                            "body": body,
                            "continues": True,
                            "masking_applied": apply_masking,
                            "reason": reason,
                        }
                    last_error = f"{url} -> HTTP {status_code}"
                except urllib.error.HTTPError as exc:
                    raw = exc.read().decode("utf-8", errors="replace")
                    last_error = f"{url} -> HTTP {exc.code}: {raw[:200]}"
                    # 404/409 on submit may still succeed via reprocess (or next base).
                    continue
                except Exception as exc:
                    last_error = f"{url} -> {exc}"
                    break
        if last_error:
            return {"triggered": False, "reason": f"dms_http_failed: {last_error}", "continues": False}
        return None

    def _trigger_processing(
        self,
        item: dict[str, Any],
        *,
        reason: str,
        apply_masking: bool = False,
    ) -> dict[str, Any]:
        """Resume document processing after reviewer allow/mask decision."""
        document_id = str(item.get("document_id") or "").strip()
        if not document_id:
            return {"triggered": False, "reason": "missing_document_id", "continues": False}

        # 1) Prefer DMS HTTP submit (works from Streamlit host; survives processor restarts).
        http_result = self._trigger_processing_via_dms_http(
            document_id,
            apply_masking=apply_masking,
            reason=reason,
        )
        if http_result and http_result.get("triggered"):
            return http_result

        # 2) In-process DocumentReceiverService (when running inside DMS).
        try:
            import asyncio

            from src.features.documents.application.document_service import DocumentReceiverService

            svc = DocumentReceiverService()

            async def _run() -> dict[str, Any]:
                try:
                    result = await svc.reprocess_document(document_id)
                    return {
                        "triggered": True,
                        "mode": "reprocess",
                        "status": getattr(result, "status", None),
                        "continues": True,
                        "masking_applied": apply_masking,
                    }
                except Exception:
                    result = await svc.submit_document(document_id)
                    return {
                        "triggered": True,
                        "mode": "submit",
                        "status": getattr(result, "status", None),
                        "continues": True,
                        "masking_applied": apply_masking,
                    }

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                loop.create_task(_run())
                return {
                    "triggered": True,
                    "mode": "async",
                    "reason": reason,
                    "continues": True,
                    "masking_applied": apply_masking,
                }
            return asyncio.run(_run())
        except Exception:
            pass

        # 3) Last resort: direct processor POST (fragile on Windows/Docker).
        return self._trigger_processing_via_processor_api(
            item,
            apply_masking=apply_masking,
            reason=reason,
        )

    def _trigger_processing_via_processor_api(
        self,
        item: dict[str, Any],
        *,
        reason: str,
        apply_masking: bool = False,
    ) -> dict[str, Any]:
        """Fallback for test harness: POST directly to chunking processor."""
        import json
        import urllib.error
        import urllib.request

        document_id = str(item.get("document_id") or "").strip()
        if not document_id:
            return {"triggered": False, "reason": "missing_document_id", "continues": False}
        document_path = self._resolve_document_path(document_id, str(item.get("document_name") or ""))
        if not document_path:
            return {"triggered": False, "reason": "document_file_not_found", "continues": False}
        if apply_masking:
            self._mask_document_file(document_path)
        processor_url = (
            os.getenv("CHUNKING_PROCESSOR_URL")
            or os.getenv("PROCESSOR_API_URL")
            or "http://localhost:3100"
        ).rstrip("/")
        api_document_path = self._processor_api_document_path(document_path)
        payload = {
            "processor_type": "chunking_vectorizing",
            "document_id": document_id,
            "document_name": document_path.name,
            "original_file_name": document_path.name,
            "document_path": api_document_path,
            "collection_name": "DocumentChunk",
            "tenant_id": "default",
            "repository_id": "default",
            "security_prevalidated": True,
        }
        try:
            request = urllib.request.Request(
                f"{processor_url}/process",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                status_code = response.getcode()
                raw = response.read().decode("utf-8", errors="replace")
            body: Any = {}
            try:
                body = json.loads(raw)
            except Exception:
                body = {"raw": raw[:500]}
            return {
                "triggered": status_code in {200, 202},
                "mode": "processor_api",
                "status_code": status_code,
                "body": body,
                "continues": status_code in {200, 202},
                "masking_applied": apply_masking,
                "reason": reason,
            }
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            body: Any = {}
            try:
                body = json.loads(raw)
            except Exception:
                body = {"raw": raw[:500]}
            return {
                "triggered": False,
                "mode": "processor_api",
                "status_code": exc.code,
                "body": body,
                "continues": False,
                "masking_applied": apply_masking,
                "reason": reason,
            }
        except Exception as exc:
            return {"triggered": False, "reason": str(exc), "continues": False}

    def _normalize(self, item: dict[str, Any]) -> dict[str, Any]:
        item = self._with_document_type_context(item)
        status = _normalize_queue_status(str(item.get("status") or STATE_PENDING))
        normalized_dlp = str(item.get("dlp_decision") or "HUMAN_REVIEW").upper()
        severity = canonical_severity_for_dlp(str(item.get("severity") or "medium"), normalized_dlp)
        # Compute risk from canonicalized queue values. Legacy rows may miss
        # dlp_decision/risk_level, but review-queue entries are human-review by design.
        normalized_for_risk = {
            **item,
            "status": status,
            "severity": severity,
            "dlp_decision": normalized_dlp,
        }
        risk_level = effective_risk_level(normalized_for_risk)
        available = effective_available_actions(
            {
                **normalized_for_risk,
                "risk_level": risk_level,
            }
        )
        return {
            **item,
            "status": status,
            "review_status": status,
            "severity": severity,
            "risk_level": risk_level,
            "dlp_decision": normalized_dlp,
            "available_actions": available,
            "decision_history": list(item.get("decision_history") or item.get("audit_trail") or []),
        }

    def _with_document_type_context(self, item: dict[str, Any]) -> dict[str, Any]:
        document_type_id = str(item.get("document_type_id") or "").strip() or None
        document_type_name = str(item.get("document_type_name") or "").strip() or None
        if document_type_id and document_type_name:
            return item

        document_id = str(item.get("document_id") or "").strip()
        if not document_id:
            return item

        cached = self._document_type_context_cache.get(document_id)
        if cached is not None:
            cached_type_id, cached_type_name = cached
            enriched = dict(item)
            if not document_type_id and cached_type_id:
                enriched["document_type_id"] = cached_type_id
            if not document_type_name and cached_type_name:
                enriched["document_type_name"] = cached_type_name
            return enriched

        resolved_type_id: str | None = document_type_id
        resolved_type_name: str | None = document_type_name
        try:
            from src.features.document_types.infrastructure.document_type_repository import get_document_type_store

            store = get_document_type_store()
            if store is not None:
                instance = store.get_document_instance(document_id)
                if instance and not resolved_type_id:
                    resolved_type_id = str(instance.get("document_type_id") or "").strip() or None
                if resolved_type_id and not resolved_type_name:
                    record = store.get_type_by_id(resolved_type_id)
                    if record is not None:
                        resolved_type_name = str(record.name or "").strip() or None
        except Exception:
            pass

        self._document_type_context_cache[document_id] = (resolved_type_id, resolved_type_name)
        enriched = dict(item)
        if not document_type_id and resolved_type_id:
            enriched["document_type_id"] = resolved_type_id
        if not document_type_name and resolved_type_name:
            enriched["document_type_name"] = resolved_type_name
        return enriched
