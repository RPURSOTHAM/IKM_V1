from __future__ import annotations

import logging
import os
import uuid
import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.security.dlp.sensitive_data_detector import scan_document_for_api, mask_text_content
from src.features.security.dlp.ner_detector import NERDetector
from src.features.security.classification.hybrid_document_classifier import classify_document_for_pipeline_hybrid
from src.features.security.classification.topic_classifier import TopicClassifier
from src.features.security.classification.confidential_similarity_checker import ConfidentialSimilarityChecker
from src.features.security.moderation.llm_moderator import LLMContentModerator
from src.features.security.dlp.dlp_engine import DLPPolicyEngine
from src.features.security.review.human_review_queue import HumanReviewQueue, get_reviewer_override_for_document
from src.features.security.dlp.policy_loader import scan_keyword_policies
from src.features.security.audit.security_event_logger import EVENT_UPLOAD_SCAN, log_security_event
from src.features.security.pipeline.pdf_preprocessor import preprocess_for_ner
from src.features.security.domain.security_scores import build_egress_security, build_ingress_security
from src.features.security.review.review_decisions import (
    enrich_policy_result,
    should_queue_for_review,
)

logger = logging.getLogger(__name__)


class _SafeStream:
    """Swallow Windows/Streamlit OSError on write/flush (Errno 22)."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def write(self, data: Any) -> int:
        try:
            return int(self._wrapped.write(data) or 0)
        except OSError:
            return len(data) if isinstance(data, (str, bytes, bytearray)) else 0

    def flush(self) -> None:
        try:
            self._wrapped.flush()
        except OSError:
            pass

    def reconfigure(self, *args: Any, **kwargs: Any) -> None:
        reconfigure = getattr(self._wrapped, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(*args, **kwargs)
            except (OSError, ValueError):
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


@contextmanager
def _safe_stdio() -> Any:
    import sys

    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout = _SafeStream(original_out)
    sys.stderr = _SafeStream(original_err)
    # Prevent transformers/tqdm progress bars from touching a broken console.
    previous_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        sys.stdout = original_out
        sys.stderr = original_err
        if previous_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = previous_tqdm


def _safe_log(message: str) -> None:
    """Log without crashing Streamlit's Windows stdout wrapper (Errno 22)."""
    try:
        logger.info(message)
    except Exception:
        pass
    try:
        print(message)
    except OSError:
        pass


def _documents_dir() -> Path:
    from src.shared.networking.document_paths import resolve_security_documents_dir

    return resolve_security_documents_dir()


class UploadSecurityPipeline:
    """Orchestrates the multi-stage upload security validation pipeline."""

    def __init__(self) -> None:
        docs_dir = _documents_dir()
        docs_dir.mkdir(parents=True, exist_ok=True)
        self.ner_detector = NERDetector()
        self.topic_classifier = TopicClassifier()
        self.similarity_checker = ConfidentialSimilarityChecker()
        self.llm_moderator = LLMContentModerator()
        self.policy_engine = DLPPolicyEngine()
        self.review_queue = HumanReviewQueue(queue_file=str(docs_dir / "human_review_queue.json"))
        self._audit_log_path = docs_dir / "secure_audit_log.jsonl"

    def scan_file(
        self,
        file_path: Path,
        document_id: str | None = None,
        *,
        document_type_id: str | None = None,
        document_type_name: str | None = None,
        honor_reviewer_decision: bool = False,
    ) -> Dict[str, Any]:
        doc_id = document_id or str(uuid.uuid4())
        doc_name = file_path.name

        _safe_log("SECURITY_PIPELINE_STARTED")

        # --- Early-exit: reviewer BLOCK is permanent across all scans ---
        # Check the queue before running expensive ML stages.
        # If the reviewer chose Block (REJECTED), refuse immediately regardless
        # of honor_reviewer_decision. The document must be re-uploaded with a
        # new document_id to start fresh.
        try:
            early_override = get_reviewer_override_for_document(doc_id)
            if early_override and early_override.get("pipeline_status") == "block":
                reason = str(early_override.get("reason") or "Blocked by reviewer decision.")
                _safe_log("SECURITY_DETECTIONS count=0")
                _safe_log("SECURITY_HIGHEST_SEVERITY=high")
                _safe_log("SECURITY_DECISION=block")
                _safe_log("PROCESSOR_SUBMIT_BLOCKED_BY_SECURITY=true")
                _safe_log(f"SECURITY_REVIEWER_BLOCK_ENFORCED doc_id={doc_id}")
                return self._make_block_decision(reason, doc_name)
        except Exception:
            pass

        # --- Stage 1: File Validation ---
        # Keep in sync with DocumentReceiver allowed_extensions / CIH formats.
        allowed_exts = {
            ".txt",
            ".text",
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".wav",
            ".mp3",
            ".m4a",
            ".aac",
            ".flac",
            ".ogg",
            ".opus",
            ".wma",
            ".mp4",
            ".webm",
            ".mov",
            ".avi",
            ".mkv",
            ".mpeg",
            ".m4v",
            ".wmv",
        }
        try:
            from src.features.configuration.platform_settings import get_settings

            configured = tuple(get_settings().allowed_extensions or ())
            if configured:
                allowed_exts |= {
                    str(x).lower() if str(x).startswith(".") else f".{str(x).lower()}"
                    for x in configured
                }
        except Exception:
            pass

        media_exts = {
            ".wav",
            ".mp3",
            ".m4a",
            ".aac",
            ".flac",
            ".ogg",
            ".opus",
            ".wma",
            ".mp4",
            ".webm",
            ".mov",
            ".avi",
            ".mkv",
            ".mpeg",
            ".m4v",
            ".wmv",
        }

        ext = file_path.suffix.lower()
        if ext not in allowed_exts:
            _safe_log("SECURITY_DETECTIONS count=0")
            _safe_log("SECURITY_HIGHEST_SEVERITY=high")
            _safe_log("SECURITY_DECISION=block")
            _safe_log("PROCESSOR_SUBMIT_BLOCKED_BY_SECURITY=true")
            return self._make_block_decision(
                f"Unsupported file extension ({ext}). Allowed: {', '.join(sorted(allowed_exts))}.",
                doc_name,
            )

        if not file_path.exists():
            _safe_log("SECURITY_DETECTIONS count=0")
            _safe_log("SECURITY_HIGHEST_SEVERITY=high")
            _safe_log("SECURITY_DECISION=block")
            _safe_log("PROCESSOR_SUBMIT_BLOCKED_BY_SECURITY=true")
            return self._make_block_decision("File does not exist.", doc_name)

        # Size limit: prefer platform max_upload_bytes (CIH default 200MB).
        max_mb = 200.0
        try:
            from src.features.configuration.platform_settings import get_settings

            max_mb = max(1.0, float(get_settings().max_upload_bytes) / (1024 * 1024))
        except Exception:
            raw_max = os.getenv("SECURITY_UPLOAD_MAX_MB", "").strip()
            if raw_max:
                try:
                    max_mb = float(raw_max)
                except ValueError:
                    pass
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > max_mb:
            _safe_log("SECURITY_DETECTIONS count=0")
            _safe_log("SECURITY_HIGHEST_SEVERITY=high")
            _safe_log("SECURITY_DECISION=block")
            _safe_log("PROCESSOR_SUBMIT_BLOCKED_BY_SECURITY=true")
            return self._make_block_decision(
                f"File size exceeds the {max_mb:.0f}MB limit.",
                doc_name,
            )

        # Load file blocks. Audio/video have no text until media_transcription —
        # treat as empty extractable text so upload can proceed.
        blocks = []
        doc_metadata: dict[str, Any] = {}
        full_text = ""
        if ext in media_exts:
            _safe_log(f"SECURITY_MEDIA_UPLOAD_DEFERRED_TEXT_SCAN ext={ext}")
            full_text = ""
        else:
            try:
                blocks, doc_metadata = load_document_blocks(file_path, mask_sensitive=False)
                full_text = " ".join(block.text for block in blocks if block.text)
            except Exception as exc:
                _safe_log("SECURITY_DETECTIONS count=0")
                _safe_log("SECURITY_HIGHEST_SEVERITY=high")
                _safe_log("SECURITY_DECISION=block")
                _safe_log("PROCESSOR_SUBMIT_BLOCKED_BY_SECURITY=true")
                return self._make_block_decision(f"File could not be parsed: {str(exc)}", doc_name)

        text_len = len(full_text.strip())
        _safe_log(f"SECURITY_TEXT_EXTRACTED length={text_len}")

        pdf_preprocess = preprocess_for_ner(
            full_text,
            blocks=blocks,
            file_name=doc_name,
            metadata=doc_metadata,
        )
        ner_input_text = pdf_preprocess.get("ner_input_text") or full_text
        doc_metadata = pdf_preprocess.get("document_metadata") or doc_metadata

        if text_len == 0:
            status = "allow"
            severity = "low"
            reason = "Document contains no extractable text; no sensitive content detected."
            empty_policy = enrich_policy_result(
                {
                    "status": status,
                    "severity": severity,
                    "reason": reason,
                    "requires_human_review": False,
                    "actions": ["allow_processing"],
                }
            )
            _safe_log("SECURITY_DETECTIONS count=0")
            _safe_log(f"SECURITY_HIGHEST_SEVERITY={severity}")
            _safe_log(f"SECURITY_DECISION={status}")
            _safe_log("PROCESSOR_SUBMIT_BLOCKED_BY_SECURITY=false")

            self._write_secure_audit_log(
                document_id=doc_id,
                document_name=doc_name,
                severity=severity,
                status=status,
                detected_categories=[],
                detections=[]
            )
            log_security_event(
                EVENT_UPLOAD_SCAN,
                decision=status,
                reason=reason,
                severity=severity,
                policy="upload_empty_content",
                document_id=doc_id,
                document_name=doc_name,
            )

            return {
                **empty_policy,
                "detected_categories": [],
                "detections": [],
                "document_type": "Unknown",
                "topics": [{"topic": "General", "confidence": 1.0}],
                "similarity_matches": [],
                "moderation_result": {},
                "masked_text": "",
                "pipeline_stages": [
                    "file_validation",
                ],
                "pipeline_debug": [
                    {
                        "stage": "file_validation",
                        "findings": [],
                        "severity": "low",
                        "decision": "allow",
                        "reason": reason,
                    }
                ],
            }

        # --- Stage 2: Sensitive Scanning ---
        # Reuse existing validator
        scanner_res = scan_document_for_api(file_path)
        raw_detections = scanner_res.get("detections", [])
        credential_matches = sum(
            1 for det in raw_detections if str(det.get("type") or "").lower() == "token"
        )
        pii_types = {"passport", "aadhaar", "pan", "ssn", "email address", "phone number"}
        pii_matches = sum(
            1 for det in raw_detections if str(det.get("type") or "").lower() in pii_types
        )
        _safe_log(
            "SECURITY_REGEX_STAGE "
            + json.dumps(
                {
                    "detection_count": len(raw_detections),
                    "credential_matches": credential_matches,
                    "pii_matches": pii_matches,
                },
                ensure_ascii=True,
            )
        )

        # Check if the existing scanner flagged block (High/Critical)
        # Note: We let the policy engine decide, but we extract its categories
        detected_categories = set(scanner_res.get("detected_categories", []))

        # --- Stage 2a: Document-level prompt injection scan (before indexing) ---
        prompt_injection_res: dict[str, Any] = {"allowed": True, "action": "allow", "hits": []}
        try:
            from src.features.security.moderation.hybrid_prompt_guard import HybridPromptGuard

            prompt_injection_res = HybridPromptGuard().check_document_text(full_text, pipeline="upload")
            if prompt_injection_res.get("hit_count"):
                for hit in prompt_injection_res.get("hits") or []:
                    raw_detections.append(
                        {
                            "category": "Prompt Injection",
                            "type": "document_prompt_injection",
                            "severity": prompt_injection_res.get("severity") or "medium",
                            "masked_value": "[PROMPT_INJECTION]",
                            "location": "Document text",
                            "reason": prompt_injection_res.get("reason"),
                            "pattern": hit.get("pattern"),
                        }
                    )
                detected_categories.add("Prompt Injection")
        except Exception as exc:
            _safe_log(f"SECURITY_DOCUMENT_PROMPT_INJECTION_ERROR={exc}")

        doc_injection_action = str(prompt_injection_res.get("action") or "allow").lower()
        try:
            from src.features.security.dlp.policy_loader import is_auto_block_enabled

            _auto_block = is_auto_block_enabled()
        except Exception:
            _auto_block = False
        if (
            doc_injection_action == "block"
            and not prompt_injection_res.get("allowed", True)
            and _auto_block
        ):
            block_policy = enrich_policy_result(
                {
                    "status": "block",
                    "severity": "high",
                    "reason": prompt_injection_res.get("reason")
                    or "Document contains embedded prompt-injection instructions.",
                    "requires_human_review": False,
                    "actions": ["block_upload"],
                }
            )
            self._write_secure_audit_log(
                document_id=doc_id,
                document_name=doc_name,
                severity="high",
                status="block",
                detected_categories=list(detected_categories),
                detections=raw_detections,
            )
            log_security_event(
                EVENT_UPLOAD_SCAN,
                decision="block",
                reason=str(block_policy.get("reason")),
                severity="high",
                policy="document_prompt_injection",
                document_id=doc_id,
                document_name=doc_name,
                metadata={
                    "prompt_guard_version": prompt_injection_res.get("prompt_guard_version"),
                    "hit_count": prompt_injection_res.get("hit_count"),
                },
            )
            return {
                **block_policy,
                "detected_categories": sorted(list(detected_categories)),
                "detections": raw_detections,
                "document_type": "Unknown",
                "topics": [{"topic": "General", "confidence": 1.0}],
                "similarity_matches": [],
                "moderation_result": {},
                "masked_text": "",
                "prompt_injection": prompt_injection_res,
                "pipeline_stages": [
                    "file_validation",
                    "document_prompt_injection",
                    "regex_sensitive_scan",
                    "keyword_blacklist_allowlist",
                    "ner",
                    "document_classification",
                    "topic_classification",
                    "embedding_similarity",
                    "llm_moderation",
                    "dlp_policy",
                ],
            }

        # --- Stage 2b: Keyword blacklist / allowlist ---
        keyword_res = scan_keyword_policies(full_text)
        keyword_detections = list(keyword_res.get("matches") or [])
        _safe_log(
            "SECURITY_KEYWORD_STAGE "
            + json.dumps(
                {
                    "action": keyword_res.get("action"),
                    "match_count": len(keyword_detections),
                    "categories": keyword_res.get("categories") or [],
                },
                ensure_ascii=True,
            )
        )

        # SECURITY_PIPELINE_LITE selects lighter *implementations*, never skips controls.
        lite = os.getenv("SECURITY_PIPELINE_LITE", "").strip().lower() in {"1", "true", "yes", "on"}

        # --- Stage 3: NER (mandatory; LITE uses rules strategy) ---
        # NER runs on layout-cleaned text; regex/confidentiality scans use full document.
        try:
            ner_detections = self.ner_detector.detect_entities(
                ner_input_text,
                strategy="rules" if lite else "auto",
            )
        except Exception:
            ner_detections = self.ner_detector.detect_entities(ner_input_text, strategy="rules")
        all_detections = list(raw_detections) + keyword_detections + ner_detections

        # --- Stage 4: Document Classification (hybrid metadata + structure + ML) ---
        try:
            doc_classification = classify_document_for_pipeline_hybrid(
                full_text,
                file_name=doc_name,
                document_id=doc_id,
                lite=lite,
            )
        except Exception:
            doc_classification = {"document_type": "Unknown", "confidence": 0.0, "all_scores": {}}

        # --- Stage 5: Topic Classification (always) ---
        try:
            topic_classification = self.topic_classifier.classify_topics(full_text)
        except Exception:
            topic_classification = {"topics": [{"topic": "General", "confidence": 1.0}]}

        # --- Stage 6: Similarity Filtering (always — hashing embedder if models unavailable) ---
        try:
            similarity_res = self.similarity_checker.check_similarity(
                full_text,
                exclude_ids={doc_id, f"seeded-{doc_id}"},
                exclude_document_names={doc_name},
            )
        except Exception as exc:
            _safe_log(f"SECURITY_SIMILARITY_ERROR={exc}")
            similarity_res = {"matches": [], "max_similarity": 0.0}

        # --- Stage 7: LLM Moderation ---
        try:
            moderation_res = self.llm_moderator.moderate_content(full_text)
        except Exception as exc:
            _safe_log(f"SECURITY_LLM_MODERATION_ERROR={exc}")
            moderation_res = {
                "risk": "low",
                "confidence": 0.0,
                "action": "allow",
                "flags": [],
                "reason": f"moderation_error:{exc}",
                "configured": True,
            }
        _safe_log(
            "SECURITY_MODERATION_STAGE "
            + json.dumps(
                {
                    "risk": moderation_res.get("risk"),
                    "action": moderation_res.get("action"),
                    "confidence": moderation_res.get("confidence"),
                    "reason": moderation_res.get("reason"),
                },
                ensure_ascii=True,
            )
        )

        # --- Stage 8: DLP Policy Engine ---
        policy_decision = self.policy_engine.evaluate_policy(
            detections=all_detections,
            doc_classification=doc_classification,
            topic_classification=topic_classification,
            similarity_results=similarity_res,
            moderation_results=moderation_res
        )
        _safe_log(
            "SECURITY_DLP_STAGE "
            + json.dumps(
                {
                    "status": policy_decision.get("status"),
                    "severity": policy_decision.get("severity"),
                    "risk_level": policy_decision.get("risk_level"),
                    "dlp_decision": policy_decision.get("dlp_decision"),
                    "trace_rules": [
                        step.get("rule")
                        for step in (policy_decision.get("decision_trace") or [])
                        if isinstance(step, dict)
                    ],
                },
                ensure_ascii=True,
            )
        )

        status = policy_decision["status"]
        severity = policy_decision["severity"]
        reason = policy_decision["reason"]
        requires_human_review = policy_decision["requires_human_review"]

        # Document prompt-injection escalation — only when actual injection hits were found.
        prompt_needs_review = bool(
            prompt_injection_res.get("hit_count")
            and doc_injection_action in {"human_review", "quarantine", "flag", "block"}
        )
        if prompt_needs_review:
            if status == "allow" or (doc_injection_action == "block" and not _auto_block):
                status = "human_review"
                requires_human_review = True
                reason = str(
                    prompt_injection_res.get("reason")
                    or "Document contains embedded prompt-injection instructions."
                )
                policy_decision = enrich_policy_result(
                    {
                        **policy_decision,
                        "status": status,
                        "severity": max(
                            str(severity),
                            str(prompt_injection_res.get("severity") or "medium"),
                            key=lambda s: {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(str(s).lower(), 1),
                        ),
                        "reason": reason,
                        "requires_human_review": True,
                        "actions": list(policy_decision.get("actions") or []) + ["route_to_human_review"],
                    }
                )
                severity = policy_decision["severity"]

        # Safety net: never return content hard-block when auto_block is disabled.
        if status == "block" and not _auto_block:
            status = "human_review"
            requires_human_review = True
            reason = str(
                reason
                or "Elevated risk signals detected. Routing to human review — "
                "choose Allow / Mask and Allow / Block."
            )
            policy_decision = enrich_policy_result(
                {
                    **policy_decision,
                    "status": status,
                    "severity": severity if severity in {"high", "critical"} else "high",
                    "reason": reason,
                    "requires_human_review": True,
                    "actions": ["route_to_human_review"],
                }
            )
            severity = policy_decision["severity"]
            reason = policy_decision["reason"]

            severity = policy_decision["severity"]
            reason = policy_decision["reason"]

        # Post-review processing (security_prevalidated): honor terminal Allow/Mask/Block
        # without reopening the queue — otherwise Allow re-triggers human_review and never processes.
        # Fresh uploads: reopen queue first so stale APPROVED cannot auto-allow HIGH re-uploads.
        override_applied: dict[str, Any] | None = None
        if honor_reviewer_decision:
            reviewer_override = get_reviewer_override_for_document(doc_id)
            if reviewer_override:
                status = str(reviewer_override["pipeline_status"])
                reason = str(reviewer_override.get("reason") or reason)
                requires_human_review = status == "human_review"
                policy_decision = enrich_policy_result(
                    {
                        **policy_decision,
                        "status": status,
                        "reason": reason,
                        "requires_human_review": requires_human_review,
                    }
                )
                severity = policy_decision.get("severity") or severity
                override_applied = {
                    "skipped": False,
                    "honored": True,
                    "prior": reviewer_override,
                }
                _safe_log(
                    f"SECURITY_REVIEWER_OVERRIDE_HONORED doc_id={doc_id} status={status}"
                )
            elif should_queue_for_review(status):
                self._enqueue_review(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    policy=policy_decision,
                    reason=reason,
                    detections=all_detections,
                    document_type_id=document_type_id,
                    document_type_name=document_type_name,
                )
        else:
            if should_queue_for_review(status):
                self._enqueue_review(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    policy=policy_decision,
                    reason=reason,
                    detections=all_detections,
                    document_type_id=document_type_id,
                    document_type_name=document_type_name,
                )

            reviewer_override = get_reviewer_override_for_document(doc_id)
            if reviewer_override:
                from src.features.security.review.review_decisions import (
                    reviewer_override_applies_on_scan,
                )

                override_status = reviewer_override["pipeline_status"]
                live_risk = str(policy_decision.get("risk_level") or severity or "").upper()
                live_dlp = str(policy_decision.get("dlp_decision") or "").upper()
                apply_override = reviewer_override_applies_on_scan(
                    live_status=status,
                    live_dlp_decision=live_dlp,
                    live_risk_level=live_risk,
                    requires_human_review=requires_human_review,
                )
                if not apply_override:
                    _safe_log(
                        f"SECURITY_REVIEWER_OVERRIDE_SKIPPED doc_id={doc_id} "
                        f"live_status={status} live_dlp={live_dlp} live_risk={live_risk} "
                        f"prior={override_status}"
                    )
                    override_applied = {
                        "skipped": True,
                        "prior": reviewer_override,
                        "live_status": status,
                        "live_risk": live_risk,
                    }
                else:
                    status = override_status
                    reason = str(reviewer_override.get("reason") or reason)
                    requires_human_review = status == "human_review"
                    policy_decision = enrich_policy_result(
                        {
                            **policy_decision,
                            "status": status,
                            "reason": reason,
                            "requires_human_review": requires_human_review,
                        }
                    )
                    override_applied = {"skipped": False, "prior": reviewer_override}

        _safe_log(f"SECURITY_DETECTIONS count={len(all_detections)}")
        _safe_log(f"SECURITY_HIGHEST_SEVERITY={severity}")
        _safe_log(f"SECURITY_DECISION={status}")
        blocked_flag = "true" if status in ("block", "human_review") else "false"
        _safe_log(f"PROCESSOR_SUBMIT_BLOCKED_BY_SECURITY={blocked_flag}")

        # Seed confidential corpus only on hard block — never on human_review.
        # Seeding on human_review caused re-uploads of the same PDF to self-match
        # (>0.95) and flip PENDING+Allow into a hard DLP block.
        if status == "block" and full_text.strip():
            try:
                self.similarity_checker.add_sensitive_document(
                    full_text[:12000],
                    document_id=f"seeded-{doc_id}",
                    document_name=doc_name,
                    metadata={"seeded_from_decision": status, "severity": severity},
                )
            except Exception as exc:
                _safe_log(f"SECURITY_CORPUS_SEED_ERROR={exc}")

        # Mask output text if allowed with masking
        masked_text = ""
        if status == "mask_and_allow":
            masked_text = mask_text_content(full_text)
            self._persist_masked_text(file_path, masked_text)

        # Add categories from other stages to detected_categories
        for det in ner_detections:
            detected_categories.add(det["type"])
        for det in keyword_detections:
            detected_categories.add(det.get("category") or det.get("type"))

        # Create secure audit log entry
        self._write_secure_audit_log(
            document_id=doc_id,
            document_name=doc_name,
            severity=severity,
            status=status,
            detected_categories=list(detected_categories),
            detections=all_detections
        )
        log_security_event(
            EVENT_UPLOAD_SCAN,
            decision=status,
            reason=reason,
            severity=severity,
            policy="upload_security_pipeline",
            document_id=doc_id,
            document_name=doc_name,
            metadata={
                "detection_count": len(all_detections),
                "document_type": doc_classification.get("document_type"),
                "similarity_max": similarity_res.get("max_similarity"),
                "keyword_action": keyword_res.get("action"),
            },
        )

        stage_results = {
            "file_validation": {"status": "ok", "extension": ext, "size_mb": round(size_mb, 3)},
            "pdf_preprocessor": {
                "confidentiality_markers": pdf_preprocess.get("confidentiality_markers") or [],
                "removed_regions_count": len(pdf_preprocess.get("removed_regions") or []),
                "ner_input_length": len(ner_input_text),
            },
            "document_prompt_injection": {
                "status": prompt_injection_res.get("action") or "allow",
                "hit_count": prompt_injection_res.get("hit_count") or 0,
                "degraded": bool(prompt_injection_res.get("degraded")),
                "reason": prompt_injection_res.get("reason"),
            },
            "regex_sensitive_scan": {
                "status": "ok",
                "detection_count": len(raw_detections),
                "categories": sorted(list(detected_categories)),
            },
            "ner": {"status": "ok", "detection_count": len(ner_detections)},
            "classifier": {
                "document_type": doc_classification.get("document_type"),
                "confidence": doc_classification.get("confidence"),
                "strategy": doc_classification.get("strategy"),
                "evidence": doc_classification.get("evidence") or [],
            },
            "topic_classification": {"topics": topic_classification.get("topics")},
            "embedding_similarity": {
                "max_similarity": similarity_res.get("max_similarity"),
                "matches": (similarity_res.get("matches") or [])[:5],
            },
            "llm_moderation": {
                "risk": moderation_res.get("risk"),
                "action": moderation_res.get("action"),
                "confidence": moderation_res.get("confidence"),
                "reason": moderation_res.get("reason"),
            },
            "dlp": {
                "status": policy_decision.get("status"),
                "severity": policy_decision.get("severity"),
                "reason": policy_decision.get("reason"),
                "decision_trace": policy_decision.get("decision_trace") or [],
                "critical_hits": policy_decision.get("critical_hits") or [],
            },
            "reviewer_override": override_applied,
            "final_action": status,
            "final_reason": reason,
            "final_severity": severity,
        }
        _safe_log(
            f"SECURITY_STAGE_FINAL action={status} severity={severity} "
            f"trace_rules={len(policy_decision.get('decision_trace') or [])}"
        )

        ingress_security = build_ingress_security(
            policy_decision=policy_decision,
            keyword_res=keyword_res,
            similarity_res=similarity_res,
            prompt_injection_res=prompt_injection_res,
            moderation_res=moderation_res,
            detection_count=len(all_detections),
        )
        egress_security = build_egress_security()

        pipeline_debug = self._build_pipeline_debug(
            raw_detections=raw_detections,
            ner_detections=ner_detections,
            keyword_detections=keyword_detections,
            doc_classification=doc_classification,
            topic_classification=topic_classification,
            similarity_res=similarity_res,
            moderation_res=moderation_res,
            policy_decision=policy_decision,
            prompt_injection_res=prompt_injection_res,
            final_status=status,
            final_severity=severity,
        )

        return {
            "status": status,
            "severity": severity,
            "reason": reason,
            "risk_level": policy_decision.get("risk_level"),
            "dlp_decision": policy_decision.get("dlp_decision"),
            "available_reviewer_actions": policy_decision.get("available_reviewer_actions"),
            "ingress_security": ingress_security,
            "egress_security": egress_security,
            "document_metadata": doc_metadata,
            "detected_categories": sorted(list(detected_categories)),
            "detections": [
                {
                    "category": d.get("category"),
                    "type": d.get("type"),
                    "severity": d.get("severity"),
                    "masked_value": d.get("masked_value"),
                    "location": d.get("location", "Text Block")
                }
                for d in all_detections
            ],
            "document_type": doc_classification["document_type"],
            "topics": topic_classification["topics"],
            "similarity_matches": similarity_res.get("matches") or [],
            "moderation_result": moderation_res,
            "keyword_policy": keyword_res,
            "actions": policy_decision["actions"],
            "masked_text": masked_text,
            "requires_human_review": requires_human_review,
            "prompt_injection": prompt_injection_res,
            "decision_trace": policy_decision.get("decision_trace") or [],
            "stage_results": stage_results,
            "pipeline_debug": pipeline_debug,
            "pipeline_stages": [
                "file_validation",
                "pdf_preprocessor",
                "document_prompt_injection",
                "regex_sensitive_scan",
                "keyword_blacklist_allowlist",
                "ner",
                "document_classification",
                "topic_classification",
                "embedding_similarity",
                "llm_moderation",
                "dlp_policy",
            ],
        }

    @staticmethod
    def _build_pipeline_debug(
        *,
        raw_detections: list[dict[str, Any]],
        ner_detections: list[dict[str, Any]],
        keyword_detections: list[dict[str, Any]],
        doc_classification: dict[str, Any],
        topic_classification: dict[str, Any],
        similarity_res: dict[str, Any],
        moderation_res: dict[str, Any],
        policy_decision: dict[str, Any],
        prompt_injection_res: dict[str, Any],
        final_status: str,
        final_severity: str,
    ) -> list[dict[str, Any]]:
        """Structured per-stage summary for debugging risk aggregation."""
        sim_score = float(similarity_res.get("max_similarity") or 0.0)
        mod_risk = str(moderation_res.get("risk") or "low")
        return [
            {
                "stage": "file_validation",
                "findings": [],
                "severity": "low",
                "decision": "allow",
                "reason": "Supported file parsed successfully.",
            },
            {
                "stage": "regex_sensitive_scan",
                "findings": [
                    {"type": d.get("type"), "severity": d.get("severity")}
                    for d in raw_detections
                ],
                "severity": "low" if not raw_detections else max(
                    (str(d.get("severity") or "low") for d in raw_detections),
                    key=lambda s: {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(s, 1),
                    default="low",
                ),
                "decision": "allow" if not raw_detections else "review_signal",
                "reason": "No regex matches." if not raw_detections else f"{len(raw_detections)} regex match(es).",
            },
            {
                "stage": "document_prompt_injection",
                "findings": prompt_injection_res.get("hits") or [],
                "severity": str(prompt_injection_res.get("severity") or "low"),
                "decision": str(prompt_injection_res.get("action") or "allow"),
                "reason": prompt_injection_res.get("reason") or "No prompt injection detected.",
            },
            {
                "stage": "keyword_blacklist_allowlist",
                "findings": keyword_detections,
                "severity": "low" if not keyword_detections else "high",
                "decision": "allow" if not keyword_detections else "review_signal",
                "reason": "No blacklist matches." if not keyword_detections else f"{len(keyword_detections)} keyword match(es).",
            },
            {
                "stage": "ner",
                "findings": [
                    {"type": d.get("type"), "severity": d.get("severity")}
                    for d in ner_detections
                ],
                "severity": "low",
                "decision": "allow",
                "reason": "Informational entities only." if ner_detections else "No entities detected.",
            },
            {
                "stage": "document_classification",
                "findings": [{"document_type": doc_classification.get("document_type"), "confidence": doc_classification.get("confidence")}],
                "severity": "low",
                "decision": "allow",
                "reason": f"Classified as {doc_classification.get('document_type') or 'Unknown'}.",
            },
            {
                "stage": "topic_classification",
                "findings": topic_classification.get("topics") or [],
                "severity": "low",
                "decision": "allow",
                "reason": "Topic labels are informational unless a configured review policy matches.",
            },
            {
                "stage": "embedding_similarity",
                "findings": similarity_res.get("matches") or [],
                "severity": "low" if sim_score < 0.75 else "medium",
                "decision": "allow" if sim_score < 0.75 else "review_signal",
                "reason": f"max_similarity={sim_score:.3f}",
            },
            {
                "stage": "llm_moderation",
                "findings": moderation_res.get("flags") or [],
                "severity": mod_risk,
                "decision": str(moderation_res.get("action") or "allow"),
                "reason": moderation_res.get("reason") or "Moderation completed.",
            },
            {
                "stage": "dlp_policy",
                "findings": policy_decision.get("decision_trace") or [],
                "severity": policy_decision.get("severity"),
                "decision": policy_decision.get("dlp_decision") or policy_decision.get("status"),
                "reason": policy_decision.get("reason"),
            },
            {
                "stage": "final_aggregation",
                "findings": [],
                "severity": final_severity,
                "decision": final_status,
                "reason": policy_decision.get("reason"),
            },
        ]

    @staticmethod
    def _persist_masked_text(file_path: Path, masked_text: str) -> None:
        """Write masked plain-text back to disk so downstream processing uses redacted content."""
        if not masked_text or file_path.suffix.lower() not in {".txt", ".text"}:
            return
        try:
            file_path.write_text(masked_text, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist masked text for %s: %s", file_path, exc)

    def _enqueue_review(
        self,
        *,
        doc_id: str,
        doc_name: str,
        policy: dict[str, Any],
        reason: str,
        detections: list[dict[str, Any]],
        document_type_id: str | None = None,
        document_type_name: str | None = None,
    ) -> None:
        self.review_queue.add_to_queue(
            document_id=doc_id,
            document_name=doc_name,
            document_type_id=document_type_id,
            document_type_name=document_type_name,
            severity=str(policy.get("severity") or "medium"),
            reason=reason,
            detections=detections,
            risk_level=str(policy.get("risk_level") or "MEDIUM"),
            dlp_decision=str(policy.get("dlp_decision") or policy.get("status") or "human_review"),
            available_actions=list(policy.get("available_reviewer_actions") or []),
            dlp_reason=str(policy.get("reason") or reason),
        )

    def _make_block_decision(self, reason: str, doc_name: str) -> Dict[str, Any]:
        try:
            log_security_event(
                EVENT_UPLOAD_SCAN,
                decision="block",
                reason=reason,
                severity="high",
                policy="upload_file_validation",
                document_name=doc_name,
            )
        except Exception:
            pass
        return {
            "status": "block",
            "severity": "high",
            "reason": reason,
            "detected_categories": [],
            "detections": [],
            "document_type": "Unknown",
            "topics": [{"topic": "General", "confidence": 1.0}],
            "similarity_matches": [],
            "moderation_result": {},
            "actions": ["block_upload"],
            "masked_text": "",
            "requires_human_review": False
        }

    def _write_secure_audit_log(
        self,
        document_id: str,
        document_name: str,
        severity: str,
        status: str,
        detected_categories: List[str],
        detections: List[Dict[str, Any]]
    ) -> None:
        # Create a masked representation of detections for safe logging
        safe_detections = []
        for det in detections:
            safe_detections.append({
                "category": det.get("category"),
                "type": det.get("type"),
                "severity": det.get("severity"),
                "masked_value": det.get("masked_value")
            })

        log_payload = {
            "timestamp": time.time(),
            "document_id": document_id,
            "document_name": document_name,
            "severity": severity,
            "status": status,
            "detected_categories": detected_categories,
            "detections": safe_detections
        }
        
        # Write to secure audit logs file
        audit_log_path = getattr(self, "_audit_log_path", None) or (_documents_dir() / "secure_audit_log.jsonl")
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        try:
            with open(audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_payload) + "\n")
        except Exception:
            pass


# Lazy singleton so Streamlit reloads pick up code changes without a stale instance.
_pipeline: UploadSecurityPipeline | None = None


def run_security_pipeline(
    file_path: Path,
    document_id: str | None = None,
    *,
    document_type_id: str | None = None,
    document_type_name: str | None = None,
    honor_reviewer_decision: bool = False,
) -> Dict[str, Any]:
    global _pipeline
    if _pipeline is None:
        _pipeline = UploadSecurityPipeline()
    with _safe_stdio():
        return _pipeline.scan_file(
            file_path,
            document_id,
            document_type_id=document_type_id,
            document_type_name=document_type_name,
            honor_reviewer_decision=honor_reviewer_decision,
        )


def reset_security_pipeline() -> None:
    """Test helper to drop the cached pipeline instance."""
    global _pipeline
    _pipeline = None
