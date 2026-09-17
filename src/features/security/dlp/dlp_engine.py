"""Enterprise DLP policy engine — regex/NER/blacklist/similarity/topic/moderation."""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)

# Hard-block ONLY for these detection types (plus multi gov-ID / patient+PHI rules).
_CRITICAL_TYPES = frozenset(
    {
        "authentication credentials/api keys",
        "api key",
        "private key",
        "password",
        "-----begin private key-----",
    }
)

# Named entities and watermark-style labels are informational — they must not alone
# elevate DLP severity or route clean documents to human review.
_NER_INFORMATIONAL_TYPES = frozenset(
    {
        "person",
        "org",
        "gpe",
        "location/gpe",
        "date",
        "product",
        "facility",
        "loc",
        "fac",
    }
)
_INFORMATIONAL_LABEL_TYPES = frozenset(
    {
        "confidentiality label",
        "sop number",
        "batch number",
        "product name",
        "equipment name",
    }
)
_ACTIONABLE_MEDIUM_TYPES = frozenset(
    {
        "email address",
        "phone number",
        "employee id",
        "supplier contact",
        "internal url",
        "internal project name",
        "patient_context",
        "employee_context",
        "document_prompt_injection",
    }
)


def _is_informational_detection(type_l: str, cat_l: str) -> bool:
    if cat_l == "ner" and type_l in _NER_INFORMATIONAL_TYPES:
        return True
    return type_l in _INFORMATIONAL_LABEL_TYPES


class DLPPolicyEngine:
    """Enforces enterprise DLP policies over multi-stage security detections."""

    def evaluate_policy(
        self,
        detections: list[dict[str, Any]],
        doc_classification: dict[str, Any],
        topic_classification: dict[str, Any],
        similarity_results: dict[str, Any],
        moderation_results: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            from src.features.security.dlp.policy_loader import load_dlp_policies

            policies = load_dlp_policies() or {}
        except Exception:
            policies = {}

        status = "allow"
        reason = "Document contains no sensitive content."
        requires_human_review = False
        actions: list[str] = []
        warnings: list[str] = []
        decision_trace: list[dict[str, Any]] = []

        severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        highest_rank = 1

        has_critical_secrets = False
        has_high_secrets = False
        has_medium_secrets = False
        gov_ids_detected = 0
        confidential_labels_count = 0
        confidential_label_values: list[str] = []
        real_sensitive_content_count = 0
        blacklist_block = False
        blacklist_review = False
        critical_hits: list[dict[str, Any]] = []

        high_gov_types = {
            "aadhaar", "aadhaar number", "pan", "pan number", "passport", "passport number",
            "ssn", "social security number (ssn)", "bank account", "bank account number",
            "credit/debit card", "credit/debit card number", "cvv",
        }

        for det in detections:
            cat = str(det.get("category") or "")
            det_type = str(det.get("type") or "")
            det_severity = str(det.get("severity") or "low").lower()
            type_l = det_type.lower().strip()
            cat_l = cat.lower().strip()
            matched = str(det.get("matched_value") or det.get("masked_value") or "")[:120]

            if type_l.startswith("blacklist:") or cat_l in {
                "financial", "credentials", "pharma", "pii_keywords", "privileged", "blacklist",
            }:
                bl_action = ""
                bl_cats = policies.get("blacklist_categories") if isinstance(policies.get("blacklist_categories"), dict) else {}
                cat_key = cat_l if cat_l in (bl_cats or {}) else (
                    type_l.split(":", 1)[-1] if type_l.startswith("blacklist:") else cat_l
                )
                cfg = bl_cats.get(cat_key) if isinstance(bl_cats, dict) else None
                if isinstance(cfg, dict):
                    bl_action = str(cfg.get("action") or "").lower()
                if bl_action == "block" or (not bl_action and det_severity in {"critical", "high"}):
                    blacklist_block = True
                    decision_trace.append(
                        {
                            "rule": "blacklist_high",
                            "action_hint": "human_review",
                            "confidence": 1.0,
                            "matched": matched or type_l,
                            "type": det_type,
                        }
                    )
                elif bl_action == "human_review" or (not bl_action and det_severity == "medium"):
                    blacklist_review = True
                elif bl_action in {"warning", "warn"}:
                    has_medium_secrets = True
                    highest_rank = max(highest_rank, 2)
                else:
                    blacklist_review = True
                # Fall through so Private Key / API key types still set has_critical_secrets.
            # Critical: explicit credential/key types only — NOT bare severity==critical.
            is_critical_type = (
                type_l in _CRITICAL_TYPES
                or cat_l in _CRITICAL_TYPES
                or "private key" in type_l
                or "private key" in matched.lower()
                or type_l in {"api key", "authentication credentials/api keys"}
            )
            if _is_informational_detection(type_l, cat_l):
                if type_l == "confidentiality label":
                    confidential_labels_count += 1
                    confidential_label_values.append((matched or det_type).strip().lower())
                continue

            if is_critical_type:
                has_critical_secrets = True
                highest_rank = max(highest_rank, 4)
                hit = {
                    "rule": "critical_credential_type",
                    "confidence": 1.0,
                    "matched": matched or det_type,
                    "type": det_type,
                    "severity": det_severity,
                }
                critical_hits.append(hit)
                decision_trace.append({**hit, "action_hint": "block"})
            elif type_l in high_gov_types or det_severity == "high":
                has_high_secrets = True
                highest_rank = max(highest_rank, 3)
                if type_l in high_gov_types:
                    gov_ids_detected += 1
                real_sensitive_content_count += 1
                decision_trace.append(
                    {
                        "rule": "high_sensitive_or_gov_id",
                        "action_hint": "human_review",
                        "confidence": 0.9,
                        "matched": matched or det_type,
                        "type": det_type,
                    }
                )
            elif type_l in _ACTIONABLE_MEDIUM_TYPES:
                has_medium_secrets = True
                highest_rank = max(highest_rank, 2)
                real_sensitive_content_count += 1
                decision_trace.append(
                    {
                        "rule": "medium_sensitive",
                        "action_hint": "human_review",
                        "confidence": 0.7,
                        "matched": matched or det_type,
                        "type": det_type,
                    }
                )
            else:
                highest_rank = max(highest_rank, severity_rank.get(det_severity, 1))

        has_phi = any(
            det.get("type") == "PHI/medical record" or det.get("category") == "Protected Health Information (PHI)"
            for det in detections
        )
        has_patient = any(
            "patient" in str(det.get("matched_value", "")).lower() or det.get("type") == "PATIENT_CONTEXT"
            for det in detections
        )
        if has_phi:
            if has_patient:
                has_critical_secrets = True
                highest_rank = max(highest_rank, 4)
                decision_trace.append(
                    {
                        "rule": "phi_with_patient_context",
                        "action_hint": "block",
                        "confidence": 1.0,
                        "matched": "PHI+patient",
                    }
                )
            else:
                has_high_secrets = True
                highest_rank = max(highest_rank, 3)
                decision_trace.append(
                    {
                        "rule": "phi_without_patient",
                        "action_hint": "human_review",
                        "confidence": 0.9,
                        "matched": "PHI",
                    }
                )

        has_privileged = any(
            det.get("type") == "Confidential legal/privileged content"
            or "attorney-client" in str(det.get("matched_value", "")).lower()
            or "trade secret" in str(det.get("matched_value", "")).lower()
            for det in detections
        )
        if has_privileged:
            has_high_secrets = True
            highest_rank = max(highest_rank, 3)
            decision_trace.append(
                {
                    "rule": "privileged_content",
                    "action_hint": "human_review",
                    "confidence": 0.9,
                    "matched": "privileged",
                }
            )

        doc_type = str(doc_classification.get("document_type") or "Unknown")
        doc_rules = policies.get("document_types") if isinstance(policies.get("document_types"), dict) else {}
        doc_rule = doc_rules.get(doc_type) if isinstance(doc_rules, dict) else None
        escalate = ""
        if isinstance(doc_rule, dict):
            escalate = str(doc_rule.get("escalate") or "").lower()
        elif doc_type in ("NDA", "Contract", "Medical Record", "Regulatory Document"):
            escalate = "high" if doc_type in ("NDA", "Medical Record") else "medium"
        if escalate == "high":
            has_high_secrets = True
            highest_rank = max(highest_rank, 3)
            reason = f"Document type '{doc_type}' requires elevated DLP controls."
            decision_trace.append(
                {
                    "rule": "document_type_escalate_high",
                    "action_hint": "human_review",
                    "confidence": float(doc_classification.get("confidence") or 0.0),
                    "matched": doc_type,
                }
            )
        elif escalate == "medium":
            has_medium_secrets = True
            highest_rank = max(highest_rank, 2)
            warnings.append(f"Document type '{doc_type}' elevates sensitivity.")

        if gov_ids_detected > 1:
            # Multiple ID-like hits are common false positives in long SOPs (e.g. bare \d{12}).
            # Route to human review — do not hard-block unless true critical credential types fired.
            has_high_secrets = True
            highest_rank = max(highest_rank, 3)
            decision_trace.append(
                {
                    "rule": "multiple_government_id_signals",
                    "action_hint": "human_review",
                    "confidence": 0.85,
                    "matched": f"gov_id_count={gov_ids_detected}",
                }
            )

        similarity_max = float(similarity_results.get("max_similarity") or 0.0)
        for match in similarity_results.get("matches") or []:
            similarity_max = max(similarity_max, float(match.get("similarity") or 0.0))
        sim_cfg = policies.get("similarity") if isinstance(policies.get("similarity"), dict) else {}
        block_above = float(sim_cfg.get("block_above", 0.95))
        human_min = float(sim_cfg.get("human_review_min", 0.85))
        warn_min = float(sim_cfg.get("warning_min", 0.75))
        if similarity_max >= warn_min:
            top = (similarity_results.get("matches") or [{}])[0]
            decision_trace.append(
                {
                    "rule": "embedding_similarity",
                    "action_hint": "human_review",
                    "confidence": similarity_max,
                    "matched": str(top.get("document_name") or top.get("reference_id") or ""),
                }
            )

        topic_action = "allow"
        topic_reason = ""
        topic_rules = policies.get("topics") if isinstance(policies.get("topics"), dict) else {}
        for topic in topic_classification.get("topics") or []:
            name = str(topic.get("topic") or "")
            conf = float(topic.get("confidence") or 0.0)
            rule = topic_rules.get(name) if isinstance(topic_rules, dict) else None
            if not isinstance(rule, dict):
                continue
            min_conf = float(rule.get("min_confidence", 0.7))
            if conf < min_conf:
                continue
            action = str(rule.get("action") or "allow").lower()
            if action == "block":
                topic_action = "block"
                topic_reason = f"Topic '{name}' (confidence={conf:.2f}) requires block."
                decision_trace.append(
                    {
                        "rule": "topic_policy_block_to_human_review",
                        "action_hint": "human_review",
                        "confidence": conf,
                        "matched": name,
                    }
                )
                break
            if action == "human_review" and topic_action != "block":
                topic_action = "human_review"
                topic_reason = f"Topic '{name}' (confidence={conf:.2f}) requires human review."
                decision_trace.append(
                    {
                        "rule": "topic_policy_human_review",
                        "action_hint": "human_review",
                        "confidence": conf,
                        "matched": name,
                    }
                )
            elif action in {"warning", "warn"} and topic_action == "allow":
                topic_action = "warning"
                topic_reason = f"Topic '{name}' (confidence={conf:.2f}) raises a warning."

        mod_risk = str(moderation_results.get("risk") or "low").lower()
        mod_action = str(moderation_results.get("action") or "").lower()
        mod_confidence = float(moderation_results.get("confidence") or 0.0)

        distinct_label_values = len(set(confidential_label_values)) or 1
        watermark_like_labels = (
            confidential_labels_count >= 5
            and confidential_labels_count >= 5 * distinct_label_values
            and real_sensitive_content_count == 0
            and not has_critical_secrets
            and not has_high_secrets
        )

        if (
            (mod_risk in {"high", "critical"} and mod_confidence >= 0.8)
            or mod_action in {"block", "mask"}
        ) and not has_high_secrets and not has_critical_secrets:
            requires_human_review = True
        # Medium/high moderation (including local warn rules for self-harm ideation)
        # must escalate to human review — not silently allow.
        if mod_action in {"block", "warn", "mask"} or (
            mod_risk in {"medium", "high", "critical"} and mod_confidence >= 0.7
        ):
            if mod_risk in {"high", "critical"} or mod_action == "block":
                has_high_secrets = True
                highest_rank = max(highest_rank, 3)
            else:
                has_medium_secrets = True
                highest_rank = max(highest_rank, 2)
            decision_trace.append(
                {
                    "rule": "llm_moderation_elevated",
                    "action_hint": "human_review",
                    "confidence": mod_confidence,
                    "matched": mod_action or mod_risk,
                }
            )

        # Final status resolution (config-driven; no silent hard-blocks by default).
        try:
            from src.features.security.dlp.policy_loader import is_auto_block_enabled

            auto_block_enabled = is_auto_block_enabled()
        except Exception:
            enf = policies.get("enforcement") if isinstance(policies.get("enforcement"), dict) else {}
            auto_block_enabled = bool(enf.get("auto_block_enabled", False))

        # Hard block only when explicitly enabled in policy/env.
        if has_critical_secrets and auto_block_enabled:
            status = "block"
            reason = (
                "Document contains critical-risk content (credentials, private keys, "
                "or patient PHI with medical context). Upload blocked."
            )
            actions.append("block_upload")
            highest_rank = max(highest_rank, 4)
            triggered = " | ".join(
                f"{h.get('rule')}={h.get('matched')}" for h in critical_hits[:5]
            ) or "critical_rule"
            _logger.warning(
                "DLP_DECISION rule_triggered=%s confidence=1.0 matched=%s final_action=block",
                triggered,
                (critical_hits[0].get("matched") if critical_hits else "critical"),
            )
        elif (
            watermark_like_labels
            and not has_high_secrets
            and not has_critical_secrets
            and not blacklist_block
            and not blacklist_review
            and similarity_max < warn_min
            and topic_action == "allow"
        ):
            status = "allow"
            reason = "Controlled-document watermark labels only. No security violations found."
            actions.append("allow_processing")
        elif (
            has_critical_secrets
            or has_high_secrets
            or (has_medium_secrets and not watermark_like_labels)
            or (has_medium_secrets and real_sensitive_content_count > 0)
            or requires_human_review
            or blacklist_review
            or blacklist_block
            or topic_action in {"block", "human_review"}
            or similarity_max >= warn_min
            or (mod_risk in {"medium", "high", "critical"} and mod_confidence >= 0.7)
            or mod_action in {"block", "warn", "mask"}
        ):
            status = "human_review"
            if has_critical_secrets:
                reason = (
                    "Critical-risk signals detected. Routing to human review — "
                    "choose Allow / Mask and Allow / Block."
                )
                highest_rank = max(highest_rank, 4)
            elif topic_reason:
                reason = topic_reason
            elif blacklist_block or blacklist_review:
                reason = (
                    "Document matched sensitive keyword/blacklist policy. "
                    "Routing to human review."
                )
            elif similarity_max >= human_min:
                reason = (
                    "Document is similar to known confidential content. "
                    "Routing to human review."
                )
            elif has_high_secrets:
                reason = (
                    "Document contains restricted identifiers or elevated sensitivity signals. "
                    "Routing to human review."
                )
            elif mod_action in {"warn", "block", "mask"} or mod_risk in {"medium", "high", "critical"}:
                reason = (
                    "Content moderation flagged safety-sensitive signals. "
                    "Routing to human review for Allow / Mask / Block."
                )
            else:
                reason = (
                    "Document shows medium-risk sensitive signals. "
                    "Routing to human review for Allow / Mask / Block."
                )
            if warnings and not topic_reason and not has_critical_secrets:
                reason = reason + " " + " ".join(warnings)
            actions.append("route_to_human_review")
            requires_human_review = True
            if has_high_secrets or blacklist_block or similarity_max > block_above:
                highest_rank = max(highest_rank, 3)
            top_trace = decision_trace[-1] if decision_trace else {}
            if has_critical_secrets:
                decision_trace.append(
                    {
                        "rule": "critical_routed_to_human_review",
                        "action_hint": "human_review",
                        "confidence": 1.0,
                        "matched": (critical_hits[0].get("matched") if critical_hits else "critical"),
                    }
                )
            _logger.info(
                "DLP_DECISION rule_triggered=%s confidence=%.3f matched=%s final_action=human_review",
                top_trace.get("rule") or ("critical_routed_to_human_review" if has_critical_secrets else "sensitivity_band"),
                float(top_trace.get("confidence") or (1.0 if has_critical_secrets else 0.0)),
                str(top_trace.get("matched") or (critical_hits[0].get("matched") if critical_hits else ""))[:80],
            )
        else:
            status = "allow"
            reason = "Clear document. No security violations found."
            actions.append("allow_processing")
            _logger.info(
                "DLP_DECISION rule_triggered=none confidence=1.0 matched= none final_action=allow"
            )

        if status == "block" and highest_rank < 4:
            highest_rank = 4
        severity = {1: "low", 2: "medium", 3: "high", 4: "critical"}[highest_rank]

        from src.features.security.review.review_decisions import enrich_policy_result

        result = enrich_policy_result(
            {
                "status": status,
                "severity": severity,
                "reason": reason,
                "requires_human_review": requires_human_review or (status == "human_review"),
                "actions": actions,
                "similarity_max": similarity_max,
                "topic_action": topic_action,
                "decision_trace": decision_trace[-25:],
                "critical_hits": critical_hits[:10],
            }
        )
        return result
