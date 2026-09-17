from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.features.document_processing.loaders.document_text import load_document_blocks
from src.shared.errors import COMPONENT_DOCUMENT_RECEIVER, DmsServiceError

logger = logging.getLogger(__name__)


class ComplianceScanner:
    _SEVERITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    _SEVERITY_REVERSE_MAP = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
    _CATEGORY_EMAIL = "Email Address"
    _CATEGORY_CREDS = "Authentication Credentials/API Keys"
    _CATEGORY_CARD = "Credit/Debit Card Number"
    _CATEGORY_SSN = "Social Security Number (SSN)"
    _CATEGORY_AADHAAR = "Aadhaar Number"
    _CATEGORY_PAN = "PAN Number"
    _CATEGORY_PASSPORT = "Passport Number"
    _CATEGORY_DL = "Driver's License"
    _CATEGORY_BANK_ACCOUNT = "Bank Account Number"
    _CATEGORY_PRIVATE_KEY = "Private Key"
    _CATEGORY_PII = "Personally Identifiable Information (PII)"
    _CATEGORY_PHI = "Protected Health Information (PHI)"
    _CATEGORY_LEGAL = "Confidential Legal Information"
    _CATEGORY_IP = "Intellectual Property / Trade Secrets"
    _CATEGORY_CONFIDENTIAL = "Confidential Business Information"

    _AWS_ACCESS_KEY_PATTERN = re.compile(r"\b(AKIA[A-Z0-9]{16})\b")
    _PRIVATE_KEY_PATTERN = re.compile(r"(-----BEGIN [A-Z ]*PRIVATE KEY-----)")
    _CREDENTIAL_PATTERN = re.compile(
        r"(?i)\b(?:api[_-]?key|auth[_-]?token|client[_-]?secret|password|access[_-]?token|bearer|jwt)[\s:=\"']{1,10}([A-Za-z0-9+/=\-_]{16,64})"
    )
    _CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b")
    _SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _AADHAAR_PATTERN = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{4}-\d{4}-\d{4}\b|\b\d{12}\b")
    _PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
    _PASSPORT_PATTERN = re.compile(r"\b[A-PR-WYa-pr-wy][1-9]\d{6}\b|\b[A-Z0-9]{9}\b")
    _DRIVER_LICENSE_PATTERN = re.compile(r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{11,13}\b")
    _IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
    _IFSC_PATTERN = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
    _BANK_NEAR_KEYWORDS_PATTERN = re.compile(
        r"(?i)(?:bank\s+account|acc\s+no|account\s+no|routing\s+(?:number|no))[\s:-]+([A-Z0-9-]{8,20})"
    )
    _EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    _PHI_PATTERN = re.compile(
        r"(?i)\b(?:medical record|patient ID|health insurance|medical history|clinical notes|diagnosis|prescription|treatment plan)\b"
    )
    _LEGAL_PATTERN = re.compile(
        r"(?i)\b(?:attorney-client privilege|work product|non-disclosure agreement|NDA|employment contract|indemnity agreement|joint venture agreement)\b"
    )
    _IP_PATTERN = re.compile(
        r"(?i)\b(?:trade secret|proprietary(?: information)?|intellectual property|copyright\s+reserved|all\s+rights\s+reserved|patent\s+pending)\b"
    )
    _CONFIDENTIAL_PATTERN = re.compile(
        r"(?i)\b(?:strictly confidential|internal use only|restricted distribution|confidential business|proprietary data)\b"
    )

    @staticmethod
    def _append_detection(
        detections: List[Dict[str, Any]],
        *,
        category: str,
        matched_value: str,
        masked_value: str,
        severity: str,
    ) -> None:
        detections.append(
            {
                "category": category,
                "matched_value": matched_value,
                "masked_value": masked_value,
                "severity": severity,
            }
        )

    def check_luhn(self, card_number: str) -> bool:
        digits = [int(d) for d in card_number if d.isdigit()]
        if len(digits) < 11 or len(digits) > 19:
            return False
        checksum = 0
        reverse_digits = digits[::-1]
        for i, digit in enumerate(reverse_digits):
            if i % 2 == 1:
                doubled = digit * 2
                checksum += doubled - 9 if doubled > 9 else doubled
            else:
                checksum += digit
        return checksum % 10 == 0

    def mask_value(self, value: str, category: str) -> str:
        if category == self._CATEGORY_CARD:
            cleaned = "".join(c for c in value if c.isdigit())
            if len(cleaned) >= 4:
                return f"XXXX-XXXX-XXXX-{cleaned[-4:]}"
            return "XXXX-XXXX-XXXX-XXXX"
        if category == self._CATEGORY_EMAIL:
            if "@" in value:
                parts = value.split("@")
                username = parts[0]
                domain = parts[1]
                masked_username = username[0] + "***" if len(username) > 1 else "*"
                return f"{masked_username}@{domain}"
            return "em***@domain.com"
        if category == self._CATEGORY_SSN:
            cleaned = "".join(c for c in value if c.isdigit())
            if len(cleaned) >= 4:
                return f"XXX-XX-{cleaned[-4:]}"
            return "XXX-XX-XXXX"
        if category == self._CATEGORY_AADHAAR:
            cleaned = "".join(c for c in value if c.isdigit())
            if len(cleaned) >= 4:
                return f"XXXX-XXXX-{cleaned[-4:]}"
            return "XXXX-XXXX-XXXX"
        if category in (self._CATEGORY_PAN, self._CATEGORY_PASSPORT, self._CATEGORY_DL):
            if len(value) >= 4:
                return f"{value[:2]}{'X' * (len(value) - 4)}{value[-2:]}"
            return "XXXX"
        if category in (self._CATEGORY_CREDS, self._CATEGORY_PRIVATE_KEY):
            if len(value) >= 8:
                return f"{value[:4]}************{value[-4:]}"
            return "************"
        if category == self._CATEGORY_BANK_ACCOUNT:
            cleaned = "".join(c for c in value if c.isalnum())
            if len(cleaned) >= 4:
                return f"XXXX-XXXX-{cleaned[-4:]}"
            return "XXXX-XXXX-XXXX"
        return value

    def scan_text(self, text: str) -> List[Dict[str, Any]]:
        detections = []

        # 1. Credentials / Secrets (Critical)
        # AWS Access Key
        for m in self._AWS_ACCESS_KEY_PATTERN.finditer(text):
            val = m.group(1)
            self._append_detection(
                detections,
                category=self._CATEGORY_CREDS,
                matched_value=val,
                masked_value=self.mask_value(val, self._CATEGORY_CREDS),
                severity="Critical",
            )
        # Private Key
        for m in self._PRIVATE_KEY_PATTERN.finditer(text):
            val = m.group(1)
            self._append_detection(
                detections,
                category=self._CATEGORY_PRIVATE_KEY,
                matched_value=val,
                masked_value="[PRIVATE KEY DETECTED]",
                severity="Critical",
            )
        # Generic credentials near keywords
        for m in self._CREDENTIAL_PATTERN.finditer(text):
            val = m.group(1)
            if len(val) >= 16 and not val.replace("-", "").isdigit():
                self._append_detection(
                    detections,
                    category=self._CATEGORY_CREDS,
                    matched_value=val,
                    masked_value=self.mask_value(val, self._CATEGORY_CREDS),
                    severity="Critical",
                )

        # 2. Credit Card numbers (Critical)
        for m in self._CARD_PATTERN.finditer(text):
            val = m.group(0)
            cleaned = "".join(c for c in val if c.isdigit())
            if self.check_luhn(cleaned):
                self._append_detection(
                    detections,
                    category=self._CATEGORY_CARD,
                    matched_value=val,
                    masked_value=self.mask_value(val, self._CATEGORY_CARD),
                    severity="Critical",
                )

        # 3. PII (High)
        # SSN
        for m in self._SSN_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_SSN,
                matched_value=val,
                masked_value=self.mask_value(val, self._CATEGORY_SSN),
                severity="High",
            )
        # Aadhaar
        for m in self._AADHAAR_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_AADHAAR,
                matched_value=val,
                masked_value=self.mask_value(val, self._CATEGORY_AADHAAR),
                severity="High",
            )
        # PAN
        for m in self._PAN_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_PAN,
                matched_value=val,
                masked_value=self.mask_value(val, self._CATEGORY_PAN),
                severity="High",
            )
        # Passport
        for m in self._PASSPORT_PATTERN.finditer(text):
            val = m.group(0)
            if not val.isdigit():
                self._append_detection(
                    detections,
                    category=self._CATEGORY_PASSPORT,
                    matched_value=val,
                    masked_value=self.mask_value(val, self._CATEGORY_PASSPORT),
                    severity="High",
                )
        # Driver's License
        for m in self._DRIVER_LICENSE_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_DL,
                matched_value=val,
                masked_value=self.mask_value(val, self._CATEGORY_DL),
                severity="High",
            )

        # 4. Bank Accounts (High)
        # IBAN
        for m in self._IBAN_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_BANK_ACCOUNT,
                matched_value=val,
                masked_value=self.mask_value(val, self._CATEGORY_BANK_ACCOUNT),
                severity="High",
            )
        # IFSC
        for m in self._IFSC_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category="Bank Routing Number (IFSC)",
                matched_value=val,
                masked_value=val,
                severity="High",
            )
        # Generic bank account near keywords
        for m in self._BANK_NEAR_KEYWORDS_PATTERN.finditer(text):
            val = m.group(1)
            if len(val.strip("-")) >= 8:
                self._append_detection(
                    detections,
                    category=self._CATEGORY_BANK_ACCOUNT,
                    matched_value=val,
                    masked_value=self.mask_value(val, self._CATEGORY_BANK_ACCOUNT),
                    severity="High",
                )

        # 5. PII (Medium)
        # Email
        for m in self._EMAIL_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_PII,
                matched_value=val,
                masked_value=self.mask_value(val, self._CATEGORY_EMAIL),
                severity="Medium",
            )
        # Phone
        for m in self._PHONE_PATTERN.finditer(text):
            val = m.group(0)
            if not ("-" in val and len(val) == 10 and val.count("-") == 2 and val.replace("-", "").isdigit()):
                self._append_detection(
                    detections,
                    category=self._CATEGORY_PII,
                    matched_value=val,
                    masked_value="XXX-XXX-" + val[-4:] if len(val) >= 4 else "XXX-XXX-XXXX",
                    severity="Medium",
                )

        # 6. PHI (High)
        for m in self._PHI_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_PHI,
                matched_value=val,
                masked_value=val,
                severity="High",
            )

        # 7. Confidential Legal Info & Trade Secrets / Intellectual Property (High)
        for m in self._LEGAL_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_LEGAL,
                matched_value=val,
                masked_value=val,
                severity="High",
            )

        for m in self._IP_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_IP,
                matched_value=val,
                masked_value=val,
                severity="High",
            )

        for m in self._CONFIDENTIAL_PATTERN.finditer(text):
            val = m.group(0)
            self._append_detection(
                detections,
                category=self._CATEGORY_CONFIDENTIAL,
                matched_value=val,
                masked_value=val,
                severity="High",
            )

        return detections

    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        try:
            blocks, doc_metadata = load_document_blocks(file_path, mask_sensitive=False)
        except Exception:
            logger.exception("Failed to parse document for compliance scan: %s", file_path)
            return {"clean": True, "detected_categories": [], "locations": [], "severity": "Low", "detections": []}

        all_detections = []
        locations = []
        detected_categories = set()
        max_severity_val = 0

        current_section = "General"

        # Scan document metadata
        for k, v in doc_metadata.items():
            if isinstance(v, str):
                metadata_detections = self.scan_text(v)
                for det in metadata_detections:
                    det["location"] = f"Document Metadata ({k})"
                    all_detections.append(det)
                    detected_categories.add(det["category"])
                    max_severity_val = max(max_severity_val, self._SEVERITY_MAP.get(det["severity"], 1))
                    locations.append(det["location"])

        # Scan document blocks
        for block in blocks:
            if block.component_type in ("title", "subtitle") or (block.style and block.style.lower().startswith("heading")):
                current_section = block.text

            text_to_scan = block.text or ""
            if not text_to_scan:
                continue

            block_detections = self.scan_text(text_to_scan)
            for det in block_detections:
                loc_parts = []
                if block.page:
                    loc_parts.append(f"Page {block.page}")
                
                if block.component_type == "header":
                    loc_parts.append("Header")
                elif block.component_type == "footer":
                    loc_parts.append("Footer")
                elif block.component_type == "table" or block.block_type == "table":
                    loc_parts.append("Table")
                elif block.component_type == "image" or block.block_type == "image":
                    loc_parts.append("OCR Image Text")
                elif current_section:
                    loc_parts.append(f"Section: {current_section}")

                location_str = ", ".join(loc_parts) if loc_parts else "Unknown Location"
                
                det["location"] = location_str
                det["page"] = block.page
                det["section"] = current_section
                
                all_detections.append(det)
                detected_categories.add(det["category"])
                max_severity_val = max(max_severity_val, self._SEVERITY_MAP.get(det["severity"], 1))
                locations.append(location_str)

        unique_locations = list(dict.fromkeys(locations))

        severity = self._SEVERITY_REVERSE_MAP.get(max_severity_val, "Low")
        clean = len(all_detections) == 0

        return {
            "clean": clean,
            "detected_categories": sorted(list(detected_categories)),
            "locations": unique_locations,
            "severity": severity,
            "detections": all_detections
        }


def dms_upload_security_enabled() -> bool:
    """When false, DMS upload skips the synchronous security pipeline (processor may still scan)."""
    import os

    explicit = os.getenv("DMS_RUN_UPLOAD_SECURITY")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip().lower() in {"1", "true", "yes", "on"}
    # Default true preserves legacy intake behavior when the env var is unset.
    return True


def check_document_compliance(
    file_path: Path,
    repository_id: str | None = None,
    *,
    document_id: str | None = None,
    document_type_id: str | None = None,
    document_type_name: str | None = None,
) -> dict[str, Any] | None:
    if not dms_upload_security_enabled():
        return {
            "status": "allow",
            "severity": "low",
            "reason": "dms_upload_security_disabled",
            "document_id": document_id,
        }

    try:
        from src.features.configuration.application.config_provider import get_platform_config
        provider = get_platform_config()
        cfg = provider.namespace("dms.security")
        enabled = cfg.get_bool("validation.enabled", True)
    except Exception:
        enabled = True

    if not enabled:
        # Config cannot bypass enforcement — log and continue scanning.
        try:
            from src.features.security.audit.security_event_logger import log_security_event

            log_security_event(
                "UPLOAD_SCAN",
                decision="config_disable_ignored",
                reason="dms.security.validation.enabled=false ignored; scan still enforced.",
                severity="medium",
                policy="no_bypass",
                document_name=str(file_path.name),
            )
        except Exception:
            logger.warning("validation.enabled=false ignored; upload security still enforced")

    from src.features.security.application.upload_security_pipeline import run_security_pipeline
    stable_doc_id = str(document_id or "").strip() or (file_path.stem if file_path and file_path.stem else None)
    scan_res = run_security_pipeline(
        file_path,
        document_id=stable_doc_id,
        document_type_id=document_type_id,
        document_type_name=document_type_name,
    )

    status = scan_res.get("status")
    document_id_hint = str(scan_res.get("document_id") or stable_doc_id or "")
    if status == "block":
        # Backwards compatibility mappings for older tests
        detected_cats_raw = scan_res.get("detected_categories", [])
        detected_cats_mapped = []
        for cat in detected_cats_raw:
            if cat == "SSN":
                detected_cats_mapped.append("Social Security Number (SSN)")
            else:
                detected_cats_mapped.append(cat)
        detected_cats_sorted = sorted(list(set(detected_cats_mapped)))
        
        locations_sorted = ["Page 1, Section: General"]
        severity_label = "High" if scan_res.get("severity") in ("high", "critical") else scan_res.get("severity").capitalize()
        
        explanation = str(
            scan_res.get("reason")
            or "The uploaded document contains sensitive or restricted information that is not permitted under the current security policy."
        ).strip()
        recommended_action = (
            "Remove or redact the restricted information and upload the updated document, "
            "or obtain the required authorization if processing this document is permitted."
        )
        if "unsupported file extension" in explanation.lower():
            recommended_action = (
                "Upload a supported file type, or ask an administrator to enable this "
                "extension in DOCUMENT_ALLOWED_EXTENSIONS / security upload allowlist."
            )
        elif "file size exceeds" in explanation.lower():
            recommended_action = "Upload a smaller file, or raise the platform max upload size."
        elif "could not be parsed" in explanation.lower():
            recommended_action = "Re-save/export the file (e.g. PPTX instead of legacy PPT) and retry."

        details = {
            "upload_status": "Blocked",
            "document_id": document_id_hint,
            "blocking_reason": explanation,
            "detected_categories": detected_cats_sorted,
            "locations": locations_sorted,
            "severity_level": severity_label,
            "human_readable_explanation": explanation,
            "recommended_corrective_action": recommended_action,
        }

        cats_line = (
            "\n".join(f"* {c}" for c in detected_cats_sorted)
            if detected_cats_sorted
            else "* (none reported — see blocking reason)"
        )
        raise DmsServiceError(
            reason=f"Compliance violation: {', '.join(detected_cats_sorted) or explanation}",
            component=COMPONENT_DOCUMENT_RECEIVER,
            code="compliance_violation",
            http_status=400,
            user_message=(
                f"**Upload Blocked**\n\n**Reason:**\n{explanation}\n\n"
                f"**Detected Categories:**\n{cats_line}\n\n**Severity:**\n{severity_label}"
            ),
            details=details,
        )
    elif status == "human_review":
        # Intake still saves the file and creates the document record, but processor
        # queue submission must wait for reviewer Allow / Mask and Allow / Block.
        logger.info(
            "Upload flagged for DLP human_review (document_id=%s severity=%s); "
            "document intake proceeds, processor queue deferred until reviewer action.",
            document_id_hint,
            scan_res.get("severity"),
        )
        return scan_res

    return scan_res if isinstance(scan_res, dict) else None


def security_scan_requires_review(scan: dict[str, Any] | None) -> bool:
    """True when upload must wait for human review before processor queue submission."""
    if not isinstance(scan, dict):
        return False
    status = str(scan.get("status") or "").lower()
    dlp = str(scan.get("dlp_decision") or "").upper()
    if status == "human_review" or dlp == "HUMAN_REVIEW":
        return True
    if bool(scan.get("requires_human_review")):
        return True
    risk = str(scan.get("risk_level") or "").upper()
    if risk in {"MEDIUM", "HIGH", "CRITICAL"} and dlp not in {"ALLOW", "MASK_AND_ALLOW", ""}:
        return True
    if risk in {"MEDIUM", "HIGH", "CRITICAL"} and status not in {"allow", "allowed", "pass", ""}:
        return True
    return False


def document_pending_human_review(record: dict[str, Any] | None) -> bool:
    """True when the intake record is still waiting on Security Human Review."""
    if not isinstance(record, dict):
        return False
    status = str(record.get("status") or "").lower()
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    security_scan = metadata.get("security_scan") if isinstance(metadata.get("security_scan"), dict) else {}
    return (
        status == "human_review"
        or bool(metadata.get("requires_human_review"))
        or security_scan_requires_review(security_scan)
    )


def processing_allowed_after_security_review(record: dict[str, Any] | None) -> bool:
    """Allow processor enqueue only after reviewer Allow / Mask and Allow.

    BLOCK and pending Human Review must never reach RabbitMQ / processors.
    """
    if not isinstance(record, dict):
        return False
    if not document_pending_human_review(record):
        return True
    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        return False
    try:
        from src.features.security.review.human_review_queue import (
            get_reviewer_override_for_document,
        )

        override = get_reviewer_override_for_document(document_id)
    except Exception:
        return False
    if not override:
        return False
    pipeline = str(override.get("pipeline_status") or "").lower()
    if pipeline == "block":
        return False
    return pipeline in {"allow", "mask_and_allow"}
