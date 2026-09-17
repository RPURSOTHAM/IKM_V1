from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DocumentComplianceError(Exception):
    def __init__(self, response_payload: Dict[str, Any]) -> None:
        super().__init__("Document compliance violation")
        self.response_payload = response_payload


class ComplianceScanner:
    _CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
        r"(?i)\b("
        r"password|api[_-]?key|access[_-]?token|secret|client[_-]?secret|private[_-]?key|auth(?:entication)?[_-]?token"
        r")\b[\s:=\"']{1,8}([A-Za-z0-9][A-Za-z0-9_\-./+=]{5,127})"
    )
    _PASSPORT_GENERIC_PATTERN = re.compile(r"\b[A-Z]\d{7}\b|\b[A-Z]{2}\d{7}\b")
    _PASSPORT_LABELED_PATTERN = re.compile(
        r"(?i)\bpassport(?:\s*(?:number|no|num))?\s*[:#-]?\s*([A-Z0-9]{8,9})\b"
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

    def mask_value(self, value: str, type_name: str) -> str:
        if type_name == "Credit/Debit Card Number":
            cleaned = "".join(c for c in value if c.isdigit())
            if len(cleaned) >= 4:
                return f"XXXX-XXXX-XXXX-{cleaned[-4:]}"
            return "XXXX-XXXX-XXXX-XXXX"
        elif type_name == "Email Address":
            if "@" in value:
                parts = value.split("@")
                username = parts[0]
                domain = parts[1]
                masked_username = username[0] + "***" if len(username) > 1 else "*"
                return f"{masked_username}@{domain}"
            return "em***@domain.com"
        elif type_name == "SSN":
            cleaned = "".join(c for c in value if c.isdigit())
            if len(cleaned) >= 4:
                return f"XXX-XX-{cleaned[-4:]}"
            return "XXX-XX-XXXX"
        elif type_name == "Aadhaar":
            cleaned = "".join(c for c in value if c.isdigit())
            if len(cleaned) >= 4:
                return f"XXXX-XXXX-{cleaned[-4:]}"
            return "XXXX-XXXX-XXXX"
        elif type_name in ("PAN", "Passport"):
            if len(value) >= 4:
                return f"{value[:2]}{'X' * (len(value) - 4)}{value[-2:]}"
            return "XXXX"
        elif type_name in ("API Key", "Token", "Password", "Private Key"):
            if len(value) >= 8:
                return f"{value[:4]}************{value[-4:]}"
            return "************"
        elif type_name == "Bank Account Number":
            cleaned = "".join(c for c in value if c.isalnum())
            if len(cleaned) >= 4:
                return f"XXXX-XXXX-{cleaned[-4:]}"
            return "XXXX-XXXX-XXXX"
        elif type_name == "Employee ID":
            return "EMP-XXXX"
        elif type_name == "Supplier Contact":
            return "SUPPLIER-XXXX"
        elif type_name == "Internal URL":
            return "[INTERNAL URL]"
        elif type_name == "Internal Project Name":
            return "Project-XXXX"
        return value

    def scan_text(self, text: str) -> List[Dict[str, Any]]:
        detections = []

        # --- HIGH RISK ---
        # SSN
        for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text):
            val = m.group(0)
            detections.append({
                "category": "High Risk",
                "type": "SSN",
                "matched_value": val,
                "masked_value": self.mask_value(val, "SSN"),
                "severity": "high",
                "reason": "Social Security Number (SSN) detected.",
                "recommended_action": "Redact SSN before uploading."
            })
        # Aadhaar — prefer spaced/hyphenated forms; bare 12 digits are too noisy for SOPs.
        for m in re.finditer(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{4}-\d{4}-\d{4}\b", text):
            val = m.group(0)
            detections.append({
                "category": "High Risk",
                "type": "Aadhaar",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Aadhaar"),
                "severity": "high",
                "reason": "Aadhaar number detected.",
                "recommended_action": "Redact Aadhaar number before uploading."
            })
        # Explicit Aadhaar label + 12 digits (avoid bare \d{12} false positives).
        for m in re.finditer(
            r"(?i)\b(?:aadhaar|aadhar|uidai)[\s:#-]*(\d{4}[\s-]?\d{4}[\s-]?\d{4}|\d{12})\b",
            text,
        ):
            val = m.group(1)
            if any(d.get("matched_value") == val or d.get("matched_value") == m.group(0) for d in detections):
                continue
            detections.append({
                "category": "High Risk",
                "type": "Aadhaar",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Aadhaar"),
                "severity": "high",
                "reason": "Aadhaar number detected.",
                "recommended_action": "Redact Aadhaar number before uploading."
            })
        # PAN
        for m in re.finditer(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text):
            val = m.group(0)
            detections.append({
                "category": "High Risk",
                "type": "PAN",
                "matched_value": val,
                "masked_value": self.mask_value(val, "PAN"),
                "severity": "high",
                "reason": "PAN number detected.",
                "recommended_action": "Redact PAN before uploading."
            })
        # Passport
        passport_pattern = r"\b[A-Z][1-9]\d{6}\b"
        for m in re.finditer(passport_pattern, text):
            val = m.group(0)
            detections.append({
                "category": "High Risk",
                "type": "Passport",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Passport"),
                "severity": "high",
                "reason": "Passport number detected.",
                "recommended_action": "Redact Passport number before uploading."
            })
        for m in self._PASSPORT_GENERIC_PATTERN.finditer(text):
            val = m.group(0)
            if any(d.get("type") == "Passport" and d.get("matched_value") == val for d in detections):
                continue
            detections.append({
                "category": "High Risk",
                "type": "Passport",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Passport"),
                "severity": "high",
                "reason": "Passport number detected.",
                "recommended_action": "Redact Passport number before uploading."
            })
        for m in self._PASSPORT_LABELED_PATTERN.finditer(text):
            val = (m.group(1) or "").strip()
            if len(val) < 8:
                continue
            if any(d.get("type") == "Passport" and d.get("matched_value") == val for d in detections):
                continue
            detections.append({
                "category": "High Risk",
                "type": "Passport",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Passport"),
                "severity": "high",
                "reason": "Passport number detected.",
                "recommended_action": "Redact Passport number before uploading."
            })
        # CVV
        for m in re.finditer(r"(?i)\b(?:cvv2?|cvc2?|card\s*verification\s*code|security\s*code)[\s:-]+([0-9]{3,4})\b", text):
            val = m.group(1)
            detections.append({
                "category": "High Risk",
                "type": "CVV",
                "matched_value": val,
                "masked_value": "XXX",
                "severity": "high",
                "reason": "Credit card security code (CVV) detected.",
                "recommended_action": "Redact CVV before uploading."
            })
        # Credit/Debit Card Number
        card_pattern = r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b"
        for m in re.finditer(card_pattern, text):
            val = m.group(0)
            cleaned = "".join(c for c in val if c.isdigit())
            if self.check_luhn(cleaned):
                detections.append({
                    "category": "High Risk",
                    "type": "Credit/Debit Card Number",
                    "matched_value": val,
                    "masked_value": self.mask_value(val, "Credit/Debit Card Number"),
                    "severity": "high",
                    "reason": "Credit/debit card number detected.",
                    "recommended_action": "Redact card numbers before uploading."
                })
        # Bank Account Number
        bank_pattern = r"(?i)(?:bank\s+account|acc\s+no|account\s+no|routing\s+(?:number|no))[\s:-]+([A-Z0-9-]{8,20})"
        for m in re.finditer(bank_pattern, text):
            val = m.group(1)
            if len(val.strip("-")) >= 8:
                detections.append({
                    "category": "High Risk",
                    "type": "Bank Account Number",
                    "matched_value": val,
                    "masked_value": self.mask_value(val, "Bank Account Number"),
                    "severity": "high",
                    "reason": "Bank account number detected.",
                    "recommended_action": "Redact bank account details."
                })
        # AWS Key / Secrets
        for m in re.finditer(r"\b(AKIA[A-Z0-9]{16})\b", text):
            val = m.group(1)
            detections.append({
                "category": "High Risk",
                "type": "API Key",
                "matched_value": val,
                "masked_value": self.mask_value(val, "API Key"),
                "severity": "high",
                "reason": "AWS Access Key ID detected.",
                "recommended_action": "Remove credentials from files before uploading."
            })
        # Private Key block
        for m in re.finditer(r"(-----BEGIN [A-Z ]*PRIVATE KEY-----)", text):
            val = m.group(1)
            detections.append({
                "category": "High Risk",
                "type": "Private Key",
                "matched_value": val,
                "masked_value": "[PRIVATE KEY DETECTED]",
                "severity": "high",
                "reason": "Private Key block detected.",
                "recommended_action": "Remove private keys before uploading."
            })
        # Passwords / Tokens keywords
        for m in self._CREDENTIAL_ASSIGNMENT_PATTERN.finditer(text):
            key_name = (m.group(1) or "").strip().lower()
            val = (m.group(2) or "").strip()
            if not val:
                continue
            # Suppress obvious placeholders and pure short numerics.
            lowered_val = val.lower()
            if lowered_val in {"none", "null", "n/a", "na", "xxxxxx", "changeme", "placeholder"}:
                continue
            if val.isdigit() and len(val) < 8:
                continue
            detections.append({
                "category": "High Risk",
                "type": "Token",
                "matched_value": f"{key_name}={val}",
                "masked_value": f"{key_name}={self.mask_value(val, 'Token')}",
                "severity": "high",
                "reason": "Authentication token/key detected.",
                "recommended_action": "Remove credentials from documents."
            })

        # Patient Info / PHI
        phi_pattern = r"(?i)\b(?:medical record|patient ID|health insurance|medical history|clinical notes|diagnosis|prescription|treatment plan)\b"
        for m in re.finditer(phi_pattern, text):
            val = m.group(0)
            detections.append({
                "category": "High Risk",
                "type": "PHI/medical record",
                "matched_value": val,
                "masked_value": val,
                "severity": "high",
                "reason": "Protected Health Information (PHI) or medical record keyword detected.",
                "recommended_action": "Remove PHI content."
            })
        # Privileged Legal Content / Trade Secrets (High Risk)
        privileged_pattern = r"(?i)\b(?:attorney-client privilege|attorney-client privileged|trade secret|intellectual property)\b"
        for m in re.finditer(privileged_pattern, text):
            val = m.group(0)
            detections.append({
                "category": "High Risk",
                "type": "Confidential legal/privileged content",
                "matched_value": val,
                "masked_value": val,
                "severity": "high",
                "reason": f"Genuinely sensitive or privileged content ({val}) detected.",
                "recommended_action": "Remove or redact privileged content before uploading."
            })

        # --- MEDIUM RISK ---
        # Email Address
        for m in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
            val = m.group(0)
            detections.append({
                "category": "Medium Risk",
                "type": "Email address",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Email Address"),
                "severity": "medium",
                "reason": "Email address detected.",
                "recommended_action": "Check if email is public or redact."
            })
        # Phone Number
        for m in re.finditer(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
            val = m.group(0)
            if not ("-" in val and len(val) == 10 and val.count("-") == 2 and val.replace("-", "").isdigit()):
                detections.append({
                    "category": "Medium Risk",
                    "type": "Phone number",
                    "matched_value": val,
                    "masked_value": "XXX-XXX-" + val[-4:] if len(val) >= 4 else "XXX-XXX-XXXX",
                    "severity": "medium",
                    "reason": "Phone number detected.",
                    "recommended_action": "Redact phone numbers."
                })
        # Employee ID
        for m in re.finditer(r"(?i)\b(?:emp(?:loyee)?\s*(?:id|no|code)|staff\s*(?:id|no))[\s:-]+([A-Z0-9-]{4,15})\b", text):
            val = m.group(0)
            detections.append({
                "category": "Medium Risk",
                "type": "Employee ID",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Employee ID"),
                "severity": "medium",
                "reason": "Employee identification code detected.",
                "recommended_action": "Redact employee identifiers."
            })
        # Supplier Contact
        for m in re.finditer(r"(?i)\b(?:supplier|vendor)\s*(?:contact|phone|email)\b", text):
            val = m.group(0)
            detections.append({
                "category": "Medium Risk",
                "type": "Supplier contact",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Supplier Contact"),
                "severity": "medium",
                "reason": "Supplier or vendor contact info marker detected.",
                "recommended_action": "Redact supplier contact details."
            })
        # Internal URL
        for m in re.finditer(r"\b(?:https?://)?(?:localhost|internal|intranet|dev|stage|prod|local|corp|secure)\.[a-z0-9.-]+\b", text):
            val = m.group(0)
            detections.append({
                "category": "Medium Risk",
                "type": "Internal URL",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Internal URL"),
                "severity": "medium",
                "reason": "Internal development or corporate intranet URL detected.",
                "recommended_action": "Redact internal system URLs."
            })
        # Internal Project Name
        # Match explicit labels (Project Name/Code/ID) or strong code-like tokens,
        # but avoid generic phrases like "project update" / "project status".
        project_name_pattern = (
            r"\b(?:(?i:project\s*(?:name|code|id)\s*[:#-]\s*[A-Za-z0-9_-]{3,24})"
            r"|(?:Project|project)\s+[A-Z]{2,}[A-Z0-9_-]{1,23})\b"
        )
        for m in re.finditer(project_name_pattern, text):
            val = m.group(0)
            detections.append({
                "category": "Medium Risk",
                "type": "Internal project name",
                "matched_value": val,
                "masked_value": self.mask_value(val, "Internal project name"),
                "severity": "medium",
                "reason": "Internal corporate project codename detected.",
                "recommended_action": "Redact internal project names."
            })

        # General Confidentiality / Legal Labels (informational — do not route alone to review)
        conf_label_pattern = r"(?i)\b(?:confidential|strictly confidential|internal use only|restricted distribution|confidential business|proprietary data|work product|non-disclosure agreement|NDA|employment contract|indemnity agreement|joint venture agreement|confidential draft|reference copy|internal reference copy)\b"
        negated_conf_pattern = r"(?i)\b(?:no|not|without|non[-\s]?)\s+(?:confidential|restricted|proprietary)\b"
        for m in re.finditer(conf_label_pattern, text):
            val = m.group(0)
            start = max(0, m.start() - 24)
            context = text[start : m.end() + 8]
            if re.search(negated_conf_pattern, context):
                continue
            detections.append({
                "category": "Low Risk",
                "type": "Confidentiality label",
                "matched_value": val,
                "masked_value": val,
                "severity": "low",
                "reason": "Confidentiality label or general watermark keyword detected.",
                "recommended_action": "Informational only; review only when combined with sensitive content."
            })

        # --- LOW RISK ---
        # SOP Number
        for m in re.finditer(r"(?i)\bSOP\s*[-#0-9A-Z]{3,15}\b", text):
            val = m.group(0)
            detections.append({
                "category": "Low Risk",
                "type": "SOP number",
                "matched_value": val,
                "masked_value": val,
                "severity": "low",
                "reason": "SOP ID number identified.",
                "recommended_action": "Allow standard procedure metadata."
            })
        # Batch Number
        for m in re.finditer(r"(?i)\bbatch\s*[-#0-9A-Z]{3,15}\b", text):
            val = m.group(0)
            detections.append({
                "category": "Low Risk",
                "type": "Batch number",
                "matched_value": val,
                "masked_value": val,
                "severity": "low",
                "reason": "Manufacturing batch run number detected.",
                "recommended_action": "Allow run numbers."
            })
        # Product Name
        for m in re.finditer(r"(?i)\bproduct\s*[-#0-9A-Z]{3,15}\b", text):
            val = m.group(0)
            detections.append({
                "category": "Low Risk",
                "type": "Product name",
                "matched_value": val,
                "masked_value": val,
                "severity": "low",
                "reason": "Product model name identified.",
                "recommended_action": "Allow product identifier references."
            })
        # Equipment Name
        for m in re.finditer(r"(?i)\bequipment\s*[-#0-9A-Z]{3,15}\b", text):
            val = m.group(0)
            detections.append({
                "category": "Low Risk",
                "type": "Equipment name",
                "matched_value": val,
                "masked_value": val,
                "severity": "low",
                "reason": "Facility equipment name identified.",
                "recommended_action": "Allow machinery references."
            })
        # Company Name
        for m in re.finditer(r"(?i)\b(?:Company|Corp|Inc|Ltd|Enterprise)\s+[A-Z][A-Za-z0-9_]{2,20}\b", text):
            val = m.group(0)
            detections.append({
                "category": "Low Risk",
                "type": "Company name",
                "matched_value": val,
                "masked_value": val,
                "severity": "low",
                "reason": "Corporate entity name identified.",
                "recommended_action": "Allow company names."
            })

        self._log_detection_summary(detections)
        return detections

    @staticmethod
    def _log_detection_summary(detections: list[dict[str, Any]]) -> None:
        if not detections:
            logger.info("SECURITY_REGEX_MATCHES total=0")
            return
        credential_matches = sum(1 for d in detections if str(d.get("type") or "").lower() == "token")
        pii_matches = sum(
            1
            for d in detections
            if str(d.get("type") or "").lower() in {"aadhaar", "pan", "passport", "ssn", "email address", "phone number"}
        )
        logger.info(
            "SECURITY_REGEX_MATCHES total=%d credential_matches=%d pii_matches=%d",
            len(detections),
            credential_matches,
            pii_matches,
        )

    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        from src.features.document_processing.loaders.document_text import load_document_blocks
        try:
            blocks, doc_metadata = load_document_blocks(file_path, mask_sensitive=False)
        except Exception:
            logger.exception("Failed to parse document for compliance scan: %s", file_path)
            return {"clean": True, "detected_categories": [], "severity": "low", "detections": []}

        all_detections = []
        detected_categories = set()
        max_severity_val = 0
        severity_map = {"low": 1, "medium": 2, "high": 3}
        severity_reverse_map = {1: "low", 2: "medium", 3: "high"}

        current_section = "General"

        # Scan document metadata
        for k, v in doc_metadata.items():
            if isinstance(v, str):
                metadata_detections = self.scan_text(v)
                for det in metadata_detections:
                    det["location"] = f"Document Metadata ({k})"
                    all_detections.append(det)
                    detected_categories.add(det["type"])
                    max_severity_val = max(max_severity_val, severity_map.get(det["severity"].lower(), 1))

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
                all_detections.append(det)
                detected_categories.add(det["type"])
                max_severity_val = max(max_severity_val, severity_map.get(det["severity"].lower(), 1))

        severity = severity_reverse_map.get(max_severity_val, "low")
        clean = len(all_detections) == 0

        # Construct clean public detections
        public_detections = []
        for det in all_detections:
            public_detections.append({
                "category": det["category"],
                "type": det["type"],
                "masked_value": det["masked_value"],
                "location": det["location"],
                "reason": det["reason"],
                "recommended_action": det["recommended_action"]
            })

        return {
            "clean": clean,
            "detected_categories": sorted(list(detected_categories)),
            "severity": severity,
            "detections": public_detections,
            "raw_detections": all_detections
        }


def scan_document_for_api(file_path: Path) -> Dict[str, Any]:
    scanner = ComplianceScanner()
    scan_res = scanner.scan_file(file_path)

    if scan_res["clean"]:
        return {"status": "accepted"}

    # Determine highest severity
    severities = [det["severity"].lower() for det in scan_res["raw_detections"]]
    severity_rank = {"low": 1, "medium": 2, "high": 3}
    highest_rank = max(severity_rank.get(s, 1) for s in severities)
    highest_severity = {1: "low", 2: "medium", 3: "high"}[highest_rank]

    # Map raw detections to filtered detections based on highest severity
    # For high severity: block all high-risk detections
    # For medium severity: allow and return warnings for medium-risk detections
    detections_to_return = []
    detected_types = set()
    for det in scan_res["raw_detections"]:
        if det["severity"] == highest_severity:
            detections_to_return.append({
                "category": det["category"],
                "type": det["type"],
                "severity": det["severity"],
                "masked_value": det["masked_value"],
                "location": det["location"],
                "reason": det["reason"],
                "recommended_action": det["recommended_action"]
            })
            detected_types.add(det["type"])

    if highest_severity == "high":
        detections_to_return = []
        for det in scan_res["raw_detections"]:
            if det["severity"] == "high":
                detections_to_return.append({
                    "category": det["category"],
                    "type": det["type"],
                    "severity": "high",
                    "masked_value": det["masked_value"],
                    "location": det["location"],
                    "reason": det["reason"],
                    "recommended_action": det["recommended_action"]
                })
                detected_types.add(det["type"])
        return {
            "status": "upload_blocked",
            "reason": "Sensitive or restricted data was detected in the uploaded document.",
            "detected_categories": sorted(list(detected_types)),
            "severity": "high",
            "detections": detections_to_return,
            "recommended_action": "Remove or redact the restricted information and upload the revised document, or request authorized secure processing."
        }
    elif highest_severity == "medium":
        detections_to_return = []
        for det in scan_res["raw_detections"]:
            if det["severity"] == "medium":
                detections_to_return.append({
                    "category": det["category"],
                    "type": det["type"],
                    "severity": "medium",
                    "masked_value": det["masked_value"],
                    "location": det["location"],
                    "action": "masked_and_allowed"
                })
                detected_types.add(det["type"])
        return {
            "status": "allowed_with_warning",
            "severity": "medium",
            "warning": "Medium-risk sensitive data was detected. The document was allowed, but values may be masked before processing.",
            "detected_categories": sorted(list(detected_types)),
            "detections": detections_to_return
        }
    else:
        return {"status": "accepted"}


def validate_document_content(file_path: Path) -> None:
    # Retain validate_document_content wrapper raise pattern for compatibility
    res = scan_document_for_api(file_path)
    if res.get("status") == "upload_blocked":
        raise DocumentComplianceError(res)


def mask_text_content(text: str) -> str:
    if not text:
        return text

    scanner = ComplianceScanner()

    # Mask credit cards
    def replace_cards(m):
        val = m.group(0)
        cleaned = "".join(c for c in val if c.isdigit())
        if scanner.check_luhn(cleaned):
            return scanner.mask_value(val, "Credit/Debit Card Number")
        return val
    text = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b", replace_cards, text)

    # Mask Emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", lambda m: scanner.mask_value(m.group(0), "Email Address"), text)

    # Mask Phone numbers
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "XXX-XXX-XXXX", text)

    # Mask SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "XXX-XX-XXXX", text)

    # Mask Aadhaar
    text = re.sub(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{4}-\d{4}-\d{4}\b|\b\d{12}\b", "XXXX-XXXX-XXXX", text)

    # Mask PAN
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "XXXXXX", text)

    # Mask Passports
    text = re.sub(r"\b[A-Z][1-9]\d{6}\b|\b[A-Z]\d{7}\b|\b[A-Z]{2}\d{7}\b", "XXXXXX", text)

    # Mask AWS keys
    text = re.sub(r"\b(AKIA[A-Z0-9]{16})\b", "AKIA************", text)

    # Mask Private Keys
    text = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "[PRIVATE KEY]", text)

    # Mask Employee ID
    text = re.sub(r"(?i)\b(?:emp(?:loyee)?\s*(?:id|no|code)|staff\s*(?:id|no))[\s:-]+([A-Z0-9-]{4,15})\b", lambda m: m.group(0).replace(m.group(1), "XXXX"), text)

    # Mask Supplier contact keyword references
    text = re.sub(r"(?i)\b(?:supplier|vendor)\s*(?:contact|phone|email)\b", "SUPPLIER-XXXX", text)

    # Mask Internal URLs
    text = re.sub(r"\b(?:https?://)?(?:localhost|internal|intranet|dev|stage|prod|local|corp|secure)\.[a-z0-9.-]+\b", "[INTERNAL URL]", text)

    # Mask Internal project names
    text = re.sub(r"(?i)\bproject\s+[A-Za-z0-9_-]{3,15}\b", "Project-XXXX", text)

    return text
