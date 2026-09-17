from __future__ import annotations

import re
from typing import Any, Dict, List

class NERDetector:
    """Fallback-safe Named Entity Recognition detector.

    Always performs NER. ``strategy`` only selects implementation:
    - ``auto``: spaCy when available, else rules
    - ``rules``: lightweight regex NER (used by SECURITY_PIPELINE_LITE)
    - ``spacy``: spaCy only with rules fallback on failure
    """

    def __init__(self) -> None:
        # Check if spaCy is installed and loaded (optional integration)
        self.nlp = None
        try:
            import spacy
            # Try to load a lightweight model if available
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                pass
        except ImportError:
            pass

    def detect_entities(self, text: str, *, strategy: str = "auto") -> List[Dict[str, Any]]:
        mode = (strategy or "auto").strip().lower()
        if mode == "rules":
            return self._detect_with_rules(text)
        if mode == "spacy" and self.nlp:
            return self._detect_with_spacy(text) or self._detect_with_rules(text)
        if self.nlp and mode in {"auto", "spacy"}:
            return self._detect_with_spacy(text) or self._detect_with_rules(text)
        return self._detect_with_rules(text)

    def _detect_with_spacy(self, text: str) -> List[Dict[str, Any]] | None:
        if not self.nlp:
            return None
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                severity = "low"
                # Classify PERSON alone as low risk
                if ent.label_ == "PERSON":
                    severity = "low"
                elif ent.label_ in ("ORG", "GPE", "LOC", "FAC", "PRODUCT", "DATE"):
                    severity = "low"
                entities.append({
                    "category": "NER",
                    "type": ent.label_,
                    "matched_value": ent.text,
                    "masked_value": ent.text,
                    "severity": severity,
                    "location": "Text Content",
                    "reason": f"Named entity of type {ent.label_} detected by spaCy."
                })
            # spaCy has no patient/employee context labels; the DLP policy
            # engine relies on these to escalate PHI, so always add them.
            entities.extend(self._detect_domain_contexts(text))
            return entities
        except Exception:
            return None

    def _detect_with_rules(self, text: str) -> List[Dict[str, Any]]:
        # Fallback rule-based matching
        detections = []

        # 1. PERSON: Title (Dr/Mr/Mrs/Ms) + Name or standard capitalized double word names
        person_patterns = [
            r"\b(?:Dr\.|Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
        ]
        for pattern in person_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                # Ignore common headers / false matches (e.g. standard capitalization at start of sentences)
                if val.lower() not in ("standard operating", "operating procedure", "confidential draft", "internal use", "reference copy"):
                    detections.append({
                        "category": "NER",
                        "type": "PERSON",
                        "matched_value": val,
                        "masked_value": val,
                        "severity": "low",
                        "reason": "Person name detected."
                    })

        # 2. ORG: Capitalized words followed by Corp/Inc/LLC/Ltd/Co/Company
        org_patterns = [
            r"\b[A-Z][A-Za-z0-9_]*\s+(?:Inc\.|Corp\.|Corporation|LLC|Ltd\.|Co\.|Company)\b",
            r"\b(?:BioPharma\s+Solutions|PharmaCorp|TechCorp|HealthCare\s+Inc)\b"
        ]
        for pattern in org_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                detections.append({
                    "category": "NER",
                    "type": "ORG",
                    "matched_value": val,
                    "masked_value": val,
                    "severity": "low",
                    "reason": "Organization name detected."
                })

        # 3. LOCATION/GPE: Common cities/countries or street suffixes
        gpe_patterns = [
            r"\b(?:New\s+York|London|Paris|Tokyo|Berlin|India|USA|United\s+States|Germany|Boston|California)\b",
            r"\b\d+\s+[A-Z][a-z]+\s+(?:St\.|Street|Rd\.|Road|Ave\.|Avenue|Blvd\.|Boulevard)\b"
        ]
        for pattern in gpe_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                detections.append({
                    "category": "NER",
                    "type": "LOCATION/GPE",
                    "matched_value": val,
                    "masked_value": val,
                    "severity": "low",
                    "reason": "Location/GPE detected."
                })

        # 4. DATE: YYYY-MM-DD, MM/DD/YYYY, or Month Name DD, YYYY
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{2}/\d{2}/\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b"
        ]
        for pattern in date_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                detections.append({
                    "category": "NER",
                    "type": "DATE",
                    "matched_value": val,
                    "masked_value": val,
                    "severity": "low",
                    "reason": "Date detected."
                })

        # 5. PRODUCT: Product/Model name references
        product_patterns = [
            r"(?i)\b(?:product|model|brand|drug)\s+[A-Z0-9-]{3,15}\b",
            r"\b(?:PharmaLine-[A-Z0-9]+)\b"
        ]
        for pattern in product_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                detections.append({
                    "category": "NER",
                    "type": "PRODUCT",
                    "matched_value": val,
                    "masked_value": val,
                    "severity": "low",
                    "reason": "Product name/reference detected."
                })

        # 6. FACILITY: Facility/Cleanroom/Suite names
        facility_patterns = [
            r"(?i)\b(?:facility|cleanroom|suite|lab|building|plant|warehouse)\s+[A-Z0-9-]{1,10}\b"
        ]
        for pattern in facility_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                detections.append({
                    "category": "NER",
                    "type": "FACILITY",
                    "matched_value": val,
                    "masked_value": val,
                    "severity": "low",
                    "reason": "Facility/laboratory reference detected."
                })

        # 7-8. Patient and employee context
        detections.extend(self._detect_domain_contexts(text))

        return detections

    @staticmethod
    def _detect_domain_contexts(text: str) -> List[Dict[str, Any]]:
        """Detect patient/employee context regardless of the NER backend in use."""
        detections: List[Dict[str, Any]] = []

        patient_patterns = [
            r"(?i)\bpatient\s+(?:ID|name|subject|code)?[\s:-]*([A-Za-z0-9-]{3,15})\b",
            r"(?i)\bsubject\s+(?:ID|number|code)?[\s:-]*([A-Za-z0-9-]{3,15})\b"
        ]
        for pattern in patient_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                detections.append({
                    "category": "NER",
                    "type": "PATIENT_CONTEXT",
                    "matched_value": val,
                    "masked_value": val,
                    "severity": "medium", # Patient context has higher default severity
                    "reason": "Patient context identified."
                })

        employee_patterns = [
            r"(?i)\bemployee\s+(?:ID|name|staff|code)?[\s:-]*([A-Za-z0-9-]{3,15})\b"
        ]
        for pattern in employee_patterns:
            for m in re.finditer(pattern, text):
                val = m.group(0)
                detections.append({
                    "category": "NER",
                    "type": "EMPLOYEE_CONTEXT",
                    "matched_value": val,
                    "masked_value": val,
                    "severity": "medium",
                    "reason": "Employee context identified."
                })

        return detections
