from __future__ import annotations

import re
from typing import Any, Dict, List

class TopicClassifier:
    """Classifies document contents into topics using keyword weighting."""

    def __init__(self) -> None:
        self.rules: Dict[str, List[str]] = {
            "Manufacturing": ["production", "manufacturing", "batch", "packaging", "operator", "yield", "compounding", "filling"],
            "Quality Assurance": ["qa", "quality assurance", "sop", "capa", "deviation", "audit", "compliance", "change control", "non-conformance"],
            "Quality Control": ["qc", "quality control", "test", "analysis", "specification", "assay", "laboratory", "retest", "analytical"],
            "Validation": ["validation", "qualification", "protocol", "iq", "oq", "pq", "commissioning", "revalidation"],
            "Regulatory": ["regulatory", "fda", "gmp", "submission", "ema", "guideline", "ind", "nda", "regulatory affairs"],
            "Engineering": ["engineering", "maintenance", "calibration", "equipment", "hvac", "utility", "facility", "spare parts"],
            "Warehouse": ["warehouse", "inventory", "shipping", "receiving", "storage", "dispensing", "logistics", "pallet"],
            "Procurement": ["procurement", "purchase", "supplier", "vendor", "contractor", "material request", "sourcing"],
            "Finance": ["finance", "invoice", "billing", "payment", "cost", "revenue", "budget", "accounts payable"],
            "HR": ["hr", "human resources", "employee", "training", "hiring", "personnel", "recruitment", "benefits"],
            "Legal": ["legal", "nda", "agreement", "contract", "privileged", "litigation", "intellectual property", "counsel"],
            "Medical / Clinical": ["medical", "clinical", "patient", "phi", "doctor", "health", "hospital", "study protocol", "informed consent"],
            "IT / Security": ["it", "security", "network", "firewall", "cybersecurity", "server", "access control", "user account"]
        }

    def classify_topics(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        topic_scores: Dict[str, float] = {}

        for topic, keywords in self.rules.items():
            matched_score = 0.0
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                if count > 0:
                    matched_score += min(count * 0.3, 1.0)
            if matched_score > 0:
                topic_scores[topic] = min(matched_score, 1.0)

        # Normalize and sort results
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for topic, score in sorted_topics:
            results.append({
                "topic": topic,
                "confidence": round(score, 2)
            })

        if not results:
            results.append({
                "topic": "General",
                "confidence": 1.0
            })

        return {
            "topics": results
        }
