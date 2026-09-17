"""Hybrid enterprise document classification — metadata + structure + ML scoring."""

from __future__ import annotations

import re
from typing import Any

from src.features.security.classification.document_classifier import classify_document_for_pipeline
from src.features.security.dlp.policy_loader import load_document_classification_config


def _normalize_scores(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    max_val = max(raw.values()) if raw else 0.0
    if max_val <= 0:
        return {k: 0.0 for k in raw}
    if max_val <= 1.0:
        return {k: round(min(1.0, v), 4) for k, v in raw.items()}
    return {k: round(min(1.0, v / max_val), 4) for k, v in raw.items()}


def _metadata_scores(
    *,
    file_name: str,
    document_id: str,
    folder_path: str,
    uploaded_metadata: dict[str, Any] | None,
    patterns: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    scores: dict[str, float] = {}
    evidence: list[str] = []
    haystack = " ".join(
        filter(
            None,
            [
                file_name,
                document_id,
                folder_path,
                str((uploaded_metadata or {}).get("document_type") or ""),
                str((uploaded_metadata or {}).get("folder_path") or ""),
            ],
        )
    )
    upper_hay = haystack.upper()

    for doc_type, spec in patterns.items():
        if not isinstance(spec, dict):
            continue
        score = 0.0
        type_evidence: list[str] = []
        tokens = spec.get("filename_tokens") or []
        token_hits = sum(1 for tok in tokens if str(tok).upper() in upper_hay)
        id_matched = False
        for pattern in spec.get("id_patterns") or []:
            try:
                if re.search(str(pattern), haystack, flags=re.IGNORECASE):
                    id_matched = True
                    score = max(score, 0.85)
                    type_evidence.append(f"document id pattern {pattern}")
                    break
            except re.error:
                continue

        if tokens and token_hits:
            token_score = min(1.0, token_hits / max(1, len(tokens)) + 0.35)
            score = max(score, token_score)
            type_evidence.append(f"filename pattern {tokens[0]}")

        if token_hits and id_matched:
            score = max(score, 0.95)

        if score > 0:
            scores[str(doc_type)] = round(min(1.0, score), 4)
            evidence.extend(type_evidence)

    return scores, evidence


def _structure_scores(text: str, patterns: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    scores: dict[str, float] = {}
    evidence: list[str] = []
    clean = (text or "").strip()
    if not clean:
        return scores, evidence

    for doc_type, spec in patterns.items():
        if not isinstance(spec, dict):
            continue
        sections = [str(s) for s in (spec.get("sections") or []) if str(s).strip()]
        if not sections:
            continue
        hits = 0
        hit_names: list[str] = []
        for section in sections:
            if re.search(rf"(?i)\b{re.escape(section)}\b", clean):
                hits += 1
                hit_names.append(section)
        if hits:
            ratio = hits / len(sections)
            scores[str(doc_type)] = round(min(0.99, 0.55 + ratio * 0.44), 4)
            if ratio >= 0.4:
                evidence.append(f"{doc_type} section structure ({', '.join(hit_names[:3])})")

    return scores, evidence


def _ml_scores(text: str, *, lite: bool) -> dict[str, float]:
    result = classify_document_for_pipeline(text, lite=lite)
    all_scores = dict(result.get("all_scores") or {})
    doc_type = str(result.get("document_type") or "Unknown")
    confidence = float(result.get("confidence") or 0.0)
    if doc_type != "Unknown" and doc_type not in all_scores:
        all_scores[doc_type] = confidence
    if not all_scores and doc_type != "Unknown":
        all_scores[doc_type] = confidence
    return _normalize_scores({str(k): float(v) for k, v in all_scores.items()})


def classify_document_hybrid(
    text: str,
    *,
    file_name: str = "",
    document_id: str = "",
    folder_path: str = "",
    uploaded_metadata: dict[str, Any] | None = None,
    lite: bool = False,
) -> dict[str, Any]:
    """Combine metadata, structural, and ML classifiers with configured weights."""
    cfg = load_document_classification_config()
    weights = cfg.get("weights") or {}
    w_meta = float(weights.get("metadata", 0.4))
    w_struct = float(weights.get("structure", 0.4))
    w_ml = float(weights.get("ml", 0.2))
    threshold = float(cfg.get("confidence_threshold", 0.70))

    meta_patterns = cfg.get("metadata_patterns") or {}
    struct_patterns = cfg.get("structure_patterns") or {}

    meta_scores, meta_evidence = _metadata_scores(
        file_name=file_name,
        document_id=document_id,
        folder_path=folder_path,
        uploaded_metadata=uploaded_metadata,
        patterns=meta_patterns,
    )
    struct_scores, struct_evidence = _structure_scores(text, struct_patterns)
    ml_scores = _ml_scores(text, lite=lite)

    all_labels = sorted(set(meta_scores) | set(struct_scores) | set(ml_scores))
    combined: dict[str, float] = {}
    for label in all_labels:
        combined[label] = round(
            meta_scores.get(label, 0.0) * w_meta
            + struct_scores.get(label, 0.0) * w_struct
            + ml_scores.get(label, 0.0) * w_ml,
            4,
        )

    evidence = meta_evidence + struct_evidence
    if not combined:
        return {
            "document_type": "Unknown",
            "confidence": 0.0,
            "all_scores": {},
            "evidence": evidence,
            "strategy": "hybrid",
            "component_scores": {"metadata": meta_scores, "structure": struct_scores, "ml": ml_scores},
        }

    best_label, best_score = max(combined.items(), key=lambda item: item[1])
    peak_component = max(
        meta_scores.get(best_label, 0.0),
        struct_scores.get(best_label, 0.0),
        ml_scores.get(best_label, 0.0),
    )
    effective_confidence = max(best_score, peak_component)
    document_type = best_label if effective_confidence >= threshold else "Unknown"

    return {
        "document_type": document_type,
        "confidence": round(effective_confidence, 4),
        "all_scores": combined,
        "evidence": evidence,
        "strategy": "hybrid",
        "component_scores": {
            "metadata": meta_scores,
            "structure": struct_scores,
            "ml": ml_scores,
        },
        "weights": {"metadata": w_meta, "structure": w_struct, "ml": w_ml},
    }


def classify_document_for_pipeline_hybrid(
    text: str,
    *,
    file_name: str = "",
    document_id: str = "",
    folder_path: str = "",
    uploaded_metadata: dict[str, Any] | None = None,
    lite: bool = False,
) -> dict[str, Any]:
    """Drop-in adapter for upload pipeline — always returns hybrid classification."""
    try:
        return classify_document_hybrid(
            text,
            file_name=file_name,
            document_id=document_id,
            folder_path=folder_path,
            uploaded_metadata=uploaded_metadata,
            lite=lite,
        )
    except Exception:
        fallback = classify_document_for_pipeline(text, lite=lite)
        fallback["strategy"] = "hybrid_fallback"
        return fallback
