from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_LABELS = [
    "NDA",
    "Contract",
    "Medical Record",
    "SOP",
    "Publication",
    "Research Paper",
    "Regulatory Document",
    "Policy",
    "Invoice",
    "Report",
    "Manual",
    "Other",
]

DEFAULT_MODEL_NAME = "facebook/bart-large-mnli"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_MAX_INPUT_CHARS = 12000

PipelineFactory = Callable[..., Any]


@dataclass(frozen=True)
class DocumentClassifierConfig:
    model_name: str
    candidate_labels: list[str]
    confidence_threshold: float
    max_input_chars: int


def _parse_candidate_labels(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_CANDIDATE_LABELS.copy()

    value = raw.strip()
    if not value:
        return DEFAULT_CANDIDATE_LABELS.copy()

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            labels = [str(item).strip() for item in parsed if str(item).strip()]
            if labels:
                return labels
    except json.JSONDecodeError:
        pass

    labels = [item.strip() for item in value.split(",") if item.strip()]
    return labels or DEFAULT_CANDIDATE_LABELS.copy()


def load_config() -> DocumentClassifierConfig:
    # 1. Start with defaults
    model_name = DEFAULT_MODEL_NAME
    candidate_labels = DEFAULT_CANDIDATE_LABELS.copy()
    threshold = DEFAULT_CONFIDENCE_THRESHOLD
    max_chars = DEFAULT_MAX_INPUT_CHARS

    # 2. Try loading from JSON configuration file
    config_path = Path(__file__).parent / "classifier_config.json"
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "model_name" in data:
                    model_name = str(data["model_name"]).strip()
                if "confidence_threshold" in data:
                    threshold = float(data["confidence_threshold"])
                if "max_input_chars" in data:
                    max_chars = int(data["max_input_chars"])
                if "candidate_labels" in data and isinstance(data["candidate_labels"], list):
                    candidate_labels = [str(lbl).strip() for lbl in data["candidate_labels"] if str(lbl).strip()]
        except Exception as e:
            logger.warning("Failed to load configuration from %s: %s", config_path, e)

    # 3. Environment variable overrides
    env_model = os.getenv("DOCUMENT_CLASSIFIER_MODEL", "")
    if env_model:
        model_name = env_model.strip()

    env_threshold = os.getenv("DOCUMENT_CLASSIFIER_CONFIDENCE_THRESHOLD", "")
    if env_threshold:
        try:
            threshold = float(env_threshold)
        except ValueError:
            logger.warning("Invalid DOCUMENT_CLASSIFIER_CONFIDENCE_THRESHOLD=%r; using %.2f", env_threshold, threshold)

    env_max_chars = os.getenv("DOCUMENT_CLASSIFIER_MAX_INPUT_CHARS", "")
    if env_max_chars:
        try:
            max_chars = int(env_max_chars)
        except ValueError:
            logger.warning("Invalid DOCUMENT_CLASSIFIER_MAX_INPUT_CHARS=%r; using %d", env_max_chars, max_chars)

    env_labels = os.getenv("DOCUMENT_CLASSIFIER_LABELS", "")
    if env_labels:
        candidate_labels = _parse_candidate_labels(env_labels)

    return DocumentClassifierConfig(
        model_name=model_name,
        candidate_labels=candidate_labels,
        confidence_threshold=threshold,
        max_input_chars=max(1000, max_chars),
    )



def _detect_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


class DocumentClassifier:
    """Zero-shot document classifier backed by a Hugging Face NLI model."""

    def __init__(
        self,
        config: DocumentClassifierConfig | None = None,
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.config = config or load_config()
        self.device = _detect_device()
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Load the zero-shot pipeline once for this classifier instance."""
        if self._pipeline is not None:
            return

        with self._load_lock:
            if self._pipeline is not None:
                return

            logger.info(
                "Loading zero-shot document classifier",
                extra={
                    "model_name": self.config.model_name,
                    "device": self.device,
                    "candidate_labels": self.config.candidate_labels,
                    "confidence_threshold": self.config.confidence_threshold,
                },
            )
            try:
                factory = self._pipeline_factory
                if factory is None:
                    from transformers import pipeline

                    factory = pipeline
                self._pipeline = factory(
                    "zero-shot-classification",
                    model=self.config.model_name,
                    device=self.device,
                )
                logger.info(
                    "Zero-shot document classifier loaded",
                    extra={"model_name": self.config.model_name, "device": self.device},
                )
            except Exception:
                logger.exception(
                    "Failed to load zero-shot document classifier",
                    extra={"model_name": self.config.model_name, "device": self.device},
                )
                raise

    def classify_document(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        clean_text = (text or "").strip()

        if not clean_text:
            return self._unknown_result(0.0, start, reason="empty_document")

        if len(clean_text) > self.config.max_input_chars:
            logger.info(
                "Truncating document before classification",
                extra={
                    "input_chars": len(clean_text),
                    "max_input_chars": self.config.max_input_chars,
                },
            )
            clean_text = clean_text[: self.config.max_input_chars]

        try:
            self.load()
            result = self._pipeline(
                clean_text,
                candidate_labels=self.config.candidate_labels,
                multi_label=False,
            )
            labels = [str(label) for label in result.get("labels", [])]
            scores = [float(score) for score in result.get("scores", [])]
            all_scores = {
                label: round(score, 4)
                for label, score in zip(labels, scores)
            }
            best_label = labels[0] if labels else "Unknown"
            confidence = round(scores[0], 4) if scores else 0.0
            document_type = (
                best_label
                if confidence >= self.config.confidence_threshold
                else "Unknown"
            )

            payload = {
                "document_type": document_type,
                "confidence": confidence,
                "classification_time_ms": self._elapsed_ms(start),
                "all_scores": all_scores,
            }
            logger.info(
                "Document classification completed",
                extra={
                    "document_type": document_type,
                    "confidence": confidence,
                    "classification_time_ms": payload["classification_time_ms"],
                },
            )
            return payload
        except Exception as exc:
            logger.exception("Document classification failed")
            payload = self._unknown_result(0.0, start, reason="classification_failed")
            payload["error"] = str(exc)
            return payload

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int(round((time.perf_counter() - start) * 1000))

    def _unknown_result(
        self,
        confidence: float,
        start: float,
        *,
        reason: str,
    ) -> dict[str, Any]:
        logger.info(
            "Document classification returned Unknown",
            extra={"reason": reason, "confidence": confidence},
        )
        return {
            "document_type": "Unknown",
            "confidence": round(confidence, 4),
            "classification_time_ms": self._elapsed_ms(start),
            "all_scores": {},
        }


_classifier: DocumentClassifier | None = None
_classifier_lock = threading.Lock()


def get_document_classifier() -> DocumentClassifier:
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = DocumentClassifier()
    return _classifier


def preload_document_classifier() -> None:
    get_document_classifier().load()


def classify_document(text: str) -> dict[str, Any]:
    return get_document_classifier().classify_document(text)


def classify_document_lite(text: str) -> dict[str, Any]:
    """Mandatory lightweight document classification (keyword heuristics).

    Used when SECURITY_PIPELINE_LITE selects a low-RAM strategy. Still returns
    document_type aligned with DLP document_types policy keys.
    """
    start = time.perf_counter()
    clean = (text or "").strip()
    if not clean:
        return {
            "document_type": "Unknown",
            "confidence": 0.0,
            "classification_time_ms": 0,
            "all_scores": {},
            "strategy": "lite_heuristic",
        }

    lowered = clean.lower()
    rules: list[tuple[str, list[str], float]] = [
        ("NDA", ["non-disclosure", "non disclosure", "confidentiality agreement", "nda "], 0.85),
        ("Medical Record", ["patient", "medical record", "phi", "diagnosis", "clinical notes"], 0.82),
        ("Contract", ["hereinafter", "party of the first part", "terms and conditions", "agreement between"], 0.8),
        ("Regulatory Document", ["fda", "ema", "regulatory submission", "21 cfr", "guidance document"], 0.8),
        ("SOP", ["standard operating procedure", "sop ", "purpose:", "scope:", "responsibility:"], 0.78),
        ("Research Paper", ["abstract", "methodology", "references", "et al", "hypothesis"], 0.75),
        ("Publication", ["published in", "journal", "doi:", "copyright"], 0.72),
        ("Invoice", ["invoice", "amount due", "bill to", "payment terms"], 0.8),
        ("Policy", ["policy statement", "shall comply", "company policy"], 0.7),
        ("Manual", ["user manual", "instructions for use", "installation guide"], 0.7),
        ("Report", ["executive summary", "findings", "recommendations"], 0.65),
    ]
    scores: dict[str, float] = {}
    for label, keywords, base in rules:
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits:
            scores[label] = min(0.99, base + 0.05 * (hits - 1))
    if not scores:
        return {
            "document_type": "Other",
            "confidence": 0.55,
            "classification_time_ms": int(round((time.perf_counter() - start) * 1000)),
            "all_scores": {"Other": 0.55},
            "strategy": "lite_heuristic",
        }
    best = max(scores.items(), key=lambda item: item[1])
    return {
        "document_type": best[0],
        "confidence": round(best[1], 4),
        "classification_time_ms": int(round((time.perf_counter() - start) * 1000)),
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
        "strategy": "lite_heuristic",
    }


def classify_document_for_pipeline(text: str, *, lite: bool = False) -> dict[str, Any]:
    """Always classify; lite selects heuristic strategy instead of zero-shot model."""
    if lite:
        return classify_document_lite(text)
    try:
        return classify_document(text)
    except Exception:
        # Never skip classification coverage — fall back to heuristic.
        result = classify_document_lite(text)
        result["fallback"] = "lite_after_model_error"
        return result

