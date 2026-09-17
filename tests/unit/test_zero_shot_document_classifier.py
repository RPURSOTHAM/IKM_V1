from __future__ import annotations

from typing import Any

from src.features.security.classification.document_classifier import (
    DocumentClassifier,
    DocumentClassifierConfig,
)


class FakeZeroShotPipeline:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        text: str,
        *,
        candidate_labels: list[str],
        multi_label: bool,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "text": text,
                "candidate_labels": candidate_labels,
                "multi_label": multi_label,
            }
        )
        ordered = sorted(
            candidate_labels,
            key=lambda label: self.scores.get(label, 0.0),
            reverse=True,
        )
        return {
            "labels": ordered,
            "scores": [self.scores.get(label, 0.0) for label in ordered],
        }


def _classifier(fake_pipeline: FakeZeroShotPipeline) -> DocumentClassifier:
    config = DocumentClassifierConfig(
        model_name="fake-zero-shot-model",
        candidate_labels=[
            "SOP",
            "Policy",
            "Contract",
            "Invoice",
            "Purchase Order",
            "Report",
            "Resume",
            "Manual",
            "Specification",
            "Certificate",
            "Form",
            "Other",
        ],
        confidence_threshold=0.70,
        max_input_chars=200,
    )
    return DocumentClassifier(
        config=config,
        pipeline_factory=lambda *args, **kwargs: fake_pipeline,
    )


def test_normal_document_classifies_as_sop() -> None:
    fake = FakeZeroShotPipeline({"SOP": 0.96, "Policy": 0.02, "Invoice": 0.01})
    result = _classifier(fake).classify_document(
        "Standard operating procedure for controlled document handling."
    )

    assert result["document_type"] == "SOP"
    assert result["confidence"] == 0.96
    assert result["classification_time_ms"] >= 0
    assert result["all_scores"]["SOP"] == 0.96
    assert fake.calls[0]["multi_label"] is False


def test_empty_document_returns_unknown_without_model_call() -> None:
    fake = FakeZeroShotPipeline({"SOP": 0.96})
    result = _classifier(fake).classify_document("   ")

    assert result["document_type"] == "Unknown"
    assert result["confidence"] == 0.0
    assert result["all_scores"] == {}
    assert fake.calls == []


def test_very_large_document_is_truncated_before_classification() -> None:
    fake = FakeZeroShotPipeline({"Report": 0.91})
    result = _classifier(fake).classify_document("A" * 1000)

    assert result["document_type"] == "Report"
    assert len(fake.calls[0]["text"]) == 200


def test_non_english_document_can_be_classified() -> None:
    fake = FakeZeroShotPipeline({"Manual": 0.88, "Other": 0.10})
    result = _classifier(fake).classify_document(
        "Manual de usuario para operar el equipo de laboratorio."
    )

    assert result["document_type"] == "Manual"
    assert result["confidence"] == 0.88


def test_low_confidence_document_returns_unknown_with_scores() -> None:
    fake = FakeZeroShotPipeline({"Other": 0.42, "Report": 0.31, "Form": 0.27})
    result = _classifier(fake).classify_document(
        "Short ambiguous text with no clear document structure."
    )

    assert result["document_type"] == "Unknown"
    assert result["confidence"] == 0.42
    assert result["all_scores"]["Other"] == 0.42

