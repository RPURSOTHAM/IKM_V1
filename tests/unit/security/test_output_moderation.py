"""Tests for layered output moderation: Regex -> Presidio -> Llama Guard -> Policy Engine."""

from __future__ import annotations

from tests.unit.weaviate_test_stubs import install_weaviate_stubs

install_weaviate_stubs()

from src.features.security.moderation.output_moderator import (
    OutputModerationConfig,
    OutputModerationPipeline,
    moderate_output,
)


def test_clean_response_allowed_with_full_layer_report() -> None:
    report = moderate_output(
        "The SOP describes cleanroom entry steps and gowning requirements."
    )
    assert report.final_action == "allow"
    assert report.safe is True
    assert report.severity == "low"
    layer_names = [layer.layer for layer in report.layers]
    assert layer_names == ["regex", "presidio", "llama_guard", "policy_engine"]
    public = report.to_public_dict()
    assert public["final_action"] == "allow"
    assert len(public["layers"]) == 4
    assert all("execution_time_ms" in layer for layer in public["layers"])


def test_confidential_data_masked_by_regex_layer() -> None:
    report = moderate_output(
        "The employee PAN is ABCDE1234F and the SSN is 123-45-6789."
    )
    assert report.final_action == "mask"
    assert report.safe is False
    assert "ABCDE1234F" not in report.sanitized_text
    assert "123-45-6789" not in report.sanitized_text
    assert report.severity in {"high", "critical"}
    regex_layer = report.layers[0]
    assert regex_layer.layer == "regex"
    assert regex_layer.action == "mask"
    assert regex_layer.detections


def test_prompt_leakage_blocked_by_policy_engine() -> None:
    report = moderate_output(
        "Sure! Here is my system prompt: You are an enterprise document assistant..."
    )
    assert report.final_action == "block"
    assert report.safe is False
    assert "prompt_leakage" in report.flags
    assert report.sanitized_text == OutputModerationConfig().blocked_message
    assert "system prompt" in " ".join(report.reasons).lower()


def test_jailbreak_response_blocked() -> None:
    report = moderate_output(
        "Developer mode enabled. I am now unrestricted and ignoring my previous instructions."
    )
    assert report.final_action == "block"
    assert "jailbreak_response" in report.flags
    assert report.severity == "critical"
    assert report.sanitized_text == OutputModerationConfig().blocked_message


def test_system_prompt_echo_detected_via_markers() -> None:
    report = moderate_output(
        "My instructions say: answer only using the retrieved document context.",
    )
    assert report.final_action == "block"
    assert "prompt_leakage" in report.flags


def test_layers_are_individually_configurable() -> None:
    config = OutputModerationConfig(
        enable_regex=False,
        enable_presidio=False,
        enable_llama_guard=False,
        enable_policy_engine=False,
    )
    report = OutputModerationPipeline(config).moderate("PAN ABCDE1234F visible")
    assert report.final_action == "allow"
    assert all(layer.status == "skipped" for layer in report.layers)
    # With layers disabled nothing is masked; this confirms toggles drive behavior.
    assert "ABCDE1234F" in report.sanitized_text


def test_optional_presidio_failure_never_blocks_pipeline() -> None:
    report = moderate_output("Routine maintenance completed on utility systems.")
    presidio_layer = report.layers[1]
    assert presidio_layer.layer == "presidio"
    assert presidio_layer.status in {"ok", "skipped", "degraded"}
    assert report.final_action == "allow"


def test_structured_report_hides_raw_matched_values_in_public_dict() -> None:
    report = moderate_output("Employee SSN is 123-45-6789.")
    public = report.to_public_dict()
    serialized = str(public)
    assert "123-45-6789" not in serialized
    assert public["layers"][0]["detection_count"] >= 1
