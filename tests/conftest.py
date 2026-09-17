"""Shared pytest fixtures for security pipeline tests."""

from __future__ import annotations

import pytest


class _CleanModelGuardStub:
    """Healthy model guard stub — keeps legacy allow-path tests deterministic."""

    def validate_prompt_sync(self, text: str):
        from src.features.security.domain.guard_result import GuardResult

        return GuardResult(allowed=True, action="allow", provider="test_stub", confidence=0.99)

    def validate_document_text_sync(self, text: str):
        from src.features.security.domain.guard_result import GuardResult

        return GuardResult(allowed=True, action="allow", provider="test_stub", confidence=0.99)


@pytest.fixture(autouse=True)
def _stub_available_model_guard_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent fail-closed unavailable escalation unless a test opts out."""
    if request.node.get_closest_marker("model_guard_unavailable"):
        return
    monkeypatch.setattr(
        "src.features.security.moderation.hybrid_prompt_guard.get_model_prompt_guard",
        lambda: _CleanModelGuardStub(),
    )
