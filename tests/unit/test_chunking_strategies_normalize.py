from __future__ import annotations

import pytest

from src.features.chunking.domain.chunking_strategy import (
    CHUNKING_STRATEGIES,
    normalize_chunking_strategy,
    resolve_chunking_strategy,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sentence", "sentence-based"),
        ("SENTENCE", "sentence-based"),
        ("fixed", "fixed-overlap-based"),
        ("section", "section-based"),
        ("semantic-based", "semantic"),
        ("sentence-based", "sentence-based"),
        ("section-based", "section-based"),
        ("semantic", "semantic"),
    ],
)
def test_normalize_chunking_strategy_aliases(raw: str, expected: str) -> None:
    assert normalize_chunking_strategy(raw) == expected


def test_resolve_chunking_strategy_accepts_aliases() -> None:
    assert resolve_chunking_strategy("sentence") == "sentence-based"
    assert resolve_chunking_strategy("fixed") == "fixed-overlap-based"


def test_resolve_chunking_strategy_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        resolve_chunking_strategy("not-a-real-strategy")


def test_supported_strategies_remain_unchanged() -> None:
    for strategy in CHUNKING_STRATEGIES:
        assert resolve_chunking_strategy(strategy) == strategy


def test_process_request_normalizes_legacy_chunking_strategy() -> None:
    from src.features.document_processing.core.contract import ProcessRequest

    request = ProcessRequest(
        document_id="doc-1",
        document_name="sample.pdf",
        chunking_strategy="sentence",
    )
    assert request.chunking_strategy == "sentence-based"
