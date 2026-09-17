"""Unit tests for chunking catalog helpers, preview service, and HTTP routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.chunking.api.chunking_routes import router as chunking_router
from src.features.chunking.application.chunking_preview_service import (
    preview_chunking,
    text_to_preview_blocks,
)
from src.features.chunking.domain.chunking_strategy import CHUNKING_STRATEGIES
from src.features.repositories.configuration.chunking_strategy_catalog import (
    build_chunking_strategy_catalog,
    get_chunking_strategy,
)


SAMPLE_TEXT = """
1. Purpose
This document describes the batch release procedure for finished products.
It covers sampling, testing, and release authorization steps.

2. Scope
The procedure applies to all commercial batches manufactured at the site.
Additional controls apply for sterile and high-potency products.
""".strip()


def _chunking_client() -> TestClient:
    app = FastAPI()
    app.include_router(chunking_router, prefix="/api/v1")
    return TestClient(app)


def test_catalog_lists_all_eight_strategies() -> None:
    catalog = build_chunking_strategy_catalog()
    assert catalog["count"] == 8
    assert set(catalog["strategy_ids"]) == set(CHUNKING_STRATEGIES)
    assert len(catalog["strategies"]) == 8
    descriptions = {item["strategy_id"]: item["description"] for item in catalog["strategies"]}
    assert "fixed word windows" in descriptions["fixed-overlap-based"].lower()
    assert "complete sentences" in descriptions["sentence-based"].lower()
    assert "never breaks mid-paragraph" in descriptions["paragraph-based"].lower()
    assert "heading boundaries" in descriptions["section-based"].lower()
    assert "Heading 1 > Heading 2" in descriptions["hierarchical"]
    assert "cosine similarity" in descriptions["semantic"].lower()
    assert "Section > Subsection > Paragraph > Table > Image Caption" in descriptions["semantic-hierarchy"]
    assert "step size" in descriptions["sliding-window"].lower()


def test_get_chunking_strategy_supports_aliases() -> None:
    detail = get_chunking_strategy("section")
    assert detail["strategy_id"] == "section-based"
    assert detail["config_fields"]
    assert detail["example_settings"]["chunking_strategy"] == "section-based"


def test_get_chunking_strategy_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_chunking_strategy("not-a-real-strategy")


def test_text_to_preview_blocks_marks_numbered_headings() -> None:
    blocks = text_to_preview_blocks(SAMPLE_TEXT)
    assert len(blocks) >= 4
    assert blocks[0].style == "Heading 1"
    assert blocks[0].text.startswith("1. Purpose")


@pytest.mark.parametrize(
    "strategy",
    sorted(CHUNKING_STRATEGIES),
)
def test_preview_chunking_runs_for_each_strategy(strategy: str) -> None:
    result = preview_chunking(
        strategy=strategy,
        text=SAMPLE_TEXT,
        chunk_size=40,
        chunk_overlap=1,
        min_content_words=5,
        citation_retainment=True,
    )
    assert result["strategy_id"] == strategy
    assert result["chunk_count"] >= 1
    assert len(result["chunks"]) == result["chunk_count"]
    assert result["chunks"][0]["text"]
    assert result["chunks"][0]["strategy_name"] == strategy


def test_http_list_strategies() -> None:
    client = _chunking_client()
    response = client.get("/api/v1/chunking/strategies")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 8
    assert len(body["strategies"]) == 8


def test_http_get_strategy_detail_and_404() -> None:
    client = _chunking_client()
    ok = client.get("/api/v1/chunking/strategies/sentence-based")
    assert ok.status_code == 200
    assert ok.json()["strategy_id"] == "sentence-based"
    missing = client.get("/api/v1/chunking/strategies/not-real")
    assert missing.status_code == 404


def test_http_preview_chunking() -> None:
    client = _chunking_client()
    response = client.post(
        "/api/v1/chunking/preview",
        json={
            "strategy": "sentence-based",
            "text": SAMPLE_TEXT,
            "chunk_size": 40,
            "chunk_overlap": 1,
            "min_content_words": 5,
            "chunking_config": {"max_sentences_per_chunk": 3},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["strategy_id"] == "sentence-based"
    assert body["chunk_count"] >= 1
    assert body["chunks"][0]["text"]
