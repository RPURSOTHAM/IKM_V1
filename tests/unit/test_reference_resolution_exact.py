from __future__ import annotations

import pytest

from src.features.references.infrastructure.neo4j_reference_store import Neo4jReferenceStore, build_document_reference_payload
from src.features.references.extraction.reference_extractor import normalize_document_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GL-CQA-ANN-0427", "GL-CQA-ANN-0427"),
        ("gl-cqa-ann-0427", "GL-CQA-ANN-0427"),
        ("GL-CQA-ANN-0427.pdf", "GL-CQA-ANN-0427"),
        ("GL_CQA_ANN_0427", "GL-CQA-ANN-0427"),
        ("GL-CQA-ANN-0427.DOCX", "GL-CQA-ANN-0427"),
    ],
)
def test_normalize_document_id_canonical(raw: str, expected: str) -> None:
    assert normalize_document_id(raw) == expected


def test_build_document_reference_payload_requires_exact_normalized_match() -> None:
    payload = build_document_reference_payload(
        source_document_id="gl-cqa-gop-0030",
        reference={
            "reference_text": "GL-CQA-ANN-0427",
            "extracted_target_id": "GL-CQA-ANN-0427",
            "normalized_target_id": "GL-CQA-ANN-0427",
            "source_sentence": "Refer to GL-CQA-ANN-0427.",
            "reference_type": "document",
            "confidence": 0.92,
            "extraction_method": "pre_chunk_keyword_regex",
            "source_scope": "document",
        },
        resolved_target={
            "document_id": "gl-cqa-gop-0042",
            "document_name": "GL-CQA-GOP-0042.pdf",
            "normalized_document_id": "GL-CQA-GOP-0042",
        },
        tenant_id="default",
        repository_id="default",
    )
    assert payload["create_refers_to"] is False
    assert payload["resolved"] is False


def test_build_document_reference_payload_exact_match_resolves() -> None:
    payload = build_document_reference_payload(
        source_document_id="gl-cqa-gop-0030",
        reference={
            "reference_text": "GL-CQA-ANN-0427",
            "extracted_target_id": "GL-CQA-ANN-0427",
            "normalized_target_id": "GL-CQA-ANN-0427",
            "source_sentence": "Refer to GL-CQA-ANN-0427.",
            "reference_type": "document",
            "confidence": 0.92,
            "extraction_method": "pre_chunk_keyword_regex",
            "source_scope": "document",
        },
        resolved_target={
            "document_id": "gl-cqa-ann-0427",
            "document_name": "GL-CQA-ANN-0427.pdf",
            "normalized_document_id": "GL-CQA-ANN-0427",
        },
        tenant_id="default",
        repository_id="default",
    )
    assert payload["create_refers_to"] is True
    assert payload["resolved_target_normalized_id"] == "GL-CQA-ANN-0427"
    assert payload["normalized_target_id"] == payload["resolved_target_normalized_id"]


def test_attempt_resolution_uses_exact_normalized_document_id(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeResult:
        def __iter__(self):
            return iter(
                [
                    {
                        "document_id": "gl-cqa-ann-0427",
                        "document_name": "GL-CQA-ANN-0427.pdf",
                        "normalized_document_id": "GL-CQA-ANN-0427",
                    }
                ]
            )

    class FakeSession:
        def run(self, cypher: str, **params):
            captured.append({"cypher": cypher, "params": params})
            return FakeResult()

    store = Neo4jReferenceStore(
        config=type("Cfg", (), {"uri": "bolt://test", "user": "neo4j", "password": "pass", "database": None})()
    )
    result = store._attempt_resolution(
        FakeSession(),
        source_document_id="gl-cqa-gop-0030",
        extracted_target_id="GL-CQA-ANN-0427",
        normalized_target_id="GL-CQA-ANN-0427",
    )
    assert result is not None
    assert result["normalized_document_id"] == "GL-CQA-ANN-0427"
    assert "normalized_document_id: $normalized_target_id" in captured[0]["cypher"]
    assert captured[0]["params"]["normalized_target_id"] == "GL-CQA-ANN-0427"


def test_attempt_resolution_rejects_partial_match_candidate(monkeypatch) -> None:
    class FakeResult:
        def __iter__(self):
            return iter(
                [
                    {
                        "document_id": "gl-cqa-gop-0042",
                        "document_name": "GL-CQA-GOP-0042.pdf",
                        "normalized_document_id": "GL-CQA-GOP-0042",
                    }
                ]
            )

    class FakeSession:
        def run(self, _cypher: str, **_params):
            return FakeResult()

    store = Neo4jReferenceStore(
        config=type("Cfg", (), {"uri": "bolt://test", "user": "neo4j", "password": "pass", "database": None})()
    )
    result = store._attempt_resolution(
        FakeSession(),
        source_document_id="gl-cqa-gop-0030",
        extracted_target_id="GL-CQA-ANN-0427",
        normalized_target_id="GL-CQA-ANN-0427",
    )
    assert result is None


def test_attempt_resolution_rejects_ambiguous_matches() -> None:
    class FakeResult:
        def __iter__(self):
            return iter(
                [
                    {
                        "document_id": "doc-a",
                        "document_name": "GL-CQA-ANN-0427.pdf",
                        "normalized_document_id": "GL-CQA-ANN-0427",
                    },
                    {
                        "document_id": "doc-b",
                        "document_name": "GL-CQA-ANN-0427-copy.pdf",
                        "normalized_document_id": "GL-CQA-ANN-0427",
                    },
                ]
            )

    class FakeSession:
        def run(self, _cypher: str, **_params):
            return FakeResult()

    store = Neo4jReferenceStore(
        config=type("Cfg", (), {"uri": "bolt://test", "user": "neo4j", "password": "pass", "database": None})()
    )
    result = store._attempt_resolution(
        FakeSession(),
        source_document_id="gl-cqa-gop-0030",
        extracted_target_id="GL-CQA-ANN-0427",
        normalized_target_id="GL-CQA-ANN-0427",
    )
    assert result is None


def test_save_document_reference_graph_maps_only_exact_target(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []

    class FakeResult:
        def single(self):
            return None

    class FakeSession:
        def run(self, cypher: str, **params):
            captured.append((cypher, params))
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeDriver:
        def session(self, **_kwargs):
            return FakeSession()

        def close(self):
            return None

    def fake_attempt(_session, *, source_document_id, extracted_target_id, normalized_target_id):
        targets = {
            "GL-CQA-ANN-0427": {
                "document_id": "gl-cqa-ann-0427",
                "document_name": "GL-CQA-ANN-0427.pdf",
                "normalized_document_id": "GL-CQA-ANN-0427",
            },
            "GL-CQA-GOP-0042": {
                "document_id": "gl-cqa-gop-0042",
                "document_name": "GL-CQA-GOP-0042.pdf",
                "normalized_document_id": "GL-CQA-GOP-0042",
            },
        }
        return targets.get(normalized_target_id)

    store = Neo4jReferenceStore(
        config=type("Cfg", (), {"uri": "bolt://test", "user": "neo4j", "password": "pass", "database": None})()
    )
    monkeypatch.setattr(store, "_driver", lambda: FakeDriver())
    monkeypatch.setattr(store, "_attempt_resolution", fake_attempt)

    stats = store.save_document_reference_graph(
        source_document_id="gl-cqa-gop-0030",
        source_title="GL-CQA-GOP-0030.pdf",
        tenant_id="default",
        repository_id="default",
        references=[
            {
                "reference_text": "GL-CQA-ANN-0427",
                "extracted_target_id": "GL-CQA-ANN-0427",
                "normalized_target_id": "GL-CQA-ANN-0427",
                "source_sentence": "Refer to GL-CQA-ANN-0427.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            },
            {
                "reference_text": "GL-CQA-GOP-0042",
                "extracted_target_id": "GL-CQA-GOP-0042",
                "normalized_target_id": "GL-CQA-GOP-0042",
                "source_sentence": "See GL-CQA-GOP-0042.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            },
        ],
    )

    assert stats["resolved"] == 2
    refers_to_blocks = [(block, params) for block, params in captured if "REFERS_TO" in block]
    assert len(refers_to_blocks) == 2
    for block, params in refers_to_blocks:
        assert "normalized_document_id: $normalized_target_id" in block
        assert params["normalized_target_id"] in {"GL-CQA-ANN-0427", "GL-CQA-GOP-0042"}


def test_save_document_reference_graph_uses_normalized_merge_keys(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []

    class FakeResult:
        def single(self):
            return None

    class FakeSession:
        def run(self, cypher: str, **params):
            captured.append((cypher, params))
            return FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeDriver:
        def session(self, **_kwargs):
            return FakeSession()

        def close(self):
            return None

    store = Neo4jReferenceStore(
        config=type("Cfg", (), {"uri": "bolt://test", "user": "neo4j", "password": "pass", "database": None})()
    )
    monkeypatch.setattr(store, "_driver", lambda: FakeDriver())
    monkeypatch.setattr(store, "_attempt_resolution", lambda _session, **_: None)

    store.save_document_reference_graph(
        source_document_id="gl-cqa-gop-0030",
        source_title="GL-CQA-GOP-0030.pdf",
        tenant_id="default",
        repository_id="default",
        references=[
            {
                "reference_text": "GL-CQA-ANN-0427",
                "extracted_target_id": "GL-CQA-ANN-0427",
                "normalized_target_id": "GL-CQA-ANN-0427",
                "source_sentence": "Refer to GL-CQA-ANN-0427.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            }
        ],
    )

    document_merge = next(block for block, _ in captured if "MERGE (src:Document" in block)
    reference_merge = next(block for block, _ in captured if "MERGE (ref:Reference" in block)
    assert "normalized_document_id: $source_normalized_id" in document_merge
    assert "source_document_id: $source_document_id" in reference_merge
    assert "normalized_target_id: $normalized_target_id" in reference_merge
    assert "reference_key:" not in reference_merge.split("MERGE")[1].split("SET")[0]


def test_debug_resolve_target_endpoint(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from src.workers.reference_processor import api_server

    class FakeStore:
        def debug_resolve_target(self, target_id: str):
            return {
                "input_target_id": target_id,
                "normalized_target_id": "GL-CQA-ANN-0427",
                "exact_matches": [
                    {
                        "document_id": "gl-cqa-ann-0427",
                        "document_name": "GL-CQA-ANN-0427.pdf",
                        "normalized_document_id": "GL-CQA-ANN-0427",
                    }
                ],
                "would_resolve": True,
            }

    monkeypatch.setattr(api_server, "get_reference_store", lambda: FakeStore())
    client = TestClient(api_server.app)
    payload = client.get("/debug/resolve/GL-CQA-ANN-0427").json()
    assert payload["would_resolve"] is True
    assert payload["exact_matches"][0]["normalized_document_id"] == "GL-CQA-ANN-0427"
