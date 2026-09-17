from __future__ import annotations

import pytest

from src.features.chunking.application.chunking_service import Chunk
from src.features.references.infrastructure.neo4j_reference_store import (
    Neo4jReferenceStore,
    build_document_reference_payload,
)
from src.features.references.extraction.reference_extractor import (
    extract_document_references,
    extract_references_from_chunk,
    extract_references_from_chunks,
    extract_references_from_text,
)


def test_extract_document_references_regex_fields() -> None:
    text = """
    Document No: SOP-001
    Document Number: QMS-002
    Document ID: VAL/003.01
    Document Reference: ABC-123
    Ref: SOP/QA/001
    Related Document: WI-005
    SOP No: SOP-099
    WI No: WI-100
    Form No: FRM-222
    """

    references = extract_document_references(text)
    ids = {item["reference_id"] for item in references}

    assert {
        "SOP-001",
        "QMS-002",
        "VAL/003.01",
        "ABC-123",
        "SOP/QA/001",
        "WI-005",
        "SOP-099",
        "WI-100",
        "FRM-222",
    } <= ids
    first = references[0]
    assert first["matched_text"]
    assert isinstance(first["start_index"], int)
    assert isinstance(first["end_index"], int)
    assert first["confidence"] > 0
    assert first["extraction_method"] == "regex"


def test_extract_document_references_removes_duplicates_case_insensitive() -> None:
    text = "Document No: SOP-001\nref: sop-001\nRelated Document: SOP-001"

    references = extract_document_references(text)

    assert [item["reference_id"] for item in references] == ["SOP-001"]


def test_extract_references_from_chunk_uses_chunk_text() -> None:
    chunk = Chunk(
        id="chunk-1",
        doc_name="sample.pdf",
        page=2,
        section_name="Validation",
        section_path="Procedure > Validation",
        text="Per SOP No: SOP-001, complete Section 3.2 and Annexure A before release.",
    )
    references = extract_references_from_chunk(chunk)
    types = {item["reference_type"] for item in references}
    assert "sop" in types
    assert "section" in types
    assert "annexure" in types
    assert all(item["chunk_id"] == "chunk-1" for item in references)
    assert all(item["source_section"] == "Procedure > Validation" for item in references)


def test_extract_references_from_chunks_runs_for_every_chunk() -> None:
    chunks = [
        Chunk(
            id="chunk-a",
            doc_name="sample.pdf",
            page=1,
            section_name="Intro",
            section_path="Intro",
            text="Document No: SOP-001",
        ),
        Chunk(
            id="chunk-b",
            doc_name="sample.pdf",
            page=2,
            section_name="Methods",
            section_path="Methods",
            text="Refer to Section 4.1 and Appendix B.",
        ),
    ]

    references = extract_references_from_chunks(chunks, source_document_id="doc-1")

    assert len(references) == 3
    assert {item["chunk_id"] for item in references} == {"chunk-a", "chunk-b"}
    assert all(item["source_document_id"] == "doc-1" for item in references)


def test_extract_references_from_text_keeps_unresolved_document_ids_null_for_sections() -> None:
    references = extract_references_from_text("Refer to Section 4.1 for acceptance criteria.")
    section_refs = [item for item in references if item["reference_type"] == "section"]
    assert len(section_refs) == 1
    assert section_refs[0]["target_document_id"] is None
    assert section_refs[0]["target_section"] == "4.1"


def test_build_document_reference_payload_resolved() -> None:
    payload = build_document_reference_payload(
        source_document_id="source-doc",
        reference={
            "reference_text": "Refer to GL-CQA-ANN-0427",
            "normalized_reference": "GL-CQA-ANN-0427",
            "trigger_phrase": "refer to",
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

    assert payload["resolved"] is True
    assert payload["create_refers_to"] is True
    assert payload["reference_key"] == "source-doc::GL-CQA-ANN-0427"


def test_build_document_reference_payload_unresolved() -> None:
    payload = build_document_reference_payload(
        source_document_id="source-doc",
        reference={
            "reference_text": "Refer to GL-CQA-ANN-0427",
            "normalized_reference": "GL-CQA-ANN-0427",
            "trigger_phrase": "refer to",
            "source_sentence": "Refer to GL-CQA-ANN-0427.",
            "reference_type": "document",
            "confidence": 0.92,
            "extraction_method": "pre_chunk_keyword_regex",
            "source_scope": "document",
        },
        resolved_target=None,
        tenant_id="default",
        repository_id="default",
    )

    assert payload["resolved"] is False
    assert payload["create_refers_to"] is False


def test_save_document_reference_graph_unresolved_does_not_create_refers_to(monkeypatch) -> None:
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

    stats = store.save_document_reference_graph(
        source_document_id="source-doc",
        source_title="Sample",
        tenant_id="default",
        repository_id="repo-1",
        references=[
            {
                "reference_text": "Refer to GL-CQA-ANN-0427",
                "normalized_reference": "GL-CQA-ANN-0427",
                "target_document_id": "GL-CQA-ANN-0427",
                "trigger_phrase": "refer to",
                "source_sentence": "Refer to GL-CQA-ANN-0427.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            }
        ],
    )

    assert stats["saved"] == 1
    assert stats["unresolved"] == 1
    cypher_blocks = [block for block, _params in captured]
    assert any("MENTIONS_REFERENCE" in block for block in cypher_blocks)
    assert not any("HAS_SECTION" in block for block in cypher_blocks)
    assert not any("HAS_CHUNK" in block for block in cypher_blocks)
    assert not any("MERGE (chunk:Chunk" in block for block in cypher_blocks)
    assert not any("REFERS_TO" in block for block in cypher_blocks)


def test_save_document_reference_graph_multiple_references_use_extracted_target(monkeypatch) -> None:
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
        mapping = {
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
        return mapping.get(normalized_target_id)

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
                "reference_text": "Refer to GL-CQA-ANN-0427",
                "normalized_reference": "GL-CQA-ANN-0427",
                "extracted_target_id": "GL-CQA-ANN-0427",
                "trigger_phrase": "refer to",
                "source_sentence": "Refer to GL-CQA-ANN-0427.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            },
            {
                "reference_text": "See GL-CQA-GOP-0042",
                "normalized_reference": "GL-CQA-GOP-0042",
                "extracted_target_id": "GL-CQA-GOP-0042",
                "trigger_phrase": "see",
                "source_sentence": "See GL-CQA-GOP-0042.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            },
        ],
    )

    assert stats["saved"] == 2
    assert stats["resolved"] == 2
    refers_to_blocks = [params for block, params in captured if "REFERS_TO" in block]
    assert len(refers_to_blocks) == 2
    extracted_targets = {params["normalized_target_id"] for params in refers_to_blocks}
    assert extracted_targets == {"GL-CQA-ANN-0427", "GL-CQA-GOP-0042"}
    assert all(params["source_normalized_id"] == "GL-CQA-GOP-0030" for params in refers_to_blocks)


def test_save_document_reference_graph_skips_self_reference(monkeypatch) -> None:
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
    monkeypatch.setattr(
        store,
        "_attempt_resolution",
        lambda _session, **_: {
            "document_id": "gl-cqa-gop-0030",
            "document_name": "GL-CQA-GOP-0030.pdf",
            "normalized_document_id": "GL-CQA-GOP-0030",
        },
    )

    stats = store.save_document_reference_graph(
        source_document_id="gl-cqa-gop-0030",
        source_title="GOP-0030",
        tenant_id="default",
        repository_id="repo-1",
        references=[
            {
                "reference_text": "Refer to GL-CQA-GOP-0030",
                "normalized_reference": "GL-CQA-GOP-0030",
                "extracted_target_id": "GL-CQA-GOP-0030",
                "trigger_phrase": "refer to",
                "source_sentence": "Refer to GL-CQA-GOP-0030.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            }
        ],
    )

    assert stats["saved"] == 0
    assert not any("REFERS_TO" in block for block, _params in captured)


def test_save_document_reference_graph_resolved_creates_refers_to(monkeypatch) -> None:
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
    monkeypatch.setattr(
        store,
        "_attempt_resolution",
        lambda _session, **_: {
            "document_id": "gl-cqa-ann-0427",
            "document_name": "GL-CQA-ANN-0427.pdf",
            "normalized_document_id": "GL-CQA-ANN-0427",
        },
    )

    stats = store.save_document_reference_graph(
        source_document_id="source-doc",
        source_title="Sample",
        tenant_id="default",
        repository_id="repo-1",
        references=[
            {
                "reference_text": "Refer to GL-CQA-ANN-0427",
                "normalized_reference": "GL-CQA-ANN-0427",
                "target_document_id": "GL-CQA-ANN-0427",
                "trigger_phrase": "refer to",
                "source_sentence": "Refer to GL-CQA-ANN-0427.",
                "reference_type": "document",
                "confidence": 0.92,
                "extraction_method": "pre_chunk_keyword_regex",
                "source_scope": "document",
            }
        ],
    )

    assert stats["resolved"] == 1
    assert any("REFERS_TO" in block for block, _params in captured)
    assert any("TARGETS" in block for block, _params in captured)


def test_neo4j_unavailable_raises_from_reference_pipeline(monkeypatch) -> None:
    class BrokenStore:
        @property
        def enabled(self):
            return False

    monkeypatch.setattr("src.features.references.application.reference_pipeline.get_reference_store", lambda: BrokenStore())
    monkeypatch.setattr(
        "src.features.references.application.reference_pipeline.load_document_blocks",
        lambda _path: ([type("Block", (), {"text": "Refer to GL-CQA-GOP-0030."})()], {}),
    )

    from pathlib import Path

    from src.features.document_processing.core.contract import ProcessRequest
    from src.features.references.application.reference_pipeline import run_pre_chunk_reference_extraction

    request = ProcessRequest(document_id="doc-1", document_name="sample.pdf")
    with pytest.raises(RuntimeError, match="Neo4j is not configured"):
        run_pre_chunk_reference_extraction(
            request,
            Path("sample.pdf"),
            set_status=lambda *_a, **_k: None,
            check_stop=lambda: False,
        )


def test_reference_graph_api_response_format(monkeypatch) -> None:
    pytest.importorskip("weaviate")
    from fastapi.testclient import TestClient

    from src.processor_service import api_server

    class FakeStore:
        def references_for(self, document_id: str):
            return [{"document_id": "GL-CQA-ANN-0427", "relationship": "REFERS_TO"}]

        def referenced_by(self, document_id: str):
            return [{"document_id": "SRC-001", "relationship": "REFERS_TO"}]

        def reference_graph(self, document_id: str):
            return {
                "nodes": [
                    {"id": document_id, "label": "Document"},
                    {"id": "GL-CQA-ANN-0427", "label": "Document"},
                ],
                "edges": [{"source": document_id, "target": "GL-CQA-ANN-0427", "type": "REFERS_TO"}],
            }

    monkeypatch.setattr(api_server, "get_reference_store", lambda: FakeStore())
    client = TestClient(api_server.app)

    references = client.get("/api/v1/documents/DOC-1/references").json()
    referenced_by = client.get("/api/v1/documents/DOC-1/referenced-by").json()
    graph = client.get("/api/v1/documents/DOC-1/reference-graph").json()

    assert references["document_id"] == "DOC-1"
    assert isinstance(references["references"], list)
    assert referenced_by["referenced_by"][0]["document_id"] == "SRC-001"
    assert set(graph) >= {"document_id", "nodes", "edges"}
    assert graph["edges"][0]["type"] == "REFERS_TO"
