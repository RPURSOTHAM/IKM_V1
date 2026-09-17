from __future__ import annotations

import pytest

from src.features.references.infrastructure.neo4j_reference_store import (
    Neo4jReferenceStore,
    build_document_reference_payload,
)
from src.features.references.extraction.reference_extractor import (
    DEFAULT_TRIGGER_PHRASES,
    extract_pre_chunk_references,
    load_trigger_phrases,
    normalize_reference_id,
    split_document_text_into_sentences,
)


def test_trigger_phrase_detection_filters_non_reference_sentences() -> None:
    text = """
    Operator performs cleaning.
    Refer to GL-CQA-GOP-0030 before release.
    The procedure is important.
    """
    references = extract_pre_chunk_references(
        text,
        source_document_id="gl-cqa-gop-0042",
        trigger_phrases=["refer to"],
    )
    assert len(references) == 1
    assert references[0]["normalized_reference"] == "GL-CQA-GOP-0030"
    assert references[0]["trigger_phrase"] == "refer to"
    assert references[0]["source_scope"] == "document"
    assert references[0]["extraction_method"] == "pre_chunk_keyword_regex"


def test_multiple_references_in_one_sentence() -> None:
    text = "Refer to GL-CQA-ANN-0427 and GL-CQA-GOP-0042."
    references = extract_pre_chunk_references(
        text,
        source_document_id="gl-cqa-gop-0030",
        trigger_phrases=["refer to"],
    )
    normalized = {item["normalized_reference"] for item in references}
    assert normalized == {"GL-CQA-ANN-0427", "GL-CQA-GOP-0042"}


def test_multiple_references_across_document() -> None:
    text = """
    Refer to GL-CQA-ANN-0427.
    See GL-CQA-GOP-0042.
    As per GL-CQA-FRM-0101.
    """
    references = extract_pre_chunk_references(
        text,
        source_document_id="gl-cqa-gop-0030",
        trigger_phrases=["refer to", "see", "as per"],
    )
    normalized = {item["normalized_reference"] for item in references}
    assert normalized == {"GL-CQA-ANN-0427", "GL-CQA-GOP-0042", "GL-CQA-FRM-0101"}
    assert len(references) == 3


def test_self_reference_skipped_by_default() -> None:
    text = """
    GL-CQA-GOP-0030 Rev 1.0
    Refer to GL-CQA-GOP-0030 for details.
    Refer to GL-CQA-ANN-0427.
    """
    references = extract_pre_chunk_references(
        text,
        source_document_id="gl-cqa-gop-0030",
        trigger_phrases=["refer to", "gop"],
    )
    normalized = {item["normalized_reference"] for item in references}
    assert "GL-CQA-GOP-0030" not in normalized
    assert normalized == {"GL-CQA-ANN-0427"}


def test_different_references_do_not_collapse() -> None:
    text = """
    Refer to GL-CQA-ANN-0427.
    Refer to GL-CQA-GOP-0042.
    Refer to GL-CQA-FRM-0101.
    """
    references = extract_pre_chunk_references(
        text,
        source_document_id="gl-cqa-gop-0030",
        trigger_phrases=["refer to"],
    )
    keys = {f"{item['source_document_id']}::{item['normalized_reference']}" for item in references}
    assert len(keys) == 3
    assert keys == {
        "gl-cqa-gop-0030::GL-CQA-ANN-0427",
        "gl-cqa-gop-0030::GL-CQA-GOP-0042",
        "gl-cqa-gop-0030::GL-CQA-FRM-0101",
    }


def test_gop_trigger_does_not_match_document_id_substring() -> None:
    text = "GL-CQA-GOP-0030 Quality Procedure"
    references = extract_pre_chunk_references(
        text,
        source_document_id="gl-cqa-gop-0030",
        trigger_phrases=["gop"],
    )
    assert references == []


def test_build_document_reference_payload_skips_self_refers_to() -> None:
    payload = build_document_reference_payload(
        source_document_id="gl-cqa-gop-0030",
        reference={
            "reference_text": "Refer to GL-CQA-GOP-0030",
            "normalized_reference": "GL-CQA-GOP-0030",
            "extracted_target_id": "GL-CQA-GOP-0030",
            "trigger_phrase": "refer to",
            "source_sentence": "Refer to GL-CQA-GOP-0030.",
            "reference_type": "document",
            "confidence": 0.92,
            "extraction_method": "pre_chunk_keyword_regex",
            "source_scope": "document",
        },
        resolved_target={
            "document_id": "gl-cqa-gop-0030",
            "document_name": "GL-CQA-GOP-0030.pdf",
            "normalized_document_id": "GL-CQA-GOP-0030",
        },
        tenant_id="default",
        repository_id="default",
    )
    assert payload["create_refers_to"] is False
    assert payload["resolved"] is False


def test_find_exact_target_documents_uses_normalized_property(monkeypatch) -> None:
    captured: list[dict] = []

    class FakeResult:
        def __iter__(self):
            return iter([])

    class FakeSession:
        def run(self, _cypher: str, **params):
            captured.append(params)
            return FakeResult()

    store = Neo4jReferenceStore(
        config=type("Cfg", (), {"uri": "bolt://test", "user": "neo4j", "password": "pass", "database": None})()
    )
    store._find_exact_target_documents(FakeSession(), "GL-CQA-ANN-0427")
    assert captured[0]["normalized_target_id"] == "GL-CQA-ANN-0427"


def test_multiple_references_in_one_sentence_labeled_and_implicit() -> None:
    text = "Refer to GL-CQA-GOP-0030. Document No: GL-CQA-ANN-0427."
    references = extract_pre_chunk_references(
        text,
        source_document_id="source-doc",
        trigger_phrases=["refer to", "document no"],
    )
    normalized = {item["normalized_reference"] for item in references}
    assert normalized == {"GL-CQA-GOP-0030", "GL-CQA-ANN-0427"}


def test_duplicate_handling_merges_evidence() -> None:
    text = """
    Refer to GL-CQA-GOP-0030.
    Refer to GL-CQA-GOP-0030 again.
    """
    references = extract_pre_chunk_references(
        text,
        source_document_id="source-doc",
        trigger_phrases=["refer to"],
    )
    assert len(references) == 1
    assert len(references[0]["evidence_sentences"]) == 2


def test_existing_regex_reused_for_labeled_reference() -> None:
    text = "As per SOP No: SOP-001, complete validation."
    references = extract_pre_chunk_references(
        text,
        source_document_id="source-doc",
        trigger_phrases=["as per"],
    )
    assert references[0]["normalized_reference"] == "SOP-001"
    assert references[0]["reference_type"] == "sop"


def test_build_document_reference_payload_unresolved() -> None:
    payload = build_document_reference_payload(
        source_document_id="gl-cqa-gop-0030",
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
    assert payload["reference_key"] == "gl-cqa-gop-0030::GL-CQA-ANN-0427"


def test_build_document_reference_payload_resolved() -> None:
    payload = build_document_reference_payload(
        source_document_id="gl-cqa-gop-0030",
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


def test_resolve_pending_references_on_future_upload(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []

    class FakeResult:
        def __init__(self, rows=None):
            self._rows = rows or []

        def __iter__(self):
            return iter(self._rows)

    class FakeSession:
        def run(self, cypher: str, **params):
            captured.append((cypher, params))
            if "coalesce(ref.resolved, false) = false" in cypher:
                return FakeResult(
                    [
                        {
                            "source_document_id": "gl-cqa-gop-0030",
                            "extracted_target_id": "GL-CQA-ANN-0427",
                            "normalized_target_id": "GL-CQA-ANN-0427",
                            "reference_type": "document",
                            "confidence": 0.92,
                        }
                    ]
                )
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

    resolved = store.resolve_pending_references(uploaded_document_id="gl-cqa-ann-0427")

    assert resolved == 1
    assert any("REFERS_TO" in block for block, _params in captured)
    assert any("TARGETS" in block for block, _params in captured)


def test_save_document_reference_graph_does_not_create_fake_target_document(monkeypatch) -> None:
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
        source_document_id="gl-cqa-gop-0030",
        source_title="GOP-0030",
        tenant_id="default",
        repository_id="default",
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
    assert not any("MERGE (target:Document" in block and "extracted_target_id" not in str(params) for block, params in captured if "REFERS_TO" in block)


def test_load_trigger_phrases_defaults() -> None:
    phrases = load_trigger_phrases()
    assert "refer to" in phrases
    assert len(phrases) >= len(DEFAULT_TRIGGER_PHRASES)


def test_split_document_text_into_sentences() -> None:
    sentences = split_document_text_into_sentences("First sentence. Second sentence!\nThird line.")
    assert sentences == ["First sentence. Second sentence!", "Third line."]


def test_split_preserves_outline_reference_lines() -> None:
    sentences = split_document_text_into_sentences(
        "8.1.1. Deviation Recurrence Check Instructions: GL-CQA-ANN-0383."
    )
    assert sentences == ["8.1.1. Deviation Recurrence Check Instructions: GL-CQA-ANN-0383."]


def test_extract_related_document_list_from_gop_pdf_page() -> None:
    text = """
    7.9 GOP for Management of Notification and Escalation in BioQuest: GL-CQA-GOP-0034.
    7.10 GOP for Continuous Improvement and CAPA Management: GL-CQA-GOP-0035.
    7.14 SOP for Trend Analysis of Quality Parameter: GL-CQA-SOP-0070.
    7.16 SOP for Investigation Procedure for Out of Level results in Environmental and Utility Monitoring: GL-QC-SOP-0320.
    8.1.1. Deviation Recurrence Check Instructions: GL-CQA-ANN-0383.
    """
    references = extract_pre_chunk_references(
        text,
        source_document_id="gl-cqa-gop-0030",
    )
    normalized = {item["normalized_target_id"] for item in references}
    assert normalized == {
        "GL-CQA-GOP-0034",
        "GL-CQA-GOP-0035",
        "GL-CQA-SOP-0070",
        "GL-QC-SOP-0320",
        "GL-CQA-ANN-0383",
    }
    assert not any(item.startswith("7.") for item in normalized)


def test_debug_references_endpoint(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from src.workers.reference_processor import api_server

    class FakeStore:
        def debug_references_for(self, document_id: str):
            return [
                {
                    "extracted_target_id": "GL-CQA-ANN-0427",
                    "reference_text": "GL-CQA-ANN-0427",
                    "source_sentence": "Refer to GL-CQA-ANN-0427.",
                    "resolved": True,
                    "target_document_id": "gl-cqa-ann-0427",
                }
            ]

    monkeypatch.setattr(api_server, "get_reference_store", lambda: FakeStore())
    client = TestClient(api_server.app)
    payload = client.get("/debug/references/gl-cqa-gop-0030").json()
    assert payload["document_id"] == "gl-cqa-gop-0030"
    assert payload["references"][0]["extracted_target_id"] == "GL-CQA-ANN-0427"


def test_reference_extraction_status_endpoint() -> None:
    from fastapi.testclient import TestClient

    from src.workers.reference_processor import api_server

    client = TestClient(api_server.app)
    payload = client.get("/status").json()
    assert payload["state"] == "idle"
    assert "reference_count" in payload
    assert "last_job_terminal_status" in payload


def test_processor_type_alias_reference_extraction() -> None:
    from src.features.document_processing.shared_processor.types import normalize_processor_type

    assert normalize_processor_type("reference_extraction") == "reference_document_extraction"


def test_labeled_reference_without_trigger_phrase() -> None:
    text = "The system uses Document ID: GL-CQA-GOP-0030 to perform checks."
    references = extract_pre_chunk_references(
        text,
        source_document_id="some-other-id",
        source_document_name="some-other-name",
    )
    assert len(references) == 1
    assert references[0]["normalized_reference"] == "GL-CQA-GOP-0030"


def test_labeled_reference_without_hyphen() -> None:
    text = "The procedure complies with SOP No: SOP001."
    references = extract_pre_chunk_references(
        text,
        source_document_id="some-other-id",
        source_document_name="some-other-name",
    )
    assert len(references) == 1
    assert references[0]["normalized_reference"] == "SOP001"


def test_self_reference_skipped_by_filename() -> None:
    text = "Please refer to GL-CQA-GOP-0030."
    references = extract_pre_chunk_references(
        text,
        source_document_id="uuid-1234-5678",
        source_document_name="GL-CQA-GOP-0030.pdf",
    )
    assert references == []

