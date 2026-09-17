"""Unit tests for template → Neo4j reference graph bridge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.features.references.graph.reference_graph_bridge import (
    adapt_template_referenced_documents,
    persist_template_references,
)


def test_adapt_template_referenced_documents_maps_pdf_reader_shape() -> None:
    refs = [
        {
            "document_id": "GL-CQA-ANN-0427",
            "title": "Annexure Control Form",
            "ref_number": "1",
            "sources": ["references_section"],
            "pages": [12, 13],
            "contexts": ["Refer to GL-CQA-ANN-0427 for annexure control."],
        },
        {
            "document_id": "GL-CQA-ANN-0427",
            "title": "duplicate should be skipped",
            "sources": ["inline"],
            "pages": [14],
            "contexts": ["again"],
        },
        {
            "document_id": "",
            "title": "missing id ignored",
        },
    ]

    adapted = adapt_template_referenced_documents(refs)
    assert len(adapted) == 1
    item = adapted[0]
    assert item["target_document_id"] == "GL-CQA-ANN-0427"
    assert item["normalized_target_id"] == "GL-CQA-ANN-0427"
    assert item["reference_text"] == "Annexure Control Form"
    assert item["source_sentence"] == "Refer to GL-CQA-ANN-0427 for annexure control."
    assert item["reference_type"] == "document"
    assert item["confidence"] >= 0.9
    assert item["extraction_method"]
    assert item["evidence_sentences"] == ["Refer to GL-CQA-ANN-0427 for annexure control."]
    assert item["pages"] == [12, 13]


def test_adapt_template_referenced_documents_inline_defaults() -> None:
    adapted = adapt_template_referenced_documents(
        [
            {
                "document_id": "SOP/QA/001",
                "sources": ["inline"],
                "contexts": [],
            }
        ]
    )
    assert len(adapted) == 1
    assert adapted[0]["normalized_target_id"] == "SOP/QA/001"
    assert adapted[0]["reference_text"] == "SOP/QA/001"
    assert adapted[0]["confidence"] == 0.75


def test_persist_template_references_returns_none_for_empty() -> None:
    assert (
        persist_template_references(
            document_id="doc-1",
            repository_id="repo-1",
            tenant_id="tenant-1",
            display_name="Sample",
            referenced_documents=[],
        )
        is None
    )


def test_persist_template_references_extracts_from_template_text() -> None:
    store = MagicMock()
    store.enabled = True
    store.save_document_reference_graph.return_value = {
        "saved": 1,
        "resolved": 0,
        "unresolved": 1,
    }
    template_data = {
        "content_blocks": [
            {"text": "Refer to Document No: SOP-QA-001 for related controls."},
            {"text": "Related Document: WI-OPS-010"},
        ],
        "referenced_documents": [],
    }
    with patch(
        "src.features.references.infrastructure.neo4j_reference_store.get_reference_store",
        return_value=store,
    ):
        stats = persist_template_references(
            document_id="doc-text-refs",
            repository_id="repo-1",
            tenant_id=None,
            display_name="Sample",
            referenced_documents=[],
            template_data=template_data,
        )
    assert stats is not None
    refs = store.save_document_reference_graph.call_args.kwargs["references"]
    targets = {item["normalized_target_id"] for item in refs}
    assert "SOP-QA-001" in targets
    assert "WI-OPS-010" in targets
    assert "ERENCE" not in targets
    assert "REFERENCES" not in targets
    assert "ERENCES" not in targets


def test_persist_template_references_calls_store_methods() -> None:
    store = MagicMock()
    store.enabled = True
    store.save_document_reference_graph.return_value = {
        "saved": 1,
        "resolved": 0,
        "unresolved": 1,
    }

    refs = [
        {
            "document_id": "WI-005",
            "title": "Work Instruction",
            "sources": ["references_section"],
            "pages": [3],
            "contexts": ["See WI-005."],
        }
    ]

    with patch(
        "src.features.references.infrastructure.neo4j_reference_store.get_reference_store",
        return_value=store,
    ):
        stats = persist_template_references(
            document_id="source-doc",
            repository_id="repo-1",
            tenant_id="tenant-1",
            display_name="Source Display Name",
            referenced_documents=refs,
        )

    assert stats == {"saved": 1, "resolved": 0, "unresolved": 1}
    store.register_document.assert_called_once_with(
        document_id="source-doc",
        document_name="Source Display Name",
        source_title="Source Display Name",
        tenant_id="tenant-1",
        repository_id="repo-1",
    )
    store.save_document_reference_graph.assert_called_once()
    kwargs = store.save_document_reference_graph.call_args.kwargs
    assert kwargs["source_document_id"] == "source-doc"
    assert kwargs["references"][0]["normalized_target_id"] == "WI-005"


def test_persist_template_references_soft_fails_on_store_error() -> None:
    store = MagicMock()
    store.enabled = True
    store.register_document.side_effect = RuntimeError("neo4j down")

    with patch(
        "src.features.references.infrastructure.neo4j_reference_store.get_reference_store",
        return_value=store,
    ):
        result = persist_template_references(
            document_id="source-doc",
            repository_id="repo-1",
            tenant_id=None,
            display_name="Doc",
            referenced_documents=[{"document_id": "SOP-1", "sources": ["inline"]}],
        )
    assert result is None


def test_persist_template_references_skips_when_store_disabled() -> None:
    store = MagicMock()
    store.enabled = False

    with patch(
        "src.features.references.infrastructure.neo4j_reference_store.get_reference_store",
        return_value=store,
    ):
        result = persist_template_references(
            document_id="source-doc",
            repository_id="repo-1",
            tenant_id=None,
            display_name="Doc",
            referenced_documents=[{"document_id": "SOP-1"}],
        )
    assert result is None
    store.register_document.assert_not_called()
    store.save_document_reference_graph.assert_not_called()
