"""Unit tests for DocumentTemplate Neo4j verification guarantees."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_store_template_graph_rejects_empty_document_id() -> None:
    from src.infrastructure.document_databases.neo4j_store import store_template_graph

    with pytest.raises(RuntimeError, match="non-empty document_id"):
        store_template_graph(
            document_id="",
            repository_id="repo-1",
            template_data={"document_tree": []},
        )


def test_store_template_graph_raises_when_verification_count_zero() -> None:
    from src.infrastructure.document_databases.neo4j_store import store_template_graph

    zero_record = MagicMock()
    zero_record.__getitem__ = lambda self, key: 0
    zero_result = MagicMock()
    zero_result.single.return_value = zero_record

    mock_tx = MagicMock()
    mock_tx.run.return_value = zero_result

    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.execute_write = lambda fn: fn(mock_tx)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    mock_gdb = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    with patch.dict("sys.modules", {"neo4j": MagicMock(GraphDatabase=mock_gdb)}):
        with pytest.raises(RuntimeError, match="no nodes exist"):
            store_template_graph(
                document_id="missing-template",
                repository_id="repo-1",
                template_data={"metadata": {}, "document_tree": []},
            )


def test_processor_does_not_return_location_when_store_fails() -> None:
    from pathlib import Path
    from unittest.mock import MagicMock

    from src.features.document_processing.core.contract import ProcessRequest
    from src.features.document_processing.processors.template_extraction import TemplateExtractionProcessor

    processor = TemplateExtractionProcessor()
    request = ProcessRequest(
        document_id="fail-te",
        document_name="sample.docx",
        repository_id="repo-1",
    )
    fake = Path("sample.docx")

    with patch(
        "src.features.document_processing.extractors.template_compliance.extract_template",
        return_value={"document": {"metadata": {}, "sections": []}},
    ), patch(
        "src.features.document_processing.extractors.template_compliance.compliance_to_legacy_template",
        return_value={"metadata": {}, "document_tree": [], "outline": [], "tables": [], "source": "template_compliance"},
    ), patch(
        "src.features.document_processing.processors.template_extraction.store_template_graph",
        side_effect=RuntimeError("Template graph write reported success but no nodes exist."),
    ):
        with pytest.raises(RuntimeError, match="no nodes exist"):
            processor.run(request, fake, set_status=MagicMock(), check_stop=lambda: False)


def test_fetch_template_graph_returns_payload() -> None:
    from src.infrastructure.document_databases.neo4j_store import fetch_template_graph

    row = {
        "template_json": json.dumps(
            {"document_tree": [{"type": "heading", "text": "Revision History"}], "outline": []}
        ),
        "metadata_json": json.dumps({"title": "SOP"}),
        "document_type_id": "type-1",
        "document_type_name": "SOP",
        "document_title": "SOP",
        "repository_id": "repo-1",
        "updated_at": "2026-01-01",
    }

    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value.single.return_value = row

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    mock_gdb = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    with (
        patch(
            "src.infrastructure.document_databases.neo4j_store._neo4j_credentials",
            return_value=("bolt://localhost:7687", "neo4j", "password"),
        ),
        patch.dict("sys.modules", {"neo4j": MagicMock(GraphDatabase=mock_gdb)}),
        patch(
            "src.infrastructure.document_databases.neo4j_store._require_neo4j_driver",
            return_value=mock_gdb,
        ),
    ):
        payload = fetch_template_graph("doc-1")

    assert payload is not None
    assert payload["document_tree"][0]["text"] == "Revision History"
    assert payload["_template"]["result_location"] == "neo4j://DocumentTemplate/doc-1"


def test_get_document_template_prefers_graph_then_artifact() -> None:
    from fastapi import HTTPException

    from src.features.documents.application.document_service import DocumentReceiverService

    service = DocumentReceiverService()
    service.get_document_record = MagicMock(return_value={"document_id": "doc-1", "metadata": {}})  # type: ignore[method-assign]

    with (
        patch(
            "src.features.documents.application.document_metadata_service.check_document_access"
        ),
        patch(
            "src.infrastructure.document_databases.neo4j_store.fetch_template_graph",
            return_value=None,
        ),
        patch(
            "src.infrastructure.document_databases.neo4j_store.fetch_document_artifact",
            return_value={"skeleton": {"sections": [{"title": "Intro"}]}, "_artifact": {"processor_type": "template_extraction"}},
        ),
    ):
        result = service.get_document_template("doc-1")

    assert result["source"] == "document_artifact"
    assert result["template"]["skeleton"]["sections"][0]["title"] == "Intro"

    with (
        patch(
            "src.features.documents.application.document_metadata_service.check_document_access"
        ),
        patch(
            "src.infrastructure.document_databases.neo4j_store.fetch_template_graph",
            return_value=None,
        ),
        patch(
            "src.infrastructure.document_databases.neo4j_store.fetch_document_artifact",
            return_value=None,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            service.get_document_template("doc-1")
    assert exc.value.status_code == 404


def test_start_template_extraction_publishes_processor_job() -> None:
    from src.features.documents.application.document_service import DocumentReceiverService

    service = DocumentReceiverService()
    record = {
        "document_id": "doc-te-1",
        "document_name": "sop.docx",
        "original_file_name": "sop.docx",
        "collection_name": "Document",
        "processing": {},
        "metadata": {},
        "status": "completed",
    }
    service.get_document_record = MagicMock(return_value=record)  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    service._compute_enabled_processor_types = MagicMock(return_value=["chunking_vectorizing"])  # type: ignore[method-assign]
    service.queue_payload = MagicMock(return_value={"document_id": "doc-te-1"})  # type: ignore[method-assign]
    store = MagicMock()
    service.get_store = MagicMock(return_value=store)  # type: ignore[method-assign]
    publisher = MagicMock()

    with (
        patch("src.features.documents.application.document_metadata_service.check_document_access"),
        patch.object(
            DocumentReceiverService,
            "_processing_allowed_after_security_review",
            return_value=True,
        ),
        patch.object(DocumentReceiverService, "publisher", new=property(lambda self: publisher)),
    ):
        result = service.start_template_extraction("doc-te-1")

    assert result["queued"] is True
    assert result["processor_type"] == "template_extraction"
    published = publisher.publish_document_job.call_args[0][0]
    assert published["processor_type"] == "template_extraction"
    assert published["template_extraction"] is True
    assert "template_extraction" in published["enabled_processor_types"]


def test_start_template_extraction_allows_stale_human_review_after_allow() -> None:
    from src.features.documents.application.document_service import DocumentReceiverService

    service = DocumentReceiverService()
    record = {
        "document_id": "doc-te-hr",
        "document_name": "sop.docx",
        "collection_name": "Document",
        "processing": {},
        "status": "completed",
        "metadata": {
            "requires_human_review": True,
            "security_scan": {"status": "human_review", "dlp_decision": "HUMAN_REVIEW"},
        },
    }
    service.get_document_record = MagicMock(return_value=record)  # type: ignore[method-assign]
    service._job_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    service._compute_enabled_processor_types = MagicMock(return_value=["chunking_vectorizing"])  # type: ignore[method-assign]
    service.queue_payload = MagicMock(return_value={"document_id": "doc-te-hr"})  # type: ignore[method-assign]
    service.get_store = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    publisher = MagicMock()

    with (
        patch("src.features.documents.application.document_metadata_service.check_document_access"),
        patch.object(
            DocumentReceiverService,
            "_processing_allowed_after_security_review",
            return_value=True,
        ),
        patch.object(DocumentReceiverService, "publisher", new=property(lambda self: publisher)),
    ):
        result = service.start_template_extraction("doc-te-hr")

    assert result["queued"] is True
    publisher.publish_document_job.assert_called_once()


def test_start_template_extraction_blocks_undecided_human_review() -> None:
    from fastapi import HTTPException

    from src.features.documents.application.document_service import DocumentReceiverService

    service = DocumentReceiverService()
    service.get_document_record = MagicMock(  # type: ignore[method-assign]
        return_value={"document_id": "doc-te-block", "metadata": {}, "processing": {}}
    )

    with (
        patch("src.features.documents.application.document_metadata_service.check_document_access"),
        patch.object(
            DocumentReceiverService,
            "_processing_allowed_after_security_review",
            return_value=False,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            service.start_template_extraction("doc-te-block")

    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "human_review_pending"
