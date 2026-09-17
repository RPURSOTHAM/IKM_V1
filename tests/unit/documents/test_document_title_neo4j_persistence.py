"""Unit tests for document_title persistence on Neo4j Document / template nodes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.infrastructure.document_databases.neo4j_store import (
    resolve_document_title_from_payload,
    store_document_graph,
    store_template_graph,
)


def test_resolve_document_title_from_payload_top_level_and_nested() -> None:
    assert resolve_document_title_from_payload({"document_title": "Alpha Guide"}) == "Alpha Guide"
    assert resolve_document_title_from_payload({"title": "Beta Guide"}) == "Beta Guide"
    assert (
        resolve_document_title_from_payload(
            {"fields": [{"field_name": "document_title", "value": "Gamma Guide"}]}
        )
        == "Gamma Guide"
    )
    assert (
        resolve_document_title_from_payload({"metadata": {"Title": "Delta Guide"}})
        == "Delta Guide"
    )
    assert resolve_document_title_from_payload({"fields": {}}) is None


def test_store_document_graph_sets_document_title_param() -> None:
    session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    graph_db = MagicMock()
    graph_db.driver.return_value = driver

    with (
        patch(
            "src.infrastructure.document_databases.neo4j_store._require_neo4j_driver",
            return_value=graph_db,
        ),
        patch(
            "src.infrastructure.document_databases.neo4j_store._neo4j_credentials",
            return_value=("bolt://localhost:7687", "neo4j", "password"),
        ),
    ):
        store_document_graph(
            document_id="doc-1",
            repository_id="repo-1",
            processor_type="key_field_extraction",
            payload={
                "fields": [{"field_name": "document_title", "value": "Cleaning SOP"}],
            },
        )

    assert session.run.called
    kwargs = session.run.call_args.kwargs
    assert kwargs.get("document_title") == "Cleaning SOP"
    assert "d.document_title" in session.run.call_args.args[0]
    assert "a.document_title" in session.run.call_args.args[0]


def test_store_template_graph_sets_document_title_param() -> None:
    session = MagicMock()
    verify_row = {"c": 1}
    session.run.return_value.single.return_value = verify_row

    def _execute_write(fn):
        tx = MagicMock()
        tx.run.return_value.single.return_value = verify_row
        return fn(tx)

    session.execute_write.side_effect = _execute_write
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    graph_db = MagicMock()
    graph_db.driver.return_value = driver

    captured: dict = {}

    def _capture_write(fn):
        tx = MagicMock()

        def _tx_run(cypher, **params):
            if "DocumentTemplate" in cypher and "MERGE (doc:Document" in cypher:
                captured["cypher"] = cypher
                captured["params"] = params
            result = MagicMock()
            result.single.return_value = verify_row
            return result

        tx.run.side_effect = _tx_run
        return fn(tx)

    session.execute_write.side_effect = _capture_write

    with (
        patch(
            "src.infrastructure.document_databases.neo4j_store._require_neo4j_driver",
            return_value=graph_db,
        ),
        patch(
            "src.infrastructure.document_databases.neo4j_store._neo4j_credentials",
            return_value=("bolt://localhost:7687", "neo4j", "password"),
        ),
        patch(
            "src.features.references.graph.reference_graph_bridge.persist_template_references",
            return_value=None,
        ),
    ):
        store_template_graph(
            document_id="doc-2",
            repository_id="repo-1",
            template_data={
                "metadata": {"title": "Equipment Qualification Protocol"},
                "document_tree": [],
            },
        )

    assert captured["params"]["document_title"] == "Equipment Qualification Protocol"
    assert "doc.document_title" in captured["cypher"]
    assert "t.document_title" in captured["cypher"]
