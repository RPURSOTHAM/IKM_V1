import pytest
pytest.skip("legacy docx_template/pdf_template removed; replaced by template_compliance", allow_module_level=True)

"""
Integration test for the DOCX template extraction pipeline.

Verifies:
 1. extract_template() produces the expected JSON schema
 2. store_docx_template_graph() creates Heading / Table / HAS_CHILD nodes
 3. The default (non-DOCX) template extraction path is unchanged
 4. The existing RAG indexing pipeline is not affected
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------
# 1. Extractor unit-level: produces correct schema
# --------------------------------------------------------------------------

@pytest.fixture()
def sample_docx(tmp_path: Path) -> Path:
    """Create a minimal DOCX with headings and a table for testing."""
    try:
        from docx import Document
    except ImportError:
        pytest.skip("python-docx not installed")
    doc = Document()
    doc.add_heading("Document Title", level=0)
    doc.add_heading("Section One", level=1)
    doc.add_paragraph("Body text under section one.")
    doc.add_heading("Subsection 1.1", level=2)
    table = doc.add_table(rows=3, cols=2)
    table.rows[0].cells[0].text = "Col A"
    table.rows[0].cells[1].text = "Col B"
    table.rows[1].cells[0].text = "val1"
    table.rows[1].cells[1].text = "val2"
    doc.add_heading("Section Two", level=1)
    out = tmp_path / "test_sample.docx"
    doc.save(str(out))
    return out


def test_extract_template_schema(sample_docx: Path):
    from src.features.document_processing.extractors.docx_template import extract_template

    result = extract_template(sample_docx)

    assert isinstance(result, dict)
    for key in (
        "source_file",
        "metadata",
        "outline",
        "possible_headings",
        "tables",
        "document_tree",
        "procedure_steps",
        "responsibilities",
        "approvals",
        "revisions",
        "images",
    ):
        assert key in result, f"Missing key: {key}"

    assert result["source_file"] == "test_sample.docx"
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["outline"], list)
    assert len(result["outline"]) >= 2
    assert isinstance(result["tables"], list)
    assert len(result["tables"]) >= 1
    assert isinstance(result["document_tree"], list)
    assert len(result["document_tree"]) >= 1


def test_extract_template_heading_hierarchy(sample_docx: Path):
    from src.features.document_processing.extractors.docx_template import extract_template

    result = extract_template(sample_docx)
    tree = result["document_tree"]

    def _collect_texts(nodes):
        for n in nodes:
            if n.get("type") == "heading":
                yield n["text"]
                yield from _collect_texts(n.get("children", []))

    all_texts = list(_collect_texts(tree))
    assert any("Section One" in t for t in all_texts), f"Heading texts: {all_texts}"
    assert len(tree) >= 1


# --------------------------------------------------------------------------
# 2. Neo4j storage adapter
# --------------------------------------------------------------------------

@pytest.fixture()
def _mock_neo4j_driver():
    """Patch GraphDatabase.driver so tests run without a live Neo4j instance."""
    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: 1  # fake node id

    mock_result = MagicMock()
    mock_result.single.return_value = mock_record

    mock_tx = MagicMock()
    mock_tx.run.return_value = mock_result

    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.execute_write = lambda fn: fn(mock_tx)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("src.infrastructure.document_databases.neo4j_store.GraphDatabase", create=True) as gdb:
        # neo4j is imported lazily inside the function, so we patch the local import
        import importlib
        import src.infrastructure.document_databases.neo4j_store as mod

        with patch.dict("sys.modules", {"neo4j": MagicMock(GraphDatabase=MagicMock(driver=MagicMock(return_value=mock_driver)))}):
            # Re-import to pick up the mock
            original = mod.store_docx_template_graph.__module__
            yield mock_tx, mock_driver


def test_store_docx_template_graph_creates_nodes(sample_docx: Path):
    """Verify Cypher statements are issued for Heading and Table nodes."""
    from src.features.document_processing.extractors.docx_template import extract_template

    template_data = extract_template(sample_docx)

    mock_record = MagicMock()
    mock_record.__getitem__ = lambda self, key: 42

    mock_result = MagicMock()
    mock_result.single.return_value = mock_record

    mock_tx = MagicMock()
    mock_tx.run.return_value = mock_result

    verify_record = MagicMock()
    verify_record.__getitem__ = lambda self, key: 1
    verify_result = MagicMock()
    verify_result.single.return_value = verify_record

    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.execute_write = lambda fn: fn(mock_tx)
    mock_session.run.return_value = verify_result

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    mock_gdb = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    with patch.dict("sys.modules", {"neo4j": MagicMock(GraphDatabase=mock_gdb)}):
        from src.infrastructure.document_databases.neo4j_store import store_docx_template_graph

        location = store_docx_template_graph(
            document_id="test-doc-001",
            repository_id="repo-001",
            template_data=template_data,
        )

    assert location == "neo4j://DocumentTemplate/test-doc-001"
    assert mock_tx.run.call_count >= 3  # MERGE DocumentTemplate, clear children, headings/tables, verify

    cypher_calls = [str(call) for call in mock_tx.run.call_args_list]
    cypher_text = " ".join(cypher_calls)
    assert "DocumentTemplate" in cypher_text
    assert "MERGE" in cypher_text
    assert "HAS_HEADING" in cypher_text or "HAS_TABLE" in cypher_text or "HAS_CHILD" in cypher_text
    mock_session.run.assert_called()  # post-commit verification


# --------------------------------------------------------------------------
# 3. Processor routing: .docx goes to DOCX extractor, others use default
# --------------------------------------------------------------------------

def test_processor_routes_docx(sample_docx: Path):
    """TemplateExtractionProcessor must call extract_template for .docx files."""
    from src.features.document_processing.processors.template_extraction import TemplateExtractionProcessor
    from src.features.document_processing.core.contract import ProcessRequest

    processor = TemplateExtractionProcessor()
    request = ProcessRequest(
        document_id="docx-route-test",
        document_name="test_sample.docx",
        repository_id="repo-001",
    )

    with patch(
        "src.features.document_processing.extractors.docx_template.extract_template"
    ) as mock_extract, patch(
        "src.features.document_processing.processors.template_extraction.store_template_graph"
    ) as mock_store:
        mock_extract.return_value = {
            "source_file": "test_sample.docx",
            "metadata": {},
            "outline": [],
            "possible_headings": [],
            "tables": [],
            "document_tree": [],
        }
        mock_store.return_value = "neo4j://DocumentTemplate/docx-route-test"

        result = processor.run(
            request,
            sample_docx,
            set_status=MagicMock(),
            check_stop=lambda: False,
        )

    mock_extract.assert_called_once()
    mock_store.assert_called_once()
    assert result.document_id == "docx-route-test"
    assert "DocumentTemplate" in result.result_location


def test_processor_routes_non_template_to_default(tmp_path: Path):
    """Non-DOCX/PDF files must use the existing skeleton path."""
    from src.features.document_processing.processors.template_extraction import TemplateExtractionProcessor
    from src.features.document_processing.core.contract import ProcessRequest

    fake_txt = tmp_path / "test.txt"
    fake_txt.write_text("Some heading\nBody text\n", encoding="utf-8")

    processor = TemplateExtractionProcessor()
    request = ProcessRequest(
        document_id="txt-route-test",
        document_name="test.txt",
        repository_id="repo-001",
    )

    mock_block = MagicMock()
    mock_block.text = "Some heading"
    mock_block.bold = True
    mock_block.font_size = 14
    mock_block.style = "Heading 1"
    mock_block.page = 1

    with patch(
        "src.features.document_processing.processors.template_extraction.load_document_blocks",
        return_value=([mock_block], {"pages": 1}),
    ), patch(
        "src.features.document_processing.processors.template_extraction.store_document_graph",
        return_value="neo4j://DocumentArtifact/txt-route-test/template_extraction",
    ) as mock_store:
        result = processor.run(
            request,
            fake_txt,
            set_status=MagicMock(),
            check_stop=lambda: False,
        )

    mock_store.assert_called_once()
    assert result.document_id == "txt-route-test"
    assert "DocumentArtifact" in result.result_location
