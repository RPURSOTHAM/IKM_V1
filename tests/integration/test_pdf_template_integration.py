import pytest
pytest.skip("legacy docx_template/pdf_template removed; replaced by template_compliance", allow_module_level=True)

"""
Integration tests for the PDF template extraction pipeline.

Verifies:
 1. extract_template() produces the expected JSON schema
 2. Headings, tables, and hierarchy are detected
 3. store_template_graph() creates Heading / Table / Image relationships
 4. TemplateExtractionProcessor routes .pdf to the PDF extractor
 5. ChunkingVectorizingProcessor still uses PdfLoader / load_document_blocks unchanged
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Create a PDF with headings and a grid table for template extraction."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        pytest.skip("reportlab not installed")

    out = tmp_path / "test_sample.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter)
    styles = getSampleStyleSheet()
    data = [["Header A", "Header B"], ["r1c1", "r1c2"], ["r2c1", "r2c2"]]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    story = [
        Paragraph("1. Purpose", styles["Heading1"]),
        Spacer(1, 8),
        Paragraph("Body under purpose.", styles["Normal"]),
        Spacer(1, 8),
        Paragraph("1.1 Scope", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph("Body under scope.", styles["Normal"]),
        Spacer(1, 12),
        table,
    ]
    doc.build(story)
    return out


def test_extract_template_schema(sample_pdf: Path):
    from src.features.document_processing.extractors.pdf_template import extract_template

    result = extract_template(sample_pdf)

    assert isinstance(result, dict)
    for key in ("source_file", "metadata", "outline", "possible_headings", "tables", "document_tree"):
        assert key in result, f"Missing key: {key}"

    assert result["source_file"] == "test_sample.pdf"
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["outline"], list)
    assert isinstance(result["tables"], list)
    assert isinstance(result["document_tree"], list)


def test_extract_template_headings_and_tables(sample_pdf: Path):
    from src.features.document_processing.extractors.pdf_template import extract_template

    result = extract_template(sample_pdf)

    outline_texts = [item.get("text", "") for item in result["outline"]]
    assert any("Purpose" in t for t in outline_texts), f"outline={outline_texts}"
    assert any(item.get("detection_method") == "pdf_loader" for item in result["outline"])

    assert len(result["tables"]) >= 1
    table = result["tables"][0]
    for key in ("table_index", "rows", "columns", "header_row", "is_nested", "table_caption", "page"):
        assert key in table

    def _collect_types(nodes):
        for n in nodes:
            yield n.get("type")
            yield from _collect_types(n.get("children") or [])

    types = list(_collect_types(result["document_tree"]))
    assert "heading" in types
    assert "table" in types


def test_extract_template_hierarchy(sample_pdf: Path):
    from src.features.document_processing.extractors.pdf_template import extract_template

    result = extract_template(sample_pdf)
    tree = result["document_tree"]
    assert len(tree) >= 1

    def _collect_heading_texts(nodes):
        for n in nodes:
            if n.get("type") == "heading":
                yield n["text"]
                yield from _collect_heading_texts(n.get("children") or [])

    texts = list(_collect_heading_texts(tree))
    assert any("Purpose" in t for t in texts), f"tree texts={texts}"


def test_store_template_graph_creates_relationships():
    from src.infrastructure.document_databases.neo4j_store import store_template_graph

    template_data = {
        "metadata": {"title": "Sample"},
        "document_tree": [
            {
                "type": "heading",
                "text": "1. Purpose",
                "level": 1,
                "style": "title",
                "detection_method": "pdf_loader",
                "children": [
                    {
                        "type": "table",
                        "table_index": 1,
                        "is_nested": False,
                        "rows": 2,
                        "columns": 2,
                        "table_caption": "",
                        "header_row": ["A", "B"],
                        "children": [],
                    },
                    {
                        "type": "image",
                        "image_index": 1,
                        "page": 1,
                        "position": {"x0": 0, "top": 10, "x1": 100, "bottom": 50},
                        "width": 100,
                        "height": 40,
                        "caption": "Figure 1",
                        "children": [],
                    },
                ],
            }
        ],
    }

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
        location = store_template_graph(
            document_id="pdf-neo4j-test",
            repository_id="repo-001",
            template_data=template_data,
        )

    assert location == "neo4j://DocumentTemplate/pdf-neo4j-test"
    assert mock_tx.run.call_count >= 3
    cypher_text = " ".join(str(call) for call in mock_tx.run.call_args_list)
    assert "DocumentTemplate" in cypher_text
    assert "HAS_HEADING" in cypher_text or "HAS_TABLE" in cypher_text or "HAS_IMAGE" in cypher_text or "HAS_CHILD" in cypher_text
    mock_session.run.assert_called()


def test_processor_routes_pdf(sample_pdf: Path):
    from src.features.document_processing.core.contract import ProcessRequest
    from src.features.document_processing.processors.template_extraction import TemplateExtractionProcessor

    processor = TemplateExtractionProcessor()
    request = ProcessRequest(
        document_id="pdf-route-test",
        document_name="test_sample.pdf",
        repository_id="repo-001",
    )

    with patch(
        "src.features.document_processing.extractors.pdf_template.extract_template"
    ) as mock_extract, patch(
        "src.features.document_processing.processors.template_extraction.store_template_graph"
    ) as mock_store:
        mock_extract.return_value = {
            "source_file": "test_sample.pdf",
            "metadata": {},
            "outline": [{"text": "1. Purpose", "level": 1}],
            "possible_headings": [],
            "tables": [{"table_index": 1}],
            "document_tree": [],
        }
        mock_store.return_value = "neo4j://DocumentTemplate/pdf-route-test"

        result = processor.run(
            request,
            sample_pdf,
            set_status=MagicMock(),
            check_stop=lambda: False,
        )

    mock_extract.assert_called_once()
    mock_store.assert_called_once()
    assert result.document_id == "pdf-route-test"
    assert "DocumentTemplate" in result.result_location


def test_chunking_path_unaffected():
    """Chunking module must keep using load_document_blocks, not pdf_template."""
    import inspect

    import src.features.document_processing.processors.chunking_vectorizing as chunking_mod

    source = inspect.getsource(chunking_mod)
    assert "load_document_blocks" in source
    assert "pdf_template" not in source
    assert "extract_template" not in source

    # PdfLoader itself must remain the ingestion parser entry for PDFs.
    import src.features.document_processing.loaders.document_text as document_text

    doc_source = inspect.getsource(document_text)
    assert "PdfLoader" in doc_source
    assert "pdf_template" not in doc_source
