"""Unit tests for SOP structure enrichment (procedure steps, roles, approvals, revisions, images)."""

from __future__ import annotations

from src.features.document_processing.extractors.sop_structure import enrich_template
from src.features.document_processing.extractors.sop_structure.approvals import extract_approvals
from src.features.document_processing.extractors.sop_structure.images import extract_images
from src.features.document_processing.extractors.sop_structure.procedure_steps import extract_procedure_steps
from src.features.document_processing.extractors.sop_structure.responsibilities import extract_responsibilities
from src.features.document_processing.extractors.sop_structure.revisions import extract_revisions


def _block(text: str, *, style: str = "Normal", page: int | None = 1, level: int | None = None):
    return {
        "text": text,
        "style": style,
        "page": page,
        "block_type": "text",
        "component_type": "heading" if style.lower().startswith("heading") else "paragraph",
        "bold": False,
        "level": level,
    }


def test_procedure_step_extraction_decimal_under_procedure() -> None:
    template = {
        "document_tree": [
            {
                "type": "heading",
                "text": "5 Procedure",
                "level": 1,
                "children": [],
            }
        ],
        "outline": [{"text": "5 Procedure", "level": 1}],
        "tables": [],
    }
    blocks = [
        _block("5 Procedure", style="Heading 1", level=1),
        _block("5.1 Switch ON machine"),
        _block("5.2 Verify pressure"),
        _block("5.3 Record observations"),
    ]
    steps = extract_procedure_steps(template, blocks)
    assert [s["step_number"] for s in steps] == ["5.1", "5.2", "5.3"]
    assert steps[0]["text"] == "Switch ON machine"
    assert steps[0]["detection_method"] == "numbered_decimal"
    assert steps[0]["section_title"] == "5 Procedure"
    assert steps[0]["order"] == 1


def test_nested_and_numbered_lettered_roman_steps() -> None:
    template = {
        "document_tree": [{"type": "heading", "text": "Procedure", "level": 1, "children": []}],
        "outline": [{"text": "Procedure", "level": 1}],
        "tables": [],
    }
    blocks = [
        _block("Procedure", style="Heading 1", level=1),
        _block("1. Prepare materials"),
        _block("2. Run the batch"),
        _block("A. Collect sample"),
        _block("B. Label sample"),
        _block("I. Final check"),
        _block("II. Sign log"),
    ]
    steps = extract_procedure_steps(template, blocks)
    assert len(steps) >= 6
    methods = {s["detection_method"] for s in steps}
    assert "numbered_decimal" in methods
    assert "lettered" in methods
    assert "roman" in methods
    # NEXT_STEP order is sequential
    assert [s["order"] for s in steps] == list(range(1, len(steps) + 1))


def test_procedure_steps_empty_when_no_structures() -> None:
    template = {
        "document_tree": [{"type": "heading", "text": "Purpose", "level": 1, "children": []}],
        "outline": [{"text": "Purpose", "level": 1}],
        "tables": [],
    }
    blocks = [
        _block("Purpose", style="Heading 1", level=1),
        _block("This document describes the purpose of the SOP."),
    ]
    assert extract_procedure_steps(template, blocks) == []


def test_responsibilities_extraction() -> None:
    template = {
        "document_tree": [
            {"type": "heading", "text": "4 Responsibilities", "level": 1, "children": []}
        ],
        "outline": [{"text": "4 Responsibilities", "level": 1}],
        "tables": [],
    }
    blocks = [
        _block("4 Responsibilities", style="Heading 1", level=1),
        _block("QA: Approve batch records"),
        _block("QC - Perform in-process checks"),
        _block("Production: Execute the procedure"),
        _block("Engineering: Maintain equipment"),
        _block("Warehouse: Issue materials"),
    ]
    roles = extract_responsibilities(template, blocks)
    role_names = {r["role"].upper() for r in roles}
    assert {"QA", "QC", "PRODUCTION", "ENGINEERING", "WAREHOUSE"} <= role_names
    qa = next(r for r in roles if r["role"].upper() == "QA")
    assert "Approve" in qa["description"]


def test_approval_block_extraction() -> None:
    template = {"document_tree": [], "outline": [], "tables": []}
    blocks = [
        _block("Prepared By: Alice Smith"),
        _block("Reviewed By: Bob Jones"),
        _block("Approved By: Carol Lee"),
        _block("Approved Date: 2024-01-15"),
    ]
    approvals = extract_approvals(template, blocks)
    assert len(approvals) == 1
    assert approvals[0]["prepared_by"] == "Alice Smith"
    assert approvals[0]["reviewed_by"] == "Bob Jones"
    assert approvals[0]["approved_by"] == "Carol Lee"
    assert approvals[0]["approved_date"] == "2024-01-15"


def test_revision_history_from_table() -> None:
    template = {
        "document_tree": [],
        "outline": [],
        "tables": [
            {
                "table_index": 1,
                "table_caption": "Revision History",
                "header_row": ["Version", "Date", "Description", "Author"],
                "row_data": [
                    ["1.0", "2023-01-01", "Initial release", "QA"],
                    ["1.1", "2024-06-01", "Updated procedure steps", "QC"],
                ],
                "page": 1,
            }
        ],
    }
    revisions = extract_revisions(template)
    assert len(revisions) == 2
    assert revisions[0]["version"] == "1.0"
    assert revisions[1]["description"] == "Updated procedure steps"
    assert revisions[1]["author"] == "QC"


def test_images_and_figure_captions() -> None:
    template = {
        "document_tree": [
            {
                "type": "image",
                "image_index": 0,
                "page": 2,
                "width": 100,
                "height": 80,
                "caption": "",
                "children": [],
            }
        ],
        "outline": [{"text": "Equipment", "level": 1}],
        "tables": [],
        "images": [],
    }
    blocks = [
        _block("Equipment", style="Heading 1", level=1, page=2),
        {
            "text": "",
            "page": 2,
            "style": "",
            "block_type": "image",
            "component_type": "image",
            "width": 100,
            "height": 80,
            "image_index": 0,
        },
        _block("Figure 1: Equipment Layout", page=2),
    ]
    images = extract_images(template, blocks)
    assert images
    assert any(i.get("figure_number") == "1" for i in images)
    assert any("Equipment Layout" in (i.get("caption") or "") for i in images)


def test_enrich_template_empty_document() -> None:
    enriched = enrich_template(
        {"document_tree": [], "outline": [], "tables": [], "metadata": {}},
        [],
    )
    assert enriched["procedure_steps"] == []
    assert enriched["responsibilities"] == []
    assert enriched["approvals"] == []
    assert enriched["revisions"] == []
    assert enriched["images"] == []


def test_enrich_template_preserves_existing_tree() -> None:
    tree = [{"type": "heading", "text": "Purpose", "level": 1, "children": []}]
    enriched = enrich_template(
        {
            "document_tree": tree,
            "outline": [{"text": "Purpose", "level": 1}],
            "tables": [{"table_index": 1, "header_row": ["A", "B"], "row_data": [["1", "2"]]}],
            "metadata": {"title": "SOP"},
        },
        [_block("Purpose", style="Heading 1", level=1), _block("Body text only.")],
    )
    assert enriched["document_tree"] == tree
    assert enriched["tables"][0]["header_row"] == ["A", "B"]
    assert enriched["procedure_steps"] == []
    assert enriched["revisions"] == []  # not a revision table


def test_store_template_graph_writes_sop_nodes() -> None:
    """Integration-style unit test with mocked Neo4j driver capturing Cypher labels."""
    from unittest.mock import MagicMock, patch

    from src.infrastructure.document_databases.neo4j_store import store_template_graph

    created_labels: list[str] = []

    class _Result:
        def __init__(self, payload=None):
            self._payload = payload or {"node_id": "eid-1", "c": 1}

        def single(self):
            return self._payload

    def _run(cypher, **params):
        for label in (
            "ProcedureStep",
            "Responsibility",
            "Approval",
            "Revision",
            "Heading",
            "Table",
            "Image",
            "DocumentTemplate",
        ):
            if f":{label}" in cypher and "CREATE" in cypher:
                created_labels.append(label)
        if "RETURN count(d)" in cypher or "RETURN count(d) AS c" in cypher:
            return _Result({"c": 1})
        if "RETURN elementId" in cypher:
            return _Result({"node_id": f"eid-{len(created_labels)}"})
        return _Result({"c": 1})

    mock_tx = MagicMock()
    mock_tx.run.side_effect = _run

    mock_session = MagicMock()
    mock_session.__enter__ = lambda s: s
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.execute_write = lambda fn: fn(mock_tx)
    mock_session.run.side_effect = _run

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    mock_gdb = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    template_data = {
        "metadata": {},
        "document_tree": [
            {"type": "heading", "text": "5 Procedure", "level": 1, "style": "Heading 1", "children": []}
        ],
        "procedure_steps": [
            {
                "step_number": "5.1",
                "text": "Switch ON",
                "order": 1,
                "page": 1,
                "section_title": "5 Procedure",
                "section_level": 1,
                "detection_method": "numbered_decimal",
            }
        ],
        "responsibilities": [
            {"role": "QA", "description": "Approve", "page": 1, "section_title": "Responsibilities"}
        ],
        "approvals": [
            {
                "prepared_by": "A",
                "reviewed_by": "B",
                "approved_by": "C",
                "approved_date": "2024-01-01",
                "review_date": "",
                "prepared_date": "",
            }
        ],
        "revisions": [
            {"version": "1.0", "date": "2024-01-01", "description": "Initial", "author": "QA"}
        ],
        "images": [
            {
                "image_index": 0,
                "page": 2,
                "caption": "Equipment Layout",
                "figure_number": "1",
                "width": 10,
                "height": 10,
                "image_type": "figure_caption",
                "section_title": "5 Procedure",
            }
        ],
        "tables": [],
    }

    with patch.dict("sys.modules", {"neo4j": MagicMock(GraphDatabase=mock_gdb)}):
        location = store_template_graph(
            document_id="sop-doc-1",
            repository_id="repo-1",
            template_data=template_data,
        )

    assert location == "neo4j://DocumentTemplate/sop-doc-1"
    assert "ProcedureStep" in created_labels
    assert "Responsibility" in created_labels
    assert "Approval" in created_labels
    assert "Revision" in created_labels
    assert "Image" in created_labels
    assert "Heading" in created_labels
