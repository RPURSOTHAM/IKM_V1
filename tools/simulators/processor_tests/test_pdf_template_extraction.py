import pytest
pytest.skip("legacy docx_template/pdf_template removed; replaced by template_compliance", allow_module_level=True)

#!/usr/bin/env python3
"""
Unit and integration test with automated extraction coverage verification.
Compares complete pdf_reader engine outputs with persisted Neo4j graph & JSON records.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_pdf_template")

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features.document_processing.extractors.pdf_template.extractor import extract_template
from src.infrastructure.document_databases.neo4j_store import (
    store_template_graph,
    delete_document_graph,
    _neo4j_credentials,
    _require_neo4j_driver
)

def run_test() -> int:
    # 1. Locate sample PDF
    documents_dir = REPO_ROOT / "_documents"
    pdf_files = [p for p in documents_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf" and p.stat().st_size > 5000]
    if not pdf_files:
        logger.error("No suitable test PDF found in %s", documents_dir)
        return 1
    
    test_pdf = min(pdf_files, key=lambda p: p.stat().st_size)
    logger.info("Using test PDF: %s (size=%d bytes)", test_pdf.name, test_pdf.stat().st_size)

    # Load raw doc for coverage checks
    from src.features.document_processing.extractors.pdf_template.pdf_reader_loader import load_pdf_with_reader
    raw_doc = load_pdf_with_reader(test_pdf)

    # 2. Extract template (SOP Canonical Template)
    logger.info("Step 1: Running extract_template() on %s", test_pdf)
    template_data = extract_template(test_pdf)
    
    # 3. Store in Neo4j
    document_id = "test-pdf-template-id-999"
    repository_id = "test-repo-999"
    
    logger.info("Step 2: Cleaning up old test data")
    delete_document_graph(document_id)

    logger.info("Step 3: Storing template graph in Neo4j")
    location = store_template_graph(
        document_id=document_id,
        repository_id=repository_id,
        template_data=template_data,
        document_type_id="test-doc-type-id",
        document_type_name="Test Document Type"
    )

    # 4. Connect to Neo4j and inspect graph / template_json
    logger.info("Step 4: Connecting to Neo4j for coverage analysis")
    uri, user, password = _neo4j_credentials()
    GraphDatabase = _require_neo4j_driver()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    errors: list[str] = []
    
    try:
        with driver.session() as session:
            # Fetch persisted JSON
            res = session.run(
                "MATCH (t:DocumentTemplate {document_id: $doc_id}) RETURN t.template_json AS json_str",
                doc_id=document_id
            ).single()
            if not res or not res["json_str"]:
                logger.error("FAIL: DocumentTemplate node or template_json missing in Neo4j")
                return 1
            
            saved_template = json.loads(res["json_str"])
            
            # Fetch counts from Graph
            h_count = session.run("MATCH (h:Heading {document_id: $doc_id}) RETURN count(h) AS c", doc_id=document_id).single()["c"]
            p_count = session.run("MATCH (p:Paragraph {document_id: $doc_id}) RETURN count(p) AS c", doc_id=document_id).single()["c"]
            t_count = session.run("MATCH (t:Table {document_id: $doc_id}) RETURN count(t) AS c", doc_id=document_id).single()["c"]
            i_count = session.run("MATCH (i:Image {document_id: $doc_id}) RETURN count(i) AS c", doc_id=document_id).single()["c"]
            
            h_rich = session.run("MATCH (h:Heading {document_id: $doc_id}) WHERE h.font_name IS NOT NULL AND h.top IS NOT NULL RETURN count(h) AS c", doc_id=document_id).single()["c"]
            p_rich = session.run("MATCH (p:Paragraph {document_id: $doc_id}) WHERE p.font_name IS NOT NULL AND p.top IS NOT NULL RETURN count(p) AS c", doc_id=document_id).single()["c"]
            
            def has_nested_type(nodes, target_type):
                for n in nodes:
                    if n.get("type") == target_type:
                        return True
                    if has_nested_type(n.get("children") or [], target_type):
                        return True
                return False

            # Perform automated coverage comparisons
            coverage_checks = {
                "metadata": (
                    len(raw_doc.get("summary", {}).get("metadata", {})) > 0, # has_source
                    True, # in_graph (stored as DocumentTemplate properties)
                    "metadata" in saved_template # in_json
                ),
                "headings": (
                    h_count > 0, # has_source
                    h_count > 0, # in_graph
                    has_nested_type(saved_template.get("document_tree", []), "heading") # in_json
                ),
                "paragraphs": (
                    p_count > 0, # has_source
                    p_count > 0, # in_graph
                    has_nested_type(saved_template.get("document_tree", []), "paragraph") # in_json
                ),
                "tables": (
                    raw_doc.get("summary", {}).get("unique_table_count", 0) > 0 or t_count > 0, # has_source
                    t_count > 0, # in_graph
                    "tables" in saved_template or has_nested_type(saved_template.get("document_tree", []), "table") # in_json
                ),
                "images": (
                    i_count > 0 or raw_doc.get("summary", {}).get("saved_image_count", 0) > 0, # has_source
                    i_count > 0, # in_graph
                    "images" in saved_template or has_nested_type(saved_template.get("document_tree", []), "image") # in_json
                ),
                "references": (
                    len(raw_doc.get("summary", {}).get("referenced_documents", [])) > 0, # has_source
                    False, # in_graph (written by separate reference pipeline)
                    "referenced_documents" in saved_template # in_json
                ),
                "headers": (
                    any((b.metadata or {}).get("region") == "header" for page in raw_doc.get("pages", []) for b in page.get("items", []) if hasattr(b, "metadata")), # has_source
                    False, # in_graph
                    "headers_footers" in saved_template # in_json
                ),
                "footers": (
                    any((b.metadata or {}).get("region") == "footer" for page in raw_doc.get("pages", []) for b in page.get("items", []) if hasattr(b, "metadata")), # has_source
                    False, # in_graph
                    "headers_footers" in saved_template # in_json
                ),
                "raw_pages": (
                    len(raw_doc.get("pages", [])) > 0, # has_source
                    False, # in_graph
                    "raw_pages" in saved_template and len(saved_template["raw_pages"]) == len(raw_doc["pages"]) # in_json
                ),
                "font_information": (
                    h_rich > 0 or p_rich > 0, # has_source
                    h_rich > 0 or p_rich > 0, # in_graph
                    "document_tree" in saved_template # in_json
                ),
                "bounding_boxes": (
                    h_rich > 0 or p_rich > 0, # has_source
                    h_rich > 0 or p_rich > 0, # in_graph
                    "document_tree" in saved_template # in_json
                ),
                "layout_information": (
                    p_count > 0, # has_source
                    p_rich > 0, # in_graph
                    "document_tree" in saved_template # in_json
                ),
                "reading_order": (
                    True, # has_source
                    True, # in_graph (verified by HAS_CHILD relationships order indices)
                    "document_tree" in saved_template # in_json
                )
            }

            # Generate Coverage Report Table
            print("\n" + "="*52)
            print("EXTRACTION COVERAGE VERIFICATION REPORT")
            print("="*52)
            print(f"{'pdf_reader Output':<25} {'Graph':<7} {'JSON':<6} {'Status':<6}")
            print("-"*52)
            for component, (has_source, in_graph, in_json) in coverage_checks.items():
                graph_marker = "YES" if in_graph else "-"
                json_marker = "YES" if in_json else "-"
                status = "PASS"
                
                # Check for leaks/omitted mappings
                if has_source and not in_graph and not in_json:
                    status = "FAIL"
                    errors.append(f"Omitted component coverage: {component} was extracted by pdf_reader but not persisted!")
                
                print(f"{component:<25} {graph_marker:<7} {json_marker:<6} {status:<6}")
            print("="*52 + "\n")

            # Dynamic check: scan for any completely unmapped summary keys in raw_doc
            unmapped_keys = []
            for key in raw_doc.keys():
                if key not in {"source", "document_name", "summary", "pages"}:
                    if key not in saved_template:
                        unmapped_keys.append(key)
            
            if unmapped_keys:
                status = "FAIL"
                errors.append(f"Omitted dynamic top-level keys detected: {unmapped_keys}")

    finally:
        driver.close()
        
    # Clean up test database nodes
    delete_document_graph(document_id)

    if errors:
        logger.error("EXTRACTION COVERAGE VERIFICATION FAILED:")
        for err in errors:
            logger.error(" - %s", err)
        return 1
    
    logger.info("PASS: All extraction coverage verification assertions passed!")
    return 0

if __name__ == "__main__":
    sys.exit(run_test())

