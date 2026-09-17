"""Neo4j persistence for metadata, template, and reference extraction processors."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DOCUMENT_TITLE_KEYS = (
    "document_title",
    "title",
    "Title",
    "Document Title",
    "documentTitle",
)


def _coerce_title_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, dict):
        for nested_key in ("value", "text", "title", "document_title"):
            nested = _coerce_title_value(value.get(nested_key))
            if nested:
                return nested
        return None
    cleaned = str(value).strip()
    return cleaned or None


def resolve_document_title_from_payload(payload: dict[str, Any] | None) -> str | None:
    """Extract a document title from a generic payload/template metadata dict."""
    if not isinstance(payload, dict):
        return None

    for key in _DOCUMENT_TITLE_KEYS:
        title = _coerce_title_value(payload.get(key))
        if title:
            return title

    fields = payload.get("fields")
    if isinstance(fields, dict):
        for key in _DOCUMENT_TITLE_KEYS:
            title = _coerce_title_value(fields.get(key))
            if title:
                return title
    elif isinstance(fields, list):
        for item in fields:
            if not isinstance(item, dict):
                continue
            field_name = str(item.get("field_name") or item.get("name") or "").strip().lower()
            if field_name in {"document_title", "title", "document title"}:
                title = _coerce_title_value(item.get("value") if "value" in item else item)
                if title:
                    return title

    for list_key in ("extracted_fields", "key_fields"):
        items = payload.get(list_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            field_name = str(item.get("field_name") or item.get("name") or "").strip().lower()
            if field_name in {"document_title", "title", "document title"}:
                title = _coerce_title_value(item.get("value") if "value" in item else item)
                if title:
                    return title

    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata is not payload:
        return resolve_document_title_from_payload(metadata)
    return None


def _neo4j_credentials() -> tuple[str, str, str]:
    uri = (os.getenv("NEO4J_URI") or os.getenv("NEO4J_BOLT_URI") or "bolt://localhost:7687").strip()
    user = (os.getenv("NEO4J_USER") or "neo4j").strip()
    password = (os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_AUTH", "neo4j/password").split("/")[-1]).strip()
    return uri, user, password


def _require_neo4j_driver():
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "neo4j driver is not installed. Add 'neo4j>=5.0.0' to processor-service dependencies."
        ) from exc
    return GraphDatabase


def store_document_graph(
    *,
    document_id: str,
    repository_id: str | None,
    processor_type: str,
    payload: dict[str, Any],
    document_type_id: str | None = None,
    document_type_name: str | None = None,
) -> str:
    """Upsert a processor artifact node and return a stable result location string."""
    uri, user, password = _neo4j_credentials()
    location = f"neo4j://DocumentArtifact/{document_id}/{processor_type}"
    GraphDatabase = _require_neo4j_driver()

    resolved_document_type_id = str(
        document_type_id
        or payload.get("document_type_id")
        or ""
    ).strip() or None
    resolved_document_type_name = str(
        document_type_name
        or payload.get("document_type_name")
        or ""
    ).strip() or None
    document_title = resolve_document_title_from_payload(payload)

    cypher = """
    MERGE (d:Document {document_id: $document_id})
    SET d.repository_id = $repository_id,
        d.document_title = coalesce($document_title, d.document_title),
        d.updated_at = datetime()
    MERGE (a:DocumentArtifact {document_id: $document_id, processor_type: $processor_type})
    SET a.repository_id = $repository_id,
        a.document_title = coalesce($document_title, a.document_title),
        a.payload_json = $payload_json,
        a.updated_at = datetime()
    MERGE (d)-[:HAS_ARTIFACT]->(a)
    WITH d, a
    FOREACH (_ IN CASE WHEN $document_type_id IS NULL THEN [] ELSE [1] END |
      MERGE (t:DocumentType {document_type_id: $document_type_id})
      SET t.name = coalesce($document_type_name, t.name),
          t.updated_at = datetime()
      MERGE (d)-[:HAS_TYPE]->(t)
    )
    RETURN a.document_id AS document_id
    """
    params = {
        "document_id": document_id,
        "repository_id": repository_id,
        "processor_type": processor_type,
        "payload_json": json.dumps(payload, default=str),
        "document_type_id": resolved_document_type_id,
        "document_type_name": resolved_document_type_name,
        "document_title": document_title,
    }
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            logger.info(
                "Creating DocumentArtifact node document_id=%s processor_type=%s params=%s",
                document_id,
                processor_type,
                {k: v for k, v in params.items() if k != "payload_json"},
            )
            session.run(cypher, **params)
        logger.info(
            "PIPELINE_TRACE %s",
            {
                "stage": "neo4j_persistence_complete",
                "document_id": document_id,
                "processor_type": processor_type,
                "result_location": location,
                "document_type_id": resolved_document_type_id,
            },
        )
    except Exception:
        logger.exception("Neo4j write failed for DocumentArtifact document_id=%s", document_id)
        raise
    finally:
        driver.close()
    return location


def fetch_document_artifact(
    document_id: str,
    processor_type: str,
) -> dict[str, Any] | None:
    """Load a DocumentArtifact payload_json from Neo4j, or None if missing."""
    document_id = str(document_id or "").strip()
    processor_type = str(processor_type or "").strip()
    if not document_id or not processor_type:
        return None
    uri, user, password = _neo4j_credentials()
    GraphDatabase = _require_neo4j_driver()
    cypher = """
    MATCH (a:DocumentArtifact {document_id: $document_id, processor_type: $processor_type})
    RETURN a.payload_json AS payload_json, a.updated_at AS updated_at
    LIMIT 1
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            row = session.run(
                cypher,
                document_id=document_id,
                processor_type=processor_type,
            ).single()
            if not row:
                return None
            payload_raw = row.get("payload_json")
            if not payload_raw:
                return None
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            if not isinstance(payload, dict):
                return None
            payload = dict(payload)
            payload.setdefault("_artifact", {})
            if isinstance(payload["_artifact"], dict):
                payload["_artifact"]["updated_at"] = str(row.get("updated_at") or "")
                payload["_artifact"]["processor_type"] = processor_type
            return payload
    except Exception:
        logger.debug(
            "Neo4j artifact fetch failed document_id=%s processor_type=%s",
            document_id,
            processor_type,
            exc_info=True,
        )
        return None
    finally:
        driver.close()


def fetch_template_graph(document_id: str) -> dict[str, Any] | None:
    """Load the stored DocumentTemplate payload from Neo4j, or None if missing."""
    document_id = str(document_id or "").strip()
    if not document_id:
        return None
    uri, user, password = _neo4j_credentials()
    GraphDatabase = _require_neo4j_driver()
    cypher = """
    MATCH (t:DocumentTemplate {document_id: $document_id})
    RETURN t.template_json AS template_json,
           t.metadata_json AS metadata_json,
           t.document_type_id AS document_type_id,
           t.document_type_name AS document_type_name,
           t.document_title AS document_title,
           t.repository_id AS repository_id,
           t.updated_at AS updated_at
    LIMIT 1
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            row = session.run(cypher, document_id=document_id).single()
            if not row:
                return None
            raw = row.get("template_json")
            if not raw:
                return None
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(payload, dict):
                return None
            payload = dict(payload)
            meta_raw = row.get("metadata_json")
            if meta_raw and not payload.get("metadata"):
                try:
                    parsed_meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                except Exception:
                    parsed_meta = None
                if isinstance(parsed_meta, dict):
                    payload["metadata"] = parsed_meta
            payload.setdefault("_template", {})
            if isinstance(payload["_template"], dict):
                payload["_template"].update(
                    {
                        "document_id": document_id,
                        "repository_id": row.get("repository_id"),
                        "document_type_id": row.get("document_type_id"),
                        "document_type_name": row.get("document_type_name"),
                        "document_title": row.get("document_title"),
                        "updated_at": str(row.get("updated_at") or ""),
                        "result_location": f"neo4j://DocumentTemplate/{document_id}",
                    }
                )
            return payload
    except Exception:
        logger.debug(
            "Neo4j template fetch failed document_id=%s",
            document_id,
            exc_info=True,
        )
        return None
    finally:
        driver.close()


def store_template_graph(
    *,
    document_id: str,
    repository_id: str | None,
    template_data: dict[str, Any],
    document_type_id: str | None = None,
    document_type_name: str | None = None,
) -> str:
    """Persist DOCX/PDF template extractor output as a verified DocumentTemplate graph.

    Creates ``(:DocumentTemplate {document_id})`` plus Heading/Table/Image/Paragraph children
    and optional SOP enrichment nodes (ProcedureStep, Responsibility, Approval, Revision).
    Returns the result location only after a post-write verification query confirms
    the DocumentTemplate node exists. Exceptions are never swallowed.
    """
    document_id = str(document_id or "").strip()
    if not document_id:
        raise RuntimeError("store_template_graph requires a non-empty document_id.")

    uri, user, password = _neo4j_credentials()
    location = f"neo4j://DocumentTemplate/{document_id}"
    GraphDatabase = _require_neo4j_driver()

    metadata: dict[str, Any] = {}
    for k, v in (template_data.get("metadata") or {}).items():
        if k == "custom_properties":
            metadata["custom_properties"] = json.dumps(v, default=str)
        else:
            metadata[k] = v

    document_title = resolve_document_title_from_payload(
        {
            **({} if not isinstance(template_data, dict) else dict(template_data)),
            "metadata": metadata,
        }
    ) or resolve_document_title_from_payload(metadata)

    document_tree = list(template_data.get("document_tree") or [])
    node_count = _count_template_tree_nodes(document_tree)
    template_json_str = json.dumps(template_data, default=str)
    
    logger.info("Template extraction started document_id=%s", document_id)
    logger.info("Template graph nodes=%d (tree roots=%d)", node_count, len(document_tree))
    logger.info("Template JSON size: %d characters", len(template_json_str))
    logger.info(
        "Calling store_template_graph() document_id=%s repository_id=%s uri=%s",
        document_id,
        repository_id,
        uri,
    )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    counters = {
        "headings": 0,
        "tables": 0,
        "images": 0,
        "paragraphs": 0,
        "steps": 0,
        "responsibilities": 0,
        "approvals": 0,
        "revisions": 0,
    }
    try:
        with driver.session() as session:
            def _write_template(tx):
                logger.info(
                    "Creating DocumentTemplate node document_id=%s params=%s",
                    document_id,
                    {
                        "document_id": document_id,
                        "repository_id": repository_id,
                        "document_type_id": document_type_id,
                        "document_type_name": document_type_name,
                        "metadata_keys": sorted(metadata.keys()),
                        "template_json_size": len(template_json_str),
                    },
                )
                tx.run(
                    """
                    MERGE (doc:Document {document_id: $document_id})
                    ON CREATE SET doc.created_at = datetime()
                    SET doc.repository_id = $repository_id,
                        doc.document_title = coalesce($document_title, doc.document_title),
                        doc.updated_at = datetime()
                    MERGE (t:DocumentTemplate {document_id: $document_id})
                    ON CREATE SET t.created_at = datetime()
                    SET t.repository_id = $repository_id,
                        t.document_type_id = $document_type_id,
                        t.document_type_name = $document_type_name,
                        t.document_title = coalesce($document_title, t.document_title),
                        t.metadata_json = $metadata_json,
                        t.template_json = $template_json,
                        t.updated_at = datetime()
                    MERGE (doc)-[:HAS_TEMPLATE]->(t)
                    """,
                    document_id=document_id,
                    repository_id=repository_id,
                    document_type_id=document_type_id,
                    document_type_name=document_type_name,
                    document_title=document_title,
                    metadata_json=json.dumps(metadata, default=str),
                    template_json=template_json_str,
                )

                logger.info("Clearing previous template children document_id=%s", document_id)
                tx.run(
                    """
                    MATCH (t:DocumentTemplate {document_id: $document_id})
                    OPTIONAL MATCH (t)-[:HAS_HEADING|HAS_TABLE|HAS_IMAGE|HAS_PARAGRAPH|HAS_CHILD|HAS_STEP|HAS_RESPONSIBILITY|HAS_APPROVAL|HAS_REVISION*0..]->(child)
                    WHERE child IS NOT NULL
                      AND NOT child:DocumentTemplate
                      AND NOT child:Document
                    DETACH DELETE child
                    """,
                    document_id=document_id,
                )
                # Safety: remove any orphaned SOP nodes left from prior schema versions
                tx.run(
                    """
                    MATCH (n)
                    WHERE n.document_id = $document_id
                      AND (
                        n:ProcedureStep OR n:Responsibility OR n:Approval OR n:Revision
                        OR n:Heading OR n:Table OR n:Image OR n:Paragraph
                      )
                    DETACH DELETE n
                    """,
                    document_id=document_id,
                )

                logger.info("Creating Heading/Table/Image/Paragraph nodes document_id=%s", document_id)
                local_counters = {
                    "headings": 0,
                    "tables": 0,
                    "images": 0,
                    "paragraphs": 0,
                    "steps": 0,
                    "responsibilities": 0,
                    "approvals": 0,
                    "revisions": 0,
                }
                heading_registry: list[dict[str, Any]] = []
                image_registry: list[dict[str, Any]] = []
                ctx = {
                    "document_id": document_id,
                    "repository_id": repository_id,
                    "counters": local_counters,
                    "heading_registry": heading_registry,
                    "image_registry": image_registry,
                }
                for idx, child in enumerate(document_tree):
                    if not isinstance(child, dict):
                        continue
                    order = idx + 1
                    node_type = child.get("type")
                    if node_type == "heading":
                        _create_heading_root(tx, ctx, child, order)
                    elif node_type == "table":
                        _create_table_root(tx, ctx, child, order)
                    elif node_type == "image":
                        _create_image_root(tx, ctx, child, order)
                    elif node_type == "paragraph":
                        _create_paragraph_root(tx, ctx, child, order)

                _create_sop_enrichment(tx, ctx, template_data)

                logger.info(
                    "Creating relationships complete document_id=%s counters=%s",
                    document_id,
                    local_counters,
                )

                verify = tx.run(
                    """
                    MATCH (d:DocumentTemplate {document_id: $document_id})
                    RETURN count(d) AS c
                    """,
                    document_id=document_id,
                ).single()
                count = int(verify["c"]) if verify else 0
                if count == 0:
                    raise RuntimeError(
                        "Template graph write reported success but no nodes exist."
                    )
                return local_counters

            try:
                counters = session.execute_write(_write_template)
                logger.info("Neo4j transaction committed document_id=%s", document_id)
            except Exception:
                logger.exception("Neo4j write failed document_id=%s", document_id)
                raise

            # Post-commit verification (read-your-writes on the same session).
            verify_row = session.run(
                """
                MATCH (d:DocumentTemplate {document_id: $document_id})
                RETURN count(d) AS c
                """,
                document_id=document_id,
            ).single()
            verified = int(verify_row["c"]) if verify_row else 0
            logger.info(
                "Post-commit DocumentTemplate verification document_id=%s count=%d",
                document_id,
                verified,
            )
            if verified == 0:
                raise RuntimeError(
                    "Template graph write reported success but no nodes exist."
                )
    finally:
        driver.close()

    referenced_documents = template_data.get("referenced_documents") or []
    try:
        from src.features.references.graph.reference_graph_bridge import persist_template_references

        display_name = document_title or str(
            (metadata.get("title") if isinstance(metadata, dict) else None) or document_id
        )
        tenant_id = None
        if isinstance(template_data.get("tenant_id"), str):
            tenant_id = template_data.get("tenant_id")
        elif isinstance(metadata, dict) and isinstance(metadata.get("tenant_id"), str):
            tenant_id = metadata.get("tenant_id")
        persist_template_references(
            document_id=document_id,
            repository_id=repository_id,
            tenant_id=tenant_id,
            display_name=display_name,
            referenced_documents=referenced_documents if isinstance(referenced_documents, list) else [],
            template_data=template_data,
        )
    except Exception:
        logger.exception(
            "Template reference graph persistence soft-failed document_id=%s",
            document_id,
        )

    logger.info(
        "store_template_graph returned %s headings=%d tables=%d images=%d paragraphs=%d "
        "steps=%d responsibilities=%d approvals=%d revisions=%d JSON_size=%d",
        location,
        counters.get("headings", 0),
        counters.get("tables", 0),
        counters.get("images", 0),
        counters.get("paragraphs", 0),
        counters.get("steps", 0),
        counters.get("responsibilities", 0),
        counters.get("approvals", 0),
        counters.get("revisions", 0),
        len(template_json_str),
    )
    return location


def store_docx_template_graph(
    *,
    document_id: str,
    repository_id: str | None,
    template_data: dict[str, Any],
    document_type_id: str | None = None,
    document_type_name: str | None = None,
) -> str:
    """Backward-compatible alias for :func:`store_template_graph`."""
    return store_template_graph(
        document_id=document_id,
        repository_id=repository_id,
        template_data=template_data,
        document_type_id=document_type_id,
        document_type_name=document_type_name,
    )


def _count_template_tree_nodes(nodes: list[Any]) -> int:
    total = 0
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        total += 1
        total += _count_template_tree_nodes(node.get("children") or [])
    return total


def _normalize_heading_key(text: str | None) -> str:
    import re

    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", cleaned)
    return cleaned.strip().lower()


def _create_heading_root(tx, ctx: dict[str, Any], node: dict, order: int):
    document_id = ctx["document_id"]
    repository_id = ctx["repository_id"]
    counters = ctx["counters"]
    pos = node.get("position") or {}
    params = {
        "document_id": document_id,
        "repository_id": repository_id,
        "text": node.get("text", ""),
        "level": node.get("level"),
        "style": node.get("style", ""),
        "detection_method": node.get("detection_method", "style"),
        "page": node.get("page"),
        "font_name": node.get("font_name", ""),
        "font_size": node.get("font_size", 0.0),
        "bold": node.get("bold", False),
        "italic": node.get("italic", False),
        "x0": pos.get("x0"),
        "top": pos.get("top"),
        "x1": pos.get("x1"),
        "bottom": pos.get("bottom"),
        "order": order,
    }
    logger.info("Creating Heading node params=%s", params)
    res = tx.run(
        """
        MATCH (t:DocumentTemplate {document_id: $document_id})
        CREATE (h:Heading {
          document_id: $document_id,
          repository_id: $repository_id,
          text: $text,
          level: $level,
          style: $style,
          detection_method: $detection_method,
          page: $page,
          font_name: $font_name,
          font_size: $font_size,
          bold: $bold,
          italic: $italic,
          x0: $x0,
          top: $top,
          x1: $x1,
          bottom: $bottom,
          created_at: datetime(),
          updated_at: datetime()
        })
        CREATE (t)-[:HAS_HEADING {order: $order}]->(h)
        RETURN elementId(h) AS node_id
        """,
        **params,
    )
    row = res.single()
    if row is None:
        raise RuntimeError(f"Failed to create Heading for document_id={document_id}")
    node_id = row["node_id"]
    counters["headings"] += 1
    ctx["heading_registry"].append(
        {
            "element_id": node_id,
            "text": str(node.get("text") or ""),
            "norm": _normalize_heading_key(node.get("text")),
            "level": node.get("level"),
        }
    )
    for ci, child in enumerate(node.get("children", [])):
        _create_child_node(tx, ctx, node_id, child, ci + 1)


def _create_paragraph_root(tx, ctx: dict[str, Any], node: dict, order: int):
    document_id = ctx["document_id"]
    repository_id = ctx["repository_id"]
    counters = ctx["counters"]
    pos = node.get("position") or {}
    params = {
        "document_id": document_id,
        "repository_id": repository_id,
        "text": node.get("text", ""),
        "page": node.get("page"),
        "style": node.get("style", ""),
        "font_name": node.get("font_name", ""),
        "font_size": node.get("font_size", 0.0),
        "bold": node.get("bold", False),
        "italic": node.get("italic", False),
        "alignment": node.get("alignment", "left"),
        "indent_left": node.get("indent_left", 0.0),
        "x0": pos.get("x0"),
        "top": pos.get("top"),
        "x1": pos.get("x1"),
        "bottom": pos.get("bottom"),
        "order": order,
    }
    logger.info("Creating Paragraph root node")
    res = tx.run(
        """
        MATCH (t:DocumentTemplate {document_id: $document_id})
        CREATE (p:Paragraph {
          document_id: $document_id,
          repository_id: $repository_id,
          text: $text,
          page: $page,
          style: $style,
          font_name: $font_name,
          font_size: $font_size,
          bold: $bold,
          italic: $italic,
          alignment: $alignment,
          indent_left: $indent_left,
          x0: $x0,
          top: $top,
          x1: $x1,
          bottom: $bottom,
          created_at: datetime(),
          updated_at: datetime()
        })
        CREATE (t)-[:HAS_PARAGRAPH {order: $order}]->(p)
        RETURN elementId(p) AS node_id
        """,
        **params,
    )
    row = res.single()
    if row is None:
        raise RuntimeError(f"Failed to create Paragraph root for document_id={document_id}")
    node_id = row["node_id"]
    counters["paragraphs"] = counters.get("paragraphs", 0) + 1
    for ci, child in enumerate(node.get("children", [])):
        _create_child_node(tx, ctx, node_id, child, ci + 1)


def _create_table_root(tx, ctx: dict[str, Any], node: dict, order: int):
    document_id = ctx["document_id"]
    repository_id = ctx["repository_id"]
    counters = ctx["counters"]
    params = {
        "document_id": document_id,
        "repository_id": repository_id,
        "table_index": node.get("table_index", 0),
        "is_nested": node.get("is_nested", False),
        "rows": node.get("rows", 0),
        "columns": node.get("columns", 0),
        "table_caption": node.get("table_caption", ""),
        "header_row": node.get("header_row", []),
        "order": order,
    }
    logger.info("Creating Table node params=%s", {k: v for k, v in params.items() if k != "header_row"})
    res = tx.run(
        """
        MATCH (t:DocumentTemplate {document_id: $document_id})
        CREATE (tbl:Table {
          document_id: $document_id,
          repository_id: $repository_id,
          table_index: $table_index,
          is_nested: $is_nested,
          rows: $rows,
          columns: $columns,
          table_caption: $table_caption,
          header_row: $header_row,
          created_at: datetime(),
          updated_at: datetime()
        })
        CREATE (t)-[:HAS_TABLE {order: $order}]->(tbl)
        RETURN elementId(tbl) AS node_id
        """,
        **params,
    )
    row = res.single()
    if row is None:
        raise RuntimeError(f"Failed to create Table for document_id={document_id}")
    node_id = row["node_id"]
    counters["tables"] += 1
    for ci, child in enumerate(node.get("children", [])):
        _create_child_node(tx, ctx, node_id, child, ci + 1)


def _image_props(node: dict) -> dict[str, Any]:
    position = node.get("position") if isinstance(node.get("position"), dict) else {}
    return {
        "image_index": int(node.get("image_index") or 0),
        "page": int(node.get("page") or 0),
        "x0": position.get("x0"),
        "top": position.get("top"),
        "x1": position.get("x1"),
        "bottom": position.get("bottom"),
        "width": node.get("width"),
        "height": node.get("height"),
        "caption": str(node.get("caption") or ""),
        "image_type": str(node.get("image_type") or "embedded"),
        "figure_number": str(node.get("figure_number") or ""),
    }


def _create_image_root(tx, ctx: dict[str, Any], node: dict, order: int):
    document_id = ctx["document_id"]
    repository_id = ctx["repository_id"]
    counters = ctx["counters"]
    props = _image_props(node)
    params = {
        "document_id": document_id,
        "repository_id": repository_id,
        "order": order,
        **props,
    }
    logger.info("Creating Image node params=%s", params)
    res = tx.run(
        """
        MATCH (t:DocumentTemplate {document_id: $document_id})
        CREATE (img:Image {
          document_id: $document_id,
          repository_id: $repository_id,
          image_index: $image_index,
          page: $page,
          x0: $x0,
          top: $top,
          x1: $x1,
          bottom: $bottom,
          width: $width,
          height: $height,
          caption: $caption,
          image_type: $image_type,
          figure_number: $figure_number,
          created_at: datetime(),
          updated_at: datetime()
        })
        CREATE (t)-[:HAS_IMAGE {order: $order}]->(img)
        RETURN elementId(img) AS node_id
        """,
        **params,
    )
    row = res.single()
    if row is None:
        raise RuntimeError(f"Failed to create Image for document_id={document_id}")
    node_id = row["node_id"]
    counters["images"] = counters.get("images", 0) + 1
    ctx["image_registry"].append(
        {
            "element_id": node_id,
            "image_index": props["image_index"],
            "page": props["page"],
        }
    )
    for ci, child in enumerate(node.get("children", [])):
        _create_child_node(tx, ctx, node_id, child, ci + 1)


def _create_child_node(tx, ctx: dict[str, Any], parent_id: str, node: dict, order: int):
    if not isinstance(node, dict):
        return
    document_id = ctx["document_id"]
    repository_id = ctx["repository_id"]
    counters = ctx["counters"]
    node_type = node.get("type")
    if node_type == "heading":
        pos = node.get("position") or {}
        res = tx.run(
            """
            MATCH (p) WHERE elementId(p) = $parent_id
            CREATE (h:Heading {
              document_id: $document_id,
              repository_id: $repository_id,
              text: $text,
              level: $level,
              style: $style,
              detection_method: $detection_method,
              page: $page,
              font_name: $font_name,
              font_size: $font_size,
              bold: $bold,
              italic: $italic,
              x0: $x0,
              top: $top,
              x1: $x1,
              bottom: $bottom,
              created_at: datetime(),
              updated_at: datetime()
            })
            CREATE (p)-[:HAS_CHILD {order: $order}]->(h)
            RETURN elementId(h) AS node_id
            """,
            parent_id=parent_id,
            document_id=document_id,
            repository_id=repository_id,
            text=node.get("text", ""),
            level=node.get("level"),
            style=node.get("style", ""),
            detection_method=node.get("detection_method", "style"),
            page=node.get("page"),
            font_name=node.get("font_name", ""),
            font_size=node.get("font_size", 0.0),
            bold=node.get("bold", False),
            italic=node.get("italic", False),
            x0=pos.get("x0"),
            top=pos.get("top"),
            x1=pos.get("x1"),
            bottom=pos.get("bottom"),
            order=order,
        )
        row = res.single()
        if row is None:
            raise RuntimeError(f"Failed to create child Heading for document_id={document_id}")
        node_id = row["node_id"]
        counters["headings"] += 1
        ctx["heading_registry"].append(
            {
                "element_id": node_id,
                "text": str(node.get("text") or ""),
                "norm": _normalize_heading_key(node.get("text")),
                "level": node.get("level"),
            }
        )
        for ci, child in enumerate(node.get("children", [])):
            _create_child_node(tx, ctx, node_id, child, ci + 1)
    elif node_type == "paragraph":
        pos = node.get("position") or {}
        res = tx.run(
            """
            MATCH (p) WHERE elementId(p) = $parent_id
            CREATE (prg:Paragraph {
              document_id: $document_id,
              repository_id: $repository_id,
              text: $text,
              page: $page,
              style: $style,
              font_name: $font_name,
              font_size: $font_size,
              bold: $bold,
              italic: $italic,
              alignment: $alignment,
              indent_left: $indent_left,
              x0: $x0,
              top: $top,
              x1: $x1,
              bottom: $bottom,
              created_at: datetime(),
              updated_at: datetime()
            })
            CREATE (p)-[:HAS_CHILD {order: $order}]->(prg)
            RETURN elementId(prg) AS node_id
            """,
            parent_id=parent_id,
            document_id=document_id,
            repository_id=repository_id,
            text=node.get("text", ""),
            page=node.get("page"),
            style=node.get("style", ""),
            font_name=node.get("font_name", ""),
            font_size=node.get("font_size", 0.0),
            bold=node.get("bold", False),
            italic=node.get("italic", False),
            alignment=node.get("alignment", "left"),
            indent_left=node.get("indent_left", 0.0),
            x0=pos.get("x0"),
            top=pos.get("top"),
            x1=pos.get("x1"),
            bottom=pos.get("bottom"),
            order=order,
        )
        row = res.single()
        if row is None:
            raise RuntimeError(f"Failed to create child Paragraph for document_id={document_id}")
        node_id = row["node_id"]
        counters["paragraphs"] = counters.get("paragraphs", 0) + 1
        for ci, child in enumerate(node.get("children", [])):
            _create_child_node(tx, ctx, node_id, child, ci + 1)
    elif node_type == "table":
        res = tx.run(
            """
            MATCH (p) WHERE elementId(p) = $parent_id
            CREATE (tbl:Table {
              document_id: $document_id,
              repository_id: $repository_id,
              table_index: $table_index,
              is_nested: $is_nested,
              rows: $rows,
              columns: $columns,
              table_caption: $table_caption,
              header_row: $header_row,
              created_at: datetime(),
              updated_at: datetime()
            })
            CREATE (p)-[:HAS_CHILD {order: $order}]->(tbl)
            RETURN elementId(tbl) AS node_id
            """,
            parent_id=parent_id,
            document_id=document_id,
            repository_id=repository_id,
            table_index=node.get("table_index", 0),
            is_nested=node.get("is_nested", False),
            rows=node.get("rows", 0),
            columns=node.get("columns", 0),
            table_caption=node.get("table_caption", ""),
            header_row=node.get("header_row", []),
            order=order,
        )
        row = res.single()
        if row is None:
            raise RuntimeError(f"Failed to create child Table for document_id={document_id}")
        node_id = row["node_id"]
        counters["tables"] += 1
        for ci, child in enumerate(node.get("children", [])):
            _create_child_node(tx, ctx, node_id, child, ci + 1)
    elif node_type == "image":
        props = _image_props(node)
        res = tx.run(
            """
            MATCH (p) WHERE elementId(p) = $parent_id
            CREATE (img:Image {
              document_id: $document_id,
              repository_id: $repository_id,
              image_index: $image_index,
              page: $page,
              x0: $x0,
              top: $top,
              x1: $x1,
              bottom: $bottom,
              width: $width,
              height: $height,
              caption: $caption,
              image_type: $image_type,
              figure_number: $figure_number,
              created_at: datetime(),
              updated_at: datetime()
            })
            CREATE (p)-[:HAS_CHILD {order: $order}]->(img)
            RETURN elementId(img) AS node_id
            """,
            parent_id=parent_id,
            document_id=document_id,
            repository_id=repository_id,
            order=order,
            **props,
        )
        row = res.single()
        if row is None:
            raise RuntimeError(f"Failed to create child Image for document_id={document_id}")
        node_id = row["node_id"]
        counters["images"] = counters.get("images", 0) + 1
        ctx["image_registry"].append(
            {
                "element_id": node_id,
                "image_index": props["image_index"],
                "page": props["page"],
            }
        )
        # Also link template root for discoverability when image is nested under heading
        tx.run(
            """
            MATCH (t:DocumentTemplate {document_id: $document_id})
            MATCH (img) WHERE elementId(img) = $img_id
            MERGE (t)-[:HAS_IMAGE {order: $order}]->(img)
            """,
            document_id=document_id,
            img_id=node_id,
            order=order,
        )
        for ci, child in enumerate(node.get("children", [])):
            _create_child_node(tx, ctx, node_id, child, ci + 1)


def _find_heading_element_id(registry: list[dict[str, Any]], section_title: str | None) -> str | None:
    norm = _normalize_heading_key(section_title)
    if not norm:
        return None
    for entry in reversed(registry):
        if entry.get("norm") == norm:
            return entry.get("element_id")
    # Soft match: section title contained in heading or vice versa
    for entry in reversed(registry):
        other = str(entry.get("norm") or "")
        if other and (norm in other or other in norm):
            return entry.get("element_id")
    return None


def _create_sop_enrichment(tx, ctx: dict[str, Any], template_data: dict[str, Any]) -> None:
    """Create ProcedureStep / Responsibility / Approval / Revision / enriched Images."""
    document_id = ctx["document_id"]
    repository_id = ctx["repository_id"]
    counters = ctx["counters"]
    heading_registry = ctx["heading_registry"]
    image_registry = ctx["image_registry"]

    step_ids: list[str] = []
    for step in template_data.get("procedure_steps") or []:
        if not isinstance(step, dict):
            continue
        res = tx.run(
            """
            MATCH (t:DocumentTemplate {document_id: $document_id})
            CREATE (p:ProcedureStep {
              document_id: $document_id,
              repository_id: $repository_id,
              step_number: $step_number,
              text: $text,
              order: $order,
              page: $page,
              section_title: $section_title,
              section_level: $section_level,
              detection_method: $detection_method,
              created_at: datetime(),
              updated_at: datetime()
            })
            CREATE (t)-[:HAS_STEP {order: $order}]->(p)
            RETURN elementId(p) AS node_id
            """,
            document_id=document_id,
            repository_id=repository_id,
            step_number=str(step.get("step_number") or ""),
            text=str(step.get("text") or ""),
            order=int(step.get("order") or 0),
            page=step.get("page"),
            section_title=str(step.get("section_title") or ""),
            section_level=int(step.get("section_level") or 1),
            detection_method=str(step.get("detection_method") or ""),
        )
        row = res.single()
        if row is None:
            continue
        step_id = row["node_id"]
        step_ids.append(step_id)
        counters["steps"] = counters.get("steps", 0) + 1
        heading_eid = _find_heading_element_id(heading_registry, step.get("section_title"))
        if heading_eid:
            tx.run(
                """
                MATCH (h) WHERE elementId(h) = $heading_id
                MATCH (p) WHERE elementId(p) = $step_id
                MERGE (h)-[:HAS_STEP {order: $order}]->(p)
                """,
                heading_id=heading_eid,
                step_id=step_id,
                order=int(step.get("order") or 0),
            )

    for prev_id, next_id in zip(step_ids, step_ids[1:]):
        tx.run(
            """
            MATCH (a) WHERE elementId(a) = $prev_id
            MATCH (b) WHERE elementId(b) = $next_id
            MERGE (a)-[:NEXT_STEP]->(b)
            """,
            prev_id=prev_id,
            next_id=next_id,
        )

    for idx, resp in enumerate(template_data.get("responsibilities") or [], start=1):
        if not isinstance(resp, dict):
            continue
        res = tx.run(
            """
            MATCH (t:DocumentTemplate {document_id: $document_id})
            CREATE (r:Responsibility {
              document_id: $document_id,
              repository_id: $repository_id,
              role: $role,
              description: $description,
              page: $page,
              created_at: datetime(),
              updated_at: datetime()
            })
            RETURN elementId(r) AS node_id
            """,
            document_id=document_id,
            repository_id=repository_id,
            role=str(resp.get("role") or ""),
            description=str(resp.get("description") or ""),
            page=resp.get("page"),
        )
        row = res.single()
        if row is None:
            continue
        resp_id = row["node_id"]
        counters["responsibilities"] = counters.get("responsibilities", 0) + 1
        heading_eid = _find_heading_element_id(
            heading_registry, resp.get("section_title") or "Responsibilities"
        )
        if heading_eid:
            tx.run(
                """
                MATCH (h) WHERE elementId(h) = $heading_id
                MATCH (r) WHERE elementId(r) = $resp_id
                MERGE (h)-[:HAS_RESPONSIBILITY {order: $order}]->(r)
                """,
                heading_id=heading_eid,
                resp_id=resp_id,
                order=idx,
            )

    for approval in template_data.get("approvals") or []:
        if not isinstance(approval, dict):
            continue
        tx.run(
            """
            MATCH (t:DocumentTemplate {document_id: $document_id})
            CREATE (a:Approval {
              document_id: $document_id,
              repository_id: $repository_id,
              prepared_by: $prepared_by,
              reviewed_by: $reviewed_by,
              approved_by: $approved_by,
              approved_date: $approved_date,
              review_date: $review_date,
              prepared_date: $prepared_date,
              created_at: datetime(),
              updated_at: datetime()
            })
            CREATE (t)-[:HAS_APPROVAL]->(a)
            """,
            document_id=document_id,
            repository_id=repository_id,
            prepared_by=str(approval.get("prepared_by") or ""),
            reviewed_by=str(approval.get("reviewed_by") or ""),
            approved_by=str(approval.get("approved_by") or ""),
            approved_date=str(approval.get("approved_date") or ""),
            review_date=str(approval.get("review_date") or ""),
            prepared_date=str(approval.get("prepared_date") or ""),
        )
        counters["approvals"] = counters.get("approvals", 0) + 1

    for idx, rev in enumerate(template_data.get("revisions") or [], start=1):
        if not isinstance(rev, dict):
            continue
        tx.run(
            """
            MATCH (t:DocumentTemplate {document_id: $document_id})
            CREATE (rv:Revision {
              document_id: $document_id,
              repository_id: $repository_id,
              version: $version,
              date: $date,
              description: $description,
              author: $author,
              created_at: datetime(),
              updated_at: datetime()
            })
            CREATE (t)-[:HAS_REVISION {order: $order}]->(rv)
            """,
            document_id=document_id,
            repository_id=repository_id,
            version=str(rev.get("version") or ""),
            date=str(rev.get("date") or ""),
            description=str(rev.get("description") or ""),
            author=str(rev.get("author") or ""),
            order=idx,
        )
        counters["revisions"] = counters.get("revisions", 0) + 1

    # Enrich existing images / create caption-only figures
    existing_keys = {
        (int(i.get("image_index") or 0), int(i.get("page") or 0)) for i in image_registry
    }
    for idx, img in enumerate(template_data.get("images") or []):
        if not isinstance(img, dict):
            continue
        image_index = int(img.get("image_index") if img.get("image_index") is not None else idx)
        page = int(img.get("page") or 0)
        key = (image_index, page)
        matched = next(
            (
                e
                for e in image_registry
                if int(e.get("image_index") or 0) == image_index
                or (page and int(e.get("page") or 0) == page)
            ),
            None,
        )
        if matched:
            tx.run(
                """
                MATCH (img) WHERE elementId(img) = $img_id
                SET img.caption = coalesce(nullif($caption, ''), img.caption),
                    img.figure_number = coalesce(nullif($figure_number, ''), img.figure_number),
                    img.image_type = coalesce(nullif($image_type, ''), img.image_type),
                    img.width = coalesce($width, img.width),
                    img.height = coalesce($height, img.height),
                    img.updated_at = datetime()
                """,
                img_id=matched["element_id"],
                caption=str(img.get("caption") or ""),
                figure_number=str(img.get("figure_number") or ""),
                image_type=str(img.get("image_type") or ""),
                width=img.get("width"),
                height=img.get("height"),
            )
            img_eid = matched["element_id"]
        elif key not in existing_keys or str(img.get("image_type") or "") == "figure_caption":
            res = tx.run(
                """
                MATCH (t:DocumentTemplate {document_id: $document_id})
                CREATE (img:Image {
                  document_id: $document_id,
                  repository_id: $repository_id,
                  image_index: $image_index,
                  page: $page,
                  width: $width,
                  height: $height,
                  caption: $caption,
                  image_type: $image_type,
                  figure_number: $figure_number,
                  created_at: datetime(),
                  updated_at: datetime()
                })
                CREATE (t)-[:HAS_IMAGE {order: $order}]->(img)
                RETURN elementId(img) AS node_id
                """,
                document_id=document_id,
                repository_id=repository_id,
                image_index=image_index,
                page=page,
                width=img.get("width"),
                height=img.get("height"),
                caption=str(img.get("caption") or ""),
                image_type=str(img.get("image_type") or "embedded"),
                figure_number=str(img.get("figure_number") or ""),
                order=idx + 1,
            )
            row = res.single()
            if row is None:
                continue
            img_eid = row["node_id"]
            counters["images"] = counters.get("images", 0) + 1
            existing_keys.add(key)
        else:
            continue

        heading_eid = _find_heading_element_id(heading_registry, img.get("section_title"))
        if heading_eid:
            tx.run(
                """
                MATCH (h) WHERE elementId(h) = $heading_id
                MATCH (img) WHERE elementId(img) = $img_id
                MERGE (h)-[:HAS_IMAGE]->(img)
                """,
                heading_id=heading_eid,
                img_id=img_eid,
            )


def delete_document_graph(document_id: str) -> dict[str, Any]:
    """Delete Document / DocumentTemplate / DocumentArtifact nodes and related subgraph."""
    uri, user, password = _neo4j_credentials()
    GraphDatabase = _require_neo4j_driver()

    cypher = """
    MATCH (n)
    WHERE n.document_id = $document_id
      AND (
        n:Document
        OR n:DocumentArtifact
        OR n:DocumentTemplate
        OR n:Heading
        OR n:Paragraph
        OR n:Table
        OR n:Image
        OR n:ProcedureStep
        OR n:Responsibility
        OR n:Approval
        OR n:Revision
      )
    DETACH DELETE n
    RETURN count(n) AS nodes_deleted
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes_deleted = 0
    try:
        with driver.session() as session:
            row = session.run(cypher, document_id=document_id).single()
            nodes_deleted = int(row["nodes_deleted"]) if row else 0
    finally:
        driver.close()
    return {
        "nodes_deleted": nodes_deleted,
        "deleted": nodes_deleted > 0,
    }
