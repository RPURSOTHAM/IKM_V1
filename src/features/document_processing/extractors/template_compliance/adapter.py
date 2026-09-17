"""Adapt template_compliance schema → Neo4j store_template_graph shape."""

from __future__ import annotations

from typing import Any

from .schema import SCHEMA_VERSION


def _section_to_node(section: dict[str, Any], *, level: int = 1) -> dict[str, Any]:
    number = str(section.get("number") or "").strip()
    title = str(section.get("title") or "").strip()
    label = f"{number} {title}".strip() if number else title
    children: list[dict[str, Any]] = []
    for sub in section.get("subsections") or []:
        if isinstance(sub, dict):
            children.append(_section_to_node(sub, level=level + 1))
    node: dict[str, Any] = {
        "type": "heading",
        "level": level,
        "text": label,
        "number": number,
        "title": title,
        "children": children,
    }
    indent = section.get("indentation")
    if isinstance(indent, dict) and indent:
        node["indentation"] = indent
    image_count = int(section.get("image_count") or 0)
    table_count = int(section.get("table_count") or 0)
    if image_count:
        node["image_count"] = image_count
    if table_count:
        node["table_count"] = table_count
    return node


def compliance_to_legacy_template(compliance: dict[str, Any]) -> dict[str, Any]:
    """Map ``extract_template`` output to the shape expected by ``store_template_graph``."""
    if not isinstance(compliance, dict):
        compliance = {}

    # Already adapted (idempotent).
    if compliance.get("source") == "template_compliance" and "document_tree" in compliance:
        return compliance

    doc = compliance.get("document") if isinstance(compliance.get("document"), dict) else {}
    raw_meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}

    title = str(raw_meta.get("title") or "").strip()
    document_type = str(raw_meta.get("document_type") or "").strip()
    metadata: dict[str, Any] = {
        "title": title or document_type,
        "document_title": title or document_type,
        "document_no": str(raw_meta.get("document_no") or "").strip(),
        "version_no": str(raw_meta.get("version_no") or "").strip(),
        "effective_date": str(raw_meta.get("effective_date") or "").strip(),
        "review_date": str(raw_meta.get("review_date") or "").strip(),
        "document_type": document_type,
        "file_name": str(doc.get("file_name") or "").strip(),
        "file_type": str(doc.get("file_type") or "").strip(),
    }

    sections = doc.get("sections") if isinstance(doc.get("sections"), list) else []
    document_tree: list[dict[str, Any]] = []
    outline: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        node = _section_to_node(section, level=1)
        document_tree.append(node)
        outline.append(
            {
                "number": node.get("number"),
                "title": node.get("title"),
                "text": node.get("text"),
                "level": 1,
            }
        )
        for child in node.get("children") or []:
            if isinstance(child, dict):
                outline.append(
                    {
                        "number": child.get("number"),
                        "title": child.get("title"),
                        "text": child.get("text"),
                        "level": child.get("level") or 2,
                    }
                )

    total_tables = int(doc.get("total_tables") or 0)
    total_images = int(doc.get("total_images") or 0)

    return {
        "source": "template_compliance",
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "document_tree": document_tree,
        "outline": outline,
        "tables": [{"index": i} for i in range(total_tables)],
        "images": [{"index": i} for i in range(total_images)],
        "possible_headings": [],
        "content_blocks": [],
        "procedure_steps": [],
        "responsibilities": [],
        "approvals": [],
        "revisions": [],
        "header": doc.get("header") or [],
        "footer": doc.get("footer") or [],
        "logo": doc.get("logo") or {},
        "fonts": doc.get("fonts") or {},
        "toc": doc.get("toc") or {},
        "signature_table": doc.get("signature_table") or {},
        "repeated_fields": doc.get("repeated_fields") or [],
        "total_tables": total_tables,
        "total_images": total_images,
        "section_count": len(sections),
        "compliance": compliance,
    }
