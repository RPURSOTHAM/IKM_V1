"""Enrich template JSON with SOP structural nodes before Neo4j write."""

from __future__ import annotations

from typing import Any

from src.features.document_processing.extractors.sop_structure.approvals import extract_approvals
from src.features.document_processing.extractors.sop_structure.images import extract_images
from src.features.document_processing.extractors.sop_structure.procedure_steps import extract_procedure_steps
from src.features.document_processing.extractors.sop_structure.responsibilities import extract_responsibilities
from src.features.document_processing.extractors.sop_structure.revisions import extract_revisions


def enrich_template(
    template_data: dict[str, Any],
    content_blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Add SOP structure fields to an existing template extraction payload.

    New top-level keys (additive, backward compatible):
      - procedure_steps
      - responsibilities
      - approvals
      - revisions
      - images (enriched; preserves prior image entries when present)
    """
    data = dict(template_data or {})
    blocks = list(content_blocks or data.get("content_blocks") or [])

    data["procedure_steps"] = extract_procedure_steps(data, blocks)
    data["responsibilities"] = extract_responsibilities(data, blocks)
    data["approvals"] = extract_approvals(data, blocks)
    data["revisions"] = extract_revisions(data)
    data["images"] = extract_images(data, blocks)

    # Keep content_blocks optional for debugging / re-enrichment; avoid huge payloads by default
    if content_blocks is not None and "content_blocks" not in data:
        data["content_blocks"] = blocks

    return data
