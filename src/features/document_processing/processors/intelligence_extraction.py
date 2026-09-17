"""Content intelligence extraction — template_compliance (DOCX) + text blocks.

Enabled per repository via settings flags (any of):
  - intelligent_extraction
  - content_intelligence
  - intelligence_extraction

Behaviour is driven by repository settings and environment variables — no hardcoded
paths or credentials.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_processing.shared_processor.types import ProcessorType

logger = logging.getLogger("src.features.document_processing.processors.intelligence_extraction")

_TEMPLATE_SUFFIXES = {".docx", ".pdf"}


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if value is True or value == 1:
        return True
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value is False or value == 0:
        return False
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(value)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return _truthy(raw, default=default)


def _suffix_from_name(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).suffix.lower()


def _resolve_suffix(request: ProcessRequest, document_path: Path) -> str:
    for candidate in (
        document_path.suffix.lower(),
        _suffix_from_name(request.document_name),
        _suffix_from_name(request.original_file_name),
    ):
        if candidate in _TEMPLATE_SUFFIXES or candidate in {".pptx", ".ppt", ".txt", ".text"}:
            return candidate
    try:
        with document_path.open("rb") as handle:
            header = handle.read(8)
        if header.startswith(b"%PDF"):
            return ".pdf"
        if header.startswith(b"PK"):
            import zipfile

            with zipfile.ZipFile(document_path) as zf:
                names = {name.replace("\\", "/") for name in zf.namelist()}
                if any(name.startswith("ppt/") for name in names):
                    return ".pptx"
                if any(name.startswith("word/") for name in names):
                    return ".docx"
                try:
                    types_xml = zf.read("[Content_Types].xml").decode("utf-8", errors="ignore").lower()
                except Exception:
                    types_xml = ""
                if "presentationml" in types_xml:
                    return ".pptx"
                if "wordprocessingml" in types_xml:
                    return ".docx"
    except OSError:
        pass
    return document_path.suffix.lower()


def _risk_keywords() -> list[str]:
    raw = os.getenv("CONTENT_INTELLIGENCE_RISK_KEYWORDS", "")
    text = str(raw or "").strip()
    if not text:
        return []
    return [part.strip().lower() for part in text.split(",") if part.strip()]


class IntelligenceExtractionProcessor(BaseProcessor):
    """Content intelligence processor backed by ``processor_service.extractors``."""

    processor_type = ProcessorType.INTELLIGENCE_EXTRACTION

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        logger.info(
            "Starting content intelligence extraction document_id=%s path=%s",
            request.document_id,
            document_path,
        )

        set_status("loading_document", 5.0)
        blocks, doc_meta = load_document_blocks(document_path)
        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        suffix = _resolve_suffix(request, document_path)
        template_data: dict[str, Any] | None = None
        extraction_mode = "text_blocks"
        artifacts: dict[str, Any] = {
            "page_count": len({getattr(b, "page", 1) for b in blocks if getattr(b, "page", None)} or {1}),
            "block_count": len(blocks),
            "suffix": suffix,
        }

        use_template = _env_flag("CONTENT_INTELLIGENCE_USE_TEMPLATE", True)
        if use_template and suffix == ".docx":
            set_status("docx_content_intelligence", 25.0)
            from src.features.document_processing.extractors.template_compliance import (
                compliance_to_legacy_template,
                extract_template,
            )

            compliance = extract_template(document_path)
            template_data = compliance_to_legacy_template(compliance)
            extraction_mode = "template_compliance"
        elif use_template and suffix == ".pdf":
            # PDF template_compliance is stubbed — stay on text_blocks.
            set_status("pdf_text_blocks_intelligence", 25.0)
            extraction_mode = "text_blocks"
            artifacts["template_skipped"] = "pdf_not_implemented_in_template_compliance"

        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        if (
            template_data is not None
            and template_data.get("source") != "template_compliance"
            and _env_flag("CONTENT_INTELLIGENCE_ENRICH_SOP", True)
        ):
            set_status("sop_structure_enrichment", 55.0)
            try:
                from src.features.document_processing.extractors.sop_structure import enrich_template

                template_data = enrich_template(template_data) or template_data
                artifacts["sop_enriched"] = True
            except Exception:
                logger.exception("SOP structure enrichment failed; continuing with template payload")
                artifacts["sop_enriched"] = False
        elif template_data is not None and template_data.get("source") == "template_compliance":
            artifacts["sop_enriched"] = False
            artifacts["sop_enrich_skipped"] = "template_compliance_schema"
        set_status("intelligence_scoring", 75.0)
        text = "\n".join(getattr(b, "text", "") for b in blocks if getattr(b, "text", None)).lower()
        risk_keywords = _risk_keywords()
        risk_score = None
        decision = None
        if risk_keywords:
            count = sum(text.count(word) for word in risk_keywords)
            weight = float(os.getenv("CONTENT_INTELLIGENCE_RISK_WEIGHT") or "0.1")
            threshold = float(os.getenv("CONTENT_INTELLIGENCE_RISK_THRESHOLD") or "0.5")
            risk_score = round(min(1.0, count * weight), 4)
            decision = "REJECTED" if risk_score > threshold else "APPROVED"
            artifacts["risk_score"] = risk_score
            artifacts["decision"] = decision
            artifacts["risk_keyword_hits"] = count

        result_location = f"neo4j://ContentIntelligence/{request.document_id}"
        persist_graph = _env_flag("CONTENT_INTELLIGENCE_PERSIST_NEO4J", True)
        if persist_graph and template_data is not None:
            set_status("storing_intelligence_graph", 90.0)
            try:
                from src.infrastructure.document_databases.neo4j_store import store_template_graph

                result_location = store_template_graph(
                    document_id=request.document_id,
                    repository_id=request.repository_id,
                    template_data=template_data,
                    document_type_id=request.document_type_id,
                    document_type_name=request.document_type_name,
                )
                artifacts["neo4j_persisted"] = True
            except Exception:
                logger.exception("Failed to persist content intelligence graph")
                artifacts["neo4j_persisted"] = False

        set_status("completed", 100.0)
        document_metadata: dict[str, Any] = {
            **(doc_meta if isinstance(doc_meta, dict) else {}),
            "extraction_mode": extraction_mode,
            "content_intelligence": True,
        }
        if risk_score is not None:
            document_metadata["risk_score"] = risk_score
            document_metadata["decision"] = decision
        if template_data is not None:
            tree = template_data.get("document_tree") or []
            artifacts["template_node_count"] = len(tree) if isinstance(tree, list) else 0
            document_metadata["template_summary"] = {
                "node_count": artifacts["template_node_count"],
                "mode": extraction_mode,
                "outline_count": len(template_data.get("outline") or []),
                "table_count": len(template_data.get("tables") or []),
            }
            # Keep extractor metadata without dropping intelligence flags.
            meta = template_data.get("metadata")
            if isinstance(meta, dict):
                document_metadata.update({k: v for k, v in meta.items() if k not in document_metadata})

        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=result_location,
            document_metadata=document_metadata,
            artifacts=artifacts,
        )
