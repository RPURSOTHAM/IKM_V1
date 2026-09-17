from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult, StatusCallback
from src.infrastructure.document_databases.neo4j_store import store_document_graph, store_template_graph
from src.features.document_processing.loaders.document_text import load_document_blocks
from src.features.document_processing.processors.base import BaseProcessor
from src.features.document_processing.shared_processor.types import ProcessorType

logger = logging.getLogger(__name__)

_TEMPLATE_SUFFIXES = {".docx", ".pdf"}
_MEDIA_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v", ".wmv",
}


def _walk_template_nodes(nodes: list) -> list:
    found: list = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        found.append(node)
        found.extend(_walk_template_nodes(node.get("children") or []))
    return found


def _template_node_count(template_data: dict) -> int:
    return len(_walk_template_nodes(template_data.get("document_tree") or []))


def _count_by_type(template_data: dict, node_type: str) -> int:
    return sum(
        1
        for node in _walk_template_nodes(template_data.get("document_tree") or [])
        if node.get("type") == node_type
    )


def _suffix_from_name(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).suffix.lower()


def resolve_template_route_suffix(request: ProcessRequest, document_path: Path) -> tuple[str, str]:
    """Pick .docx/.pdf routing using path, then document_name, then original_file_name."""
    candidates: list[tuple[str, str]] = [
        (document_path.suffix.lower(), "document_path"),
        (_suffix_from_name(request.document_name), "document_name"),
        (_suffix_from_name(request.original_file_name), "original_file_name"),
    ]
    for suffix, source in candidates:
        if suffix in _TEMPLATE_SUFFIXES:
            return suffix, source
    for suffix, source in candidates:
        if suffix:
            return suffix, source
    return "", "none"


def _sniff_office_or_pdf(document_path: Path) -> str:
    """Return '.pdf' / '.docx' / '.pptx' when file magic matches, else ''."""
    try:
        with document_path.open("rb") as handle:
            header = handle.read(8)
    except OSError:
        return ""
    if header.startswith(b"%PDF"):
        return ".pdf"
    if not header.startswith(b"PK"):
        return ""
    try:
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
    except Exception:
        return ""
    return ""


def _skip_template_result(
    request: ProcessRequest,
    *,
    reason: str,
    suffix: str,
) -> ProcessorResult:
    return ProcessorResult(
        processor_type=ProcessorType.TEMPLATE_EXTRACTION.value,
        document_id=request.document_id,
        storage_backend="neo4j",
        result_location=None,
        document_metadata={
            "skipped": True,
            "skip_reason": reason,
            "file_extension": suffix.lstrip("."),
            "message": f"Template extraction is not applicable for {suffix or 'this file type'}.",
        },
        artifacts={"skipped": True, "skip_reason": reason},
    )


class TemplateExtractionProcessor(BaseProcessor):
    processor_type = ProcessorType.TEMPLATE_EXTRACTION

    def run(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        suffix, suffix_source = resolve_template_route_suffix(request, document_path)
        enabled = getattr(request, "enabled_processor_types", None)
        logger.info(
            "Template extraction started document_id=%s repository_id=%s processor_type=%s "
            "enabled_processor_types=%s template_extraction_flag=%s path=%s "
            "document_name=%s original_file_name=%s route_suffix=%s route_source=%s",
            request.document_id,
            request.repository_id,
            request.normalized_processor_type(),
            enabled,
            getattr(request, "template_extraction", None),
            document_path,
            request.document_name,
            request.original_file_name,
            suffix or "(empty)",
            suffix_source,
        )
        if suffix == ".docx":
            return self._run_docx(request, document_path, set_status=set_status, check_stop=check_stop)
        if suffix == ".pdf":
            set_status("skipped_pdf", 100.0)
            return _skip_template_result(
                request,
                reason="pdf_not_implemented_in_template_compliance",
                suffix=suffix,
            )
        if suffix in {".pptx", ".ppt"}:
            set_status("skipped_pptx", 100.0)
            return _skip_template_result(
                request,
                reason="pptx_not_supported_for_docx_pdf_template",
                suffix=suffix,
            )
        if suffix in _MEDIA_SUFFIXES:
            set_status("skipped_media", 100.0)
            return _skip_template_result(
                request,
                reason="media_not_supported_for_template",
                suffix=suffix,
            )

        sniffed = _sniff_office_or_pdf(document_path)
        if sniffed == ".pptx":
            set_status("skipped_pptx", 100.0)
            return _skip_template_result(
                request,
                reason="pptx_not_supported_for_docx_pdf_template",
                suffix=".pptx",
            )
        if sniffed == ".pdf":
            set_status("skipped_pdf", 100.0)
            return _skip_template_result(
                request,
                reason="pdf_not_implemented_in_template_compliance",
                suffix=".pdf",
            )
        if sniffed == ".docx":
            logger.warning(
                "Template route suffix mismatch document_id=%s path_suffix=%s sniffed=%s; using sniffed type",
                request.document_id,
                suffix or "(empty)",
                sniffed,
            )
            return self._run_docx(request, document_path, set_status=set_status, check_stop=check_stop)

        logger.info(
            "Template extraction using default DocumentArtifact path document_id=%s "
            "reason=unsupported_or_non_template_suffix suffix=%s",
            request.document_id,
            suffix or "(empty)",
        )
        return self._run_default(request, document_path, set_status=set_status, check_stop=check_stop)

    def _persist_template_graph(
        self,
        request: ProcessRequest,
        template_data: dict,
        *,
        set_status: StatusCallback,
        status_label: str,
    ) -> str:
        headings = _count_by_type(template_data, "heading")
        tables = int(template_data.get("total_tables") or len(template_data.get("tables") or []))
        images = int(template_data.get("total_images") or len(template_data.get("images") or []))
        node_count = _template_node_count(template_data)
        logger.info(
            "Template extract counts document_id=%s headings=%d tables=%d images=%d graph_nodes=%d",
            request.document_id,
            headings,
            tables,
            images,
            node_count,
        )
        set_status(status_label, 70.0)
        if not str(request.document_id or "").strip():
            raise RuntimeError("template extraction refused to write Neo4j: document_id is empty")
        logger.info(
            "Calling store_template_graph() document_id=%s repository_id=%s graph_nodes=%d "
            "sections=%d tables=%d images=%d",
            request.document_id,
            request.repository_id,
            node_count,
            int(template_data.get("section_count") or 0),
            tables,
            images,
        )
        location = store_template_graph(
            document_id=request.document_id,
            repository_id=request.repository_id,
            template_data=template_data,
            document_type_id=request.document_type_id,
            document_type_name=request.document_type_name,
        )
        if not location or "DocumentTemplate" not in str(location):
            raise RuntimeError(
                f"store_template_graph returned invalid location={location!r} "
                f"for document_id={request.document_id}"
            )
        logger.info("store_template_graph returned %s", location)
        return location

    def _run_docx(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        from src.features.document_processing.extractors.template_compliance import (
            compliance_to_legacy_template,
            extract_template,
        )

        logger.info("DOCX template_compliance extraction started: document_id=%s", request.document_id)
        set_status("docx_template_extraction", 10.0)
        compliance = extract_template(document_path)
        template_data = compliance_to_legacy_template(compliance)
        logger.info(
            "Template object created document_id=%s outline=%d tables=%d tree_nodes=%d source=%s",
            request.document_id,
            len(template_data.get("outline") or []),
            int(template_data.get("total_tables") or 0),
            _template_node_count(template_data),
            template_data.get("source"),
        )

        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        location = self._persist_template_graph(
            request,
            template_data,
            set_status=set_status,
            status_label="storing_docx_template",
        )

        meta = template_data.get("metadata") if isinstance(template_data.get("metadata"), dict) else {}
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata={
                **meta,
                "schema_version": template_data.get("schema_version"),
                "source": template_data.get("source"),
                "section_count": template_data.get("section_count"),
                "total_tables": template_data.get("total_tables"),
                "total_images": template_data.get("total_images"),
            },
            artifacts={
                "outline_count": len(template_data.get("outline", [])),
                "section_count": int(template_data.get("section_count") or 0),
                "table_count": int(template_data.get("total_tables") or 0),
                "heading_count": _count_by_type(template_data, "heading"),
                "image_count": int(template_data.get("total_images") or 0),
                "graph_node_count": _template_node_count(template_data),
                "schema_version": template_data.get("schema_version"),
                "extractor": "template_compliance",
            },
        )

    def _run_default(
        self,
        request: ProcessRequest,
        document_path: Path,
        *,
        set_status: StatusCallback,
        check_stop: Callable[[], bool],
    ) -> ProcessorResult:
        set_status("loading_document", 10.0)
        blocks, doc_meta = load_document_blocks(document_path)
        if check_stop():
            raise RuntimeError("Processing stopped by operator.")

        set_status("building_skeleton", 45.0)
        skeleton = self._build_skeleton(blocks)
        payload = {
            "document_type_id": request.document_type_id,
            "document_type_name": request.document_type_name,
            "skeleton": skeleton,
            "source": "template_extraction",
        }

        set_status("storing", 85.0)
        logger.info(
            "Calling store_document_graph() (default path) document_id=%s — "
            "no DocumentTemplate nodes will be created",
            request.document_id,
        )
        location = store_document_graph(
            document_id=request.document_id,
            repository_id=request.repository_id,
            processor_type=self.processor_type.value,
            payload=payload,
            document_type_id=request.document_type_id,
            document_type_name=request.document_type_name,
        )
        return ProcessorResult(
            processor_type=self.processor_type.value,
            document_id=request.document_id,
            storage_backend=self.storage_backend,
            result_location=location,
            document_metadata=doc_meta,
            artifacts={"skeleton_sections": len(skeleton.get("sections", []))},
        )

    @staticmethod
    def _build_skeleton(blocks: list) -> dict:
        header_blocks = [b for b in blocks[:12]]
        footer_blocks = [b for b in blocks[-8:]] if len(blocks) > 8 else []
        sections: list[dict] = []
        for block in blocks:
            if not block.text.strip():
                continue
            text = block.text.strip()
            is_heading = (
                block.bold
                or (block.font_size and block.font_size >= 12)
                or block.style.lower().startswith("heading")
            )
            if not is_heading and re.match(r"^\d+\.\s+\S", text):
                is_heading = True
            if is_heading:
                level = "main_header" if (block.font_size or 0) >= 14 else "sub_header"
                sections.append(
                    {
                        "level": level,
                        "title": block.text.strip(),
                        "page": block.page,
                        "description": f"Section starting at page {block.page}.",
                    }
                )

        return {
            "header": [{"text": b.text.strip(), "page": b.page} for b in header_blocks if b.text.strip()],
            "footer": [{"text": b.text.strip(), "page": b.page} for b in footer_blocks if b.text.strip()],
            "sections": sections,
            "references": [
                {
                    "title": section["title"],
                    "page": section["page"],
                    "description": section["description"],
                }
                for section in sections
                if "reference" in section["title"].lower()
            ],
        }
