"""Export-time DLP: scan and redact document bytes before download."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from src.features.security.audit.security_event_logger import EVENT_EXPORT_SCAN, log_security_event
from src.features.security.dlp.policy_loader import scan_keyword_policies
from src.features.security.dlp.sensitive_data_detector import mask_text_content, scan_document_for_api

logger = logging.getLogger(__name__)


class ExportSecurityPipeline:
    """Scan original files at download time and return safe content."""

    def moderate_file(
        self,
        file_path: Path,
        *,
        document_id: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Export file not found: {path}")

        suffix = path.suffix.lower()
        scan = scan_document_for_api(path)
        detections = list(scan.get("detections") or [])

        # Keyword policy scan on extracted/raw text
        try:
            from src.features.document_processing.loaders.document_text import load_document_blocks

            blocks, _meta = load_document_blocks(path, mask_sensitive=False)
            full_text = "\n".join(b.text for b in blocks if getattr(b, "text", None))
        except Exception:
            try:
                full_text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                full_text = ""

        kw = scan_keyword_policies(full_text)
        detections.extend(kw.get("matches") or [])

        high = any(str(d.get("severity", "")).lower() in {"high", "critical"} for d in detections)
        blocked = bool(scan.get("upload_blocked")) or kw.get("action") == "block"

        decision = "block" if blocked and high else ("mask" if detections else "allow")
        if kw.get("action") == "human_review":
            decision = "mask"

        log_security_event(
            EVENT_EXPORT_SCAN,
            decision=decision,
            reason=scan.get("reason") or kw.get("action") or "export scan",
            severity="high" if high else "medium" if detections else "low",
            policy="export_dlp",
            user=user,
            document_id=document_id,
            document_name=path.name,
            metadata={"detection_count": len(detections), "extension": suffix},
        )

        if decision == "block":
            safe_text = (
                "Download blocked by export DLP policy. "
                "The document contains high-risk sensitive data."
            )
            return {
                "status": "block",
                "filename": f"{path.stem}.blocked.txt",
                "media_type": "text/plain",
                "content": safe_text.encode("utf-8"),
                "detections": detections,
                "decision": decision,
            }

        masked_text = mask_text_content(full_text) if full_text else ""
        if decision == "allow" and not detections:
            return {
                "status": "allow",
                "filename": path.name,
                "media_type": _media_type(suffix),
                "content": path.read_bytes(),
                "detections": detections,
                "decision": decision,
            }

        # Always emit a redacted text/docx-safe payload for masked exports.
        content, filename, media_type = _render_masked_export(path, masked_text or full_text, suffix)
        return {
            "status": "mask",
            "filename": filename,
            "media_type": media_type,
            "content": content,
            "detections": detections,
            "decision": decision,
        }


def _media_type(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def _render_masked_export(path: Path, masked_text: str, suffix: str) -> tuple[bytes, str, str]:
    if suffix == ".docx":
        try:
            from docx import Document

            doc = Document()
            for paragraph in (masked_text or "").splitlines() or [""]:
                doc.add_paragraph(paragraph)
            buf = io.BytesIO()
            doc.save(buf)
            return buf.getvalue(), f"{path.stem}.redacted.docx", _media_type(".docx")
        except Exception as exc:
            logger.warning("DOCX redaction fallback to TXT: %s", exc)

    # PDF binary redaction without a full rewrite stack: emit redacted text.
    name = f"{path.stem}.redacted.txt"
    return (masked_text or "").encode("utf-8"), name, "text/plain"


def moderate_export_file(file_path: Path, **kwargs: Any) -> dict[str, Any]:
    return ExportSecurityPipeline().moderate_file(file_path, **kwargs)
