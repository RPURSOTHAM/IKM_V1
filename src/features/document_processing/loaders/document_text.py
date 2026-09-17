"""Load structured blocks and lightweight document facts for all processor types."""

from __future__ import annotations

from pathlib import Path

from src.features.document_processing.loaders.docxloader import DocxLoader
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.loaders.pdfloader import PdfLoader
from src.features.documents.infrastructure.content.page_count import page_count_from_file


def _ensure_format_processor_enabled(suffix: str) -> None:
    """Raise when the deployment disables the loader required for this file type."""
    from src.features.document_processing.shared_processor.deployment import (
        ProcessorConfigurationError,
        ProcessorConfigurationProvider,
    )

    provider = ProcessorConfigurationProvider.instance()
    if suffix == ".pdf" and not provider.is_processor_enabled("pdf"):
        raise ProcessorConfigurationError(
            "PDF processor is disabled for this IKM deployment."
        )
    if suffix == ".docx" and not provider.is_processor_enabled("docx"):
        raise ProcessorConfigurationError(
            "DOCX processor is disabled for this IKM deployment."
        )
    if suffix in {".pptx", ".ppt"} and not provider.is_processor_enabled("pptx"):
        raise ProcessorConfigurationError(
            "PowerPoint processor is disabled for this IKM deployment."
        )
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        if not provider.is_processor_enabled("image"):
            raise ProcessorConfigurationError(
                "Image processor is disabled for this IKM deployment."
            )
        if not provider.is_processor_enabled("ocr"):
            raise ProcessorConfigurationError(
                "OCR processor is disabled for this IKM deployment."
            )
    if suffix in {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
        ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v", ".wmv",
    } and not provider.is_processor_enabled("media"):
        raise ProcessorConfigurationError(
            "Media transcription processor is disabled for this IKM deployment."
        )


def _resolve_ppt_path(document_path: Path) -> Path:
    """Legacy .ppt → convert to .pptx via LibreOffice when available."""
    if document_path.suffix.lower() != ".ppt":
        return document_path
    converted = document_path.with_suffix(".pptx")
    if converted.is_file():
        return converted
    import shutil
    import subprocess

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise ValueError(
            "Legacy .ppt requires LibreOffice (soffice) to convert to .pptx, "
            "or re-save the file as .pptx."
        )
    out_dir = document_path.parent
    completed = subprocess.run(
        [
            soffice,
            "--headless",
            "--convert-to",
            "pptx",
            "--outdir",
            str(out_dir),
            str(document_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not converted.is_file():
        raise ValueError(
            f"Failed converting .ppt to .pptx: {(completed.stderr or completed.stdout or '')[-500:]}"
        )
    return converted


def _blocks_from_media_transcript(document_path: Path) -> list[Block]:
    from src.features.document_processing.media.cih_media import load_transcript_sidecar

    payload = load_transcript_sidecar(document_path)
    if not payload:
        raise ValueError(
            f"No transcript available for media file {document_path.name}. "
            "Wait for media_transcription to complete."
        )
    blocks: list[Block] = []
    segments = payload.get("segments") or []
    if isinstance(segments, list) and segments:
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            index = int(seg.get("index") or (len(blocks) + 1))
            blocks.append(
                Block(
                    text=text,
                    page=index,
                    line_number=1,
                    block_type="text",
                    component_type="transcript",
                    metadata={
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "source": "media_transcription",
                    },
                )
            )
    if not blocks:
        full_text = str(payload.get("full_text") or "").strip()
        if full_text:
            blocks = [
                Block(
                    text=line.strip(),
                    page=1,
                    line_number=i,
                    block_type="text",
                    component_type="transcript",
                    metadata={"source": "media_transcription"},
                )
                for i, line in enumerate(full_text.splitlines(), start=1)
                if line.strip()
            ]
    if not blocks:
        raise ValueError(f"Transcript for {document_path.name} is empty.")
    return blocks


def load_document_blocks(
    document_path: Path,
    mask_sensitive: bool = True,
    document_id: str | None = None,
    *,
    citation_retainment: bool = True,
) -> tuple[list[Block], dict]:
    suffix = document_path.suffix.lower()
    _ensure_format_processor_enabled(suffix)
    if suffix == ".pdf":
        blocks = PdfLoader().load(document_path)
        from src.features.document_processing.loaders.scanned_pdf_ocr import apply_ocr_to_scanned_pdf

        blocks = apply_ocr_to_scanned_pdf(document_path, blocks)
    elif suffix == ".docx":
        blocks = DocxLoader().load(document_path, citation_retainment=citation_retainment)
    elif suffix in {".pptx", ".ppt"}:
        from src.features.document_processing.loaders.pptxloader import PptxLoader

        blocks = PptxLoader().load(_resolve_ppt_path(document_path))
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        from src.features.document_processing.loaders.image_ocr_loader import ImageOcrLoader

        blocks = ImageOcrLoader().load(document_path)
    elif suffix in {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
        ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg", ".m4v", ".wmv",
    }:
        blocks = _blocks_from_media_transcript(document_path)
    elif suffix in {".txt", ".text"}:
        text = document_path.read_text(encoding="utf-8", errors="replace")
        blocks = [
            Block(text=line.strip(), page=1)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        raise ValueError(f"Unsupported document extension: {suffix}")

    # When the image capability is disabled, drop image blocks entirely.
    try:
        from src.features.document_processing.shared_processor.deployment import ProcessorConfigurationProvider
        from src.features.document_processing.loaders.component_classification import is_image_block

        if not ProcessorConfigurationProvider.is_enabled("image"):
            blocks = [block for block in blocks if not is_image_block(block)]
    except Exception:
        pass

    pages = {block.page for block in blocks if block.page}
    full_text = "\n".join(block.text for block in blocks if block.text)
    file_size_bytes: int | None = None
    try:
        file_size_bytes = int(document_path.stat().st_size)
    except OSError:
        file_size_bytes = None
    page_count = max(pages) if pages else 1
    word_count = len(full_text.split())
    from_file = page_count_from_file(document_path, word_count=word_count, block_count=len(blocks))
    if from_file is not None and from_file > page_count:
        page_count = from_file
    ocr_used = any(
        isinstance(getattr(block, "metadata", None), dict) and block.metadata.get("ocr_used")
        for block in blocks
    )
    metadata = {
        "original_file_name": document_path.name,
        "document_name": document_path.stem,
        "page_count": page_count,
        "character_count": len(full_text),
        "line_count": len(blocks),
        "file_extension": suffix.lstrip("."),
        "file_size_bytes": file_size_bytes,
        "ocr_used": ocr_used,
    }
    if mask_sensitive:
        try:
            from src.features.security.review.human_review_queue import get_reviewer_override_for_document

            candidate_ids = list(
                dict.fromkeys(
                    [document_id or "", document_path.stem, document_path.name]
                )
            )
            override = None
            for candidate in candidate_ids:
                if not candidate:
                    continue
                override = get_reviewer_override_for_document(candidate)
                if override:
                    break
            if override and override.get("pipeline_status") == "block":
                raise ValueError("Document blocked by reviewer decision")
            # Honor terminal reviewer Mask only — never re-run the upload scan here
            # (that would reopen APPROVED → PENDING and break Allow/Mask processing).
            if override and override.get("pipeline_status") == "mask_and_allow":
                mask_sensitive = True
            else:
                mask_sensitive = False
        except ValueError:
            raise
        except Exception:
            mask_sensitive = False

    if mask_sensitive:
        try:
            from src.features.security.dlp.sensitive_data_detector import mask_text_content
            for block in blocks:
                if block.text:
                    block.text = mask_text_content(block.text)
            metadata["masking_applied"] = True
        except Exception:
            pass

    return blocks, metadata
