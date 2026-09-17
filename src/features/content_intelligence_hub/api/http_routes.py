"""Content Intelligence Hub API — document facade + Ollama image summary.

Mounted under ``settings.api_prefix`` as ``/cih``.
"""

from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from src.application.consumer_api.security import security_scheme
from src.features.documents.schemas.document_schemas import DocumentJobStatusResponse, UploadResponse
from src.features.documents.application.document_service import DocumentReceiverService

router = APIRouter(
    prefix="/cih",
    tags=["Content Intelligence Hub"],
    dependencies=[Depends(security_scheme)],
)

_CIH_FILE_TYPES = (
    "PDF (.pdf), Word (.docx), PowerPoint (.pptx, .ppt), "
    "images (.png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp), "
    "audio (.wav, .mp3, .m4a, .aac, .flac, .ogg), "
    "video (.mp4, .mov, .avi, .mkv, .webm)"
)

_UPLOAD_DESCRIPTION = f"""
Upload a file for Content Intelligence Hub extraction.

**How to test in Swagger**

1. Click **Authorize** and paste a Bearer token from `POST /api/v1/auth/token`.
2. Get a repository id from `GET /api/v1/repositories/options`.
3. Choose a file (see supported types below).
4. Set **repository_id**.
5. Leave **submit_for_processing** as `true`.
6. Execute. Copy `document_id` from the response.
7. Call **Get processing status** until the job is `COMPLETED`.
8. Then call **Get extracted content**, **Get chunks**, and **Get metadata**.
9. For audio or video, also call **Get transcript**.
10. For images, call **Summarize image (upload)** or **Summarize uploaded image**
    (uses Ollama `OPEN_WEIGHT_VLM_MODEL`, e.g. gemma3:27b).

**Supported files:** {_CIH_FILE_TYPES}

Transcription uses updated CIH ASR settings (`ASR_MODEL_NAME` / `CIH_FASTER_WHISPER_MODEL`).
Image summary uses Ollama (`OPEN_WEIGHT_VLM_*` / `OLLAMA_*`).
""".strip()


def _form_submit_for_processing(value: Any) -> bool:
    """Parse multipart submit_for_processing. Default True when omitted/empty."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "null", "none"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    return True


def _form_optional_id(value: Any) -> str | None:
    """Normalize multipart UUID/id fields; treat blank/null tokens as omitted."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "undefined", "string"}:
        return None
    return text


def _svc() -> DocumentReceiverService:
    return DocumentReceiverService()


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="1. Upload content",
    operation_id="cihUploadDocuments",
    description=_UPLOAD_DESCRIPTION,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "cih-upload": {
                            "summary": "Upload and start processing",
                            "description": "Set repository_id from GET /api/v1/repositories/options. Choose a supported file.",
                            "value": {
                                "repository_id": "02c5bdde-17db-4b01-b481-8872b0a8cf85",
                                "submit_for_processing": "true",
                            },
                        }
                    }
                }
            }
        }
    },
)
async def cih_upload_documents(
    files: List[UploadFile] = File(
        ...,
        description=(
            "Choose one or more files to extract. "
            f"Supported: {_CIH_FILE_TYPES}."
        ),
    ),
    repository_id: str | None = Form(
        default=None,
        description="Required. Copy a repository id from GET /api/v1/repositories/options.",
    ),
    collection_name: str | None = Form(
        default=None,
        description="Optional. Leave blank to use the repository default collection.",
    ),
    tenant_id: str | None = Form(
        default=None,
        description="Optional. Leave blank unless your environment uses tenants.",
    ),
    document_type_id: str | None = Form(
        default=None,
        description="Optional. Leave blank to use the repository default document type.",
    ),
    submit_for_processing: str = Form(
        default="true",
        description="Use true to start extraction after upload. Use false only to store the file without processing.",
    ),
) -> UploadResponse:
    return await _svc().upload_documents(
        files,
        repository_id=_form_optional_id(repository_id),
        collection_name=_form_optional_id(collection_name) if collection_name is not None else None,
        tenant_id=_form_optional_id(tenant_id) if tenant_id is not None else None,
        document_type_id=_form_optional_id(document_type_id),
        submit_for_processing=_form_submit_for_processing(submit_for_processing),
    )


@router.post(
    "/image-summary",
    summary="7. Summarize image (upload)",
    operation_id="cihSummarizeImageUpload",
    description=(
        "Upload an image and get a visual summary from the updated CIH Ollama VLM "
        "(`OPEN_WEIGHT_VLM_MODEL`, e.g. gemma3:27b on ollama.com). "
        "Does not require a repository. Authorize with a Bearer token."
    ),
)
async def cih_summarize_image_upload(
    file: UploadFile = File(..., description="Image file (.png, .jpg, .jpeg, .webp, .bmp, .tif)"),
    prompt: str | None = Form(
        default=None,
        description="Optional custom prompt. Leave blank for the default CIH image-summary prompt.",
    ),
) -> dict:
    from src.features.content_intelligence_hub.application.image_summary import summarize_upload

    return await summarize_upload(file, prompt=prompt)


@router.get(
    "/{document_id}/status",
    response_model=DocumentJobStatusResponse,
    summary="2. Get processing status",
    operation_id="cihGetDocumentStatus",
    description=(
        "Paste the document_id from the upload response. "
        "Repeat this call until the job status is COMPLETED (or failed). "
        "Do not call extraction until processing has finished."
    ),
)
def cih_get_document_status(document_id: str) -> DocumentJobStatusResponse:
    return _svc().get_document_status(document_id)


@router.get(
    "/{document_id}/extraction",
    summary="3. Get extracted content",
    operation_id="cihGetDocumentExtraction",
    description=(
        "Returns the extracted text for the uploaded file (pages, slides, or transcript blocks). "
        "Use after status is COMPLETED. "
        "For audio/video this is the transcript; returns 404 until media_transcription finishes."
    ),
)
def cih_get_document_extraction(document_id: str) -> dict:
    return _svc().get_document_extraction(document_id)


@router.get(
    "/{document_id}/chunks",
    summary="4. Get chunks",
    operation_id="cihGetDocumentChunks",
    description=(
        "Returns the searchable text pieces created from the extracted content. "
        "Use after status is COMPLETED."
    ),
)
def cih_get_document_chunks(
    document_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    include_text: bool = Query(default=True),
    include_vector: bool = Query(default=False),
) -> dict:
    payload = _svc().get_document_chunks(
        document_id,
        limit=limit,
        offset=offset,
        include_text=include_text,
        include_vector=include_vector,
    )
    if include_text:
        from src.features.security.moderation.egress_moderator import redact_chunks_payload

        payload = redact_chunks_payload(payload, document_id=document_id)
    return payload


@router.get(
    "/{document_id}/metadata",
    summary="5. Get metadata",
    operation_id="cihGetDocumentMetadata",
    description="Returns file and processing metadata for the uploaded document. Use after status is COMPLETED.",
)
def cih_get_document_metadata(document_id: str) -> dict:
    return _svc().get_document_metadata_bundle(document_id)


@router.get(
    "/{document_id}/transcript",
    summary="6. Get transcript (audio and video)",
    operation_id="cihGetDocumentTranscript",
    description=(
        "Returns timed speech text for audio and video files. "
        "Use after status is COMPLETED. "
        "This is empty or not found for PDF, Word, and PowerPoint."
    ),
)
def cih_get_document_transcript(document_id: str) -> dict:
    return _svc().get_document_transcript(document_id)


@router.get(
    "/{document_id}/image-summary",
    summary="8. Summarize uploaded image",
    operation_id="cihSummarizeDocumentImage",
    description=(
        "Summarize an already-uploaded image document with Ollama "
        "(`OPEN_WEIGHT_VLM_MODEL` / gemma3:27b). "
        "document_id must point to a PNG/JPG/etc. file on disk."
    ),
)
def cih_summarize_document_image(
    document_id: str,
    prompt: str | None = Query(default=None, description="Optional custom prompt."),
) -> dict:
    from src.features.content_intelligence_hub.application.image_summary import summarize_document_image

    return summarize_document_image(document_id, prompt=prompt)
