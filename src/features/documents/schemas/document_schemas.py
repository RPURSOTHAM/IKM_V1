from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RepositoryType = Literal["local", "s3", "azure", "sharepoint"]


class RepositoryConfigRequest(BaseModel):
    repository_type: RepositoryType = "local"
    config: dict[str, Any] = Field(default_factory=dict)


class DocumentRegisterRequest(BaseModel):
    document_name: str = Field(..., min_length=1)
    original_file_name: str | None = None
    repository_type: RepositoryType = "local"
    repository_path: str | None = None
    collection_name: str | None = None
    tenant_id: str | None = None
    repository_id: str | None = Field(default=None, min_length=36, max_length=36)
    document_type_id: str | None = Field(default=None, min_length=36, max_length=36)
    submit_for_processing: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    document_id: str
    document_name: str
    original_file_name: str
    document_type: str
    repository_type: str
    repository_path: str
    collection_name: str
    tenant_id: str | None = None
    status: str
    upload_timestamp: float
    queue_submission_timestamp: float | None = None
    processing_completion_timestamp: float | None = None
    error_details: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    metadata_updated_at: float | None = None
    repository_id: str | None = None
    document_type_id: str | None = Field(
        default=None,
        description="Effective document type UUID assigned to this document.",
    )
    document_type_name: str | None = Field(
        default=None,
        description="Resolved display name of the assigned document type.",
    )
    validation_status: str | None = Field(
        default=None,
        description="Phase 4 validation document status: VALID | WARNING | INVALID.",
    )
    job_id: str | None = None
    batch_id: str | None = None


class DocumentValidationResponse(BaseModel):
    document_id: str
    status: str | None = None
    document_status: str | None = None
    missing_required_fields: list[str] = Field(default_factory=list)
    invalid_fields: list[dict[str, Any]] = Field(default_factory=list)
    low_confidence_fields: list[dict[str, Any]] = Field(default_factory=list)
    confidence_threshold: float | None = None
    source: str | None = None
    completed_at: str | None = None


class DocumentDeletionResponse(BaseModel):
    """Result of a full document delete (source file + all indexed artifacts)."""

    document_id: str
    original_file_name: str | None = None
    document_name: str | None = None
    repository_id: str | None = None
    deleted: bool = True
    purge: dict[str, Any] = Field(default_factory=dict)


class UploadRejection(BaseModel):
    """Per-file rejection within a multi-document upload batch."""

    filename: str | None = None
    code: str
    message: str
    existing_document_id: str | None = None
    existing_document_name: str | None = None
    existing_status: str | None = None
    existing_repository_id: str | None = None


class UploadResponse(BaseModel):
    batch_id: str
    uploaded_count: int
    documents: list[DocumentResponse]
    rejected_count: int = 0
    rejected: list[UploadRejection] = Field(default_factory=list)


class DocumentJobStatusResponse(BaseModel):
    """Processing status sourced from MySQL ``document_job`` (no scheduler or queue inspection)."""

    document_id: str
    job: dict[str, Any]
    document: DocumentResponse


class UpdateDocumentMetadataRequest(BaseModel):
    metadata: dict[str, Any] = Field(
        ...,
        description="User-defined metadata key/value pairs to store on the document.",
    )
    merge: bool = Field(
        default=True,
        description="When true (default), merge into existing metadata. When false, replace user metadata but preserve system keys.",
    )
    change_reason: str | None = Field(
        default=None,
        max_length=512,
        description="Optional operator comment recorded in the audit trail.",
    )


class DocumentPreviewResponse(BaseModel):
    """Read-only preview payload for a document (HTML or native inline format)."""

    document_id: str
    document_name: str
    original_file_name: str
    format: str
    content: str | None = Field(
        default=None,
        description="HTML or text preview body when conversion was performed.",
    )
    media_type: str
    read_only: bool = True
    source: str = Field(
        description="redis_cache | inline_conversion | original_file",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunking/indexing facts including file_size_bytes and chunk_count when available.",
    )


class DocumentInfoResponse(BaseModel):
    """Facts captured during chunking, vectorization, and indexing."""

    document_id: str | None = None
    document_name: str | None = None
    original_file_name: str | None = None
    document_type: str | None = None
    file_extension: str | None = None
    file_size_bytes: int | None = None
    page_count: int | None = None
    character_count: int | None = None
    line_count: int | None = None
    chunk_count: int | None = None
    embedded_chunk_count: int | None = None
    collection_name: str | None = None
    tenant_id: str | None = None
    repository_id: str | None = None
    citation_retainment: bool | None = None
    indexing_status: str | None = None
    processor_status: str | None = None
    completed_at: str | None = None
    result_location: str | None = None
    source: str | None = None

    model_config = {"extra": "allow"}


class ExtractedFieldValue(BaseModel):
    field_name: str
    value: Any | None = None
    confidence: float | None = None
    extraction_model: str | None = None


class DocumentTypeMetadataResponse(BaseModel):
    """Structured metadata extracted via metadata_extraction processor."""

    document_id: str | None = None
    document_type_id: str | None = None
    status: str | None = None
    fields: list[ExtractedFieldValue] = Field(default_factory=list)
    extracted_field_count: int = 0
    processor_status: str | None = None
    completed_at: str | None = None
    result_location: str | None = None
    error_details: str | None = None
    source: str | None = None


class DocumentKeyFieldsResponse(BaseModel):
    """Key fields extracted via key_field_extraction processor."""

    document_id: str
    document_type_id: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    updated_at: str | None = None
    source: str | None = None


class UpdateDocumentKeyFieldsRequest(BaseModel):
    fields: dict[str, Any] = Field(
        ...,
        description="Key-field values to override by field name.",
    )
    reason: str | None = Field(
        default=None,
        max_length=512,
        description="Optional reason recorded in key-field audit events.",
    )


class DocumentKeyFieldHistoryResponse(BaseModel):
    document_id: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class DocumentMetadataBundleResponse(BaseModel):
    document_id: str
    document_info: DocumentInfoResponse
    type_metadata: DocumentTypeMetadataResponse
    processing: dict[str, Any] = Field(default_factory=dict)


class DocumentChunksResponse(BaseModel):
    """Indexed Weaviate chunks for one document."""

    document_id: str
    document_name: str
    original_file_name: str | None = None
    collection_name: str
    tenant: str | None = None
    offset: int
    limit: int
    returned: int
    include_text: bool
    include_vector: bool
    document_info: dict[str, Any] = Field(default_factory=dict)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
