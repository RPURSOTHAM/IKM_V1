"""Pydantic schemas for external evidence search/import APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

EvidenceSource = Literal["pubmed", "clinicaltrials.gov"]


class PubMedSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mesh_terms: list[str] = Field(default_factory=list, max_length=20)
    title: str | None = Field(default=None, max_length=500)
    max_results: int | None = Field(default=None, ge=1, le=100)


class ClinicalTrialsSearchRequest(BaseModel):
    condition: str | None = Field(default=None, max_length=500)
    nct_id: str | None = Field(default=None, max_length=32)
    title: str | None = Field(
        default=None,
        max_length=500,
        description="Optional title search (ClinicalTrials.gov query.titles).",
    )
    max_results: int | None = Field(default=None, ge=1, le=100)

    @field_validator("nct_id")
    @classmethod
    def _normalize_nct(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip().upper()
        if not text:
            return None
        # Match ClinicalTrialsClient.normalize_nct_id — reject early as 422.
        import re

        if not re.fullmatch(r"NCT\d{8}", text):
            raise ValueError("nct_id must match NCT######## (8 digits)")
        return text


class EvidencePreviewItem(BaseModel):
    source: EvidenceSource
    record_id: str
    title: str
    summary: str | None = None
    published_date: str | None = None
    pdf_available: bool = False
    full_text_available: bool = False
    already_imported: bool = False
    external_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceSearchResponse(BaseModel):
    source: EvidenceSource
    query: str
    max_results: int
    count: int
    results: list[EvidencePreviewItem] = Field(default_factory=list)


class EvidenceImportItem(BaseModel):
    source: EvidenceSource
    record_id: str = Field(..., min_length=1, max_length=64)


class EvidenceImportRequest(BaseModel):
    items: list[EvidenceImportItem] = Field(..., min_length=1, max_length=25)
    document_type_id: str | None = Field(default=None, min_length=36, max_length=36)
    submit_for_processing: bool = True


class EvidenceImportResultItem(BaseModel):
    source: EvidenceSource
    record_id: str
    status: Literal["imported", "already_imported", "failed"]
    document_id: str | None = None
    message: str | None = None


class EvidenceImportResponse(BaseModel):
    repository_id: str
    results: list[EvidenceImportResultItem] = Field(default_factory=list)


class EvidenceRegistryItem(BaseModel):
    repository_id: str
    source: EvidenceSource
    source_record_id: str
    published_date: str | None = None
    document_id: str | None = None
    imported_at: str | None = None
    status: str
    external_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRegistryListResponse(BaseModel):
    repository_id: str
    count: int
    items: list[EvidenceRegistryItem] = Field(default_factory=list)


class PubmedAgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    conversation: list[dict[str, str]] = Field(default_factory=list)
    model_id: str | None = None
    provider: str | None = Field(
        default=None,
        description="azure_openai (default when configured) or openai / openai_compatible",
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    reranking_enabled: bool | None = None
    query_expansion_enabled: bool | None = None
    parallel_retrieval_enabled: bool | None = None


class PubmedAgentResponse(BaseModel):
    allowed: bool
    safe: bool = True
    risk_type: str | None = None
    severity: str = "low"
    reason: str | None = None
    safe_message: str | None = None
    action: str = "allow"
    answer: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    grounded: bool = True
    evidence_sufficient: bool = True
    model_id: str | None = None
    provider: str | None = None
    repository_id: str
    citation_format: str = "[REPO:DOCUMENT:PAGE]"
