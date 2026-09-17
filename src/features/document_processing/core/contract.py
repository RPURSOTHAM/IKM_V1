"""Unified processor input/output contract — every processor type uses the same envelope."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from src.features.document_processing.shared_processor.types import ProcessorType, normalize_processor_type
from src.features.chunking.domain.chunking_strategy import coerce_optional_chunking_strategy


class RepositorySettings(BaseModel):
    """Repository-scoped processing configuration passed with every job."""

    chunk_size: int | None = None
    chunk_overlap_sentences: int | None = None
    chunking_strategy: str | None = None
    chunking_config: dict[str, Any] | None = None
    citation_retainment: bool | None = None
    embedding_model_name: str | None = None
    embedding_model_dir: str | None = None
    extraction_model: dict[str, Any] | None = None
    document_type_id: str | None = None
    document_type_name: str | None = None
    metadata_fields: list[dict[str, Any]] = Field(default_factory=list)
    key_fields: list[dict[str, Any]] = Field(default_factory=list)
    strict_key_field_page_scope: bool | None = None

    @field_validator("metadata_fields", "key_fields", mode="before")
    @classmethod
    def _coerce_field_lists(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @field_validator("chunking_strategy", mode="before")
    @classmethod
    def _normalize_chunking_strategy(cls, value: Any) -> str | None:
        return coerce_optional_chunking_strategy(None if value is None else str(value))

    @field_validator("embedding_model_name", "embedding_model_dir", mode="before")
    @classmethod
    def _blank_strings_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ProcessRequest(BaseModel):
    """Standard job payload accepted by every processor container via ``POST /process``."""

    processor_type: str = ProcessorType.CHUNKING_VECTORIZING.value
    document_id: str
    document_name: str
    original_file_name: str | None = None
    document_path: str | None = None
    collection_name: str | None = None
    tenant_id: str | None = None
    repository_id: str | None = None
    repository_settings: RepositorySettings | None = None
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None
    model_name: str | None = None
    model_dir: str | None = None
    chunk_size: int | None = None
    chunk_overlap_sentences: int | None = None
    min_content_words: int | None = None
    chunking_strategy: str | None = None
    chunking_config: dict[str, Any] | None = None
    citation_retainment: bool | None = None
    document_type_id: str | None = None
    document_type_name: str | None = None
    metadata_fields: list[dict[str, Any]] = Field(default_factory=list)
    key_fields: list[dict[str, Any]] = Field(default_factory=list)
    effective_fields: list[dict[str, Any]] = Field(default_factory=list)
    extraction_model: dict[str, Any] | None = None
    strict_key_field_page_scope: bool = False
    validation_confidence_threshold: float | None = None
    # When true, the caller already ran the upload security pipeline (e.g. Streamlit).
    # Skip the duplicate in-container scan to avoid OOM on small Docker Desktop VMs.
    security_prevalidated: bool = False
    # Repository scheduling hints (optional; used for logging / diagnostics).
    enabled_processor_types: list[str] = Field(default_factory=list)
    template_extraction: bool | None = None

    @field_validator("metadata_fields", "key_fields", "effective_fields", mode="before")
    @classmethod
    def _coerce_field_lists(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @field_validator("strict_key_field_page_scope", "security_prevalidated", mode="before")
    @classmethod
    def _coerce_bool_defaults(cls, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "null", "none"}:
                return False
            if text in {"1", "true", "yes", "on", "y"}:
                return True
            if text in {"0", "false", "no", "off", "n"}:
                return False
        return bool(value)

    @field_validator("chunking_strategy", mode="before")
    @classmethod
    def _normalize_chunking_strategy(cls, value: Any) -> str | None:
        return coerce_optional_chunking_strategy(None if value is None else str(value))

    @field_validator("model_name", "model_dir", "weaviate_url", "weaviate_api_key", mode="before")
    @classmethod
    def _blank_strings_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def normalized_processor_type(self) -> str:
        return normalize_processor_type(self.processor_type)


class ProcessorResult(BaseModel):
    """Standard processor output returned to the runtime host after a successful run."""

    processor_type: str
    document_id: str
    storage_backend: str
    result_location: str | None = None
    document_metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


StatusCallback = Callable[[str, float | None], None]
