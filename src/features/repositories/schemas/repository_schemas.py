"""Pydantic models for repository settings exposed on the Consumer API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkingStrategyId = Literal[
    "fixed-overlap-based",
    "sentence-based",
    "paragraph-based",
    "section-based",
    "hierarchical",
    "semantic",
    "semantic-hierarchy",
    "sliding-window",
]


class EmbeddingModelSettings(BaseModel):
    """Embedding model configuration shared by processor and retrieval."""

    provider: Literal["local", "openai", "azure_openai", "aws", "anthropic"] = Field(
        ...,
        description="Embedding backend: local weights under src/models/ or a cloud API.",
    )
    model_id: str = Field(..., min_length=1, max_length=256, description="Provider-specific model identifier.")
    local_model_dir: str | None = Field(
        default=None,
        max_length=256,
        description="Directory name under src/models/ when provider=local.",
    )
    dimensions: int | None = Field(
        default=None,
        ge=1,
        description="Optional output dimension override (supported cloud models).",
    )
    credential_ref: str | None = Field(
        default=None,
        max_length=512,
        description="Platform secret reference when provider is not local.",
    )


class ExtractionModelSettings(BaseModel):
    """Optional LLM/rule configuration for metadata extraction processors."""

    provider: str | None = Field(default=None, max_length=64)
    model_id: str | None = Field(default=None, max_length=256)


class RepositoryKeyFieldConfig(BaseModel):
    field_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        description="Server-assigned key field identifier.",
    )
    name: str = Field(..., min_length=1, max_length=128, description="Field name to extract.")
    type: Literal["string", "integer", "float", "boolean", "date", "number", "datetime", "text"] = Field(
        default="string",
        description="Expected field type (Phase 3: string|integer|float|boolean|date; aliases: number|datetime|text).",
    )
    required: bool = Field(default=False, description="Whether extracted value is mandatory.")
    description: str | None = Field(default=None, max_length=512)


class RepositorySettings(BaseModel):
    """Processing and retrieval settings stored per repository (MySQL repository_settings.settings)."""

    embedding_model: EmbeddingModelSettings | dict[str, Any] | None = Field(
        default=None,
        description="Embedding model selection; see GET /repositories/embedding-models for allowed values.",
    )
    chunk_size: int | None = Field(
        default=None,
        ge=200,
        le=2000,
        description="Target chunk size in characters (200–2000).",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        description="Chunk overlap; must be smaller than chunk_size.",
    )
    chunking_strategy: ChunkingStrategyId | None = Field(
        default=None,
        description="Chunking algorithm; see GET /repositories/chunking-strategies for options and per-strategy config.",
    )
    chunking_config: dict[str, Any] | None = Field(
        default=None,
        description="Strategy-specific chunking parameters (see chunking-strategies catalog config_fields).",
    )
    indexing_strategy: Literal["weaviate_upsert"] | None = Field(
        default=None,
        description="Vector index write strategy; only weaviate_upsert is supported today.",
    )
    metadata_extraction: bool | None = Field(
        default=None,
        description="When true, enqueue metadata_extraction processor jobs.",
    )
    key_field_extraction: bool | None = Field(
        default=None,
        description="When true, enqueue key_field_extraction processor jobs.",
    )
    key_field_extraction_enabled: bool | None = Field(
        default=None,
        description="Alias for key_field_extraction. Preferred repository-level switch for key field extraction.",
    )
    key_fields: list[RepositoryKeyFieldConfig] | list[dict[str, Any]] | None = Field(
        default=None,
        description="Repository-level key fields to extract when key field extraction is enabled.",
    )
    folder_hierarchy: list[str] | None = Field(
        default=None,
        description=(
            "Optional ordered key-field IDs used to create repository logical folders. "
            "Leave empty for automatic grouping from discovered structured fields."
        ),
    )
    strict_key_field_page_scope: bool | None = Field(
        default=None,
        description=(
            "When true, key field extraction is limited to first 2 and last page only, "
            "and skips extraction if page metadata is missing."
        ),
    )
    validation_enabled: bool | None = Field(
        default=None,
        description="When true, enqueue document_validation after key_field_extraction.",
    )
    validation_confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for PASS vs WARNING (default 0.85).",
    )
    citation_retainment: bool | None = Field(
        default=None,
        description="Preserve citation/source spans in chunked output.",
    )
    template_extraction: bool | None = Field(
        default=None,
        description="When true, enqueue template_extraction processor jobs.",
    )
    reference_document_extraction: bool | None = Field(
        default=None,
        description="When true, enqueue reference_document_extraction processor jobs.",
    )
    conversion_for_rendering: bool | None = Field(
        default=None,
        description="When true, enqueue conversion_for_rendering processor jobs.",
    )
    intelligent_extraction: bool | None = Field(
        default=None,
        description="When true, enqueue content intelligence (intelligence_extraction) processor jobs.",
    )
    content_intelligence: bool | None = Field(
        default=None,
        description="Alias for intelligent_extraction — enables content intelligence extraction.",
    )
    document_type_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        description="Default document type UUID for documents in this repository.",
    )
    extraction_model: ExtractionModelSettings | dict[str, Any] | None = Field(
        default=None,
        description="Model hint passed to metadata extraction processors.",
    )
    reranking: bool | None = Field(
        default=None,
        description=(
            "Optional per-repository rerank default. None inherits platform "
            "RERANKER_ENABLED; true opts in; false opts out."
        ),
    )
    retrieval_search_mode: Literal["hybrid", "vector", "keyword"] | None = Field(
        default=None,
        description="Default retrieval mode for this repository.",
    )
    lexical_composition: bool | None = Field(
        default=None,
        description="When true, hybrid/keyword search uses BM25; when false, vector-only modes apply.",
    )


class CreateRepositoryRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique repository name; also used as the Weaviate collection name.",
    )
    owner_user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Optional owner user id. When omitted, API resolves owner from auth context or uses a system fallback.",
    )
    owner_user_name: str | None = Field(
        default=None,
        max_length=256,
        description="Optional display name for the repository owner; resolved from platform users when omitted.",
    )
    default_tenant_id: str | None = Field(default=None, max_length=256)
    key_field_extraction_enabled: bool | None = Field(
        default=None,
        description="Optional shortcut for settings.key_field_extraction_enabled during repository creation.",
    )
    key_fields: list[RepositoryKeyFieldConfig] | list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional shortcut for settings.key_fields during repository creation.",
    )
    settings: RepositorySettings | dict[str, Any] | None = Field(
        default=None,
        description="Initial processing/retrieval settings; defaults applied when omitted.",
    )


class UpdateRepositoryOwnerRequest(BaseModel):
    owner_user_name: str = Field(..., min_length=1, max_length=256)


class UpdateRepositoryRequest(BaseModel):
    """PATCH body for /repositories/{id} — editable repository metadata only.

    Lifecycle status changes use dedicated activate/archive/reactivate endpoints.
    Processing/retrieval knobs use PATCH /repositories/{id}/settings.
    """

    model_config = ConfigDict(extra="forbid")

    owner_user_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Display name for the repository owner.",
    )
    default_tenant_id: str | None = Field(
        default=None,
        max_length=256,
        description="Optional default Weaviate tenant id for this repository.",
    )
    settings: RepositorySettings | dict[str, Any] | None = Field(
        default=None,
        description="Optional nested settings patch (same semantics as PATCH .../settings).",
    )


class UpdateRepositorySettingsRequest(BaseModel):
    """PATCH body for /repositories/{id}/settings — merges into stored settings."""

    status: Literal["active", "archived", "configuring", "approved"] | None = Field(
        default=None,
        description="Lifecycle status; use POST .../activate for active unless platform admin.",
    )
    embedding_model: EmbeddingModelSettings | dict[str, Any] | None = Field(
        default=None,
        description="Embedding model selection; see GET /repositories/embedding-models for allowed values.",
    )
    chunk_size: int | None = Field(default=None, ge=200, le=2000)
    chunk_overlap: int | None = Field(default=None, ge=0)
    chunking_strategy: ChunkingStrategyId | None = None
    chunking_config: dict[str, Any] | None = None
    indexing_strategy: Literal["weaviate_upsert"] | None = None
    metadata_extraction: bool | None = None
    key_field_extraction: bool | None = None
    key_field_extraction_enabled: bool | None = None
    key_fields: list[RepositoryKeyFieldConfig] | list[dict[str, Any]] | None = None
    citation_retainment: bool | None = None
    template_extraction: bool | None = None
    reference_document_extraction: bool | None = None
    conversion_for_rendering: bool | None = None
    intelligent_extraction: bool | None = None
    content_intelligence: bool | None = None
    strict_key_field_page_scope: bool | None = None
    validation_enabled: bool | None = None
    validation_confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    document_type_id: str | None = Field(default=None, min_length=36, max_length=36)
    extraction_model: ExtractionModelSettings | dict[str, Any] | None = None
    reranking: bool | None = None
    retrieval_search_mode: Literal["hybrid", "vector", "keyword"] | None = None
    lexical_composition: bool | None = None


class EmbeddingModelOption(BaseModel):
    """One selectable embedding model entry."""

    provider: str
    model_id: str
    local_model_dir: str | None = None
    hub_id: str | None = None
    dimensions: int | None = None
    dimensions_configurable: bool | None = None
    requires_credential_ref: bool | None = None
    available: bool | None = Field(
        default=None,
        description="For local models, true when weights exist under src/models/.",
    )
    description: str | None = None
    embedding_model: dict[str, Any] = Field(
        ...,
        description="Ready-to-use embedding_model object for repository settings PATCH/POST.",
    )


class EmbeddingProviderCatalog(BaseModel):
    provider: str
    label: str
    requires_credential_ref: bool
    models: list[EmbeddingModelOption]


class EmbeddingModelCatalogResponse(BaseModel):
    """All allowed embedding_model choices grouped by provider."""

    default: dict[str, Any]
    local_models_root: str
    providers: list[EmbeddingProviderCatalog]
    options: list[EmbeddingModelOption]
    models: list[EmbeddingModelOption] = Field(
        ...,
        description="Alias of `options` for clients that expect a top-level `models` array.",
    )
    count: int


class ChunkingConfigFieldSpec(BaseModel):
    key: str
    scope: Literal["settings", "chunking_config"]
    label: str
    value_type: Literal["integer", "number", "boolean", "string"]
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    default: Any | None = None
    description: str | None = None


class ChunkingStrategyOption(BaseModel):
    strategy_id: ChunkingStrategyId
    label: str
    description: str
    sequence: int = Field(..., ge=1)
    config_fields: list[ChunkingConfigFieldSpec]
    example_settings: dict[str, Any]


class ChunkingStrategyCatalogResponse(BaseModel):
    default: ChunkingStrategyId
    strategies: list[ChunkingStrategyOption]
    strategy_ids: list[ChunkingStrategyId]
    count: int


class UpsertRepositorySettingsRequest(RepositorySettings):
    """PUT body for /repositories/{id}/settings — full settings upsert with defaults."""


class CreateRepositoryKeyFieldRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    type: Literal["string", "integer", "float", "boolean", "date", "number", "datetime", "text"] = "string"
    required: bool = False
    description: str | None = Field(default=None, max_length=512)


class PatchRepositoryKeyFieldsRequest(BaseModel):
    key_fields: list[RepositoryKeyFieldConfig] | list[dict[str, Any]] = Field(
        ...,
        description="Complete replacement list for repository key field configuration.",
    )


def settings_to_dict(settings: RepositorySettings | dict[str, Any] | None) -> dict[str, Any] | None:
    if settings is None:
        return None
    if hasattr(settings, "model_dump"):
        return settings.model_dump(exclude_none=True)
    return dict(settings)
