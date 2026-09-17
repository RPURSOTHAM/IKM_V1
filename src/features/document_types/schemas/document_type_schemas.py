"""Pydantic models for repository-scoped document type and key field API payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.features.document_types.domain.models import KeyFieldType, MetadataDataType


class CreateDocumentTypeRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str | None = None
    parent_document_type_id: str | None = None
    code: str | None = Field(default=None, max_length=64)


class CreateChildDocumentTypeRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    code: str | None = Field(default=None, max_length=64)
    description: str | None = None


class UpdateDocumentTypeRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    code: str | None = Field(default=None, max_length=64)
    description: str | None = None
    is_active: bool | None = None


_KEY_FIELD_TYPE_LITERAL = Literal[
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    # Backward-compatible aliases
    "number",
    "datetime",
    "text",
]


class CreateKeyFieldRequest(BaseModel):
    field_name: str = Field(..., min_length=1, max_length=128)
    field_type: _KEY_FIELD_TYPE_LITERAL = "string"
    required: bool = False
    default_value: Any | None = None
    description: str | None = None

    @field_validator("field_type")
    @classmethod
    def validate_field_type(cls, value: str) -> str:
        if value not in KeyFieldType.values():
            raise ValueError(
                "field_type must be one of: string, integer, float, boolean, date "
                "(aliases: number, datetime, text)"
            )
        return KeyFieldType.canonical(value)


class UpdateKeyFieldRequest(BaseModel):
    field_type: _KEY_FIELD_TYPE_LITERAL | None = None
    required: bool | None = None
    default_value: Any | None = None
    description: str | None = None

    @field_validator("field_type")
    @classmethod
    def validate_field_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in KeyFieldType.values():
            raise ValueError(
                "field_type must be one of: string, integer, float, boolean, date "
                "(aliases: number, datetime, text)"
            )
        return KeyFieldType.canonical(value)


class CreateMetadataFieldRequest(BaseModel):
    field_name: str = Field(..., min_length=1, max_length=128)
    display_label: str = Field(..., min_length=1, max_length=256)
    data_type: Literal["string", "number", "boolean", "date", "datetime", "enum", "text"]
    required: bool = False
    default_value: Any | None = None
    enum_values: list[str] | None = None
    max_length: int | None = Field(default=None, ge=1)
    field_group: str = Field(default="general", min_length=1, max_length=64)
    group_label: str = Field(default="General", min_length=1, max_length=256)
    group_sequence: int = Field(default=100, ge=0, description="Lower values appear first among groups.")
    field_sequence: int = Field(default=100, ge=0, description="Lower values appear first within a group.")

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, value: str) -> str:
        if value not in MetadataDataType.values():
            raise ValueError(f"data_type must be one of: {', '.join(sorted(MetadataDataType.values()))}")
        return value


class UpdateMetadataFieldRequest(BaseModel):
    display_label: str | None = Field(default=None, min_length=1, max_length=256)
    required: bool | None = None
    default_value: Any | None = None
    enum_values: list[str] | None = None
    max_length: int | None = Field(default=None, ge=1)
    field_group: str | None = Field(default=None, min_length=1, max_length=64)
    group_label: str | None = Field(default=None, min_length=1, max_length=256)
    group_sequence: int | None = Field(default=None, ge=0)
    field_sequence: int | None = Field(default=None, ge=0)
    is_active: bool | None = None
