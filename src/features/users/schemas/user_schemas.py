"""Pydantic models for platform security API payloads exposed on the Consumer API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlatformRoleOption(BaseModel):
    role_id: str = Field(..., description="Stable platform role key.")
    label: str = Field(..., description="Display label for UI.")
    description: str = Field(..., description="Role capabilities summary.")
    sequence: int = Field(..., ge=1, description="Display order — lower values appear first.")
    assignable_on_create: bool = Field(
        ...,
        description="Whether POST /platform/users may set this role in platform_role.",
    )
    grant_endpoint: str | None = Field(
        default=None,
        description="POST path to grant this role when not assignable on create.",
    )
    revoke_endpoint: str | None = Field(
        default=None,
        description="DELETE path to revoke this role when supported.",
    )
    grant_requires_role: str | None = Field(
        default=None,
        description="Minimum platform role required to grant this role (e.g. administrator).",
    )


class TokenRequest(BaseModel):
    user_id: str = Field(..., description="Platform user id.")
    password: str = Field(..., min_length=1, description="Platform user password.")


class CreatePlatformUserRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128, description="Unique platform user id.")
    password: str = Field(..., min_length=8, max_length=256, description="Initial password. Never returned.")
    display_name: str | None = Field(default=None, max_length=256)
    platform_role: str | None = Field(
        default=None,
        description="Optional non-admin platform role. Administrator and platform_owner cannot be set here.",
    )


class GrantRepositoryRoleRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    role: str = Field(
        ...,
        description="Repository role: consumer, contributor, delegate, or owner.",
    )


class PlatformRoleCatalogResponse(BaseModel):
    roles: list[PlatformRoleOption]
    count: int = Field(..., ge=0)
    assignable_on_create: list[str] = Field(
        default_factory=list,
        description="role_id values allowed in POST /platform/users platform_role.",
    )
    role_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of role_id values for convenience.",
    )
