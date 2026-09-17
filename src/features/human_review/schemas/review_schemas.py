"""Phase 5 document review request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AssignReviewRequest(BaseModel):
    assigned_to: str = Field(..., min_length=1, max_length=256)
    comment: str | None = Field(default=None, max_length=4000)


class ReviewDecisionRequest(BaseModel):
    comment: str | None = Field(default=None, max_length=4000)
    expected_version: int | None = Field(
        default=None,
        description="Optimistic concurrency token from GET review.version",
    )


class ReopenReviewRequest(BaseModel):
    comment: str | None = Field(default=None, max_length=4000)
    expected_version: int | None = None


class ReviewSummary(BaseModel):
    review_id: str
    document_id: str
    repository_id: str | None = None
    document_type_id: str | None = None
    validation_status: str
    review_status: str
    assigned_to: str | None = None
    comment: str | None = None
    version: int = 1
    created_at: str | None = None
    updated_at: str | None = None
    decided_at: str | None = None
    decided_by: str | None = None


class ReviewListResponse(BaseModel):
    reviews: list[dict[str, Any]]
    count: int
