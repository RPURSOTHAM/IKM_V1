"""HTTP routes for Phase 5 document human review workflow."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from src.features.human_review.domain.review_exceptions import DocumentReviewError
from src.features.human_review.schemas.review_schemas import (
    AssignReviewRequest,
    ReopenReviewRequest,
    ReviewDecisionRequest,
)
from src.features.human_review.application.review_service import (
    DocumentReviewService,
    get_document_review_service,
)

router = APIRouter(tags=["Document Reviews"])


def _svc() -> DocumentReviewService:
    return get_document_review_service()


def _review_error(exc: DocumentReviewError) -> HTTPException:
    return HTTPException(
        status_code=exc.http_status,
        detail={"code": exc.code, "message": exc.message, "details": exc.details},
    )


@router.get(
    "/reviews",
    summary="List document reviews",
    operation_id="listDocumentReviews",
)
def list_reviews(
    repository: str | None = Query(default=None, alias="repository"),
    repository_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    document_type: str | None = Query(default=None, alias="document_type"),
    document_type_id: str | None = Query(default=None),
    assigned_to: str | None = Query(default=None),
    document_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.list_reviews(
            repository_id=repository_id or repository,
            status=status,
            document_type_id=document_type_id or document_type,
            assigned_to=assigned_to,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc


@router.get(
    "/reviews/{review_id}",
    summary="Get document review details",
    operation_id="getDocumentReview",
)
def get_review(
    review_id: str,
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.get_review(review_id)
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc


@router.post(
    "/reviews/{review_id}/assign",
    summary="Assign reviewer",
    operation_id="assignDocumentReview",
)
def assign_review(
    review_id: str,
    body: AssignReviewRequest,
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.assign(review_id, assigned_to=body.assigned_to, comment=body.comment)
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc


@router.post(
    "/reviews/{review_id}/approve",
    summary="Approve document review",
    operation_id="approveDocumentReview",
)
def approve_review(
    review_id: str,
    body: ReviewDecisionRequest = Body(default_factory=ReviewDecisionRequest),
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.approve(
            review_id,
            comment=body.comment,
            expected_version=body.expected_version,
        )
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc


@router.post(
    "/reviews/{review_id}/reject",
    summary="Reject document review",
    operation_id="rejectDocumentReview",
)
def reject_review(
    review_id: str,
    body: ReviewDecisionRequest = Body(default_factory=ReviewDecisionRequest),
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.reject(
            review_id,
            comment=body.comment,
            expected_version=body.expected_version,
        )
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc


@router.post(
    "/reviews/{review_id}/request-changes",
    summary="Request changes on document review",
    operation_id="requestDocumentReviewChanges",
)
def request_changes(
    review_id: str,
    body: ReviewDecisionRequest = Body(default_factory=ReviewDecisionRequest),
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.request_changes(
            review_id,
            comment=body.comment,
            expected_version=body.expected_version,
        )
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc


@router.post(
    "/reviews/{review_id}/reopen",
    summary="Reopen a closed document review",
    operation_id="reopenDocumentReview",
)
def reopen_review(
    review_id: str,
    body: ReopenReviewRequest = Body(default_factory=ReopenReviewRequest),
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.reopen(
            review_id,
            comment=body.comment,
            expected_version=body.expected_version,
        )
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc


@router.get(
    "/documents/{document_id}/review-history",
    summary="Get chronological review and metadata-edit history",
    operation_id="getDocumentReviewHistory",
    tags=["Documents"],
)
def get_document_review_history(
    document_id: str,
    service: DocumentReviewService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.get_review_history(document_id)
    except DocumentReviewError as exc:
        raise _review_error(exc) from exc
