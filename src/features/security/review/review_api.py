"""FastAPI routes for human review workflow."""

from __future__ import annotations

from typing import Any

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

_PENDING_STATUS_ALIASES = frozenset(
    {
        "PENDING",
        "PENDING_REVIEW",
        "HUMAN_REVIEW",
        "human_review",
        "pending",
        "pending_review",
    }
)

from src.features.security.review.human_review_service import HumanReviewService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/security/reviews", tags=["Human Review"])


class ReviewActionRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
    comments: str = ""


class ReviewDecisionRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
    decision: str = Field(..., min_length=1, description="ALLOW | MASK_AND_ALLOW | BLOCK")
    comments: str = ""


class ConfidentialSeedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    document_id: str | None = None
    document_name: str | None = None


def _service() -> HumanReviewService:
    return HumanReviewService()


@router.get("/decision-model")
def get_decision_model() -> dict[str, Any]:
    """Return risk levels, DLP decisions, and allowed reviewer actions per risk."""
    return _service().decision_model()


def _status_filter_matches(item: dict[str, Any], raw_status: str) -> bool:
    normalized = str(raw_status or "").strip()
    if not normalized:
        return True
    if normalized in _PENDING_STATUS_ALIASES:
        return str(item.get("status") or item.get("review_status") or "").upper() == "PENDING"
    return str(item.get("status") or "").upper() == normalized.upper()


@router.get("")
def list_reviews(
    status: str | None = Query(
        default=None,
        description=(
            "Filter by queue status. Use PENDING (or human_review) for open reviews. "
            "Terminal values: APPROVED, APPROVED_MASKED, REJECTED, REPROCESSED, COMPLETED."
        ),
    ),
    review_status: str | None = Query(
        default=None,
        description="Alias filter; same vocabulary as status for queue rows.",
    ),
    document_id: str | None = Query(
        default=None,
        description="Optional document UUID filter (matches review document_id).",
    ),
    include_all: bool = Query(
        default=False,
        description="When true, return every review row. Default lists pending reviews only.",
    ),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum rows returned."),
) -> dict[str, Any]:
    svc = _service()
    status_filter = status or review_status
    if include_all:
        items = svc.list_all()
    elif status_filter:
        if str(status_filter) in _PENDING_STATUS_ALIASES or str(status_filter).upper() == "PENDING":
            items = svc.list_pending()
        else:
            items = svc.list_all()
            if status:
                items = [i for i in items if _status_filter_matches(i, status)]
            if review_status:
                items = [i for i in items if _status_filter_matches(i, review_status)]
    else:
        items = svc.list_pending()

    if document_id:
        doc_id = str(document_id).strip()
        items = [i for i in items if str(i.get("document_id") or "") == doc_id]

    items = items[:limit]
    return {"reviews": items, "count": len(items)}


@router.post("/confidential-corpus")
def seed_confidential_corpus(body: ConfidentialSeedRequest) -> dict[str, Any]:
    """Manually seed the confidential similarity corpus."""
    from src.features.security.classification.confidential_similarity_engine import add_sensitive_document

    doc_id = add_sensitive_document(
        body.text,
        document_id=body.document_id,
        document_name=body.document_name or "manual-seed",
        metadata={"source": "api"},
    )
    return {"id": doc_id, "status": "indexed"}


@router.get("/{review_id}")
def get_review(review_id: str) -> dict[str, Any]:
    item = _service().get(review_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Review not found: {review_id}")
    return item


@router.get("/{review_id}/actions")
def get_review_actions(review_id: str) -> dict[str, Any]:
    svc = _service()
    item = svc.get(review_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Review not found: {review_id}")
    try:
        actions = svc.available_actions(review_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "review_id": review_id,
        "risk_level": item.get("risk_level"),
        "dlp_decision": item.get("dlp_decision"),
        "review_status": item.get("review_status"),
        "available_actions": actions,
    }


@router.post("/{review_id}/reopen")
def reopen_review(review_id: str) -> dict[str, Any]:
    """Reset a previously actioned review to PENDING (for fresh Allow/Mask/Block)."""
    try:
        return _service().reopen(
            review_id,
            reason="API reopen — live scan requires a new explicit reviewer decision.",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{review_id}/decide")
def apply_review_decision(review_id: str, body: ReviewDecisionRequest) -> dict[str, Any]:
    logger.info(
        "POST /security/reviews/%s/decide decision=%s reviewer=%s",
        review_id,
        body.decision,
        body.reviewer,
    )
    try:
        result = _service().apply_decision(
            review_id,
            decision=body.decision,
            reviewer=body.reviewer,
            comments=body.comments,
        )
        logger.info(
            "Review %s actioned: status=%s decision=%s",
            review_id,
            result.get("status"),
            result.get("reviewer_decision"),
        )
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{review_id}/approve")
def approve_review(review_id: str, body: ReviewActionRequest) -> dict[str, Any]:
    try:
        return _service().approve(review_id, reviewer=body.reviewer, comments=body.comments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{review_id}/reject")
def reject_review(review_id: str, body: ReviewActionRequest) -> dict[str, Any]:
    try:
        return _service().reject(review_id, reviewer=body.reviewer, comments=body.comments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{review_id}/reprocess")
def reprocess_review(review_id: str, body: ReviewActionRequest) -> dict[str, Any]:
    try:
        return _service().reprocess(review_id, reviewer=body.reviewer, comments=body.comments)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
