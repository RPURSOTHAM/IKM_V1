"""Unit tests for Phase 5 document human review workflow."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.features.documents.application.document_metadata_service import (
    resolve_document_metadata_bundle,
)
from src.features.human_review.domain.review_exceptions import (
    ConflictError,
    InvalidTransitionError,
    PermissionDeniedError,
)
from src.features.human_review.domain.review_models import (
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_CHANGES_REQUESTED,
    REVIEW_STATUS_IN_REVIEW,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    REVIEW_STATUS_REOPENED,
)
from src.features.human_review.application.review_service import DocumentReviewService


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class InMemoryReviewStore:
    """Minimal stand-in for DocumentReviewStore used by unit tests."""

    def __init__(self) -> None:
        self.reviews: dict[str, dict[str, Any]] = {}
        self.audits: list[dict[str, Any]] = []

    def ensure_schema(self) -> None:
        return None

    def insert_review(
        self,
        *,
        document_id: str,
        repository_id: str | None,
        document_type_id: str | None,
        validation_status: str,
        review_status: str = "PENDING",
        assigned_to: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        review_id = str(uuid.uuid4())
        now = _utc_iso()
        record = {
            "review_id": review_id,
            "document_id": document_id,
            "repository_id": repository_id,
            "document_type_id": document_type_id,
            "validation_status": validation_status,
            "review_status": review_status,
            "assigned_to": assigned_to,
            "comment": comment,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "decided_at": None,
            "decided_by": None,
        }
        self.reviews[review_id] = record
        return dict(record)

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        row = self.reviews.get(review_id)
        return dict(row) if row else None

    def get_open_review_for_document(self, document_id: str) -> dict[str, Any] | None:
        open_statuses = {"PENDING", "IN_REVIEW", "CHANGES_REQUESTED", "REOPENED"}
        matches = [
            r
            for r in self.reviews.values()
            if r["document_id"] == document_id and r["review_status"] in open_statuses
        ]
        if not matches:
            return None
        matches.sort(key=lambda r: r["created_at"], reverse=True)
        return dict(matches[0])

    def get_latest_review_for_document(self, document_id: str) -> dict[str, Any] | None:
        matches = [r for r in self.reviews.values() if r["document_id"] == document_id]
        if not matches:
            return None
        matches.sort(key=lambda r: r["created_at"], reverse=True)
        return dict(matches[0])

    def list_reviews(self, **filters: Any) -> list[dict[str, Any]]:
        items = list(self.reviews.values())
        if filters.get("repository_id"):
            items = [r for r in items if r.get("repository_id") == filters["repository_id"]]
        if filters.get("status"):
            items = [r for r in items if r.get("review_status") == filters["status"]]
        if filters.get("document_type_id"):
            items = [r for r in items if r.get("document_type_id") == filters["document_type_id"]]
        if filters.get("assigned_to"):
            items = [r for r in items if r.get("assigned_to") == filters["assigned_to"]]
        if filters.get("document_id"):
            items = [r for r in items if r.get("document_id") == filters["document_id"]]
        return [dict(r) for r in items]

    def update_review(
        self,
        review_id: str,
        *,
        expected_version: int | None = None,
        review_status: str | None = None,
        assigned_to: str | None = None,
        validation_status: str | None = None,
        comment: str | None = None,
        decided_by: str | None = None,
        clear_decision: bool = False,
        set_decision: bool = False,
    ) -> dict[str, Any] | None:
        row = self.reviews.get(review_id)
        if row is None:
            return None
        if expected_version is not None and int(row.get("version") or 1) != int(expected_version):
            raise ConflictError(
                "Review was modified by another user. Refresh and retry.",
                details={"review_id": review_id},
            )
        if review_status is not None:
            row["review_status"] = review_status
        if assigned_to is not None:
            row["assigned_to"] = assigned_to
        if validation_status is not None:
            row["validation_status"] = validation_status
        if comment is not None:
            row["comment"] = comment
        if set_decision:
            row["decided_at"] = _utc_iso()
            row["decided_by"] = decided_by
        if clear_decision:
            row["decided_at"] = None
            row["decided_by"] = None
        row["version"] = int(row.get("version") or 1) + 1
        row["updated_at"] = _utc_iso()
        return dict(row)

    def insert_audit(self, **kwargs: Any) -> dict[str, Any]:
        from src.features.human_review.domain.review_exceptions import ValidationError

        action = str(kwargs.get("action") or "").strip()
        field_name = str(kwargs.get("field_name") or "").strip()
        allowed = kwargs.pop("allowed_field_names", None)
        if action == "metadata_edit":
            if not field_name or field_name.lower().startswith("additionalprop"):
                raise ValidationError(
                    "Unknown metadata fields",
                    details={"fields": [field_name] if field_name else []},
                )
            if allowed is not None and field_name not in set(allowed):
                raise ValidationError(
                    "Unknown metadata fields",
                    details={"fields": [field_name]},
                )
        item = {
            "audit_id": str(uuid.uuid4()),
            "modified_at": _utc_iso(),
            **kwargs,
        }
        self.audits.append(item)
        return dict(item)

    def list_audit_for_document(self, document_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        items = [a for a in self.audits if a.get("document_id") == document_id]
        return items[:limit]

    def list_audit_for_review(self, review_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        items = [a for a in self.audits if a.get("review_id") == review_id]
        return items[:limit]


@pytest.fixture
def store() -> InMemoryReviewStore:
    return InMemoryReviewStore()


@pytest.fixture
def service(store: InMemoryReviewStore) -> DocumentReviewService:
    svc = DocumentReviewService(store=store)
    return svc


def test_review_created_for_warning(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    review = service.maybe_create_review_from_validation(
        document_id="doc-warn",
        validation_status="WARNING",
        repository_id="repo-1",
        document_type_id="type-1",
    )
    assert review is not None
    assert review["review_status"] == REVIEW_STATUS_PENDING
    assert review["validation_status"] == "WARNING"
    assert store.get_open_review_for_document("doc-warn") is not None
    assert any(a["action"] == "review_created" for a in store.audits)


def test_no_review_for_valid(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    review = service.maybe_create_review_from_validation(
        document_id="doc-valid",
        validation_status="VALID",
        repository_id="repo-1",
    )
    assert review is None
    assert store.get_latest_review_for_document("doc-valid") is None


def test_approve_flow(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-a",
        validation_status="INVALID",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        approved = service.approve(created["review_id"], comment="looks good")
    assert approved["review_status"] == REVIEW_STATUS_APPROVED
    assert approved["decided_by"] == "reviewer-1"
    assert any(a["action"] == "approve" for a in store.audits)


def test_reject_flow(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-r",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        rejected = service.reject(created["review_id"], comment="bad data")
    assert rejected["review_status"] == REVIEW_STATUS_REJECTED


def test_request_changes(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-c",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        updated = service.request_changes(created["review_id"], comment="fix date")
    assert updated["review_status"] == REVIEW_STATUS_CHANGES_REQUESTED


def test_reopen(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-re",
        validation_status="INVALID",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        service.approve(created["review_id"])
        reopened = service.reopen(created["review_id"], comment="need another look")
    assert reopened["review_status"] == REVIEW_STATUS_REOPENED
    assert reopened.get("decided_by") is None


def test_metadata_edit_writes_audit(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-edit",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "rerun_validation", return_value={"status": "PASS", "document_status": "VALID"}), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        service.record_metadata_edits(
            document_id="doc-edit",
            changes=[
                {
                    "field_name": "effective_date",
                    "old_value": None,
                    "new_value": "2026-01-01",
                }
            ],
            modified_by="reviewer-1",
            reason="filled missing date",
        )
    edits = [a for a in store.audits if a["action"] == "metadata_edit"]
    assert len(edits) == 1
    assert edits[0]["field_name"] == "effective_date"
    assert edits[0]["new_value"] == "2026-01-01"
    refreshed = store.get_review(created["review_id"])
    assert refreshed is not None
    assert refreshed["validation_status"] == "VALID"


def test_audit_history_chronological(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-hist",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        service.assign(created["review_id"], assigned_to="alice")
        service.approve(created["review_id"])
    history = store.list_audit_for_document("doc-hist")
    actions = [h["action"] for h in history]
    assert actions[0] == "review_created"
    assert "assign" in actions
    assert "approve" in actions


def test_validation_rerun_updates_status(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-vr",
        validation_status="INVALID",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "rerun_validation", return_value={"status": "WARNING", "document_status": "WARNING"}), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        service.record_metadata_edits(
            document_id="doc-vr",
            changes=[{"field_name": "version", "old_value": "1", "new_value": "2"}],
            modified_by="reviewer-1",
        )
    assert store.get_review(created["review_id"])["validation_status"] == "WARNING"
    assert any(a["action"] == "validation_rerun" for a in store.audits)


def test_metadata_api_review_section() -> None:
    record = {
        "document_id": "doc-meta",
        "metadata": {
            "review": {
                "review_id": "rev-1",
                "status": "PENDING",
                "assigned_to": "alice",
                "last_updated": "2026-07-24T00:00:00Z",
                "validation_status": "WARNING",
            },
            "validation": {"status": "WARNING", "document_status": "WARNING"},
            "validation_status": "WARNING",
        },
        "processing": {},
    }
    with patch(
        "src.features.human_review.application.review_service.get_document_review_service",
        side_effect=Exception("store offline"),
    ):
        bundle = resolve_document_metadata_bundle(record)
    assert bundle["review"]["review_id"] == "rev-1"
    assert bundle["review"]["status"] == "PENDING"
    assert bundle["review"]["validation_status"] == "WARNING"


def test_review_history_api(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-rh",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    store.insert_audit(
        document_id="doc-rh",
        review_id=created["review_id"],
        action="metadata_edit",
        field_name="title",
        old_value="A",
        new_value="B",
        modified_by="reviewer-1",
        reason="fix",
        comment="please fix",
    )
    with patch.object(service, "_require_read"), patch(
        "src.features.documents.application.document_service.DocumentReceiverService.get_document_record",
        return_value={"document_id": "doc-rh", "repository_id": "repo-1", "metadata": {}},
    ):
        payload = service.get_review_history("doc-rh")
    assert payload["count"] >= 2
    assert any(e.get("action") == "metadata_edit" for e in payload["history"])


def test_permissions_block_non_reviewer(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-perm",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", side_effect=PermissionDeniedError("denied")):
        with pytest.raises(PermissionDeniedError):
            service.approve(created["review_id"])


def test_neo4j_persistence_called(service: DocumentReviewService) -> None:
    with patch.object(service, "_persist_neo4j_artifact") as neo, patch(
        "src.features.documents.infrastructure.document_metadata_repository.MetadataStore"
    ):
        review = service.maybe_create_review_from_validation(
            document_id="doc-neo",
            validation_status="WARNING",
            repository_id="repo-1",
            document_type_id="type-1",
        )
    assert review is not None
    assert neo.called
    kwargs = neo.call_args.kwargs
    assert kwargs["action"] == "review_created"
    assert kwargs["validation_after"] == "WARNING"


def test_multiple_review_cycles(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-multi",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        service.approve(created["review_id"])
        service.reopen(created["review_id"])
        service.request_changes(created["review_id"], comment="again")
        service.assign(created["review_id"], assigned_to="bob")
        final = service.approve(created["review_id"])
    assert final["review_status"] == REVIEW_STATUS_APPROVED
    assert final["version"] >= 5


def test_concurrent_review_protection(service: DocumentReviewService, store: InMemoryReviewStore) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-conc",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        # First actor assigns (version 1 -> 2)
        service.assign(created["review_id"], assigned_to="alice")
        # Second actor still holds stale version=1
        with pytest.raises(ConflictError):
            service.approve(created["review_id"], expected_version=1)
        # Closed review cannot be approved again
        service.approve(created["review_id"], expected_version=2)
        with pytest.raises(ConflictError):
            service.approve(created["review_id"])


def test_invalid_transition_blocked(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-it",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        with pytest.raises(InvalidTransitionError):
            service.reopen(created["review_id"])
        service.approve(created["review_id"])
        with pytest.raises(ConflictError):
            service.request_changes(created["review_id"])


def test_assign_moves_to_in_review(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-as",
        validation_status="WARNING",
        repository_id="repo-1",
    )
    assert created is not None
    with patch.object(service, "_require_reviewer", return_value="reviewer-1"), patch.object(
        service, "_persist_neo4j_artifact"
    ), patch.object(service, "_sync_metadata_review"):
        updated = service.assign(created["review_id"], assigned_to="carol")
    assert updated["review_status"] == REVIEW_STATUS_IN_REVIEW
    assert updated["assigned_to"] == "carol"


def test_get_review_for_metadata(service: DocumentReviewService) -> None:
    created = service.maybe_create_review_from_validation(
        document_id="doc-gm",
        validation_status="INVALID",
        repository_id="repo-1",
    )
    assert created is not None
    section = service.get_review_for_metadata("doc-gm")
    assert section is not None
    assert section["review_id"] == created["review_id"]
    assert section["status"] == REVIEW_STATUS_PENDING
    assert section["validation_status"] == "INVALID"
