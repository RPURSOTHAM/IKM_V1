"""Phase 5 human review workflow service (document validation follow-up)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.features.human_review.domain.review_exceptions import (
    ConflictError,
    InvalidTransitionError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.features.human_review.domain.review_models import (
    ALLOWED_TRANSITIONS,
    ALL_REVIEW_STATUSES,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_CHANGES_REQUESTED,
    REVIEW_STATUS_IN_REVIEW,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_REJECTED,
    REVIEW_STATUS_REOPENED,
    REVIEW_TRIGGER_STATUSES,
)
from src.features.human_review.infrastructure.review_repository import (
    DocumentReviewStore,
    get_document_review_store,
)

logger = logging.getLogger(__name__)

HUMAN_REVIEW_PROCESSOR_TYPE = "human_review"
METADATA_EDIT_ACTION = "metadata_edit"


def _pipeline_trace(stage: str, **payload: Any) -> None:
    logger.info("PIPELINE_TRACE %s", {"stage": stage, **payload})


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_swagger_placeholder_field(field_name: str | None) -> bool:
    return str(field_name or "").strip().lower().startswith("additionalprop")


def _normalize_validation_status(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if not text:
        return None
    mapping = {
        "PASS": "VALID",
        "VALID": "VALID",
        "WARNING": "WARNING",
        "FAIL": "INVALID",
        "INVALID": "INVALID",
    }
    return mapping.get(text, text)


class DocumentReviewService:
    def __init__(self, store: DocumentReviewStore | None = None) -> None:
        self._store = store

    def initialize(self) -> None:
        store = self._require_store()
        store.ensure_schema()

    def _require_store(self) -> DocumentReviewStore:
        store = self._store or get_document_review_store()
        if store is None:
            raise ValidationError(
                "Document review store unavailable (MySQL not configured).",
                details={"code": "review_store_unavailable"},
            )
        return store

    def _require_reviewer(self, repository_id: str | None) -> str:
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.authentication.domain.authentication_exceptions import AuthorizationError
        from src.features.users.application.user_service import get_platform_security_service

        actor = get_current_user_from_context()
        if actor is None or actor.user_id in {None, "", "anonymous"}:
            raise PermissionDeniedError("Authentication required for review actions")
        if not repository_id:
            if actor.is_platform_admin or actor.is_admin_api_key:
                return str(actor.user_id)
            raise PermissionDeniedError("Repository context required for review actions")
        try:
            role = get_platform_security_service().check_document_review_access(actor, str(repository_id))
        except AuthorizationError as exc:
            raise PermissionDeniedError(
                str(exc) or "Reviewer role required (contributor or higher)",
                details={"repository_id": repository_id},
            ) from exc
        return str(actor.user_id or role or "reviewer")

    def _require_read(self, repository_id: str | None) -> None:
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.users.application.user_service import get_platform_security_service

        actor = get_current_user_from_context()
        if actor is None or not repository_id:
            return
        get_platform_security_service().check_retrieval_access(actor, str(repository_id))

    # ------------------------------------------------------------------
    # Auto-create on validation outcome
    # ------------------------------------------------------------------
    def maybe_create_review_from_validation(
        self,
        *,
        document_id: str,
        validation_status: str | None,
        repository_id: str | None = None,
        document_type_id: str | None = None,
        validation_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Create or refresh a review when validation is WARNING/INVALID. No-op for VALID."""
        status = _normalize_validation_status(validation_status)
        if status is None:
            return None
        if status not in REVIEW_TRIGGER_STATUSES and status != "INVALID":
            # VALID → do not create; still refresh open review validation_status if present
            if status == "VALID":
                try:
                    store = self._require_store()
                except ValidationError:
                    return None
                open_review = store.get_open_review_for_document(document_id)
                if open_review:
                    updated = store.update_review(
                        open_review["review_id"],
                        validation_status="VALID",
                    )
                    store.insert_audit(
                        document_id=document_id,
                        review_id=open_review["review_id"],
                        action="validation_rerun",
                        field_name="validation_status",
                        old_value=open_review.get("validation_status"),
                        new_value="VALID",
                        modified_by="system",
                        reason="document_validation completed VALID",
                        comment=None,
                    )
                    self._persist_neo4j_artifact(
                        document_id=document_id,
                        repository_id=repository_id or open_review.get("repository_id"),
                        document_type_id=document_type_id or open_review.get("document_type_id"),
                        review=updated or open_review,
                        action="validation_rerun",
                        validation_before=open_review.get("validation_status"),
                        validation_after="VALID",
                        validation_payload=validation_payload,
                        reviewer="system",
                    )
                    return updated
            return None

        try:
            store = self._require_store()
        except ValidationError:
            logger.debug("Review store unavailable; skip auto-create for %s", document_id)
            return None

        open_review = store.get_open_review_for_document(document_id)
        if open_review:
            updated = store.update_review(
                open_review["review_id"],
                validation_status=status,
            )
            store.insert_audit(
                document_id=document_id,
                review_id=open_review["review_id"],
                action="validation_updated",
                field_name="validation_status",
                old_value=open_review.get("validation_status"),
                new_value=status,
                modified_by="system",
                reason="document_validation outcome",
            )
            return updated

        review = store.insert_review(
            document_id=document_id,
            repository_id=repository_id,
            document_type_id=document_type_id,
            validation_status=status,
            review_status=REVIEW_STATUS_PENDING,
        )
        _pipeline_trace(
            "human_review_created",
            document_id=document_id,
            review_id=review.get("review_id"),
            current_status=None,
            next_status=REVIEW_STATUS_PENDING,
            validation_status=status,
        )
        store.insert_audit(
            document_id=document_id,
            review_id=review["review_id"],
            action="review_created",
            field_name="review_status",
            old_value=None,
            new_value=REVIEW_STATUS_PENDING,
            modified_by="system",
            reason=f"Auto-created from validation_status={status}",
        )
        self._persist_neo4j_artifact(
            document_id=document_id,
            repository_id=repository_id,
            document_type_id=document_type_id,
            review=review,
            action="review_created",
            validation_before=None,
            validation_after=status,
            validation_payload=validation_payload,
            reviewer="system",
        )
        try:
            from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
            from src.features.configuration.platform_settings import get_settings

            MetadataStore(get_settings().db_path).update_document_metadata(
                document_id,
                {
                    "review": {
                        "review_id": review["review_id"],
                        "status": review["review_status"],
                        "assigned_to": review.get("assigned_to"),
                        "last_updated": review.get("updated_at"),
                        "validation_status": review.get("validation_status"),
                    }
                },
                merge=True,
            )
        except Exception:
            logger.debug("Could not merge review into intake metadata", exc_info=True)
        return review

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_reviews(
        self,
        *,
        repository_id: str | None = None,
        status: str | None = None,
        document_type_id: str | None = None,
        assigned_to: str | None = None,
        document_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        if status and status not in ALL_REVIEW_STATUSES:
            raise ValidationError(f"Unknown review status: {status}")
        if repository_id:
            self._require_read(repository_id)
        store = self._require_store()
        items = store.list_reviews(
            repository_id=repository_id,
            status=status,
            document_type_id=document_type_id,
            assigned_to=assigned_to,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )
        return {"reviews": items, "count": len(items)}

    def get_review(self, review_id: str) -> dict[str, Any]:
        store = self._require_store()
        review = store.get_review(review_id)
        if review is None:
            raise NotFoundError(f"Review not found: {review_id}")
        self._require_read(review.get("repository_id"))

        document_id = review["document_id"]
        validation = self._load_validation(document_id)
        key_fields = self._load_key_fields(document_id)
        effective_fields = key_fields.get("effective_fields") or []
        extracted_metadata = key_fields.get("extracted_metadata") or {}
        allowed_fields = self._allowed_metadata_field_names(
            document_id,
            document_type_id=review.get("document_type_id"),
            effective_fields=effective_fields,
        )
        audit = self._filter_audit_events(
            store.list_audit_for_review(review_id),
            allowed_field_names=allowed_fields if allowed_fields else None,
        )
        return {
            **review,
            "validation": validation,
            "extracted_metadata": extracted_metadata,
            "effective_fields": effective_fields,
            "audit_history": audit,
        }

    def get_review_for_metadata(self, document_id: str) -> dict[str, Any] | None:
        try:
            store = self._require_store()
        except ValidationError:
            return None
        review = store.get_latest_review_for_document(document_id)
        if review is None:
            return None
        return {
            "review_id": review["review_id"],
            "status": review["review_status"],
            "assigned_to": review.get("assigned_to"),
            "last_updated": review.get("updated_at"),
            "validation_status": review.get("validation_status"),
        }

    def get_review_history(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_service import DocumentReceiverService

        record = DocumentReceiverService().get_document_record(document_id)
        repository_id = str(record.get("repository_id") or (record.get("metadata") or {}).get("repository_id") or "") or None
        self._require_read(repository_id)
        store = self._require_store()
        events = store.list_audit_for_document(document_id)
        allowed_fields = self._allowed_metadata_field_names(
            document_id,
            document_type_id=(
                record.get("document_type_id")
                or (record.get("metadata") or {}).get("document_type_id")
                or (record.get("processing") or {}).get("document_type_id")
            ),
        )
        history = self._filter_audit_events(
            events,
            allowed_field_names=allowed_fields if allowed_fields else None,
        )
        return {
            "document_id": document_id,
            "history": history,
            "count": len(history),
        }

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def assign(self, review_id: str, *, assigned_to: str, comment: str | None = None) -> dict[str, Any]:
        return self._transition(
            review_id,
            action="assign",
            target_status=REVIEW_STATUS_IN_REVIEW,
            assigned_to=assigned_to.strip(),
            comment=comment,
            set_decision=False,
        )

    def approve(self, review_id: str, *, comment: str | None = None, expected_version: int | None = None) -> dict[str, Any]:
        return self._transition(
            review_id,
            action="approve",
            target_status=REVIEW_STATUS_APPROVED,
            comment=comment,
            expected_version=expected_version,
            set_decision=True,
        )

    def reject(self, review_id: str, *, comment: str | None = None, expected_version: int | None = None) -> dict[str, Any]:
        return self._transition(
            review_id,
            action="reject",
            target_status=REVIEW_STATUS_REJECTED,
            comment=comment,
            expected_version=expected_version,
            set_decision=True,
        )

    def request_changes(
        self, review_id: str, *, comment: str | None = None, expected_version: int | None = None
    ) -> dict[str, Any]:
        return self._transition(
            review_id,
            action="request_changes",
            target_status=REVIEW_STATUS_CHANGES_REQUESTED,
            comment=comment,
            expected_version=expected_version,
            set_decision=False,
        )

    def reopen(self, review_id: str, *, comment: str | None = None, expected_version: int | None = None) -> dict[str, Any]:
        return self._transition(
            review_id,
            action="reopen",
            target_status=REVIEW_STATUS_REOPENED,
            comment=comment,
            expected_version=expected_version,
            set_decision=False,
            clear_decision=True,
        )

    def _transition(
        self,
        review_id: str,
        *,
        action: str,
        target_status: str,
        comment: str | None = None,
        assigned_to: str | None = None,
        expected_version: int | None = None,
        set_decision: bool = False,
        clear_decision: bool = False,
    ) -> dict[str, Any]:
        store = self._require_store()
        review = store.get_review(review_id)
        if review is None:
            raise NotFoundError(f"Review not found: {review_id}")

        reviewer = self._require_reviewer(review.get("repository_id"))
        current = str(review.get("review_status") or "")

        # Concurrent / closed-review protection before transition table checks.
        if expected_version is None and current in {REVIEW_STATUS_APPROVED, REVIEW_STATUS_REJECTED} and action != "reopen":
            raise ConflictError(
                "Review is already closed",
                details={"review_id": review_id, "review_status": current},
            )

        allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
        if target_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {current} to {target_status}",
                details={"from": current, "to": target_status, "action": action},
            )

        updated = store.update_review(
            review_id,
            expected_version=expected_version,
            review_status=target_status,
            assigned_to=assigned_to if assigned_to is not None else None,
            comment=comment,
            decided_by=reviewer if set_decision else None,
            set_decision=set_decision,
            clear_decision=clear_decision,
        )
        if updated is None:
            raise NotFoundError(f"Review not found: {review_id}")

        store.insert_audit(
            document_id=review["document_id"],
            review_id=review_id,
            action=action,
            field_name="review_status",
            old_value=current,
            new_value=target_status,
            modified_by=reviewer,
            reason=comment,
            comment=comment,
        )
        self._persist_neo4j_artifact(
            document_id=review["document_id"],
            repository_id=review.get("repository_id"),
            document_type_id=review.get("document_type_id"),
            review=updated,
            action=action,
            validation_before=review.get("validation_status"),
            validation_after=updated.get("validation_status"),
            reviewer=reviewer,
            comment=comment,
        )
        self._sync_metadata_review(review["document_id"], updated)
        return updated

    # ------------------------------------------------------------------
    # Metadata edit + validation re-run
    # ------------------------------------------------------------------
    def record_metadata_edits(
        self,
        *,
        document_id: str,
        changes: list[dict[str, Any]],
        modified_by: str,
        reason: str | None = None,
        allowed_field_names: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any] | None:
        """Append audit rows for key-field edits and re-run validation."""
        if not changes:
            return None
        # Always resolve configured fields; never trust caller-only filtering.
        resolved_allowed = self._allowed_metadata_field_names(document_id)
        if allowed_field_names is not None:
            # Intersect so callers cannot widen the allow-list past document-type fields.
            if resolved_allowed:
                allowed: set[str] | None = set(allowed_field_names) & resolved_allowed
            else:
                allowed = set(allowed_field_names)
        elif resolved_allowed:
            allowed = resolved_allowed
        else:
            # Document type fields unavailable (tests/legacy): reject placeholders only.
            allowed = None
        changes = [
            change
            for change in changes
            if self._is_allowed_metadata_edit_field(
                str(change.get("field_name") or "").strip(),
                allowed_field_names=allowed,
            )
        ]
        if not changes:
            return None
        try:
            store = self._require_store()
        except ValidationError:
            return None

        review = store.get_open_review_for_document(document_id) or store.get_latest_review_for_document(document_id)
        review_id = review["review_id"] if review else None
        before_status = review.get("validation_status") if review else None
        for change in changes:
            field_name = str(change.get("field_name") or "").strip()
            if not self._is_allowed_metadata_edit_field(field_name, allowed_field_names=allowed):
                continue
            try:
                store.insert_audit(
                    document_id=document_id,
                    review_id=review_id,
                    action=METADATA_EDIT_ACTION,
                    field_name=field_name,
                    old_value=change.get("old_value"),
                    new_value=change.get("new_value"),
                    modified_by=modified_by,
                    reason=reason or change.get("change_reason"),
                    allowed_field_names=allowed if allowed else None,
                )
            except ValidationError:
                logger.warning(
                    "Skipped invalid metadata_edit audit document_id=%s field_name=%s",
                    document_id,
                    field_name,
                )
                continue

        # Re-run validation; job-store hook refreshes review.validation_status when available.
        validation = self.rerun_validation(document_id, modified_by=modified_by, reason=reason)
        refreshed = store.get_open_review_for_document(document_id) or store.get_latest_review_for_document(
            document_id
        )
        after_status = None
        if validation:
            after_status = _normalize_validation_status(
                validation.get("document_status") or validation.get("status")
            )
        if refreshed and after_status and refreshed.get("validation_status") != after_status:
            refreshed = store.update_review(refreshed["review_id"], validation_status=after_status) or refreshed
            store.insert_audit(
                document_id=document_id,
                review_id=refreshed["review_id"],
                action="validation_rerun",
                field_name="validation_status",
                old_value=before_status,
                new_value=after_status,
                modified_by=modified_by,
                reason=reason or "metadata edit triggered validation re-run",
            )
        if refreshed:
            self._persist_neo4j_artifact(
                document_id=document_id,
                repository_id=refreshed.get("repository_id"),
                document_type_id=refreshed.get("document_type_id"),
                review=refreshed,
                action="metadata_edit",
                validation_before=before_status,
                validation_after=after_status or refreshed.get("validation_status"),
                validation_payload=validation,
                reviewer=modified_by,
                edited_metadata={c.get("field_name"): c.get("new_value") for c in changes},
                comment=reason,
            )
            self._sync_metadata_review(document_id, refreshed)
        return refreshed

    def rerun_validation(
        self,
        document_id: str,
        *,
        modified_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Synchronously re-execute document_validation over current key-field artifact."""
        from src.features.documents.application.document_metadata_service import (
            fetch_key_field_artifact_from_neo4j,
            resolve_key_field_metadata_bundle,
        )
        from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
        from src.features.documents.application.document_service import DocumentReceiverService
        from src.features.configuration.platform_settings import get_settings
        from src.infrastructure.document_databases.neo4j_store import store_document_graph
        from src.features.document_validation.application.validation_pipeline import (
            DEFAULT_CONFIDENCE_THRESHOLD,
            run_document_validation,
        )

        record = DocumentReceiverService().get_document_record(document_id)
        repository_id = str(
            record.get("repository_id") or (record.get("metadata") or {}).get("repository_id") or ""
        ).strip() or None
        document_type_id = str(
            record.get("document_type_id")
            or (record.get("metadata") or {}).get("document_type_id")
            or (record.get("processing") or {}).get("document_type_id")
            or ""
        ).strip() or None
        document_type_name = str(
            record.get("document_type_name")
            or (record.get("metadata") or {}).get("document_type_name")
            or (record.get("processing") or {}).get("document_type_name")
            or ""
        ).strip() or None

        artifact = fetch_key_field_artifact_from_neo4j(document_id) or {}
        payload = artifact.get("artifact_payload") if isinstance(artifact.get("artifact_payload"), dict) else {}
        extracted_fields = payload.get("fields") or payload.get("extracted_fields") or []

        key_bundle = resolve_key_field_metadata_bundle(record)
        effective_fields = key_bundle.get("effective_fields") or []
        if not effective_fields and document_type_id:
            try:
                from src.features.document_types.application.document_type_service import get_document_type_service

                bundle = get_document_type_service().resolve_effective_fields(str(document_type_id))
                fields = bundle.get("fields") if isinstance(bundle, dict) else []
                effective_fields = [f for f in fields if isinstance(f, dict)] if isinstance(fields, list) else []
            except Exception:
                logger.debug("Could not resolve effective fields during revalidation", exc_info=True)

        threshold = DEFAULT_CONFIDENCE_THRESHOLD
        processing = record.get("processing") if isinstance(record.get("processing"), dict) else {}
        raw_threshold = processing.get("validation_confidence_threshold")
        if raw_threshold is not None:
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError):
                threshold = DEFAULT_CONFIDENCE_THRESHOLD

        validation = run_document_validation(
            effective_fields=effective_fields,
            extracted_fields=extracted_fields,
            confidence_threshold=threshold,
        )
        validation.update(
            {
                "document_type_id": document_type_id,
                "document_type_name": document_type_name,
                "source": "document_validation",
                "rerun_reason": reason or "metadata_edit",
                "rerun_by": modified_by,
                "rerun_at": _utc_iso(),
            }
        )
        store_document_graph(
            document_id=document_id,
            repository_id=repository_id,
            processor_type="document_validation",
            payload=validation,
            document_type_id=document_type_id,
            document_type_name=document_type_name,
        )
        patch = {
            "validation": {
                "status": validation.get("status"),
                "document_status": validation.get("document_status"),
                "missing_required_fields": validation.get("missing_required_fields") or [],
                "invalid_fields": validation.get("invalid_fields") or [],
                "low_confidence_fields": validation.get("low_confidence_fields") or [],
                "confidence_threshold": validation.get("confidence_threshold"),
                "completed_at": _utc_iso(),
                "source": "document_validation_rerun",
            },
            "validation_status": validation.get("document_status") or validation.get("status"),
        }
        MetadataStore(get_settings().db_path).update_document_metadata(document_id, patch, merge=True)

        # Keep job scheduling metadata in sync without re-entering the review auto-create hook.
        try:
            from src.infrastructure.database.document_jobs import DocumentJobStore, mysql_params_from_env

            params = mysql_params_from_env()
            if params is not None:
                job_store = DocumentJobStore(params)
                meta = job_store._scheduling_metadata(document_id)
                results = meta.setdefault("processor_results", {})
                results["document_validation"] = {
                    "status": "COMPLETED",
                    "result_location": f"neo4j://DocumentArtifact/{document_id}/document_validation",
                    "document_metadata": dict(validation),
                    "error_details": None,
                    "completed_at": _utc_iso(),
                }
                job_store._write_scheduling_metadata(document_id, meta)
        except Exception:
            logger.debug("Could not update job store after validation re-run", exc_info=True)

        return validation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_validation(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_service import DocumentReceiverService

        try:
            return DocumentReceiverService().get_document_validation(document_id)
        except Exception:
            return {}

    def _load_key_fields(self, document_id: str) -> dict[str, Any]:
        from src.features.documents.application.document_metadata_service import (
            resolve_key_field_metadata_bundle,
        )
        from src.features.documents.application.document_service import DocumentReceiverService

        try:
            record = DocumentReceiverService().get_document_record(document_id)
            return resolve_key_field_metadata_bundle(record)
        except Exception:
            return {}

    def _allowed_metadata_field_names(
        self,
        document_id: str,
        *,
        document_type_id: str | None = None,
        effective_fields: list[dict[str, Any]] | None = None,
    ) -> set[str]:
        """Resolve configured document-type field names for metadata edit audits."""
        names: set[str] = set()
        for field in effective_fields or []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("field_name") or field.get("name") or "").strip()
            if name:
                names.add(name)
        if names:
            return names

        type_id = str(document_type_id or "").strip() or None
        if not type_id:
            try:
                from src.features.documents.application.document_service import DocumentReceiverService

                record = DocumentReceiverService().get_document_record(document_id)
                type_id = str(
                    record.get("document_type_id")
                    or (record.get("metadata") or {}).get("document_type_id")
                    or (record.get("processing") or {}).get("document_type_id")
                    or ""
                ).strip() or None
            except Exception:
                type_id = None
        if not type_id:
            bundle = self._load_key_fields(document_id)
            for field in bundle.get("effective_fields") or []:
                if not isinstance(field, dict):
                    continue
                name = str(field.get("field_name") or field.get("name") or "").strip()
                if name:
                    names.add(name)
            if names:
                return names
            return set()

        try:
            from src.features.document_types.application.document_type_service import get_document_type_service

            bundle = get_document_type_service().resolve_effective_fields(type_id)
            for field in bundle.get("fields") or []:
                if not isinstance(field, dict):
                    continue
                name = str(field.get("field_name") or field.get("name") or "").strip()
                if name:
                    names.add(name)
        except Exception:
            logger.debug(
                "Could not resolve effective fields for audit filtering document_id=%s type=%s",
                document_id,
                type_id,
                exc_info=True,
            )
        return names

    @staticmethod
    def _is_allowed_metadata_edit_field(
        field_name: str | None,
        *,
        allowed_field_names: set[str] | frozenset[str] | None,
    ) -> bool:
        name = str(field_name or "").strip()
        if not name or _is_swagger_placeholder_field(name):
            return False
        if allowed_field_names is None:
            # Document-type fields unavailable: still block Swagger placeholders.
            return True
        return name in allowed_field_names

    def _filter_audit_events(
        self,
        events: list[dict[str, Any]],
        *,
        allowed_field_names: set[str] | frozenset[str] | None,
    ) -> list[dict[str, Any]]:
        """Drop invalid metadata_edit rows; keep lifecycle/validation audit events."""
        filtered: list[dict[str, Any]] = []
        allow: set[str] | None
        if allowed_field_names is None:
            allow = None
        elif len(allowed_field_names) == 0:
            allow = set()
        else:
            allow = set(allowed_field_names)
        for event in events:
            action = str(event.get("action") or "").strip()
            if action != METADATA_EDIT_ACTION:
                filtered.append(event)
                continue
            field_name = str(event.get("field_name") or "").strip()
            if not self._is_allowed_metadata_edit_field(field_name, allowed_field_names=allow):
                continue
            filtered.append(event)
        return filtered

    def _sync_metadata_review(self, document_id: str, review: dict[str, Any]) -> None:
        try:
            from src.features.documents.infrastructure.document_metadata_repository import MetadataStore
            from src.features.configuration.platform_settings import get_settings

            MetadataStore(get_settings().db_path).update_document_metadata(
                document_id,
                {
                    "review": {
                        "review_id": review.get("review_id"),
                        "status": review.get("review_status"),
                        "assigned_to": review.get("assigned_to"),
                        "last_updated": review.get("updated_at"),
                        "validation_status": review.get("validation_status"),
                    }
                },
                merge=True,
            )
        except Exception:
            logger.debug("Could not sync review into metadata for %s", document_id, exc_info=True)

    def _persist_neo4j_artifact(
        self,
        *,
        document_id: str,
        repository_id: str | None,
        document_type_id: str | None,
        review: dict[str, Any],
        action: str,
        validation_before: Any = None,
        validation_after: Any = None,
        validation_payload: dict[str, Any] | None = None,
        reviewer: str | None = None,
        edited_metadata: dict[str, Any] | None = None,
        comment: str | None = None,
    ) -> None:
        try:
            from src.infrastructure.document_databases.neo4j_store import store_document_graph

            # Append-friendly payload: merge prior decisions when possible.
            prior: dict[str, Any] = {}
            try:
                from src.features.documents.application.document_metadata_service import (
                    fetch_validation_artifact_from_neo4j,
                )
                # Reuse generic Neo4j fetch pattern via store_document_graph upsert;
                # decisions list is rebuilt from review audits for durability.
            except Exception:
                prior = {}

            decisions = list(prior.get("decisions") or []) if isinstance(prior, dict) else []
            decisions.append(
                {
                    "action": action,
                    "reviewer": reviewer,
                    "comment": comment,
                    "at": _utc_iso(),
                    "review_status": review.get("review_status"),
                    "validation_before": validation_before,
                    "validation_after": validation_after,
                }
            )
            payload = {
                "review_id": review.get("review_id"),
                "document_id": document_id,
                "repository_id": repository_id,
                "document_type_id": document_type_id or review.get("document_type_id"),
                "review_status": review.get("review_status"),
                "validation_status": review.get("validation_status"),
                "assigned_to": review.get("assigned_to"),
                "decided_by": review.get("decided_by"),
                "comment": review.get("comment") or comment,
                "decisions": decisions,
                "edited_metadata": edited_metadata or {},
                "validation_before": validation_before,
                "validation_after": validation_after,
                "validation_payload": validation_payload,
                "reviewer": reviewer,
                "updated_at": _utc_iso(),
                "source": HUMAN_REVIEW_PROCESSOR_TYPE,
            }
            store_document_graph(
                document_id=document_id,
                repository_id=repository_id,
                processor_type=HUMAN_REVIEW_PROCESSOR_TYPE,
                payload=payload,
                document_type_id=document_type_id or review.get("document_type_id"),
            )
            _pipeline_trace(
                "human_review_complete",
                document_id=document_id,
                review_id=review.get("review_id"),
                current_status=None,
                next_status=review.get("review_status"),
                action=action,
                validation_status=review.get("validation_status"),
            )
        except Exception:
            logger.debug("Could not persist human_review Neo4j artifact", exc_info=True)


_service: DocumentReviewService | None = None


def get_document_review_service() -> DocumentReviewService:
    global _service
    if _service is None:
        _service = DocumentReviewService()
    return _service
