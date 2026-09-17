from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.features.documents.application.document_service import DocumentReceiverService


def _patch_effective_fields(*names: str):
    return patch.object(
        DocumentReceiverService,
        "_allowed_effective_field_names",
        return_value=set(names),
    )


@_patch_effective_fields("study_id")
@patch("src.features.documents.application.document_metadata_service.resolve_key_fields")
@patch("src.features.documents.application.document_metadata_service.fetch_key_field_artifact_from_neo4j")
@patch("src.features.documents.application.document_metadata_service.check_document_access")
@patch("src.application.consumer_api.context.get_current_user_from_context")
@patch("src.infrastructure.document_databases.neo4j_store.store_document_graph")
def test_update_key_fields_persists_manual_override(
    mock_store_graph: MagicMock,
    mock_actor: MagicMock,
    _mock_access: MagicMock,
    mock_fetch: MagicMock,
    mock_resolve: MagicMock,
    _mock_allowed: MagicMock,
) -> None:
    service = DocumentReceiverService()
    record = {
        "document_id": "doc-1",
        "repository_id": "repo-1",
        "metadata": {"document_type_id": "type-1", "document_type_name": "SOP"},
        "processing": {},
    }
    service.get_document_record = MagicMock(return_value=record)  # type: ignore[method-assign]
    mock_actor.return_value = None
    mock_fetch.return_value = {
        "document_type_id": "type-1",
        "artifact_payload": {
            "document_type_id": "type-1",
            "source": "key_field_extraction",
            "fields": {"study_id": {"value": "OLD", "confidence": 0.9}},
        },
    }
    mock_resolve.return_value = {"document_id": "doc-1", "fields": {"study_id": {"value": "NEW"}}}

    result = service.update_key_fields(
        "doc-1",
        {"study_id": "NEW"},
        reason="User correction",
    )

    assert result["fields"]["study_id"]["value"] == "NEW"
    mock_store_graph.assert_called_once()
    payload = mock_store_graph.call_args.kwargs["payload"]
    assert payload["source"] == "manual_override"
    assert payload["fields"]["study_id"]["value"] == "NEW"
    assert payload["fields"]["study_id"]["source"] == "manual_override"
    assert payload["fields"]["study_id"]["updated_by"] == "system"
    assert payload["history"] and payload["history"][0]["field_name"] == "study_id"


@_patch_effective_fields("study_id")
@patch("src.features.documents.application.document_metadata_service.resolve_key_fields")
@patch("src.features.documents.application.document_metadata_service.fetch_key_field_artifact_from_neo4j")
@patch("src.features.documents.application.document_metadata_service.check_document_access")
@patch("src.application.consumer_api.context.get_current_user_from_context")
@patch("src.infrastructure.document_databases.neo4j_store.store_document_graph")
def test_update_key_fields_noop_skips_write(
    mock_store_graph: MagicMock,
    mock_actor: MagicMock,
    _mock_access: MagicMock,
    mock_fetch: MagicMock,
    mock_resolve: MagicMock,
    _mock_allowed: MagicMock,
) -> None:
    service = DocumentReceiverService()
    record = {"document_id": "doc-1", "metadata": {"document_type_id": "type-1"}, "processing": {}}
    service.get_document_record = MagicMock(return_value=record)  # type: ignore[method-assign]
    mock_actor.return_value = None
    mock_fetch.return_value = {
        "document_type_id": "type-1",
        "artifact_payload": {"fields": {"study_id": {"value": "UNCHANGED"}}},
    }
    mock_resolve.return_value = {"document_id": "doc-1", "fields": {"study_id": {"value": "UNCHANGED"}}}

    out = service.update_key_fields("doc-1", {"study_id": "UNCHANGED"}, reason="noop")
    assert out["fields"]["study_id"]["value"] == "UNCHANGED"
    mock_store_graph.assert_not_called()


@_patch_effective_fields("study_id", "document_title")
@patch("src.features.documents.application.document_metadata_service.check_document_access")
@patch("src.features.users.infrastructure.user_repository.get_platform_security_store")
def test_get_key_fields_history_returns_audit_entries(
    mock_get_store: MagicMock,
    _mock_access: MagicMock,
    _mock_allowed: MagicMock,
) -> None:
    service = DocumentReceiverService()
    service.get_document_record = MagicMock(  # type: ignore[method-assign]
        return_value={
            "document_id": "doc-1",
            "repository_id": "repo-1",
            "metadata": {"document_type_id": "type-1"},
        }
    )
    fake_store = MagicMock()
    fake_store.list_audit_events.return_value = [
        {
            "audit_id": "a1",
            "event_time": "2026-07-23T09:00:00+00:00",
            "document_id": "doc-1",
            "old_value_json": {"field_name": "study_id", "value": "OLD"},
            "new_value_json": {"field_name": "study_id", "value": "NEW"},
            "actor_user_id": "alice",
            "change_reason": "Correction",
        },
        {
            "audit_id": "a2",
            "event_time": "2026-07-23T09:05:00+00:00",
            "document_id": "doc-1",
            "old_value_json": {"field_name": "additionalProp1", "value": None},
            "new_value_json": {"field_name": "additionalProp1", "value": {}},
            "actor_user_id": "alice",
            "change_reason": "swagger",
        },
    ]
    mock_get_store.return_value = fake_store

    out = service.get_key_fields_history("doc-1")
    assert out["count"] == 1
    assert out["history"][0]["field_name"] == "study_id"
    assert out["history"][0]["old_value"] == "OLD"
    assert out["history"][0]["new_value"] == "NEW"


@_patch_effective_fields("document_title", "document_number")
@patch("src.features.documents.application.document_metadata_service.resolve_key_fields")
@patch("src.features.documents.application.document_metadata_service.fetch_key_field_artifact_from_neo4j")
@patch("src.features.documents.application.document_metadata_service.check_document_access")
@patch("src.application.consumer_api.context.get_current_user_from_context")
@patch("src.infrastructure.document_databases.neo4j_store.store_document_graph")
def test_update_key_fields_rejects_unknown_swagger_placeholder(
    mock_store_graph: MagicMock,
    mock_actor: MagicMock,
    _mock_access: MagicMock,
    mock_fetch: MagicMock,
    mock_resolve: MagicMock,
    _mock_allowed: MagicMock,
) -> None:
    service = DocumentReceiverService()
    service.get_document_record = MagicMock(  # type: ignore[method-assign]
        return_value={
            "document_id": "doc-1",
            "repository_id": "repo-1",
            "metadata": {"document_type_id": "type-1", "document_type_name": "SOP"},
            "processing": {},
        }
    )
    mock_actor.return_value = None
    mock_fetch.return_value = {"artifact_payload": {"fields": {}}}

    with pytest.raises(HTTPException) as exc:
        service.update_key_fields("doc-1", {"additionalProp1": {}})

    assert exc.value.status_code == 400
    assert exc.value.detail == {
        "message": "Unknown metadata fields",
        "fields": ["additionalProp1"],
    }
    mock_store_graph.assert_not_called()


@_patch_effective_fields("document_title", "document_number")
@patch("src.features.documents.application.document_metadata_service.resolve_key_fields")
@patch("src.features.documents.application.document_metadata_service.fetch_key_field_artifact_from_neo4j")
@patch("src.features.documents.application.document_metadata_service.check_document_access")
@patch("src.application.consumer_api.context.get_current_user_from_context")
@patch("src.infrastructure.document_databases.neo4j_store.store_document_graph")
def test_update_key_fields_rejects_mixed_valid_and_invalid_fields(
    mock_store_graph: MagicMock,
    mock_actor: MagicMock,
    _mock_access: MagicMock,
    mock_fetch: MagicMock,
    mock_resolve: MagicMock,
    _mock_allowed: MagicMock,
) -> None:
    service = DocumentReceiverService()
    service.get_document_record = MagicMock(  # type: ignore[method-assign]
        return_value={
            "document_id": "doc-1",
            "repository_id": "repo-1",
            "metadata": {"document_type_id": "type-1"},
            "processing": {},
        }
    )
    mock_actor.return_value = None
    with pytest.raises(HTTPException) as exc:
        service.update_key_fields(
            "doc-1",
            {"document_title": "New Title", "additionalProp1": {}},
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["fields"] == ["additionalProp1"]
    mock_store_graph.assert_not_called()


@_patch_effective_fields("document_title", "document_number")
@patch("src.features.human_review.application.review_service.get_document_review_service")
@patch("src.features.users.infrastructure.user_repository.get_platform_security_store")
@patch("src.features.documents.application.document_metadata_service.resolve_key_fields")
@patch("src.features.documents.application.document_metadata_service.fetch_key_field_artifact_from_neo4j")
@patch("src.features.documents.application.document_metadata_service.check_document_access")
@patch("src.application.consumer_api.context.get_current_user_from_context")
@patch("src.infrastructure.document_databases.neo4j_store.store_document_graph")
def test_update_key_fields_accepts_valid_field_and_audits(
    mock_store_graph: MagicMock,
    mock_actor: MagicMock,
    _mock_access: MagicMock,
    mock_fetch: MagicMock,
    mock_resolve: MagicMock,
    mock_get_sec_store: MagicMock,
    mock_get_review: MagicMock,
    _mock_allowed: MagicMock,
) -> None:
    service = DocumentReceiverService()
    record = {
        "document_id": "doc-1",
        "repository_id": "repo-1",
        "metadata": {"document_type_id": "type-1", "document_type_name": "SOP"},
        "processing": {},
    }
    service.get_document_record = MagicMock(return_value=record)  # type: ignore[method-assign]
    mock_actor.return_value = None
    mock_fetch.return_value = {
        "document_type_id": "type-1",
        "artifact_payload": {
            "fields": {"document_title": {"value": "Old SOP", "confidence": 0.8}},
            "history": [{"field_name": "additionalProp1", "new_value": {}}],
        },
    }
    mock_resolve.return_value = {
        "document_id": "doc-1",
        "fields": {"document_title": {"value": "Updated SOP"}},
    }
    audit_service = MagicMock()
    sec_store = MagicMock()
    mock_get_sec_store.return_value = sec_store
    with patch(
        "src.features.audit.application.security_audit_service.AuditService",
        return_value=audit_service,
    ):
        result = service.update_key_fields(
            "doc-1",
            {"document_title": "Updated SOP"},
            reason="Correction",
        )

    assert result["fields"]["document_title"]["value"] == "Updated SOP"
    mock_store_graph.assert_called_once()
    payload = mock_store_graph.call_args.kwargs["payload"]
    assert "additionalProp1" not in payload["fields"]
    assert all(entry["field_name"] != "additionalProp1" for entry in payload["history"])
    assert payload["history"][-1]["field_name"] == "document_title"
    audit_service.record.assert_called_once()
    recorded = audit_service.record.call_args.kwargs
    assert recorded["new_value"]["field_name"] == "document_title"
    assert recorded["new_value"]["value"] == "Updated SOP"
    mock_get_review.return_value.record_metadata_edits.assert_called_once()
    assert mock_get_review.return_value.record_metadata_edits.call_args.kwargs["allowed_field_names"] == {
        "document_title",
        "document_number",
    }
