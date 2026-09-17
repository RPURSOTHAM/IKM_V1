from __future__ import annotations

from typing import Any

from src.features.document_types.domain.document_type_exceptions import (
    DuplicateNameError,
    MetadataInUseError,
    NotFoundError,
    SystemTypeImmutableError,
    ValidationError,
)
from src.features.document_types.application.field_group_service import (
    DEFAULT_FIELD_GROUP,
    DEFAULT_FIELD_SEQUENCE,
    DEFAULT_GROUP_LABEL,
    DEFAULT_GROUP_SEQUENCE,
)
from src.features.document_types.domain.models import MetadataFieldRecord
from src.features.document_types.infrastructure.document_type_repository import DocumentTypeStore


class MetadataDomain:
    """CRUD and usage guards for metadata field definitions."""

    def __init__(self, store: DocumentTypeStore) -> None:
        self._store = store

    def assert_not_system_field(self, field: MetadataFieldRecord) -> None:
        if field.is_system:
            raise SystemTypeImmutableError(
                "System metadata fields on the Base Document Type cannot be modified or deleted.",
                details={"metadata_field_id": field.metadata_field_id, "field_name": field.field_name},
            )

    def assert_not_in_use(self, field: MetadataFieldRecord) -> None:
        usage_count = self._store.count_metadata_value_usage(field.metadata_field_id)
        if usage_count > 0:
            raise MetadataInUseError(
                f"Metadata field cannot be changed because {usage_count} documents store a value for it.",
                details={"metadata_field_id": field.metadata_field_id, "document_count": usage_count},
            )
        doc_count = self._store.count_documents_for_type(field.document_type_id)
        if field.required and doc_count > 0:
            raise MetadataInUseError(
                "Required metadata field cannot be changed while documents reference its document type.",
                details={
                    "metadata_field_id": field.metadata_field_id,
                    "document_type_id": field.document_type_id,
                    "document_count": doc_count,
                },
            )

    def _resolve_grouping(
        self,
        document_type_id: str,
        payload: dict[str, Any],
        *,
        partial: bool,
    ) -> tuple[str, str, int, int]:
        field_group = str(payload.get("field_group") or DEFAULT_FIELD_GROUP)
        group_label = str(payload.get("group_label") or DEFAULT_GROUP_LABEL).strip() or DEFAULT_GROUP_LABEL
        group_sequence = payload.get("group_sequence", DEFAULT_GROUP_SEQUENCE)
        field_sequence = payload.get("field_sequence", DEFAULT_FIELD_SEQUENCE)

        if partial and "field_group" not in payload:
            field_group = DEFAULT_FIELD_GROUP
        if partial and "group_label" not in payload and "field_group" not in payload:
            group_label = DEFAULT_GROUP_LABEL
        if partial and "group_sequence" not in payload:
            group_sequence = DEFAULT_GROUP_SEQUENCE
        if partial and "field_sequence" not in payload:
            field_sequence = DEFAULT_FIELD_SEQUENCE

        try:
            self._store.validate_field_group(field_group)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        try:
            self._store.validate_sequence(int(group_sequence), field_name="group_sequence")
            self._store.validate_sequence(int(field_sequence), field_name="field_sequence")
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        existing_group = self._store.get_group_definition(document_type_id, field_group)
        if existing_group is not None:
            existing_label, existing_sequence = existing_group
            if "group_label" in payload and group_label != existing_label:
                raise ValidationError(
                    "group_label must match other fields in the same field_group on this document type.",
                    details={"field_group": field_group, "expected_group_label": existing_label},
                )
            if "group_sequence" in payload and int(group_sequence) != existing_sequence:
                raise ValidationError(
                    "group_sequence must match other fields in the same field_group on this document type.",
                    details={"field_group": field_group, "expected_group_sequence": existing_sequence},
                )
            group_label = existing_label
            group_sequence = existing_sequence

        return field_group, group_label, int(group_sequence), int(field_sequence)

    def validate_payload(self, payload: dict[str, Any], *, partial: bool = False) -> None:
        if not partial or "field_name" in payload:
            field_name = payload.get("field_name")
            if field_name is None and not partial:
                raise ValidationError("field_name is required.")
            if field_name is not None:
                try:
                    self._store.validate_field_name(str(field_name))
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
        if not partial or "data_type" in payload:
            data_type = payload.get("data_type")
            if data_type is None and not partial:
                raise ValidationError("data_type is required.")
            if data_type is not None:
                try:
                    self._store.validate_data_type(str(data_type), payload.get("enum_values"))
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
        if not partial or "display_label" in payload:
            label = payload.get("display_label")
            if label is None and not partial:
                raise ValidationError("display_label is required.")
            if label is not None and not str(label).strip():
                raise ValidationError("display_label is required.")
        if not partial or any(k in payload for k in ("field_group", "group_label", "group_sequence", "field_sequence")):
            if payload.get("group_label") is not None and not str(payload.get("group_label")).strip():
                raise ValidationError("group_label cannot be empty.")

    def assert_not_system_document_type(self, document_type_id: str) -> None:
        doc_type = self._store.get_type_by_id(document_type_id)
        if doc_type is None:
            raise NotFoundError("Document type not found.", details={"document_type_id": document_type_id})
        if doc_type.is_system:
            raise SystemTypeImmutableError(
                "The Base Document Type cannot be modified; foundational metadata fields are seed-managed only.",
                details={"document_type_id": document_type_id},
            )

    def create_field(self, document_type_id: str, payload: dict[str, Any]) -> MetadataFieldRecord:
        self.assert_not_system_document_type(document_type_id)
        doc_type = self._store.get_type_by_id(document_type_id)
        assert doc_type is not None
        self.validate_payload(payload)
        field_group, group_label, group_sequence, field_sequence = self._resolve_grouping(
            document_type_id,
            payload,
            partial=False,
        )
        field_name = str(payload["field_name"])
        if self._store.field_name_on_type(document_type_id, field_name):
            raise DuplicateNameError(
                "A metadata field with this name already exists on the document type.",
                details={"document_type_id": document_type_id, "field_name": field_name},
            )
        effective = self._store.effective_field_names(document_type_id)
        if field_name in effective:
            raise DuplicateNameError(
                "field_name conflicts with an inherited metadata field.",
                details={"document_type_id": document_type_id, "field_name": field_name},
            )
        return self._store.insert_metadata_field(
            document_type_id=document_type_id,
            field_name=field_name,
            display_label=str(payload["display_label"]),
            data_type=str(payload["data_type"]),
            required=bool(payload.get("required", False)),
            default_value=payload.get("default_value"),
            enum_values=payload.get("enum_values"),
            max_length=payload.get("max_length"),
            field_group=field_group,
            group_label=group_label,
            group_sequence=group_sequence,
            field_sequence=field_sequence,
            is_system=False,
        )

    def update_field(
        self,
        document_type_id: str,
        metadata_field_id: str,
        payload: dict[str, Any],
    ) -> MetadataFieldRecord:
        field = self._store.get_field_by_id(metadata_field_id)
        if field is None or field.document_type_id != document_type_id:
            raise NotFoundError(
                "Metadata field not found.",
                details={"document_type_id": document_type_id, "metadata_field_id": metadata_field_id},
            )
        self.assert_not_system_field(field)
        self.assert_not_in_use(field)
        if "field_name" in payload or "data_type" in payload or "document_type_id" in payload:
            raise ValidationError("field_name, data_type, and document_type_id are immutable after creation.")
        self.validate_payload(payload, partial=True)

        grouping_payload = {
            "field_group": payload.get("field_group", field.field_group),
            "group_label": payload.get("group_label", field.group_label),
            "group_sequence": payload.get("group_sequence", field.group_sequence),
            "field_sequence": payload.get("field_sequence", field.field_sequence),
        }
        field_group, group_label, group_sequence, field_sequence = self._resolve_grouping(
            document_type_id,
            grouping_payload,
            partial=False,
        )

        unset_default = "default_value" in payload and payload["default_value"] is None
        return self._store.update_metadata_field(
            metadata_field_id,
            display_label=payload.get("display_label"),
            required=payload.get("required"),
            default_value=payload.get("default_value"),
            enum_values=payload.get("enum_values"),
            max_length=payload.get("max_length"),
            field_group=field_group if "field_group" in payload else None,
            group_label=group_label if "group_label" in payload else None,
            group_sequence=group_sequence if "group_sequence" in payload else None,
            field_sequence=field_sequence if "field_sequence" in payload else None,
            is_active=payload.get("is_active"),
            unset_default=unset_default,
        )

    def delete_field(self, document_type_id: str, metadata_field_id: str) -> None:
        field = self._store.get_field_by_id(metadata_field_id)
        if field is None or field.document_type_id != document_type_id:
            raise NotFoundError(
                "Metadata field not found.",
                details={"document_type_id": document_type_id, "metadata_field_id": metadata_field_id},
            )
        self.assert_not_system_field(field)
        self.assert_not_in_use(field)
        self._store.delete_metadata_field(metadata_field_id)
