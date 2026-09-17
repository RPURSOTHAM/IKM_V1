"""FastAPI routes for repository-scoped document types and key fields."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.features.document_types.domain.document_type_exceptions import DocumentTypeError
from src.features.document_types.schemas.document_type_schemas import (
    CreateDocumentTypeRequest,
    CreateKeyFieldRequest,
    UpdateDocumentTypeRequest,
    UpdateKeyFieldRequest,
)
from src.features.document_types.application.document_type_service import DocumentTypeService, get_document_type_service

router = APIRouter(tags=["Document Types"])


def _svc() -> DocumentTypeService:
    return get_document_type_service()


def _doc_type_error(exc: DocumentTypeError) -> HTTPException:
    return HTTPException(
        status_code=exc.http_status,
        detail={"code": exc.code, "message": exc.message, "details": exc.details},
    )


@router.post(
    "/repositories/{repository_id}/document-types",
    status_code=201,
    summary="Create document type in repository",
    operation_id="createRepositoryDocumentType",
)
def create_repository_document_type(
    repository_id: str,
    body: CreateDocumentTypeRequest,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.create_repository_document_type(
            repository_id,
            body.model_dump(exclude_none=True),
        )
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.get(
    "/repositories/{repository_id}/document-types",
    summary="List document types for repository",
    operation_id="listRepositoryDocumentTypes",
)
def list_repository_document_types(
    repository_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.list_repository_document_types(repository_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.get(
    "/repositories/{repository_id}/document-types/tree",
    summary="Get document type hierarchy tree for repository",
    operation_id="getRepositoryDocumentTypeTree",
)
def get_repository_document_type_tree(
    repository_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.get_repository_document_type_tree(repository_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.get(
    "/repositories/{repository_id}/document-types/{document_type_id}",
    summary="Get document type in repository",
    operation_id="getRepositoryDocumentType",
)
def get_repository_document_type(
    repository_id: str,
    document_type_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.get_type(document_type_id, repository_id=repository_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.patch(
    "/repositories/{repository_id}/document-types/{document_type_id}",
    summary="Update document type in repository",
    operation_id="patchRepositoryDocumentType",
)
def patch_repository_document_type(
    repository_id: str,
    document_type_id: str,
    body: UpdateDocumentTypeRequest,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        # Enforce repository scoping before mutation.
        service.get_type(document_type_id, repository_id=repository_id)
        return service.update_type(document_type_id, body.model_dump(exclude_none=True))
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.delete(
    "/repositories/{repository_id}/document-types/{document_type_id}",
    status_code=204,
    summary="Delete document type in repository",
    operation_id="deleteRepositoryDocumentType",
)
def delete_repository_document_type(
    repository_id: str,
    document_type_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> None:
    try:
        service.initialize()
        service.get_type(document_type_id, repository_id=repository_id)
        service.delete_type(document_type_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.get(
    "/document-types/{document_type_id}",
    summary="Get document type by id",
    operation_id="getDocumentType",
)
def get_document_type(
    document_type_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.get_type(document_type_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.patch(
    "/document-types/{document_type_id}",
    summary="Update document type",
    operation_id="patchDocumentType",
)
def patch_document_type(
    document_type_id: str,
    body: UpdateDocumentTypeRequest,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.update_type(document_type_id, body.model_dump(exclude_none=True))
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.put(
    "/document-types/{document_type_id}",
    summary="Update document type",
    operation_id="updateDocumentType",
)
def update_document_type(
    document_type_id: str,
    body: UpdateDocumentTypeRequest,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.update_type(document_type_id, body.model_dump(exclude_none=True))
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.delete(
    "/document-types/{document_type_id}",
    status_code=204,
    summary="Delete document type",
    operation_id="deleteDocumentType",
)
def delete_document_type(
    document_type_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> None:
    try:
        service.initialize()
        service.delete_type(document_type_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.post(
    "/document-types/{document_type_id}/fields",
    status_code=201,
    summary="Create key field on document type",
    operation_id="createDocumentTypeKeyField",
)
def create_document_type_key_field(
    document_type_id: str,
    body: CreateKeyFieldRequest,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.create_key_field(document_type_id, body.model_dump(exclude_none=True))
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.get(
    "/document-types/{document_type_id}/fields",
    summary="List key fields defined on document type",
    operation_id="listDocumentTypeKeyFields",
)
def list_document_type_key_fields(
    document_type_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.list_key_fields(document_type_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.get(
    "/document-types/{document_type_id}/effective-fields",
    summary="Resolve inherited key fields for document type",
    operation_id="getDocumentTypeEffectiveFields",
)
def get_document_type_effective_fields(
    document_type_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.resolve_effective_fields(document_type_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.put(
    "/fields/{field_id}",
    summary="Update key field definition",
    operation_id="updateKeyField",
)
def update_key_field(
    field_id: str,
    body: UpdateKeyFieldRequest,
    service: DocumentTypeService = Depends(_svc),
) -> dict[str, Any]:
    try:
        service.initialize()
        return service.update_key_field(field_id, body.model_dump(exclude_none=True))
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc


@router.delete(
    "/fields/{field_id}",
    status_code=204,
    summary="Delete key field definition",
    operation_id="deleteKeyField",
)
def delete_key_field(
    field_id: str,
    service: DocumentTypeService = Depends(_svc),
) -> None:
    try:
        service.initialize()
        service.delete_key_field(field_id)
    except DocumentTypeError as exc:
        raise _doc_type_error(exc) from exc
