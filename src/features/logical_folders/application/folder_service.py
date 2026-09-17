"""Create/reuse logical folders from real key-field extraction results."""

from __future__ import annotations

import logging
from typing import Any

from src.features.logical_folders.application.hierarchy import (
    catalog_key_fields,
    extracted_field_names,
    hierarchy_levels_from_extracted_fields,
    hierarchy_values_from_extraction,
    parse_folder_hierarchy,
)
from src.features.logical_folders.domain.exceptions import (
    FolderAssignmentSkipped,
    FolderHierarchyConfigurationError,
    LogicalFolderError,
    LogicalFolderNotFound,
)
from src.features.logical_folders.infrastructure.folder_repository import (
    LogicalFolderStore,
    get_logical_folder_store,
)

logger = logging.getLogger(__name__)


def _document_type_fields_for_repository(repository_id: str) -> list[dict[str, Any]]:
    try:
        from src.features.document_types.application.document_type_service import DocumentTypeService
        from src.features.document_types.infrastructure.document_type_repository import get_document_type_store
    except Exception:
        return []
    store = get_document_type_store()
    if store is None:
        return []
    fields: list[dict[str, Any]] = []
    try:
        types = store.list_types_for_repository(repository_id)
    except Exception:
        logger.debug("Could not list document types for folder hierarchy", exc_info=True)
        return []
    service = DocumentTypeService()
    for record in types:
        type_id = getattr(record, "document_type_id", None)
        if not type_id:
            continue
        try:
            bundle = service.resolve_effective_fields(str(type_id))
        except Exception:
            continue
        for item in bundle.get("fields") or []:
            if isinstance(item, dict):
                fields.append(item)
    return fields


class LogicalFolderService:
    def __init__(self, store: LogicalFolderStore | None = None) -> None:
        self._store = store

    def store(self) -> LogicalFolderStore:
        if self._store is not None:
            return self._store
        store = get_logical_folder_store()
        if store is None:
            raise LogicalFolderError("Logical folder store is unavailable.")
        return store

    def assign_from_extraction(
        self,
        *,
        repository_id: str,
        document_id: str,
        settings: dict[str, Any] | None,
        extracted_fields: Any,
        document_type_fields: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        hierarchy_ids = parse_folder_hierarchy(settings)
        type_fields = document_type_fields
        if type_fields is None:
            type_fields = _document_type_fields_for_repository(repository_id)
        catalog = catalog_key_fields(settings=settings, document_type_fields=type_fields)

        if hierarchy_ids:
            levels = hierarchy_values_from_extraction(
                hierarchy_ids=hierarchy_ids,
                catalog=catalog,
                extracted_fields=extracted_fields,
            )
            require_all_values = True
        else:
            levels = hierarchy_levels_from_extracted_fields(extracted_fields)
            if catalog:
                for level in levels:
                    name = str(level.get("key_field_name") or "").strip().lower()
                    record = catalog.get(f"name:{name}") if name else None
                    if isinstance(record, dict) and record.get("field_id"):
                        level["key_field_id"] = record.get("field_id")
            require_all_values = False
            if not levels:
                logger.info(
                    "logical_folder_assignment_skipped repository_id=%s document_id=%s reason=no_usable_extracted_values",
                    repository_id,
                    document_id,
                )
                return {"status": "skipped", "reason": "no_usable_extracted_values"}

        return self._assign_levels(
            repository_id=repository_id,
            document_id=document_id,
            levels=levels,
            require_all_values=require_all_values,
        )

    def _assign_levels(
        self,
        *,
        repository_id: str,
        document_id: str,
        levels: list[dict[str, Any]],
        require_all_values: bool,
    ) -> dict[str, Any]:
        parent_id: str | None = None
        path_parts: list[str] = []
        created: list[dict[str, Any]] = []
        store = self.store()
        for level in levels:
            folder_name = level.get("folder_name")
            if not folder_name:
                logger.info(
                    "logical_folder_assignment_skipped repository_id=%s document_id=%s key_field_id=%s key_field_name=%s reason=missing_extracted_value",
                    repository_id,
                    document_id,
                    level.get("key_field_id"),
                    level.get("key_field_name"),
                )
                if require_all_values:
                    raise FolderAssignmentSkipped(
                        "Configured folder hierarchy field has no extracted value.",
                        details={
                            "repository_id": repository_id,
                            "document_id": document_id,
                            "key_field_id": level.get("key_field_id"),
                            "key_field_name": level.get("key_field_name"),
                        },
                    )
                continue
            path_parts.append(str(folder_name))
            folder = store.get_or_create_child(
                repository_id=repository_id,
                parent_folder_id=parent_id,
                name=str(folder_name),
                path="/".join(path_parts),
                metadata={
                    "key_field_id": level.get("key_field_id"),
                    "key_field_name": level.get("key_field_name"),
                },
            )
            logger.info(
                "logical_folder_ensured repository_id=%s document_id=%s key_field_id=%s key_field_name=%s parent_folder_id=%s folder_id=%s",
                repository_id,
                document_id,
                level.get("key_field_id"),
                level.get("key_field_name"),
                parent_id,
                folder.get("folder_id"),
            )
            created.append(folder)
            parent_id = folder.get("folder_id")

        if not created:
            return {"status": "skipped", "reason": "no_usable_extracted_values"}

        leaf = created[-1]
        assignment = store.assign_document(
            document_id=document_id,
            folder_id=str(leaf["folder_id"]),
            repository_id=repository_id,
        )
        result = {
            "status": "assigned",
            "repository_id": repository_id,
            "document_id": document_id,
            "folder_id": assignment["folder_id"],
            "path": leaf.get("path"),
            "folders": created,
        }
        self._persist_assignment_metadata(document_id, result)
        return result

    def list_folders(self, repository_id: str) -> dict[str, Any]:
        folders = self.store().list_by_repository(repository_id)
        by_id = {item["folder_id"]: {**item, "children": []} for item in folders}
        roots: list[dict[str, Any]] = []
        for item in by_id.values():
            parent_id = item.get("parent_folder_id")
            if parent_id and parent_id in by_id:
                by_id[parent_id]["children"].append(item)
            else:
                roots.append(item)
        return {
            "repository_id": repository_id,
            "folders": folders,
            "tree": roots,
            "count": len(folders),
        }

    def list_folder_documents(self, repository_id: str, folder_id: str) -> dict[str, Any]:
        store = self.store()
        folder = store.get_by_id(folder_id, repository_id=repository_id)
        if folder is None:
            raise LogicalFolderNotFound(
                "Logical folder not found.",
                details={"repository_id": repository_id, "folder_id": folder_id},
            )
        document_ids = store.list_document_ids(repository_id, folder_id)
        from src.features.repositories.application.repository_service import RepositoryService

        documents = RepositoryService._repository_document_entries(document_ids)
        return {
            "repository_id": repository_id,
            "folder": folder,
            "documents": documents,
            "document_count": len(documents),
        }

    def require_folders_in_repository(
        self,
        repository_id: str,
        folder_ids: list[str],
    ) -> list[str]:
        """Return unique folder IDs after verifying each belongs to ``repository_id``.

        Missing IDs and folders owned by another repository both raise
        ``LogicalFolderNotFound`` so callers cannot probe cross-repository existence.
        """
        unique_ids = list(dict.fromkeys(str(item).strip() for item in folder_ids if str(item).strip()))
        if not unique_ids:
            return []
        store = self.store()
        missing: list[str] = []
        for folder_id in unique_ids:
            if store.get_by_id(folder_id, repository_id=repository_id) is None:
                missing.append(folder_id)
        if missing:
            raise LogicalFolderNotFound(
                "Logical folder not found in this repository.",
                details={"repository_id": repository_id, "folder_ids": missing},
            )
        return unique_ids

    def assignment_for_document(self, document_id: str) -> dict[str, Any] | None:
        try:
            return self.store().get_assignment(document_id)
        except LogicalFolderError:
            return None

    @staticmethod
    def _persist_assignment_metadata(document_id: str, result: dict[str, Any]) -> None:
        try:
            from src.features.configuration.platform_settings import get_settings
            from src.features.documents.infrastructure.document_metadata_repository import MetadataStore

            MetadataStore(get_settings().db_path).update_document_metadata(
                document_id,
                {
                    "logical_folder": {
                        "folder_id": result.get("folder_id"),
                        "path": result.get("path"),
                        "status": result.get("status"),
                    }
                },
                merge=True,
            )
        except Exception:
            logger.debug("Could not persist logical folder metadata on intake record", exc_info=True)


def get_logical_folder_service() -> LogicalFolderService:
    return LogicalFolderService()


def assign_document_folders_after_extraction(
    *,
    document_id: str,
    repository_id: str | None,
    document_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Pipeline hook: consume real key-field extraction output and assign folders."""
    if not repository_id:
        try:
            from src.features.repositories.application.repository_service import get_repository_service

            repository_id = get_repository_service().get_repository_id_for_document(document_id)
        except Exception:
            repository_id = None
    if not repository_id:
        logger.info("logical_folder_assignment_skipped document_id=%s reason=no_repository", document_id)
        return None
    try:
        from src.features.repositories.application.repository_service import get_repository_service

        resolved = get_repository_service().resolve_settings(repository_id)
        settings = resolved.get("settings") if isinstance(resolved, dict) else resolved
        if not isinstance(settings, dict):
            settings = {}
    except Exception:
        logger.exception(
            "logical_folder_settings_load_failed repository_id=%s document_id=%s",
            repository_id,
            document_id,
        )
        raise

    extracted = None
    meta = document_metadata if isinstance(document_metadata, dict) else {}
    extracted = meta.get("extracted_fields")
    if extracted is None:
        extracted = (meta.get("artifacts") or {}).get("extracted_fields") if isinstance(meta.get("artifacts"), dict) else None
    if extracted is None:
        extracted = (meta.get("artifacts") or {}).get("fields") if isinstance(meta.get("artifacts"), dict) else None

    field_names = extracted_field_names(extracted)
    if field_names:
        try:
            from src.features.repositories.application.repository_service import get_repository_service

            get_repository_service().ensure_discovered_key_fields(repository_id, field_names)
            resolved = get_repository_service().resolve_settings(repository_id)
            settings = resolved.get("settings") if isinstance(resolved, dict) else resolved
            if not isinstance(settings, dict):
                settings = {}
        except Exception:
            logger.exception(
                "logical_folder_key_field_persist_failed repository_id=%s document_id=%s",
                repository_id,
                document_id,
            )

    try:
        return get_logical_folder_service().assign_from_extraction(
            repository_id=repository_id,
            document_id=document_id,
            settings=settings,
            extracted_fields=extracted,
        )
    except FolderAssignmentSkipped as exc:
        logger.info(
            "logical_folder_assignment_skipped repository_id=%s document_id=%s details=%s",
            repository_id,
            document_id,
            exc.details,
        )
        return {"status": "skipped", "reason": "missing_extracted_value", "details": exc.details}
    except FolderHierarchyConfigurationError:
        logger.exception(
            "logical_folder_hierarchy_invalid repository_id=%s document_id=%s",
            repository_id,
            document_id,
        )
        raise
