from __future__ import annotations

from typing import Any


class LogicalFolderError(Exception):
    code: str = "logical_folder_error"
    http_status: int = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class FolderHierarchyConfigurationError(LogicalFolderError):
    code = "invalid_folder_hierarchy"
    http_status = 422


class FolderAssignmentSkipped(LogicalFolderError):
    """Configured hierarchy exists but a required extracted value is absent."""

    code = "folder_assignment_skipped"
    http_status = 200


class LogicalFolderNotFound(LogicalFolderError):
    code = "not_found"
    http_status = 404
