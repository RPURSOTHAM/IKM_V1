"""Exceptions for the template_compliance component."""

from __future__ import annotations


class TemplateComplianceError(Exception):
    """Base error for template extraction."""


class UnsupportedFormatError(TemplateComplianceError):
    """Raised when the file type is not supported yet."""


class ExtractionError(TemplateComplianceError):
    """Raised when extraction fails for a supported file."""


class TemplateFileNotFoundError(TemplateComplianceError):
    """Raised when the input path does not exist."""
